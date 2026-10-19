"""
Provides third-party integration endpoints for receiving external product feeds
and review batches, analyzing authenticity in real time, and retrieving stats.
"""

from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.pagination import PageNumberPagination
from django.shortcuts import get_object_or_404
from django.db.models import Q
from products.models import Product
from reviews.models import Review
from analytics.services import evaluate_product_alerts
from ML_Model.service import analyze_review
from .serializers import (
    ProductIngestSerializer,
    SingleReviewCreateSerializer,
    ProductDetailSerializer,
    ReviewOutputSerializer,
)


class StandardResultsPagination(PageNumberPagination):
    page_size = 10
    page_size_query_param = 'page_size'
    max_page_size = 100


class ProductListCreateAPIView(APIView):
    """
    GET  /api/products/ : List products with authenticity metrics, filters, and search.
    POST /api/products/ : External platform sends product with list of reviews.
    """
    pagination_class = StandardResultsPagination

    def get(self, request):
        queryset = Product.objects.all().order_by('-created_at')

        # Filter by keyword
        q = request.GET.get('q', '').strip()
        if q:
            queryset = queryset.filter(Q(name__icontains=q) | Q(description__icontains=q))

        # Filter by category
        category = request.GET.get('category', '').strip()
        if category:
            queryset = queryset.filter(category=category)

        # Filter by platform
        platform = request.GET.get('platform', '').strip()
        if platform:
            queryset = queryset.filter(platform=platform)

        # Filter by authenticity status
        auth_status = request.GET.get('status', '').strip().lower()
        if auth_status in ['genuine', 'suspicious', 'fake']:
            filtered_ids = [
                p.id for p in queryset if p.authenticity_label().lower() == auth_status
            ]
            queryset = queryset.filter(id__in=filtered_ids)

        paginator = self.pagination_class()
        page = paginator.paginate_queryset(queryset, request)
        serializer = ProductDetailSerializer(page, many=True)
        return paginator.get_paginated_response(serializer.data)

    def post(self, request):
        """
        External platform ingest endpoint:
        Receives: {platform, name, url, category, description, image_url, reviews: [{text, rating, ip}]}
        Saves product, executes analyze_review() on all reviews, and returns summary.
        """
        serializer = ProductIngestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        data = serializer.validated_data

        # 1. Create Product
        product = Product.objects.create(
            name=data['name'],
            platform=data['platform'],
            source_url=data.get('url', ''),
            category=data.get('category', 'General'),
            description=data.get('description', ''),
            image_url=data.get('image_url', ''),
            is_external=True
        )

        # 2. Ingest and Analyze Reviews
        reviews_data = data.get('reviews', [])
        analyzed_reviews = []

        for r_item in reviews_data:
            text = r_item['text']
            rating = r_item.get('rating', 5)
            # Execute unified ML analysis service
            analysis = analyze_review(text=text, rating=rating, ip=None, record_ip=False)

            review = Review.objects.create(
                product=product,
                text=text,
                rating=rating,
                ip_address=None,
                svm_score=analysis['svm_score'],
                j48_score=analysis['j48_score'],
                sentiment=analysis['sentiment_polarity'],
                ip_flag=False,
                final_score=analysis['final_score'],
                label=analysis['label'],
                source='external_api'
            )
            analyzed_reviews.append(review)

        # 3. Evaluate Alerts
        evaluate_product_alerts(product)

        # 4. Construct Response Payload
        review_serializer = ReviewOutputSerializer(analyzed_reviews, many=True)
        product_serializer = ProductDetailSerializer(product)

        return Response({
            "status": "success",
            "message": f"Successfully ingested product and analyzed {len(analyzed_reviews)} reviews.",
            "product": product_serializer.data,
            "reviews": review_serializer.data,
        }, status=status.HTTP_201_CREATED)


class ProductDetailAPIView(APIView):
    """
    GET /api/products/<id>/ : Retrieve single product with full authenticity statistics.
    """
    def get(self, request, pk):
        product = get_object_or_404(Product, pk=pk)
        serializer = ProductDetailSerializer(product)
        return Response(serializer.data, status=status.HTTP_200_OK)


class ProductReviewsAPIView(APIView):
    """
    GET /api/products/<id>/reviews/ : Retrieve paginated review feed for a product.
    Supports ?label=Fake / ?label=Genuine filtering.
    """
    pagination_class = StandardResultsPagination

    def get(self, request, pk):
        product = get_object_or_404(Product, pk=pk)
        reviews_qs = product.reviews.all().order_by('-created_at')

        label = request.GET.get('label', '').strip().capitalize()
        if label in ['Genuine', 'Suspicious', 'Fake']:
            reviews_qs = reviews_qs.filter(label=label)

        paginator = self.pagination_class()
        page = paginator.paginate_queryset(reviews_qs, request)
        serializer = ReviewOutputSerializer(page, many=True)
        return paginator.get_paginated_response(serializer.data)


class SingleReviewCreateAPIView(APIView):
    """
    POST /api/reviews/ : Add a single review to an existing product and obtain ML score.
    """
    def post(self, request):
        serializer = SingleReviewCreateSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        data = serializer.validated_data
        product_id = data['product_id']
        product = get_object_or_404(Product, id=product_id)

        text = data['text']
        rating = data.get('rating', 5)

        # Execute unified ML analysis service
        analysis = analyze_review(text=text, rating=rating, ip=None, record_ip=False)

        user = request.user if request.user.is_authenticated else None

        review = Review.objects.create(
            product=product,
            user=user,
            text=text,
            rating=rating,
            ip_address=None,
            svm_score=analysis['svm_score'],
            j48_score=analysis['j48_score'],
            sentiment=analysis['sentiment_polarity'],
            ip_flag=False,
            final_score=analysis['final_score'],
            label=analysis['label'],
            source='api'
        )

        # Run alert check
        evaluate_product_alerts(product, trigger_review=review)

        review_serializer = ReviewOutputSerializer(review)

        return Response({
            "status": "success",
            "message": f"Review analyzed and recorded. Label: {review.label}",
            "review": review_serializer.data,
            "product_authenticity_score": product.overall_authenticity_score(),
            "product_authenticity_label": product.authenticity_label(),
        }, status=status.HTTP_201_CREATED)

