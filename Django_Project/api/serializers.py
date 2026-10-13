"""
Graze Smasher Product Hunting Tool - REST API Serializers
CS619 Final-Year Project
"""

from rest_framework import serializers
from products.models import Product
from reviews.models import Review


class ReviewItemInputSerializer(serializers.Serializer):
    """Input schema for individual reviews within an external product batch."""
    text = serializers.CharField(required=True, min_length=3)
    rating = serializers.IntegerField(required=False, default=5, min_value=1, max_value=5)
    ip = serializers.CharField(required=False, allow_blank=True, default=None)


class ProductIngestSerializer(serializers.Serializer):
    """Input schema for third-party platforms sending a product with batch reviews."""
    platform = serializers.CharField(required=True, max_length=50)
    name = serializers.CharField(required=True, max_length=255)
    url = serializers.URLField(required=False, allow_blank=True, default='')
    category = serializers.CharField(required=False, default='General', max_length=100)
    description = serializers.CharField(required=False, allow_blank=True, default='')
    image_url = serializers.URLField(required=False, allow_blank=True, default='')
    reviews = ReviewItemInputSerializer(many=True, required=False, default=[])


class SingleReviewCreateSerializer(serializers.Serializer):
    """Input schema for submitting a single review to an existing product."""
    product_id = serializers.IntegerField(required=True)
    text = serializers.CharField(required=True, min_length=3)
    rating = serializers.IntegerField(required=False, default=5, min_value=1, max_value=5)
    ip = serializers.CharField(required=False, allow_blank=True, default=None)


class ReviewOutputSerializer(serializers.ModelSerializer):
    """Output schema for analyzed reviews."""
    class Meta:
        model = Review
        fields = [
            'id',
            'product',
            'text',
            'rating',
            'ip_address',
            'svm_score',
            'j48_score',
            'sentiment',
            'ip_flag',
            'final_score',
            'label',
            'source',
            'created_at',
        ]


class ProductDetailSerializer(serializers.ModelSerializer):
    """Comprehensive product output including computed authenticity metrics."""
    review_count = serializers.IntegerField(read_only=True)
    average_rating = serializers.FloatField(read_only=True)
    overall_authenticity_score = serializers.FloatField(read_only=True)
    authenticity_label = serializers.CharField(read_only=True)
    community_trust_rating = serializers.FloatField(read_only=True)
    genuine_reviews_count = serializers.IntegerField(read_only=True)
    fake_reviews_count = serializers.IntegerField(read_only=True)
    suspicious_reviews_count = serializers.IntegerField(read_only=True)

    class Meta:
        model = Product
        fields = [
            'id',
            'name',
            'description',
            'category',
            'platform',
            'source_url',
            'image_url',
            'display_image',
            'is_external',
            'created_at',
            'review_count',
            'average_rating',
            'overall_authenticity_score',
            'authenticity_label',
            'community_trust_rating',
            'genuine_reviews_count',
            'fake_reviews_count',
            'suspicious_reviews_count',
        ]
