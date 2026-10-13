from django.shortcuts import render, get_object_or_404, redirect
from django.core.paginator import Paginator
from django.db.models import Q, Avg, Count
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import Product, SavedProduct
from .forms import ProductForm
from accounts.decorators import reviewer_required


def product_list(request):
    """
    Searchable, filterable, and paginated product catalog showcasing
    real-time ML authenticity scores.
    """
    queryset = Product.objects.annotate(
        avg_score=Avg('reviews__final_score'),
        total_revs=Count('reviews')
    )

    # Search query
    q = request.GET.get('q', '').strip()
    if q:
        queryset = queryset.filter(Q(name__icontains=q) | Q(description__icontains=q))

    # Category filter
    category = request.GET.get('category', '').strip()
    if category:
        queryset = queryset.filter(category=category)

    # Platform filter
    platform = request.GET.get('platform', '').strip()
    if platform:
        queryset = queryset.filter(platform=platform)

    # Authenticity status filter
    status = request.GET.get('status', '').strip().lower()
    if status == 'genuine':
        queryset = queryset.filter(avg_score__gte=70.0)
    elif status == 'suspicious':
        queryset = queryset.filter(avg_score__gte=40.0, avg_score__lt=70.0)
    elif status == 'fake':
        queryset = queryset.filter(avg_score__lt=40.0)

    # Sorting
    sort = request.GET.get('sort', 'newest')
    if sort == 'score_desc':
        queryset = queryset.order_by('-avg_score')
    elif sort == 'score_asc':
        queryset = queryset.order_by('avg_score')
    elif sort == 'reviews_desc':
        queryset = queryset.order_by('-total_revs')
    else:
        queryset = queryset.order_by('-created_at')

    # Pagination: 9 products per page
    paginator = Paginator(queryset, 9)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    # User saved products set
    saved_product_ids = set()
    if request.user.is_authenticated:
        saved_product_ids = set(
            SavedProduct.objects.filter(user=request.user).values_list('product_id', flat=True)
        )

    categories = Product.CATEGORY_CHOICES
    platforms = Product.PLATFORM_CHOICES

    context = {
        'page_obj': page_obj,
        'q': q,
        'category': category,
        'platform': platform,
        'status': status,
        'sort': sort,
        'categories': categories,
        'platforms': platforms,
        'saved_product_ids': saved_product_ids,
        'total_count': paginator.count,
    }
    return render(request, 'products/product_list.html', context)


def product_detail(request, pk):
    """
    Detailed product overview displaying complete review history,
    authenticity breakdown, community trust rating, and review submission form.
    """
    product = get_object_or_404(Product, pk=pk)
    reviews = product.reviews.all()

    # Filter reviews by label if requested
    label_filter = request.GET.get('label', '').strip().capitalize()
    if label_filter in ['Genuine', 'Suspicious', 'Fake']:
        reviews = reviews.filter(label=label_filter)

    total_reviews = product.review_count()
    overall_score = product.overall_authenticity_score()
    overall_label = product.authenticity_label()

    genuine_count = product.genuine_reviews_count()
    fake_count = product.fake_reviews_count()
    suspicious_count = product.suspicious_reviews_count()

    fake_pct = round((fake_count / total_reviews * 100), 1) if total_reviews > 0 else 0.0
    genuine_pct = round((genuine_count / total_reviews * 100), 1) if total_reviews > 0 else 0.0
    suspicious_pct = round((suspicious_count / total_reviews * 100), 1) if total_reviews > 0 else 0.0

    is_saved = False
    if request.user.is_authenticated:
        is_saved = SavedProduct.objects.filter(user=request.user, product=product).exists()

    # Trend timeline over reviews sequence
    import json
    chronological_reviews = list(product.reviews.order_by('created_at'))
    timeline_labels = []
    timeline_scores = []
    timeline_fakes = []
    cum_score_sum = 0
    fake_cum = 0

    for idx, r in enumerate(chronological_reviews, 1):
        timeline_labels.append(f"#{idx}")
        cum_score_sum += r.final_score
        timeline_scores.append(round(cum_score_sum / idx, 1))
        if r.label == 'Fake':
            fake_cum += 1
        timeline_fakes.append(fake_cum)

    context = {
        'product': product,
        'reviews': reviews,
        'label_filter': label_filter,
        'total_reviews': total_reviews,
        'overall_score': overall_score,
        'overall_label': overall_label,
        'genuine_count': genuine_count,
        'fake_count': fake_count,
        'suspicious_count': suspicious_count,
        'fake_pct': fake_pct,
        'genuine_pct': genuine_pct,
        'suspicious_pct': suspicious_pct,
        'trust_rating': product.community_trust_rating(),
        'is_saved': is_saved,
        'chart_labels': json.dumps(timeline_labels),
        'chart_scores': json.dumps(timeline_scores),
        'chart_fakes': json.dumps(timeline_fakes),
    }
    return render(request, 'products/product_detail.html', context)


@login_required
def toggle_save_product(request, pk):
    product = get_object_or_404(Product, pk=pk)
    saved = SavedProduct.objects.filter(user=request.user, product=product).first()
    if saved:
        saved.delete()
        messages.info(request, f"Removed '{product.name}' from your watchlist.")
    else:
        SavedProduct.objects.create(user=request.user, product=product)
        messages.success(request, f"Added '{product.name}' to your watchlist!")
    
    next_url = request.META.get('HTTP_REFERER', 'product_list')
    return redirect(next_url)


@reviewer_required
def create_product(request):
    """Permits Reviewers and Admins to add products to the tracking catalog."""
    if request.method == 'POST':
        form = ProductForm(request.POST, request.FILES)
        if form.is_valid():
            product = form.save(commit=False)
            product.created_by = request.user
            product.save()
            messages.success(request, f"Product '{product.name}' created successfully!")
            return redirect('product_detail', pk=product.pk)
    else:
        form = ProductForm()

    return render(request, 'products/product_form.html', {'form': form})


def privacy_policy_view(request):
    """
    Renders the Privacy and Data Use statement explaining IP detection heuristics,
    data retention, automated machine learning scoring, and ethical protections.
    """
    return render(request, 'products/privacy_policy.html')

