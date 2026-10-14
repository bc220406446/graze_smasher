"""
Graze Smasher Product Hunting Tool - Analytics, Dashboards & Alerts Views
CS619 Final-Year Project
"""

import os
import json
import pandas as pd
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST
from django.core.paginator import Paginator
from django.contrib import messages
from django.db.models import Avg, Count, Q
from django.utils import timezone
from products.models import Product, SavedProduct
from reviews.models import Review
from .models import Alert
from ML_Model.config import SCREENSHOTS_DIR
from accounts.decorators import reviewer_required


# ==============================================================================
# GENERAL USER DASHBOARD & VIEWS
# ==============================================================================

@login_required
def user_dashboard(request):
    """
    General User Dashboard:
    - Welcome message & prominent 'Add Product' button
    - Summary cards: Saved products, monitored products, recent alerts, created products
    - Saved & monitored products with current authenticity scores and status badges
    - Recent notifications / alerts and analysis activity
    - Authenticity trend chart where historical data exists
    """
    user = request.user

    # 1. Saved Products
    saved_items = SavedProduct.objects.filter(user=user).select_related('product')
    saved_products = [item.product for item in saved_items if not item.product.is_hidden]

    # Calculate average authenticity score of saved items
    if saved_products:
        avg_saved_score = round(
            sum(p.overall_authenticity_score() for p in saved_products) / len(saved_products), 1
        )
    else:
        avg_saved_score = 0.0

    # 2. Products submitted by this user
    user_created_products = Product.objects.filter(created_by=user).order_by('-created_at')

    # 3. Alerts for saved or created products
    saved_product_ids = [p.id for p in saved_products]
    created_product_ids = list(user_created_products.values_list('id', flat=True))
    relevant_product_ids = list(set(saved_product_ids + created_product_ids))

    alerts_qs = Alert.objects.filter(
        Q(user=user) | Q(product_id__in=relevant_product_ids)
    ).select_related('product').order_by('-created_at')

    unread_alerts_count = alerts_qs.filter(is_read=False).count()
    recent_alerts = alerts_qs[:8]

    # 4. Chart data for authenticity scores of saved products
    chart_labels = [p.name[:18] + '...' if len(p.name) > 18 else p.name for p in saved_products[:7]]
    chart_scores = [p.overall_authenticity_score() for p in saved_products[:7]]

    context = {
        'saved_products': saved_products[:6],
        'saved_count': len(saved_products),
        'monitored_count': len(saved_products),
        'avg_saved_score': avg_saved_score,
        'user_created_products': user_created_products[:5],
        'created_count': user_created_products.count(),
        'recent_alerts': recent_alerts,
        'unread_alerts_count': unread_alerts_count,
        'chart_labels': json.dumps(chart_labels),
        'chart_scores': json.dumps(chart_scores),
    }
    return render(request, 'users/general/user_dashboard.html', context)


@login_required
def saved_products_view(request):
    """Dedicated page displaying all saved products in the user's watchlist."""
    user = request.user
    q = request.GET.get('q', '').strip()

    saved_items = SavedProduct.objects.filter(user=user).select_related('product')
    if q:
        saved_items = saved_items.filter(
            Q(product__name__icontains=q) | Q(product__description__icontains=q)
        )

    products = [item.product for item in saved_items if not item.product.is_hidden]

    paginator = Paginator(products, 9)
    page_obj = paginator.get_page(request.GET.get('page'))

    context = {
        'page_obj': page_obj,
        'total_count': len(products),
        'q': q,
    }
    return render(request, 'users/general/saved_products.html', context)


@login_required
def monitored_products_view(request):
    """
    Dedicated view for actively monitored products showing live authenticity scores,
    threat levels, and recent anomaly triggers.
    """
    user = request.user
    saved_items = SavedProduct.objects.filter(user=user).select_related('product')
    monitored = [item.product for item in saved_items if not item.product.is_hidden]

    # Enrich each monitored product with alert count
    enriched = []
    for p in monitored:
        alert_count = Alert.objects.filter(product=p, is_read=False).count()
        enriched.append({
            'product': p,
            'score': p.overall_authenticity_score(),
            'label': p.authenticity_label(),
            'review_count': p.review_count(),
            'active_alerts': alert_count,
        })

    context = {
        'monitored_products': enriched,
        'total_monitored': len(enriched),
    }
    return render(request, 'users/general/monitored_products.html', context)


@login_required
def user_alerts_view(request):
    """Dedicated notification feed for security and authenticity alerts."""
    user = request.user
    status = request.GET.get('status', 'all')

    saved_product_ids = list(SavedProduct.objects.filter(user=user).values_list('product_id', flat=True))
    created_product_ids = list(Product.objects.filter(created_by=user).values_list('id', flat=True))
    relevant_ids = list(set(saved_product_ids + created_product_ids))

    alerts_qs = Alert.objects.filter(
        Q(user=user) | Q(product_id__in=relevant_ids)
    ).select_related('product').order_by('-created_at')

    if status == 'unread':
        alerts_qs = alerts_qs.filter(is_read=False)

    paginator = Paginator(alerts_qs, 12)
    page_obj = paginator.get_page(request.GET.get('page'))

    unread_count = Alert.objects.filter(
        Q(user=user) | Q(product_id__in=relevant_ids),
        is_read=False
    ).count()

    context = {
        'page_obj': page_obj,
        'status': status,
        'unread_count': unread_count,
    }
    return render(request, 'users/general/user_alerts.html', context)


@login_required
def user_personal_analytics(request):
    """Personalized analytics summary for the logged-in user."""
    user = request.user
    saved_items = SavedProduct.objects.filter(user=user).select_related('product')
    saved_products = [item.product for item in saved_items if not item.product.is_hidden]

    user_reviews = Review.objects.filter(user=user)
    user_created = Product.objects.filter(created_by=user)

    genuine_reviews = user_reviews.filter(label='Genuine').count()
    suspicious_reviews = user_reviews.filter(label='Suspicious').count()
    fake_reviews = user_reviews.filter(label='Fake').count()

    # Authenticity distribution in saved products
    genuine_products = sum(1 for p in saved_products if p.overall_authenticity_score() >= 70.0)
    suspicious_products = sum(1 for p in saved_products if 40.0 <= p.overall_authenticity_score() < 70.0)
    fake_products = sum(1 for p in saved_products if p.overall_authenticity_score() < 40.0)

    context = {
        'saved_count': len(saved_products),
        'reviews_count': user_reviews.count(),
        'created_count': user_created.count(),
        'genuine_reviews': genuine_reviews,
        'suspicious_reviews': suspicious_reviews,
        'fake_reviews': fake_reviews,
        'saved_genuine': genuine_products,
        'saved_suspicious': suspicious_products,
        'saved_fake': fake_products,
        'chart_labels': json.dumps(['Genuine', 'Suspicious', 'Manipulated / Fake']),
        'chart_data': json.dumps([genuine_products, suspicious_products, fake_products]),
    }
    return render(request, 'users/general/user_personal_analytics.html', context)


# ==============================================================================
# REVIEWER DASHBOARD & WORKFLOW
# ==============================================================================

@reviewer_required
def reviewer_dashboard(request):
    """
    Dedicated Reviewer Workspace:
    - Total reviews submitted by the reviewer
    - Recently submitted reviews with ML predictions
    - Products reviewed count
    - Review submission activity chart
    """
    user = request.user
    reviewer_reviews = Review.objects.filter(user=user).select_related('product').order_by('-created_at')

    total_submitted = reviewer_reviews.count()
    genuine_count = reviewer_reviews.filter(label='Genuine').count()
    suspicious_count = reviewer_reviews.filter(label='Suspicious').count()
    fake_count = reviewer_reviews.filter(label='Fake').count()

    # Unique products reviewed
    reviewed_product_ids = reviewer_reviews.values_list('product_id', flat=True).distinct()
    products_reviewed_count = len(set(reviewed_product_ids))

    recent_reviews = reviewer_reviews[:8]

    # Chart data: distribution of reviewer's evaluations
    distribution_labels = ['Genuine', 'Suspicious', 'Fake']
    distribution_data = [genuine_count, suspicious_count, fake_count]

    # Star rating distribution
    ratings_data = [
        reviewer_reviews.filter(rating=r).count() for r in [1, 2, 3, 4, 5]
    ]

    context = {
        'total_submitted': total_submitted,
        'genuine_count': genuine_count,
        'suspicious_count': suspicious_count,
        'fake_count': fake_count,
        'products_reviewed_count': products_reviewed_count,
        'recent_reviews': recent_reviews,
        'chart_labels': json.dumps(distribution_labels),
        'chart_data': json.dumps(distribution_data),
        'ratings_data': json.dumps(ratings_data),
    }
    return render(request, 'users/reviewer/reviewer_dashboard.html', context)


@reviewer_required
def reviewer_my_reviews(request):
    """Complete paginated history of reviews submitted by this reviewer."""
    user = request.user
    q = request.GET.get('q', '').strip()
    label_filter = request.GET.get('label', '').strip().capitalize()

    reviews_qs = Review.objects.filter(user=user).select_related('product').order_by('-created_at')

    if q:
        reviews_qs = reviews_qs.filter(
            Q(text__icontains=q) | Q(product__name__icontains=q)
        )
    if label_filter in ['Genuine', 'Suspicious', 'Fake']:
        reviews_qs = reviews_qs.filter(label=label_filter)

    paginator = Paginator(reviews_qs, 10)
    page_obj = paginator.get_page(request.GET.get('page'))

    context = {
        'page_obj': page_obj,
        'q': q,
        'label_filter': label_filter,
        'total_count': reviews_qs.count(),
    }
    return render(request, 'users/reviewer/reviewer_my_reviews.html', context)


@reviewer_required
def reviewer_reviewed_products(request):
    """Catalog of products that this reviewer has contributed reviews for."""
    user = request.user
    product_ids = Review.objects.filter(user=user).values_list('product_id', flat=True).distinct()
    products = Product.objects.filter(id__in=product_ids).order_by('-created_at')

    paginator = Paginator(products, 9)
    page_obj = paginator.get_page(request.GET.get('page'))

    context = {
        'page_obj': page_obj,
        'total_count': products.count(),
    }
    return render(request, 'users/reviewer/reviewer_reviewed_products.html', context)


# ==============================================================================
# GLOBAL PLATFORM ANALYTICS & ALERTS
# ==============================================================================

def analytics_overview(request):
    """
    Comprehensive platform analytics dashboard powered by Chart.js:
    - ML model benchmark evaluation (reads saved model_comparison_table.csv)
    - Catalog authenticity distribution (Genuine / Suspicious / Fake)
    - Platform comparison (Average authenticity across Amazon, eBay, Walmart, etc.)
    - Fake review trends over time
    - Leaderboards: Most Manipulated Products vs Most Authentic Products
    """
    csv_path = os.path.join(SCREENSHOTS_DIR, "model_comparison_table.csv")
    ml_benchmarks = []
    if os.path.exists(csv_path):
        try:
            df_comp = pd.read_csv(csv_path)
            clean_df = df_comp.rename(columns={
                'Model': 'model',
                'Accuracy': 'accuracy',
                'Precision': 'precision',
                'Recall': 'recall',
                'F1-Score': 'f1_score',
                '5-Fold CV Accuracy': 'cv_accuracy'
            })
            ml_benchmarks = clean_df.to_dict(orient="records")
        except Exception:
            ml_benchmarks = []

    total_products = Product.objects.count()
    total_reviews = Review.objects.count()
    fake_reviews_count = Review.objects.filter(label='Fake').count()
    suspicious_reviews_count = Review.objects.filter(label='Suspicious').count()
    genuine_reviews_count = Review.objects.filter(label='Genuine').count()

    overall_fake_pct = round((fake_reviews_count / total_reviews * 100), 1) if total_reviews > 0 else 0.0
    overall_genuine_pct = round((genuine_reviews_count / total_reviews * 100), 1) if total_reviews > 0 else 0.0
    overall_suspicious_pct = round((suspicious_reviews_count / total_reviews * 100), 1) if total_reviews > 0 else 0.0

    platform_stats = Product.objects.values('platform').annotate(
        avg_auth=Avg('reviews__final_score'),
        product_count=Count('id', distinct=True),
        review_count=Count('reviews')
    ).order_by('-avg_auth')

    platform_labels = [p['platform'] for p in platform_stats]
    platform_scores = [round(float(p['avg_auth'] or 0.0), 1) for p in platform_stats]

    doughnut_labels = ['Genuine Reviews', 'Suspicious Reviews', 'Fake / Generated Reviews']
    doughnut_counts = [genuine_reviews_count, suspicious_reviews_count, fake_reviews_count]

    model_names = [m.get('Model', '') for m in ml_benchmarks]
    model_accuracies = [m.get('Accuracy', 0) for m in ml_benchmarks]
    model_f1_scores = [m.get('F1-Score', 0) for m in ml_benchmarks]
    model_cv_scores = [m.get('5-Fold CV Accuracy', 0) for m in ml_benchmarks]

    all_products = list(Product.objects.filter(is_hidden=False))
    products_by_score = sorted(all_products, key=lambda p: p.overall_authenticity_score())
    top_manipulated = products_by_score[:5]
    top_authentic = list(reversed(products_by_score))[:5]

    context = {
        'total_products': total_products,
        'total_reviews': total_reviews,
        'fake_reviews_count': fake_reviews_count,
        'overall_fake_pct': overall_fake_pct,
        'overall_genuine_pct': overall_genuine_pct,
        'overall_suspicious_pct': overall_suspicious_pct,
        'ml_benchmarks': ml_benchmarks,
        'top_manipulated': top_manipulated,
        'top_authentic': top_authentic,
        'chart_doughnut_labels': json.dumps(doughnut_labels),
        'chart_doughnut_data': json.dumps(doughnut_counts),
        'chart_platform_labels': json.dumps(platform_labels),
        'chart_platform_scores': json.dumps(platform_scores),
        'chart_model_names': json.dumps(model_names),
        'chart_model_accuracies': json.dumps(model_accuracies),
        'chart_model_f1': json.dumps(model_f1_scores),
        'chart_model_cv': json.dumps(model_cv_scores),
    }
    return render(request, 'public/analytics_overview.html', context)


def suspicious_products_discovery(request):
    """Discovery page highlighting manipulated, suspicious, and high-risk products."""
    all_products = Product.objects.filter(is_hidden=False).annotate(
        total_revs=Count('reviews'),
        avg_score=Avg('reviews__final_score')
    ).filter(total_revs__gt=0)

    tab = request.GET.get('tab', 'manipulated')

    if tab == 'alerts':
        alert_product_ids = Alert.objects.values_list('product_id', flat=True).distinct()
        products_list = all_products.filter(id__in=alert_product_ids).order_by('avg_score')
    elif tab == 'trusted':
        products_list = all_products.filter(avg_score__gte=70.0).order_by('-avg_score')
    else:
        products_list = all_products.filter(avg_score__lt=70.0).order_by('avg_score')

    recent_alerts = Alert.objects.select_related('product')[:8]

    context = {
        'products': products_list,
        'tab': tab,
        'recent_alerts': recent_alerts,
        'total_flagged': products_list.count(),
    }
    return render(request, 'public/suspicious_discovery.html', context)


@login_required
@require_POST
def mark_alert_read(request, alert_id):
    """Marks an alert as read."""
    alert = get_object_or_404(Alert, id=alert_id)
    alert.is_read = True
    alert.save()
    messages.success(request, "Alert marked as read.")
    next_url = request.META.get('HTTP_REFERER', 'user_dashboard')
    return redirect(next_url)
