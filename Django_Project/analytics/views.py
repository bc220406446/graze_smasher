"""
Graze Smasher Product Hunting Tool - Analytics, Dashboards & Alerts Views
CS619 Final-Year Project
"""

import os
import json
import pandas as pd
from datetime import timedelta
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST
from django.contrib import messages
from django.db.models import Avg, Count, Q
from django.utils import timezone
from products.models import Product, SavedProduct
from reviews.models import Review
from .models import Alert
from ML_Model.config import SCREENSHOTS_DIR


@login_required
def user_dashboard(request):
    """
    Personalized user dashboard displaying:
    - Watchlist of saved products with authenticity scores and trust ratings
    - User's review submission history with ML classification scores
    - Unread system alerts for monitored products
    - Summary activity statistics
    """
    user = request.user

    # 1. Saved Products
    saved_items = SavedProduct.objects.filter(user=user).select_related('product')
    saved_products = [item.product for item in saved_items]

    # Calculate average authenticity score of saved items
    if saved_products:
        avg_saved_score = round(
            sum(p.overall_authenticity_score() for p in saved_products) / len(saved_products), 1
        )
    else:
        avg_saved_score = 0.0

    # 2. Reviews written by user
    user_reviews = Review.objects.filter(user=user).select_related('product').order_by('-created_at')

    # 3. Alerts: Alerts on products the user created or saved
    saved_product_ids = [p.id for p in saved_products]
    alerts_qs = Alert.objects.filter(
        Q(user=user) | Q(product_id__in=saved_product_ids)
    ).select_related('product').order_by('-created_at')

    unread_alerts_count = alerts_qs.filter(is_read=False).count()
    recent_alerts = alerts_qs[:10]

    context = {
        'saved_products': saved_products,
        'saved_count': len(saved_products),
        'avg_saved_score': avg_saved_score,
        'user_reviews': user_reviews,
        'reviews_count': user_reviews.count(),
        'recent_alerts': recent_alerts,
        'unread_alerts_count': unread_alerts_count,
    }
    return render(request, 'analytics/user_dashboard.html', context)


def analytics_overview(request):
    """
    Comprehensive platform analytics dashboard powered by Chart.js:
    - ML model benchmark evaluation (reads saved model_comparison_table.csv)
    - Catalog authenticity distribution (Genuine / Suspicious / Fake)
    - Platform comparison (Average authenticity across Amazon, eBay, Walmart, etc.)
    - Fake review trends over time
    - Leaderboards: Most Manipulated Products vs Most Authentic Products
    """
    # 1. Read ML model comparison results from CSV
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

    # 2. Global Catalog Metrics
    total_products = Product.objects.count()
    total_reviews = Review.objects.count()
    fake_reviews_count = Review.objects.filter(label='Fake').count()
    suspicious_reviews_count = Review.objects.filter(label='Suspicious').count()
    genuine_reviews_count = Review.objects.filter(label='Genuine').count()

    overall_fake_pct = round((fake_reviews_count / total_reviews * 100), 1) if total_reviews > 0 else 0.0
    overall_genuine_pct = round((genuine_reviews_count / total_reviews * 100), 1) if total_reviews > 0 else 0.0
    overall_suspicious_pct = round((suspicious_reviews_count / total_reviews * 100), 1) if total_reviews > 0 else 0.0

    # 3. Platform Breakdown for Chart.js
    platform_stats = Product.objects.values('platform').annotate(
        avg_auth=Avg('reviews__final_score'),
        product_count=Count('id', distinct=True),
        review_count=Count('reviews')
    ).order_by('-avg_auth')

    platform_labels = [p['platform'] for p in platform_stats]
    platform_scores = [round(float(p['avg_auth'] or 0.0), 1) for p in platform_stats]

    # 4. Review Distribution Data for Doughnut Chart
    doughnut_labels = ['Genuine Reviews', 'Suspicious Reviews', 'Fake / Generated Reviews']
    doughnut_counts = [genuine_reviews_count, suspicious_reviews_count, fake_reviews_count]

    # 5. ML Benchmark Data for Chart.js
    model_names = [m.get('Model', '') for m in ml_benchmarks]
    model_accuracies = [m.get('Accuracy', 0) for m in ml_benchmarks]
    model_f1_scores = [m.get('F1-Score', 0) for m in ml_benchmarks]
    model_cv_scores = [m.get('5-Fold CV Accuracy', 0) for m in ml_benchmarks]

    # 6. Top Manipulated vs Top Authentic Products
    all_products = list(Product.objects.all())
    # Sort by overall authenticity score
    products_by_score = sorted(all_products, key=lambda p: p.overall_authenticity_score())
    top_manipulated = products_by_score[:5]  # Lowest score
    top_authentic = list(reversed(products_by_score))[:5]  # Highest score

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
        # JSON serialized for Chart.js
        'chart_doughnut_labels': json.dumps(doughnut_labels),
        'chart_doughnut_data': json.dumps(doughnut_counts),
        'chart_platform_labels': json.dumps(platform_labels),
        'chart_platform_scores': json.dumps(platform_scores),
        'chart_model_names': json.dumps(model_names),
        'chart_model_accuracies': json.dumps(model_accuracies),
        'chart_model_f1': json.dumps(model_f1_scores),
        'chart_model_cv': json.dumps(model_cv_scores),
    }
    return render(request, 'analytics/analytics_overview.html', context)


def suspicious_products_discovery(request):
    """
    Discovery page highlighting manipulated, suspicious, and high-risk products:
    - Products with high fake review proportions
    - Products with large rating-trust discrepancies (star inflation)
    - Products with active anomaly alerts
    """
    all_products = Product.objects.annotate(
        total_revs=Count('reviews'),
        avg_score=Avg('reviews__final_score')
    ).filter(total_revs__gt=0)

    # Filter tab
    tab = request.GET.get('tab', 'manipulated')

    if tab == 'alerts':
        # Products that have active alerts
        alert_product_ids = Alert.objects.values_list('product_id', flat=True).distinct()
        products_list = all_products.filter(id__in=alert_product_ids).order_by('avg_score')
    elif tab == 'trusted':
        # Products with score >= 70%
        products_list = all_products.filter(avg_score__gte=70.0).order_by('-avg_score')
    else:
        # Default: Manipulated / Suspicious products (score < 65%)
        products_list = all_products.filter(avg_score__lt=70.0).order_by('avg_score')

    # Total active alerts in system
    recent_alerts = Alert.objects.select_related('product')[:8]

    context = {
        'products': products_list,
        'tab': tab,
        'recent_alerts': recent_alerts,
        'total_flagged': products_list.count(),
    }
    return render(request, 'analytics/suspicious_discovery.html', context)


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
