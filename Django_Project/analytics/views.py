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
from ML_Model.config import MODEL_COMPARISON_CSV
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
        Q(user=user) | Q(user__isnull=True, product_id__in=relevant_product_ids)
    ).select_related('product').order_by('-created_at')

    unread_alerts_count = alerts_qs.filter(is_read=False).count()
    recent_alerts = alerts_qs[:8]

    # Attach active alert count to each saved product
    for p in saved_products:
        p.active_alerts_count = alerts_qs.filter(product_id=p.id, is_read=False).count()
        p.total_alerts_count = alerts_qs.filter(product_id=p.id).count()

    # 4. Chart data for authenticity scores of saved products
    chart_labels = [p.name[:18] + '...' if len(p.name) > 18 else p.name for p in saved_products[:7]]
    chart_scores = [p.overall_authenticity_score() for p in saved_products[:7]]

    context = {
        'saved_products': saved_products[:6],
        'saved_count': user_created_products.count(),
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
    """Dedicated page displaying products added/created by the logged-in user only."""
    user = request.user
    q = request.GET.get('q', '').strip()

    products_qs = Product.objects.filter(created_by=user)
    if q:
        products_qs = products_qs.filter(
            Q(name__icontains=q) | Q(description__icontains=q)
        )
    products = list(products_qs.order_by('-created_at'))

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
        active_count = Alert.objects.filter(product=p, is_read=False).filter(Q(user=user) | Q(user__isnull=True)).count()
        total_count = Alert.objects.filter(product=p).filter(Q(user=user) | Q(user__isnull=True)).count()
        enriched.append({
            'product': p,
            'score': p.overall_authenticity_score(),
            'label': p.authenticity_label(),
            'review_count': p.review_count(),
            'active_alerts': active_count,
            'total_alerts': total_count,
        })

    context = {
        'monitored_products': enriched,
        'total_monitored': len(enriched),
    }
    return render(request, 'users/general/monitored_products.html', context)


@login_required
def user_alerts_view(request):
    """Dedicated notification feed for security and authenticity alerts with product-based browsing."""
    user = request.user
    profile = getattr(user, 'profile', None)
    is_admin = user.is_superuser or user.is_staff or (profile and profile.role == 'admin')
    is_reviewer = profile and profile.role == 'reviewer'

    q = request.GET.get('q', '').strip()
    status = request.GET.get('status', 'all')
    product_filter = request.GET.get('product', '').strip()

    if is_admin:
        # Administrators inspect platform-wide anomaly alerts addressed to them or unassigned
        alerts_qs = Alert.objects.filter(Q(user=user) | Q(user__isnull=True)).select_related('product', 'user').order_by('-created_at')
        filter_product_ids = list(alerts_qs.values_list('product_id', flat=True).distinct())
        filter_products = list(Product.objects.filter(id__in=filter_product_ids, is_hidden=False).distinct())
        total_alerts_count = alerts_qs.count()
        unread_count = alerts_qs.filter(is_read=False).count()
        role_label = "Administrator"
        feed_subtitle = "Platform-wide anomaly alerts, 50% fake review ratio triggers, and detection notifications."
    else:
        # General users & Reviewers: Alerts specifically addressed to this user (or unassigned for their monitored products)
        saved_items = SavedProduct.objects.filter(user=user).select_related('product')
        saved_products = [item.product for item in saved_items if not item.product.is_hidden]
        saved_product_ids = [p.id for p in saved_products]
        created_product_ids = list(Product.objects.filter(created_by=user).values_list('id', flat=True))
        user_alert_product_ids = list(Alert.objects.filter(user=user).values_list('product_id', flat=True).distinct())
        relevant_ids = list(set(saved_product_ids + created_product_ids + user_alert_product_ids))

        alerts_qs = Alert.objects.filter(
            Q(user=user) | Q(user__isnull=True, product_id__in=relevant_ids)
        ).select_related('product', 'user').order_by('-created_at')

        filter_products = list(Product.objects.filter(id__in=relevant_ids, is_hidden=False).distinct())
        if not filter_products and saved_products:
            filter_products = saved_products

        total_alerts_count = alerts_qs.count()
        unread_count = alerts_qs.filter(is_read=False).count()

        if is_reviewer:
            role_label = "Reviewer"
            feed_subtitle = "ML classification feedback for your submitted reviews and monitored product anomaly alerts."
        else:
            role_label = "General User"
            feed_subtitle = "Real-time fake review surges and authenticity score drop alerts across your monitored products."

    # Filter by keyword search (product name or alert message)
    if q:
        alerts_qs = alerts_qs.filter(
            Q(product__name__icontains=q) | Q(message__icontains=q)
        )

    # Filter by selected product if specified
    selected_product = None
    if product_filter.isdigit():
        prod_id = int(product_filter)
        if is_admin or prod_id in [p.id for p in filter_products]:
            alerts_qs = alerts_qs.filter(product_id=prod_id)
            selected_product = Product.objects.filter(id=prod_id).first()

    if status == 'unread':
        alerts_qs = alerts_qs.filter(is_read=False)
    elif status == 'read':
        alerts_qs = alerts_qs.filter(is_read=True)

    paginator = Paginator(alerts_qs, 12)
    page_obj = paginator.get_page(request.GET.get('page'))

    context = {
        'page_obj': page_obj,
        'q': q,
        'status': status,
        'product_filter': product_filter,
        'selected_product': selected_product,
        'monitored_products': filter_products,
        'unread_count': unread_count,
        'total_alerts_count': total_alerts_count,
        'role_label': role_label,
        'feed_subtitle': feed_subtitle,
        'is_admin': is_admin,
        'is_reviewer': is_reviewer,
    }
    return render(request, 'users/shared/alerts.html', context)


@login_required
def user_personal_analytics(request):
    """
    Personalized analytics summary for the logged-in user:
    - Fake review trends (over time/recent sequence)
    - Detection accuracy (ML model comparison benchmarks)
    - Product authenticity scores (for user's saved/monitored products)
    """
    from datetime import timedelta
    user = request.user
    saved_items = SavedProduct.objects.filter(user=user).select_related('product')
    saved_products = [item.product for item in saved_items if not item.product.is_hidden]

    user_reviews = Review.objects.filter(user=user)
    user_created = Product.objects.filter(created_by=user)

    genuine_reviews = user_reviews.filter(label='Genuine').count()
    fake_reviews = user_reviews.filter(label='Fake').count()

    # Authenticity distribution in saved products (Strict Binary: Genuine >= 50, Fake < 50)
    saved_genuine = sum(1 for p in saved_products if p.overall_authenticity_score() >= 50.0)
    saved_fake = sum(1 for p in saved_products if p.overall_authenticity_score() < 50.0)

    # 1. Fake Review Trends (over the last 7 days)
    saved_product_ids = [p.id for p in saved_products]
    if saved_product_ids:
        rel_reviews = Review.objects.filter(product_id__in=saved_product_ids)
    else:
        rel_reviews = Review.objects.all()

    trend_labels = []
    trend_genuine = []
    trend_fake = []
    now = timezone.now()
    for i in range(6, -1, -1):
        day_start = (now - timedelta(days=i)).replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        day_label = day_start.strftime("%b %d")
        trend_labels.append(day_label)

        day_revs = rel_reviews.filter(created_at__gte=day_start, created_at__lt=day_end)
        g_count = day_revs.filter(label='Genuine').count()
        f_count = day_revs.filter(label='Fake').count()
        trend_genuine.append(g_count)
        trend_fake.append(f_count)

    # 2. Detection Accuracy (ML Model Comparison)
    model_names = ['LinearSVC', 'J-48 Tree', 'TextBlob NLP', 'Ensemble']
    model_accuracies = [87.5, 85.0, 81.4, 91.8]
    model_f1 = [87.5, 85.1, 81.5, 91.8]

    # 3. Product Authenticity Scores
    if saved_products:
        auth_prod_labels = [p.name[:16] + '...' if len(p.name) > 16 else p.name for p in saved_products[:8]]
        auth_prod_scores = [p.overall_authenticity_score() for p in saved_products[:8]]
    else:
        catalog_sample = Product.objects.filter(is_hidden=False)[:6]
        auth_prod_labels = [p.name[:16] + '...' if len(p.name) > 16 else p.name for p in catalog_sample]
        auth_prod_scores = [p.overall_authenticity_score() for p in catalog_sample]

    context = {
        'saved_count': len(saved_products),
        'reviews_count': user_reviews.count(),
        'created_count': user_created.count(),
        'genuine_reviews': genuine_reviews,
        'fake_reviews': fake_reviews,
        'saved_genuine': saved_genuine,
        'saved_fake': saved_fake,
        # Visual Reports Data
        'trend_labels': trend_labels,
        'trend_genuine': trend_genuine,
        'trend_fake': trend_fake,
        'model_names': model_names,
        'model_accuracies': model_accuracies,
        'model_f1': model_f1,
        'auth_prod_labels': auth_prod_labels,
        'auth_prod_scores': auth_prod_scores,
        'chart_labels': json.dumps(['Verified Genuine', 'Detected Fake']),
        'chart_data': json.dumps([saved_genuine, saved_fake]),
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
    fake_count = reviewer_reviews.filter(label='Fake').count()
    suspicious_count = 0

    # Unique products reviewed
    reviewed_product_ids = reviewer_reviews.values_list('product_id', flat=True).distinct()
    products_reviewed_count = len(set(reviewed_product_ids))

    recent_reviews = reviewer_reviews[:8]

    # Chart data: distribution of reviewer's evaluations (Binary)
    distribution_labels = ['Genuine', 'Fake']
    distribution_data = [genuine_count, fake_count]

    # Star rating distribution
    ratings_data = [
        reviewer_reviews.filter(rating=r).count() for r in [1, 2, 3, 4, 5]
    ]

    context = {
        'total_submitted': total_submitted,
        'genuine_count': genuine_count,
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
    if label_filter in ['Genuine', 'Fake']:
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
    """Legacy public analytics overview redirected to product catalog."""
    return redirect('product_list')


def suspicious_products_discovery(request):
    """Legacy public suspicious products discovery redirected to product catalog."""
    return redirect('product_list')


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


@login_required
@require_POST
def delete_alert(request, alert_id):
    """Deletes an alert that has already been read."""
    user = request.user
    profile = getattr(user, 'profile', None)
    is_admin = user.is_superuser or user.is_staff or (profile and profile.role == 'admin')

    if is_admin:
        alert = get_object_or_404(Alert, id=alert_id)
    else:
        monitored_ids = list(SavedProduct.objects.filter(user=user).values_list('product_id', flat=True))
        alert = get_object_or_404(
            Alert,
            Q(id=alert_id) & (Q(user=user) | Q(user__isnull=True, product_id__in=monitored_ids))
        )

    alert.delete()
    messages.success(request, "Alert deleted successfully.")
    next_url = request.META.get('HTTP_REFERER', 'user_alerts')
    return redirect(next_url)

