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
from products.models import Product, SavedProduct, ProductFlag
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
        'total_alerts_count': alerts_qs.count(),
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
        'recent_alerts': page_obj,
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
    return render(request, 'includes/alerts.html', context)


@login_required
def user_my_reports(request):
    """
    My Reports View in User Dashboard:
    Displays the list of product reports submitted by the logged-in user.
    Accessible to General Users and Admins; Reviewers do not have access.
    """
    profile = getattr(request.user, 'profile', None)
    if profile and profile.role == 'reviewer' and not (request.user.is_staff or request.user.is_superuser or profile.is_admin):
        messages.error(request, "Access restricted: Reviewers do not have access to the product reporting page.")
        return redirect('reviewer_dashboard')

    user = request.user
    flag_status = request.GET.get('flag_status', 'all').strip().lower()

    user_flags_qs = ProductFlag.objects.filter(user=user).select_related('product').order_by('-created_at')

    total_reports_count = user_flags_qs.count()
    pending_reports_count = user_flags_qs.filter(status=ProductFlag.STATUS_PENDING).count()
    resolved_reports_count = user_flags_qs.filter(status=ProductFlag.STATUS_RESOLVED).count()

    if flag_status == 'pending':
        user_flags_qs = user_flags_qs.filter(status=ProductFlag.STATUS_PENDING)
    elif flag_status == 'resolved':
        user_flags_qs = user_flags_qs.filter(status=ProductFlag.STATUS_RESOLVED)
    elif flag_status == 'dismissed':
        user_flags_qs = user_flags_qs.filter(status=ProductFlag.STATUS_DISMISSED)

    paginator = Paginator(user_flags_qs, 15)
    page_obj = paginator.get_page(request.GET.get('page'))

    context = {
        'page_obj': page_obj,
        'product_flags': page_obj,
        'flag_status': flag_status,
        'total_reports_count': total_reports_count,
        'pending_reports_count': pending_reports_count,
        'resolved_reports_count': resolved_reports_count,
    }
    return render(request, 'includes/my_reports.html', context)



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

    # Anomaly alerts for reviewed, saved or created products
    saved_product_ids = list(SavedProduct.objects.filter(user=user).values_list('product_id', flat=True))
    created_product_ids = list(Product.objects.filter(created_by=user).values_list('id', flat=True))
    relevant_ids = list(set(list(reviewed_product_ids) + saved_product_ids + created_product_ids))
    alerts_qs = Alert.objects.filter(
        Q(user=user) | Q(user__isnull=True, product_id__in=relevant_ids)
    ).select_related('product', 'user').order_by('-created_at')

    recent_alerts = alerts_qs[:8]
    total_alerts_count = alerts_qs.count()

    context = {
        'total_submitted': total_submitted,
        'genuine_count': genuine_count,
        'fake_count': fake_count,
        'products_reviewed_count': products_reviewed_count,
        'recent_reviews': recent_reviews,
        'recent_alerts': recent_alerts,
        'total_alerts_count': total_alerts_count,
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
    rating_filter = request.GET.get('rating', '').strip()
    sort_filter = request.GET.get('sort', 'newest').strip()

    reviews_qs = Review.objects.filter(user=user).select_related('product')

    if q:
        reviews_qs = reviews_qs.filter(
            Q(text__icontains=q) | Q(product__name__icontains=q)
        )
    if label_filter in ['Genuine', 'Fake', 'Suspicious']:
        reviews_qs = reviews_qs.filter(label=label_filter)
    if rating_filter.isdigit():
        reviews_qs = reviews_qs.filter(rating=int(rating_filter))

    if sort_filter == 'oldest':
        reviews_qs = reviews_qs.order_by('created_at')
    elif sort_filter == 'score_high':
        reviews_qs = reviews_qs.order_by('-final_score')
    elif sort_filter == 'score_low':
        reviews_qs = reviews_qs.order_by('final_score')
    elif sort_filter == 'rating_high':
        reviews_qs = reviews_qs.order_by('-rating', '-created_at')
    elif sort_filter == 'rating_low':
        reviews_qs = reviews_qs.order_by('rating', '-created_at')
    else:
        reviews_qs = reviews_qs.order_by('-created_at')

    paginator = Paginator(reviews_qs, 10)
    page_obj = paginator.get_page(request.GET.get('page'))

    context = {
        'page_obj': page_obj,
        'q': q,
        'label_filter': label_filter,
        'rating_filter': rating_filter,
        'sort_filter': sort_filter,
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

