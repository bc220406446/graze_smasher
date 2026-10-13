"""
Graze Smasher Product Hunting Tool - Custom Administrator Control Panel
CS619 Final-Year Project

Provides administrator features:
- User and RBAC Role Management
- Flagged Content Moderation (Confirm / Override Labels)
- Live Inspection of ML Model Performance & Serialization Status
- Real-Time Runtime Configuration of Detection Weights, Thresholds & Rules
"""

import os
import pandas as pd
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from django.views.decorators.http import require_POST
from django.contrib.auth.models import User
from django.db.models import Count, Q
from .decorators import admin_required
from .models import UserProfile
from .admin_forms import SystemConfigurationForm
from reviews.models import Review
from products.models import Product
from analytics.models import Alert, SystemConfiguration
from ML_Model.config import (
    SCREENSHOTS_DIR,
    SVM_MODEL_PATH,
    VECTORIZER_PATH,
    J48_MODEL_PATH,
    DATASET_PATH,
)


@admin_required
def admin_control_panel(request):
    """Admin Dashboard overview."""
    total_users = User.objects.count()
    total_products = Product.objects.count()
    total_reviews = Review.objects.count()
    flagged_reviews_count = Review.objects.filter(Q(label='Fake') | Q(label='Suspicious') | Q(ip_flag=True)).count()
    active_alerts_count = Alert.objects.filter(is_read=False).count()

    system_config = SystemConfiguration.get_solo()

    context = {
        'total_users': total_users,
        'total_products': total_products,
        'total_reviews': total_reviews,
        'flagged_reviews_count': flagged_reviews_count,
        'active_alerts_count': active_alerts_count,
        'system_config': system_config,
    }
    return render(request, 'accounts/admin/control_panel.html', context)


@admin_required
def admin_users_view(request):
    """Manage platform user accounts and RBAC role assignments."""
    search = request.GET.get('q', '').strip()
    role_filter = request.GET.get('role', '').strip()

    users_qs = User.objects.select_related('profile').order_by('-date_joined')

    if search:
        users_qs = users_qs.filter(Q(username__icontains=search) | Q(email__icontains=search))
    if role_filter:
        users_qs = users_qs.filter(profile__role=role_filter)

    context = {
        'users_list': users_qs,
        'search': search,
        'role_filter': role_filter,
        'role_choices': UserProfile.ROLE_CHOICES,
    }
    return render(request, 'accounts/admin/users_list.html', context)


@admin_required
@require_POST
def admin_update_user_role(request, user_id):
    """Updates the role of a user."""
    target_user = get_object_or_404(User, id=user_id)
    new_role = request.POST.get('role', '').strip()

    if new_role in [UserProfile.ROLE_GENERAL, UserProfile.ROLE_REVIEWER, UserProfile.ROLE_ADMIN]:
        profile = target_user.profile
        profile.role = new_role
        profile.save()

        # Update staff status if promoted to admin
        if new_role == UserProfile.ROLE_ADMIN:
            target_user.is_staff = True
        elif not target_user.is_superuser:
            target_user.is_staff = False
        target_user.save()

        messages.success(request, f"Updated role for '{target_user.username}' to {profile.get_role_display()}.")
    else:
        messages.error(request, "Invalid role specified.")

    return redirect('admin_users')


@admin_required
def admin_flagged_content_view(request):
    """Moderation queue for reviews flagged by ML models or IP velocity."""
    label_filter = request.GET.get('label', '').strip().capitalize()
    ip_only = request.GET.get('ip_burst', '')

    flagged_qs = Review.objects.select_related('product', 'user').filter(
        Q(label='Fake') | Q(label='Suspicious') | Q(ip_flag=True)
    ).order_by('-created_at')

    if label_filter in ['Fake', 'Suspicious', 'Genuine']:
        flagged_qs = flagged_qs.filter(label=label_filter)
    if ip_only == '1':
        flagged_qs = flagged_qs.filter(ip_flag=True)

    context = {
        'reviews': flagged_qs,
        'label_filter': label_filter,
        'ip_only': ip_only,
        'total_flagged': flagged_qs.count(),
    }
    return render(request, 'accounts/admin/flagged_reviews.html', context)


@admin_required
@require_POST
def admin_override_review_label(request, review_id):
    """Administrator manually overrides or confirms ML classification label."""
    review = get_object_or_404(Review, id=review_id)
    new_label = request.POST.get('new_label', '').strip()

    if new_label in ['Genuine', 'Suspicious', 'Fake']:
        old_label = review.label
        review.label = new_label
        # If admin marks as Genuine, adjust final authenticity score up if it was low
        if new_label == 'Genuine' and review.final_score < 70.0:
            review.final_score = 85.0
        elif new_label == 'Fake' and review.final_score >= 40.0:
            review.final_score = 25.0
        review.save()

        messages.success(
            request,
            f"Review #{review.id} label manually changed from '{old_label}' to '{new_label}'."
        )
    else:
        messages.error(request, "Invalid classification label.")

    next_url = request.META.get('HTTP_REFERER', 'admin_flagged_content')
    return redirect(next_url)


@admin_required
def admin_settings_view(request):
    """View and update runtime detection parameters, weights, and rules."""
    config_obj = SystemConfiguration.get_solo()

    if request.method == 'POST':
        form = SystemConfigurationForm(request.POST, instance=config_obj)
        if form.is_valid():
            form.save()
            messages.success(request, "System detection settings updated successfully!")
            return redirect('admin_settings')
    else:
        form = SystemConfigurationForm(instance=config_obj)

    context = {
        'form': form,
        'config_obj': config_obj,
    }
    return render(request, 'accounts/admin/settings.html', context)


@admin_required
def admin_model_metrics_view(request):
    """Inspect serialized model files, hyperparameter specifications, and evaluation benchmarks."""
    csv_path = os.path.join(SCREENSHOTS_DIR, "model_comparison_table.csv")
    ml_benchmarks = []
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            rows = []
            for _, r in df.iterrows():
                rows.append({
                    'model': str(r.get('Model', '')),
                    'accuracy': r.get('Accuracy', 0),
                    'precision': r.get('Precision', 0),
                    'recall': r.get('Recall', 0),
                    'f1_score': r.get('F1-Score', 0),
                    'cv_accuracy': r.get('5-Fold CV Accuracy', 0),
                })
            ml_benchmarks = rows
        except Exception:
            ml_benchmarks = []

    model_artifacts = [
        {
            "name": "Calibrated SVM Classifier",
            "file": str(SVM_MODEL_PATH),
            "exists": os.path.exists(SVM_MODEL_PATH),
            "size_kb": round(os.path.getsize(SVM_MODEL_PATH) / 1024, 1) if os.path.exists(SVM_MODEL_PATH) else 0,
            "architecture": "LinearSVC(C=1.0) wrapped in CalibratedClassifierCV(cv=5)",
            "input_type": "TF-IDF Vectorized n-grams (1, 2)",
        },
        {
            "name": "TF-IDF N-Gram Vectorizer",
            "file": str(VECTORIZER_PATH),
            "exists": os.path.exists(VECTORIZER_PATH),
            "size_kb": round(os.path.getsize(VECTORIZER_PATH) / 1024, 1) if os.path.exists(VECTORIZER_PATH) else 0,
            "architecture": "TfidfVectorizer(min_df=2, sublinear_tf=True, ngram_range=(1,2))",
            "input_type": "Preprocessed & Lemmatized Review Text",
        },
        {
            "name": "J-48 Decision Tree Classifier",
            "file": str(J48_MODEL_PATH),
            "exists": os.path.exists(J48_MODEL_PATH),
            "size_kb": round(os.path.getsize(J48_MODEL_PATH) / 1024, 1) if os.path.exists(J48_MODEL_PATH) else 0,
            "architecture": "DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_leaf=5)",
            "input_type": "8 Numerical Behavioral & Sentiment Features",
        },
    ]

    context = {
        'ml_benchmarks': ml_benchmarks,
        'model_artifacts': model_artifacts,
        'dataset_exists': os.path.exists(DATASET_PATH),
    }
    return render(request, 'accounts/admin/model_metrics.html', context)
