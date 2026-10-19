from datetime import timedelta
from django.utils import timezone
from django.contrib.auth.models import User
from django.db.models import Q
from .models import Alert


def evaluate_product_alerts(product, trigger_review=None):
    """
    Evaluates and generates in-app alerts across the system according to defined rules:

    1. Reviewer Alert:
       - Notifies the reviewer with their review classification ('Fake' or 'Genuine').
       - Triggered on review submission (no cooldown).

    2. Admin & General User Fake Ratio Alerts:
       - Triggered when fake review ratio reaches or exceeds 50% on a product.
       - Admin Alert: Notifies platform administrators (24h cooldown per product).
       - General User Alert: Notifies users monitoring the product (24h cooldown per user/product).
    """
    alerts_created = []

    # --------------------------------------------------------------------------
    # 1. Reviewer Alert (Direct classification feedback on submitted review)
    # --------------------------------------------------------------------------
    if trigger_review and trigger_review.user:
        label_text = trigger_review.label.lower() if trigger_review.label else 'genuine'
        reviewer_alert = Alert.objects.create(
            product=trigger_review.product,
            user=trigger_review.user,
            type='review_classified',
            message=f"Your Review is classified as {label_text}"
        )
        alerts_created.append(reviewer_alert)

    # --------------------------------------------------------------------------
    # 2. Product Fake Review Ratio Evaluation (Threshold: >= 50%)
    # --------------------------------------------------------------------------
    total_reviews = product.reviews.count()
    fake_reviews = product.reviews.filter(label='Fake').count()
    fake_ratio = (fake_reviews / total_reviews * 100.0) if total_reviews > 0 else 0.0

    if total_reviews > 0 and fake_ratio >= 50.0:
        pct_val = int(fake_ratio) if fake_ratio.is_integer() else round(fake_ratio, 1)
        ratio_msg = f"Fake reviews for subject product is increase to {pct_val}%"
        cooldown_24h = timezone.now() - timedelta(hours=24)

        # 2A. Admin Alerts (24-hour cooldown per product)
        admin_users = list(User.objects.filter(
            Q(is_superuser=True) | Q(is_staff=True) | Q(profile__role='admin')
        ).distinct())

        admin_alert_exists = Alert.objects.filter(
            product=product,
            type='fake_spike',
            created_at__gte=cooldown_24h
        ).filter(Q(user__in=admin_users) | Q(user__isnull=True)).exists()

        if not admin_alert_exists:
            # Clear previous outdated ratio alerts for this product
            Alert.objects.filter(product=product, type='fake_spike').filter(
                Q(user__in=admin_users) | Q(user__isnull=True)
            ).delete()

            if admin_users:
                for admin in admin_users:
                    alerts_created.append(Alert.objects.create(
                        product=product,
                        user=admin,
                        type='fake_spike',
                        message=ratio_msg
                    ))
            else:
                alerts_created.append(Alert.objects.create(
                    product=product,
                    user=None,
                    type='fake_spike',
                    message=ratio_msg
                ))

        # 2B. General User Alerts (24-hour cooldown per user per product)
        monitored_users = list(User.objects.filter(
            saved_products__product=product
        ).exclude(id__in=[u.id for u in admin_users]).distinct())

        for m_user in monitored_users:
            user_alert_exists = Alert.objects.filter(
                product=product,
                user=m_user,
                type='fake_spike',
                created_at__gte=cooldown_24h
            ).exists()

            if not user_alert_exists:
                Alert.objects.filter(product=product, user=m_user, type='fake_spike').delete()
                alerts_created.append(Alert.objects.create(
                    product=product,
                    user=m_user,
                    type='fake_spike',
                    message=ratio_msg
                ))

    return alerts_created


def dispatch_reviewer_classification_alert(review):
    """Convenience wrapper delegating directly to evaluate_product_alerts."""
    if not review or not review.product:
        return None
    alerts = evaluate_product_alerts(review.product, trigger_review=review)
    return alerts[0] if alerts else None
