"""
Graze Smasher Product Hunting Tool - Analytics & Alert Engine
CS619 Final-Year Project

Monitors fake-review velocity, ratio thresholds, and authenticity shifts.
Generates system Alerts and dispatches console email notifications according to platform rules:
1. General User Alerts (Monitored products >= 50% fake review ratio, 24h cooldown per user per product)
2. Reviewer Alerts (Classification feedback on every review, no cooldown)
3. Admin Alerts (Platform-wide >= 50% fake review ratio, 24h cooldown per product)
"""

import logging
from datetime import timedelta
from django.utils import timezone
from django.conf import settings
from django.core.mail import send_mail
from .models import Alert

logger = logging.getLogger(__name__)


def dispatch_reviewer_classification_alert(review):
    """
    Notify reviewers whenever their submitted review is classified by the ML model
    as either Fake or Genuine.
    Alert Message: Your Review is classified as fake/genuine
    Send an alert for every submitted review classification.
    No cooldown or notification limit applies to reviewer alerts.
    """
    if not review or not review.user:
        return None
    label_text = review.label.lower() if review.label else 'genuine'
    msg = f"Your Review is classified as {label_text}"
    return Alert.objects.create(
        product=review.product,
        user=review.user,
        type='review_classified',
        message=msg
    )


# Alias for backward/alternative naming compatibility
dispatch_reviewer_alert = dispatch_reviewer_classification_alert


def _dispatch_alert(product, alert_type, message, alerts_created, window_hours=12):
    """
    Dispatches auxiliary heuristic alerts (e.g. IP burst, low authenticity warning) to:
    1. System Administrators.
    2. Users who are actively monitoring this product in their watchlist.
    Prevents duplicate alerts within window_hours per recipient.
    """
    from django.contrib.auth.models import User
    from django.db.models import Q

    cutoff = timezone.now() - timedelta(hours=window_hours)

    admin_users = list(User.objects.filter(
        Q(is_superuser=True) | Q(is_staff=True) | Q(profile__role='admin')
    ).distinct())

    monitored_users = list(User.objects.filter(
        saved_products__product=product
    ).exclude(id__in=[u.id for u in admin_users]).distinct())

    recipients = admin_users + monitored_users

    if not recipients:
        if not Alert.objects.filter(product=product, user__isnull=True, type=alert_type, created_at__gte=cutoff).exists():
            alert = Alert.objects.create(product=product, user=None, type=alert_type, message=message)
            alerts_created.append(alert)
        return

    for recipient in recipients:
        if not Alert.objects.filter(product=product, user=recipient, type=alert_type, created_at__gte=cutoff).exists():
            alert = Alert.objects.create(product=product, user=recipient, type=alert_type, message=message)
            alerts_created.append(alert)


def evaluate_product_alerts(product, trigger_review=None):
    """
    Evaluates alerts across the system according to defined rules:

    1. General User Alerts:
       - Notify users when a monitored product reaches or exceeds a 50% fake review ratio.
       - Alert Message: Fake reviews for subject product is increase to __%
       - Send alerts only for products the user monitors.
       - Apply a 24-hour cooldown per user per product to prevent duplicate alerts.

    2. Reviewer Alerts:
       - Notify reviewers whenever their submitted review is classified by the ML model as either Fake or Genuine.
       - Alert Message: Your Review is classified as fake/genuine
       - Send an alert for every submitted review classification.
       - No cooldown or notification limit applies to reviewer alerts.

    3. Admin Alerts:
       - Notify platform administrators whenever any product in the system reaches or exceeds a 50% fake review ratio.
       - Alert Message: Fake reviews for subject product is increase to __%
       - Apply a 24-hour cooldown per product for admin alerts to prevent repeated notifications.

    Important:
       - The 50% threshold applies to the fake review ratio.
       - Do not trigger alerts for products below 50%.
    """
    from django.contrib.auth.models import User
    from django.db.models import Q

    alerts_created = []

    # 1. Reviewer Alerts: Triggered on every submitted review classification (no cooldown)
    if trigger_review and trigger_review.user:
        rev_alert = dispatch_reviewer_classification_alert(trigger_review)
        if rev_alert:
            alerts_created.append(rev_alert)

    # 2. Product Fake Review Ratio Evaluation (Rule 1 & Rule 3)
    total_reviews = product.reviews.count()
    fake_reviews = product.reviews.filter(label='Fake').count()
    fake_ratio = (fake_reviews / total_reviews * 100.0) if total_reviews > 0 else 0.0

    # Strict 50% threshold: only trigger ratio alerts when fake review ratio >= 50%
    if total_reviews > 0 and fake_ratio >= 50.0:
        pct_val = int(fake_ratio) if fake_ratio.is_integer() else round(fake_ratio, 1)
        ratio_msg = f"Fake reviews for subject product is increase to {pct_val}%"
        cooldown_24h = timezone.now() - timedelta(hours=24)

        # 2A. Admin Alerts: Notify platform administrators (24-hour cooldown per product)
        admin_users = list(User.objects.filter(
            Q(is_superuser=True) | Q(is_staff=True) | Q(profile__role='admin')
        ).distinct())

        admin_alert_exists = Alert.objects.filter(
            product=product,
            type='fake_spike',
            created_at__gte=cooldown_24h
        ).filter(Q(user__in=admin_users) | Q(user__isnull=True)).exists()

        if not admin_alert_exists:
            # Delete previous notifications for this product with cooldown period of 24 hours
            Alert.objects.filter(product=product, type='fake_spike').filter(
                Q(user__in=admin_users) | Q(user__isnull=True)
            ).delete()

            if admin_users:
                for admin in admin_users:
                    alert = Alert.objects.create(
                        product=product,
                        user=admin,
                        type='fake_spike',
                        message=ratio_msg
                    )
                    alerts_created.append(alert)
            else:
                alert = Alert.objects.create(
                    product=product,
                    user=None,
                    type='fake_spike',
                    message=ratio_msg
                )
                alerts_created.append(alert)

        # 2B. General User Alerts: Notify users monitoring this product (24-hour cooldown per user per product)
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
                # Delete previous notifications for this user and product with cooldown period of 24 hours
                Alert.objects.filter(product=product, user=m_user, type='fake_spike').delete()

                alert = Alert.objects.create(
                    product=product,
                    user=m_user,
                    type='fake_spike',
                    message=ratio_msg
                )
                alerts_created.append(alert)

    # 5. Optional Email Notifications
    if alerts_created and getattr(settings, 'ALERT_EMAIL_NOTIFICATIONS', False):
        for alert in alerts_created:
            try:
                recipient_email = alert.user.email if alert.user and alert.user.email else getattr(settings, 'DEFAULT_ADMIN_EMAIL', 'admin@grazesmasher.com')
                send_mail(
                    subject=f"[Graze Smasher Alert] {alert.get_type_display()} - {product.name}",
                    message=(f"Attention,\n\nAn automated authenticity anomaly alert was triggered for monitored product:\n\n"
                             f"Product: {product.name}\n"
                             f"Alert Type: {alert.get_type_display()}\n"
                             f"Details: {alert.message}\n"
                             f"Timestamp: {alert.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
                             f"Visit Graze Smasher Dashboard to inspect product reviews."),
                    from_email=getattr(settings, 'DEFAULT_FROM_EMAIL', 'alerts@grazesmasher.com'),
                    recipient_list=[recipient_email],
                    fail_silently=True
                )
            except Exception as e:
                logger.error(f"Failed to dispatch alert email: {e}")

    return alerts_created
