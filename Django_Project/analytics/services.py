"""
Graze Smasher Product Hunting Tool - Analytics & Alert Engine
CS619 Final-Year Project

Monitors fake-review velocity, ratio thresholds, and authenticity shifts.
Generates system Alerts and dispatches console email notifications.
"""

import logging
from datetime import timedelta
from django.utils import timezone
from django.conf import settings
from django.core.mail import send_mail
from .models import Alert

logger = logging.getLogger(__name__)


def _dispatch_alert(product, alert_type, message, alerts_created, window_hours=12):
    """
    Dispatches alerts to:
    1. System Administrators (Admins receive alerts whenever a product submission or review submission spikes fake reviews across any product).
    2. Users who are actively monitoring / saved this product in their watchlist (plus product creator).
    Prevents duplicate alerts within window_hours per recipient.
    """
    from django.contrib.auth.models import User
    from django.db.models import Q

    cutoff = timezone.now() - timedelta(hours=window_hours)

    # 1. Admin users (all platform admins get notified of fake spikes anywhere)
    admin_users = list(User.objects.filter(
        Q(is_superuser=True) | Q(is_staff=True) | Q(profile__role='admin')
    ).distinct())

    # 2. Monitored users (only users who saved this product in their watchlist)
    monitored_users = list(User.objects.filter(
        saved_products__product=product
    ).exclude(id__in=[u.id for u in admin_users]).distinct())

    if product.created_by and product.created_by not in admin_users and product.created_by not in monitored_users:
        monitored_users.append(product.created_by)

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
    Evaluates rolling review windows for abnormal deception patterns:
    1. Fake Review Ratio Threshold: Flag if >= ALERT_FAKE_RATIO_THRESHOLD of recent reviews are Fake.
    2. Fake Review Spike: Flag if recent fake ratio jumped by >= ALERT_FAKE_SPIKE_THRESHOLD vs prior window.
    3. Critical Low Authenticity: Flag if overall product score drops below threshold (<50%).
    4. IP Burst: Flag if current review triggered an IP velocity violation.
    
    Alert Recipients:
    - Admins receive alerts across all products on the platform.
    - Regular users receive alerts only for products in their monitored watchlist.
    """
    alerts_created = []
    window_count = getattr(settings, 'ALERT_WINDOW_COUNT', 10)
    fake_threshold = getattr(settings, 'ALERT_FAKE_RATIO_THRESHOLD', 0.40)
    spike_threshold = getattr(settings, 'ALERT_FAKE_SPIKE_THRESHOLD', 0.25)

    all_reviews = list(product.reviews.order_by('-created_at')[: window_count * 2])
    total_reviews_count = len(all_reviews)

    if total_reviews_count == 0:
        return alerts_created

    # 1. IP Burst Alert (Direct trigger from submitted review)
    if trigger_review and trigger_review.ip_flag:
        msg = (f"Suspicious IP submission velocity detected from {trigger_review.ip_address} "
               f"on '{product.name}'. Multiple submissions within velocity window.")
        _dispatch_alert(product, 'ip_burst', msg, alerts_created, window_hours=6)

    # 2. Window-based Ratio & Spike Evaluation
    recent_window = all_reviews[:min(window_count, total_reviews_count)]
    prior_window = all_reviews[min(window_count, total_reviews_count):]

    recent_fake_count = sum(1 for r in recent_window if r.label == 'Fake')
    recent_fake_ratio = recent_fake_count / len(recent_window) if recent_window else 0.0

    # A) Ratio Threshold Trigger
    if len(recent_window) >= 3 and recent_fake_ratio >= fake_threshold:
        msg = (f"Deception Alert: {round(recent_fake_ratio * 100, 1)}% of recent reviews for "
               f"'{product.name}' are classified as Fake / Machine-Generated (Threshold: {int(fake_threshold*100)}%).")
        _dispatch_alert(product, 'fake_spike', msg, alerts_created, window_hours=12)

    # B) Spike vs Prior Window Trigger
    if len(prior_window) >= 3:
        prior_fake_count = sum(1 for r in prior_window if r.label == 'Fake')
        prior_fake_ratio = prior_fake_count / len(prior_window)
        spike_diff = recent_fake_ratio - prior_fake_ratio

        if spike_diff >= spike_threshold:
            msg = (f"Abnormal Fake Review Spike on '{product.name}': Fake ratio surged by "
                   f"+{round(spike_diff * 100, 1)}% compared to prior window.")
            _dispatch_alert(product, 'fake_spike', msg, alerts_created, window_hours=12)

    # C) Critical Authenticity Drop Trigger (< 50% for binary classification)
    overall_score = product.overall_authenticity_score()
    auth_threshold = getattr(settings, 'AUTHENTICITY_THRESHOLD', 50.0)
    if total_reviews_count >= 5 and overall_score < auth_threshold:
        msg = (f"Critical Authenticity Warning: Overall authenticity score for '{product.name}' "
               f"dropped to {overall_score}% (Classified as Deceptive / Fake).")
        _dispatch_alert(product, 'low_authenticity', msg, alerts_created, window_hours=24)

    # 3. Optional Email Notification
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

