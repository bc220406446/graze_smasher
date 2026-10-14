"""
Unit Tests for Analytics and Alert Heuristic Engine
CS619 Final-Year Project: Graze Smasher Product Hunting Tool
"""

from datetime import timedelta
from django.utils import timezone
from django.test import TestCase, Client
from django.contrib.auth.models import User
from products.models import Product
from reviews.models import Review
from analytics.models import Alert
from analytics.services import evaluate_product_alerts


class AlertEngineTestCase(TestCase):
    """Verifies that the alert engine creates correct Alert records under anomaly conditions."""

    def setUp(self):
        self.product = Product.objects.create(
            name="Test Noise Cancelling Headphones",
            category="Electronics",
            platform="Amazon"
        )

    def test_high_fake_ratio_alert(self):
        """When fake reviews exceed 40% of recent submissions, a fake_spike alert must be generated."""
        for i in range(3):
            Review.objects.create(
                product=self.product,
                text=f"Fake review {i}",
                rating=5,
                svm_score=0.9,
                j48_score=0.9,
                final_score=20.0,
                label=Review.LABEL_FAKE
            )
        for i in range(2):
            Review.objects.create(
                product=self.product,
                text=f"Genuine review {i}",
                rating=5,
                svm_score=0.1,
                j48_score=0.1,
                final_score=85.0,
                label=Review.LABEL_GENUINE
            )

        new_alerts = evaluate_product_alerts(self.product)
        self.assertGreaterEqual(len(new_alerts), 1)

        types = [a.type for a in new_alerts]
        self.assertIn('fake_spike', types)

    def test_ip_velocity_burst_alert(self):
        """When a review triggers an IP velocity flag, an ip_burst alert must fire."""
        flagged_review = Review.objects.create(
            product=self.product,
            text="Spam review from burst IP",
            rating=5,
            ip_address="192.0.2.77",
            ip_flag=True,
            final_score=30.0,
            label=Review.LABEL_FAKE
        )

        new_alerts = evaluate_product_alerts(self.product, trigger_review=flagged_review)
        types = [a.type for a in new_alerts]
        self.assertIn('ip_burst', types)

    def test_authenticity_drop_alert(self):
        """When the overall product authenticity score drops below 40%, a low_authenticity alert must fire."""
        for i in range(5):
            Review.objects.create(
                product=self.product,
                text=f"Low score review {i}",
                rating=1,
                final_score=25.0,
                label=Review.LABEL_FAKE
            )

        new_alerts = evaluate_product_alerts(self.product)
        types = [a.type for a in new_alerts]
        self.assertIn('low_authenticity', types)

    def test_no_duplicate_unread_alerts(self):
        """The engine must not create duplicate unread alerts of the same type within cooldown."""
        for i in range(3):
            Review.objects.create(
                product=self.product,
                text=f"Fake review {i}",
                final_score=20.0,
                label=Review.LABEL_FAKE
            )

        alerts_run_1 = evaluate_product_alerts(self.product)
        initial_count = Alert.objects.filter(product=self.product).count()

        # Run again immediately
        alerts_run_2 = evaluate_product_alerts(self.product)
        self.assertEqual(len(alerts_run_2), 0)
        self.assertEqual(Alert.objects.filter(product=self.product).count(), initial_count)


class AnalyticsViewsTestCase(TestCase):
    """Verifies that platform-wide aggregation and analytics views respond with HTTP 200."""

    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username="analytics_user", password="password123")
        p1 = Product.objects.create(name="Product 1", category="Electronics", platform="Amazon")
        p2 = Product.objects.create(name="Product 2", category="Apparel", platform="eBay")

        Review.objects.create(product=p1, text="Good", rating=5, final_score=90.0, label="Genuine")
        Review.objects.create(product=p1, text="Bad", rating=1, final_score=20.0, label="Fake")
        Review.objects.create(product=p2, text="OK", rating=3, final_score=50.0, label="Suspicious")

    def test_analytics_overview_view(self):
        """GET /analytics/ returns 200 OK with benchmark charts and metrics."""
        res = self.client.get('/analytics/')
        self.assertEqual(res.status_code, 200)
        self.assertContains(res, "Analytics & Detection Benchmarks")

    def test_suspicious_discovery_view(self):
        """GET /suspicious-products/ returns 200 OK with suspicious radar."""
        res = self.client.get('/suspicious-products/')
        self.assertEqual(res.status_code, 200)
        self.assertContains(res, "Deception Radar")

    def test_user_dashboard_requires_login(self):
        """GET /dashboard/ redirects anonymous visitors to login."""
        res = self.client.get('/dashboard/')
        self.assertEqual(res.status_code, 302)

    def test_user_dashboard_authenticated(self):
        """GET /dashboard/ returns 200 OK for logged-in users."""
        self.client.force_login(self.user)
        res = self.client.get('/dashboard/')
        self.assertEqual(res.status_code, 200)
        self.assertContains(res, "Monitored Products")
