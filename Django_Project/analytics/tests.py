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
        self.assertEqual(new_alerts[0].message, "Fake reviews for subject product is increase to 60%")

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

    def test_general_user_alert_at_50_percent_fake_ratio(self):
        """General user monitoring a product receives an alert when fake ratio reaches/exceeds 50%."""
        from products.models import SavedProduct
        user = User.objects.create_user(username="mon_user", password="password123")
        SavedProduct.objects.create(user=user, product=self.product)

        # 2 Fake, 2 Genuine -> 50% fake review ratio
        for i in range(2):
            Review.objects.create(product=self.product, text=f"Fake {i}", rating=5, label=Review.LABEL_FAKE)
            Review.objects.create(product=self.product, text=f"Gen {i}", rating=5, label=Review.LABEL_GENUINE)

        new_alerts = evaluate_product_alerts(self.product)
        user_alerts = [a for a in new_alerts if a.user == user]
        self.assertEqual(len(user_alerts), 1)
        expected_msg = "Fake reviews for subject product is increase to 50%"
        self.assertEqual(user_alerts[0].message, expected_msg)

    def test_general_user_no_alert_below_50_percent(self):
        """No ratio alert should be triggered if the product fake review ratio is below 50%."""
        from products.models import SavedProduct
        user = User.objects.create_user(username="mon_user_below", password="password123")
        SavedProduct.objects.create(user=user, product=self.product)

        # 1 Fake, 3 Genuine -> 25% fake review ratio (< 50%)
        Review.objects.create(product=self.product, text="Fake 1", rating=5, label=Review.LABEL_FAKE)
        for i in range(3):
            Review.objects.create(product=self.product, text=f"Gen {i}", rating=5, label=Review.LABEL_GENUINE)

        new_alerts = evaluate_product_alerts(self.product)
        user_alerts = [a for a in new_alerts if a.user == user and a.type == 'fake_spike']
        self.assertEqual(len(user_alerts), 0)

    def test_general_user_24h_cooldown_per_user_per_product(self):
        """24-hour cooldown prevents duplicate alerts for the same user and product."""
        from products.models import SavedProduct
        user1 = User.objects.create_user(username="user1", password="password123")
        user2 = User.objects.create_user(username="user2", password="password123")
        SavedProduct.objects.create(user=user1, product=self.product)
        SavedProduct.objects.create(user=user2, product=self.product)

        # 3 Fake out of 4 reviews -> 75% fake ratio
        for i in range(3):
            Review.objects.create(product=self.product, text=f"Fake {i}", rating=5, label=Review.LABEL_FAKE)
        Review.objects.create(product=self.product, text="Gen 1", rating=5, label=Review.LABEL_GENUINE)

        run1 = evaluate_product_alerts(self.product)
        self.assertEqual(len([a for a in run1 if a.user == user1]), 1)
        self.assertEqual(len([a for a in run1 if a.user == user2]), 1)

        # Immediate second run: cooldown active
        run2 = evaluate_product_alerts(self.product)
        self.assertEqual(len([a for a in run2 if a.user == user1]), 0)
        self.assertEqual(len([a for a in run2 if a.user == user2]), 0)

    def test_reviewer_alert_classification_feedback_no_cooldown(self):
        """Reviewers receive alerts for both Fake and Genuine classifications on every review."""
        reviewer = User.objects.create_user(username="reviewer_alice", password="password123")
        
        # 1. Fake review submission
        rev_fake = Review.objects.create(
            product=self.product,
            user=reviewer,
            text="Best product ever wow buy now!",
            rating=5,
            label=Review.LABEL_FAKE
        )
        alerts1 = evaluate_product_alerts(self.product, trigger_review=rev_fake)
        reviewer_fake_alerts = [a for a in alerts1 if a.user == reviewer and a.type == 'review_classified']
        self.assertEqual(len(reviewer_fake_alerts), 1)
        self.assertEqual(reviewer_fake_alerts[0].message, "Your Review is classified as fake")

        # 2. Genuine review submission by same reviewer on another product (or even same): must fire again (no cooldown)
        prod2 = Product.objects.create(name="Another Product", platform="Amazon")
        rev_gen = Review.objects.create(
            product=prod2,
            user=reviewer,
            text="Normal honest user experience with minor flaws.",
            rating=4,
            label=Review.LABEL_GENUINE
        )
        alerts2 = evaluate_product_alerts(prod2, trigger_review=rev_gen)
        reviewer_gen_alerts = [a for a in alerts2 if a.user == reviewer and a.type == 'review_classified']
        self.assertEqual(len(reviewer_gen_alerts), 1)
        self.assertEqual(reviewer_gen_alerts[0].message, "Your Review is classified as genuine")

    def test_admin_alert_at_50_percent_fake_ratio_with_cooldown(self):
        """Platform admins receive alerts for any product reaching >= 50% fake ratio with 24h per-product cooldown."""
        admin = User.objects.create_user(username="platform_admin", password="password123", is_staff=True)
        admin.profile.role = 'admin'
        admin.profile.save()

        # 3 Fake out of 5 -> 60%
        for i in range(3):
            Review.objects.create(product=self.product, text=f"Fake {i}", rating=5, label=Review.LABEL_FAKE)
        for i in range(2):
            Review.objects.create(product=self.product, text=f"Gen {i}", rating=5, label=Review.LABEL_GENUINE)

        run1 = evaluate_product_alerts(self.product)
        admin_alerts_1 = [a for a in run1 if a.user == admin and a.type == 'fake_spike']
        self.assertEqual(len(admin_alerts_1), 1)
        self.assertEqual(admin_alerts_1[0].message, "Fake reviews for subject product is increase to 60%")

        # Immediate rerun: 24h cooldown per product prevents duplicate admin alerts
        run2 = evaluate_product_alerts(self.product)
        admin_alerts_2 = [a for a in run2 if a.user == admin and a.type == 'fake_spike']
        self.assertEqual(len(admin_alerts_2), 0)


class AnalyticsViewsTestCase(TestCase):
    """Verifies that platform-wide aggregation and analytics views respond with HTTP 200."""

    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username="analytics_user", password="password123")
        p1 = Product.objects.create(name="Product 1", category="Electronics", platform="Amazon")
        p2 = Product.objects.create(name="Product 2", category="Apparel", platform="eBay")

        Review.objects.create(product=p1, text="Good", rating=5, final_score=90.0, label="Genuine")
        Review.objects.create(product=p1, text="Bad", rating=1, final_score=20.0, label="Fake")
        Review.objects.create(product=p2, text="OK", rating=3, final_score=50.0, label="Genuine")

    def test_analytics_overview_view(self):
        """GET /analytics/ redirects to /explore/."""
        res = self.client.get('/analytics/')
        self.assertEqual(res.status_code, 302)
        self.assertRedirects(res, '/explore/')

    def test_suspicious_discovery_view(self):
        """GET /suspicious-products/ redirects to /explore/."""
        res = self.client.get('/suspicious-products/')
        self.assertEqual(res.status_code, 302)
        self.assertRedirects(res, '/explore/')

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

    def test_mark_alert_read_and_delete_alert(self):
        """Test marking an alert as read and then deleting it."""
        product = Product.objects.first()
        alert = Alert.objects.create(
            product=product,
            user=self.user,
            type='fake_spike',
            message="Fake reviews for subject product is increase to 60%",
            is_read=False
        )

        self.client.force_login(self.user)

        # Mark as read
        read_res = self.client.post(f'/alerts/{alert.id}/read/')
        self.assertEqual(read_res.status_code, 302)
        alert.refresh_from_db()
        self.assertTrue(alert.is_read)

        # Delete alert
        del_res = self.client.post(f'/alerts/{alert.id}/delete/')
        self.assertEqual(del_res.status_code, 302)
        self.assertFalse(Alert.objects.filter(id=alert.id).exists())

