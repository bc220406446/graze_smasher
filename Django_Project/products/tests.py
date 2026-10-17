"""
Unit Tests for Products, Reviews Submission & Catalog Views
CS619 Final-Year Project: Graze Smasher Product Hunting Tool
"""

from django.test import TestCase, Client
from django.contrib.auth.models import User
from products.models import Product, SavedProduct
from reviews.models import Review
from accounts.models import UserProfile


class ProductModelAndViewsTestCase(TestCase):
    """Verifies Product model properties, catalog filtering, review submission, and privacy policy."""

    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username="test_reviewer", password="password123")
        self.user.profile.role = UserProfile.ROLE_REVIEWER
        self.user.profile.save()

        self.product = Product.objects.create(
            name="Smart LED Desk Lamp",
            category="Home & Kitchen",
            platform="Amazon",
            description="Adjustable brightness with eye protection."
        )

    def test_product_computed_metrics(self):
        """Computed properties must calculate averages and verified trust ratings accurately."""
        # 1 Genuine review (rating 5, final_score 90)
        Review.objects.create(
            product=self.product,
            text="Very bright and clean light for studying.",
            rating=5,
            final_score=90.0,
            label=Review.LABEL_GENUINE
        )
        # 1 Fake review (rating 5, final_score 20)
        Review.objects.create(
            product=self.product,
            text="Best lamp ever buy now now now!!",
            rating=5,
            final_score=20.0,
            label=Review.LABEL_FAKE
        )

        self.assertEqual(self.product.review_count(), 2)
        self.assertEqual(self.product.average_rating(), 5.0)
        self.assertEqual(self.product.overall_authenticity_score(), 55.0)
        self.assertEqual(self.product.authenticity_label(), "Genuine")
        self.assertEqual(self.product.genuine_reviews_count(), 1)
        self.assertEqual(self.product.fake_reviews_count(), 1)
        # Community trust rating considers ONLY genuine reviews:
        self.assertEqual(self.product.community_trust_rating(), 5.0)

    def test_product_list_view_and_search(self):
        """Product list view returns 200 and matches queries."""
        res = self.client.get('/')
        self.assertEqual(res.status_code, 200)

        # Search matching query
        res = self.client.get('/?q=Lamp')
        self.assertEqual(res.status_code, 200)
        self.assertContains(res, "Smart LED Desk Lamp")

        # Search non-matching query
        res = self.client.get('/?q=NonExistentObjectXYZ')
        self.assertEqual(res.status_code, 200)
        self.assertNotContains(res, "Smart LED Desk Lamp")

    def test_product_detail_view(self):
        """Product detail view displays product data and reviews."""
        res = self.client.get(f'/products/{self.product.id}/')
        self.assertEqual(res.status_code, 200)
        self.assertContains(res, "Smart LED Desk Lamp")

    def test_submit_review_via_web_interface(self):
        """Authenticated Reviewers can post reviews, triggering ML analysis."""
        self.client.force_login(self.user)
        res = self.client.post(
            f'/products/{self.product.id}/review/',
            {
                'rating': '5',
                'text': 'The flexible neck allows easy positioning for reading.'
            },
            follow=True
        )
        self.assertEqual(res.status_code, 200)

        # Verify review stored in DB with computed scores
        latest_rev = Review.objects.filter(product=self.product).latest('created_at')
        self.assertEqual(latest_rev.user, self.user)
        self.assertGreater(latest_rev.final_score, 0)
        self.assertIn(latest_rev.label, ['Genuine', 'Fake'])

    def test_watchlist_toggle(self):
        """Users can toggle products in their watchlist."""
        self.client.force_login(self.user)
        res = self.client.get(f'/products/{self.product.id}/save/', follow=True)
        self.assertEqual(res.status_code, 200)
        self.assertTrue(SavedProduct.objects.filter(user=self.user, product=self.product).exists())

        # Toggle again to remove
        res = self.client.get(f'/products/{self.product.id}/save/', follow=True)
        self.assertEqual(res.status_code, 200)
        self.assertFalse(SavedProduct.objects.filter(user=self.user, product=self.product).exists())

    def test_privacy_policy_view(self):
        """Privacy and Data policy view renders 200 OK with key disclosure sections."""
        res = self.client.get('/privacy/')
        self.assertEqual(res.status_code, 200)
        self.assertContains(res, "Privacy &amp; Data Policy")
        self.assertContains(res, "Ethical Data Privacy &amp; Zero IP Sharing")
        self.assertContains(res, "Multi-Model ML Ensemble")

    def test_reviewer_can_only_review_once(self):
        """Reviewers are strictly limited to one review per product; second attempt is rejected."""
        self.client.force_login(self.user)
        # First submission succeeds
        res1 = self.client.post(
            f'/products/{self.product.id}/review/',
            {
                'rating': '5',
                'text': 'First review submitted by reviewer.'
            },
            follow=True
        )
        self.assertEqual(res1.status_code, 200)
        self.assertEqual(Review.objects.filter(product=self.product, user=self.user).count(), 1)
        latest_rev = Review.objects.filter(product=self.product, user=self.user).first()
        self.assertIsNone(latest_rev.ip_address)  # Zero IP storage

        # Second submission by same reviewer is blocked
        res2 = self.client.post(
            f'/products/{self.product.id}/review/',
            {
                'rating': '4',
                'text': 'Attempted second review on the same product.'
            },
            follow=True
        )
        self.assertEqual(res2.status_code, 200)
        self.assertContains(res2, "You have already reviewed this product")
        self.assertEqual(Review.objects.filter(product=self.product, user=self.user).count(), 1)
