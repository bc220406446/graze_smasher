"""
Unit Tests for Role-Based Access Control (RBAC) and Authentication
CS619 Final-Year Project: Graze Smasher Product Hunting Tool
"""

from django.test import TestCase, Client
from django.contrib.auth.models import User
from accounts.models import UserProfile
from products.models import Product, ProductFlag
from reviews.models import Review


class RBACPermissionsTestCase(TestCase):
    """Verifies that role decorators enforce access control across General, Reviewer, and Admin."""

    def setUp(self):
        self.client = Client()

        # 1. Create General User
        self.general_user = User.objects.create_user(username="gen_user", password="password123")
        self.general_user.profile.role = UserProfile.ROLE_GENERAL
        self.general_user.profile.save()

        # 2. Create Reviewer
        self.reviewer_user = User.objects.create_user(username="rev_user", password="password123")
        self.reviewer_user.profile.role = UserProfile.ROLE_REVIEWER
        self.reviewer_user.profile.save()

        # 3. Create Admin User
        self.admin_user = User.objects.create_user(username="admin_user", password="password123")
        self.admin_user.profile.role = UserProfile.ROLE_ADMIN
        self.admin_user.is_staff = True
        self.admin_user.save()
        self.admin_user.profile.save()

        self.product = Product.objects.create(name="Sample Watch", category="Accessories")

    def test_anonymous_redirect_on_protected_views(self):
        """Anonymous users attempting to create a product must be redirected to login."""
        res = self.client.get('/products/create/')
        self.assertEqual(res.status_code, 302)
        self.assertIn('/accounts/login/', res.headers.get('Location'))

    def test_general_user_can_access_product_creation(self):
        """General users are permitted to access product creation with manual reviews."""
        self.client.force_login(self.general_user)
        res = self.client.get('/products/create/')
        self.assertEqual(res.status_code, 200)

    def test_reviewer_cannot_create_product(self):
        """Reviewers cannot create new products; they are redirected to their reviewer workspace."""
        self.client.force_login(self.reviewer_user)
        res = self.client.get('/products/create/')
        self.assertEqual(res.status_code, 302)
        self.assertIn('/reviewer/dashboard/', res.headers.get('Location'))

    def test_reviewer_can_access_reviewer_dashboard(self):
        """Reviewers can access the dedicated reviewer dashboard and reviews history."""
        self.client.force_login(self.reviewer_user)
        res = self.client.get('/reviewer/dashboard/')
        self.assertEqual(res.status_code, 200)

        res_revs = self.client.get('/reviewer/my-reviews/')
        self.assertEqual(res_revs.status_code, 200)

    def test_general_user_cannot_access_reviewer_dashboard(self):
        """General users cannot access the reviewer dashboard."""
        self.client.force_login(self.general_user)
        res = self.client.get('/reviewer/dashboard/')
        self.assertEqual(res.status_code, 302)

    def test_non_admin_cannot_access_admin_panel(self):
        """Neither General Users nor Reviewers can access the Admin Control Panel."""
        self.client.force_login(self.general_user)
        res = self.client.get('/accounts/admin-panel/')
        self.assertEqual(res.status_code, 302)

        self.client.force_login(self.reviewer_user)
        res = self.client.get('/accounts/admin-panel/')
        self.assertEqual(res.status_code, 302)

    def test_admin_can_access_admin_panel_and_routes(self):
        """Admins have full access to Admin Control Panel and product management."""
        self.client.force_login(self.admin_user)
        res = self.client.get('/accounts/admin-panel/')
        self.assertEqual(res.status_code, 200)

        res_prods = self.client.get('/accounts/admin-panel/products/')
        self.assertEqual(res_prods.status_code, 200)

        res_flags = self.client.get('/accounts/admin-panel/flagged/')
        self.assertEqual(res_flags.status_code, 200)

    def test_user_can_flag_product_and_admin_resolve(self):
        """General user can flag a product and admin can resolve the report."""
        self.client.force_login(self.general_user)
        res = self.client.post(
            f'/products/{self.product.id}/flag/',
            {
                'category': ProductFlag.CATEGORY_MISLEADING,
                'details': 'Deceptive advertisement and fake product photos.'
            },
            follow=True
        )
        self.assertEqual(res.status_code, 200)
        flag = ProductFlag.objects.get(product=self.product, user=self.general_user)
        self.assertEqual(flag.status, ProductFlag.STATUS_PENDING)

        # Admin resolves flag and hides product
        self.client.force_login(self.admin_user)
        res_resolve = self.client.post(
            f'/accounts/admin-panel/flags/{flag.id}/resolve/',
            {'action': 'resolve_and_hide', 'admin_notes': 'Verified deceptive listing.'},
            follow=True
        )
        self.assertEqual(res_resolve.status_code, 200)
        flag.refresh_from_db()
        self.assertEqual(flag.status, ProductFlag.STATUS_RESOLVED)
        self.product.refresh_from_db()
        self.assertTrue(self.product.is_hidden)

    def test_admin_update_user_role(self):
        """Admins can promote General Users to Reviewers."""
        self.client.force_login(self.admin_user)
        res = self.client.post(
            f'/accounts/admin-panel/users/{self.general_user.id}/role/',
            {'role': UserProfile.ROLE_REVIEWER},
            follow=True
        )
        self.assertEqual(res.status_code, 200)
        self.general_user.profile.refresh_from_db()
        self.assertEqual(self.general_user.profile.role, UserProfile.ROLE_REVIEWER)

    def test_admin_override_review_label(self):
        """Admins can override automated ML classifications in the moderation queue."""
        review = Review.objects.create(
            product=self.product,
            text="Decent watch for the price.",
            final_score=35.0,
            label=Review.LABEL_FAKE
        )

        self.client.force_login(self.admin_user)
        res = self.client.post(
            f'/accounts/admin-panel/reviews/{review.id}/override/',
            {'new_label': Review.LABEL_GENUINE, 'notes': 'Verified manual review'},
            follow=True
        )
        self.assertEqual(res.status_code, 200)
        review.refresh_from_db()
        self.assertEqual(review.label, Review.LABEL_GENUINE)
        self.assertGreaterEqual(review.final_score, 70.0)
