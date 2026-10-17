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

    def test_registration_with_fullname_and_email(self):
        """User registers using Full Name, Email, Role, and matching Passwords (no username)."""
        res = self.client.post('/accounts/register/', {
            'full_name': 'Jane Doe',
            'email': 'jane.doe@example.com',
            'role': UserProfile.ROLE_REVIEWER,
            'password': 'SecurePassword123!',
            'confirm_password': 'SecurePassword123!',
        }, follow=True)
        self.assertEqual(res.status_code, 200)

        user = User.objects.get(email='jane.doe@example.com')
        self.assertEqual(user.first_name, 'Jane')
        self.assertEqual(user.last_name, 'Doe')
        self.assertEqual(user.profile.role, UserProfile.ROLE_REVIEWER)
        self.assertTrue(user.check_password('SecurePassword123!'))

    def test_email_login_success(self):
        """User logs in using their email address and password."""
        User.objects.create_user(
            username='user_login@example.com',
            email='user_login@example.com',
            password='UserPass123!'
        )
        res = self.client.post('/accounts/login/', {
            'email': 'user_login@example.com',
            'password': 'UserPass123!'
        }, follow=True)
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.context['user'].is_authenticated)
        self.assertEqual(res.context['user'].email, 'user_login@example.com')

    def test_email_login_invalid_credentials(self):
        """Invalid email or password returns an error message."""
        res = self.client.post('/accounts/login/', {
            'email': 'nonexistent@example.com',
            'password': 'WrongPassword123'
        })
        self.assertEqual(res.status_code, 200)
        self.assertContains(res, "Invalid email address or password")

    def test_profile_update_with_fullname(self):
        """Profile update successfully splits Full Name and updates User."""
        self.client.force_login(self.general_user)
        res = self.client.post('/accounts/profile/', {
            'full_name': 'Johnathan Alexander Doe',
        }, follow=True)
        self.assertEqual(res.status_code, 200)
        self.general_user.refresh_from_db()
        self.assertEqual(self.general_user.first_name, 'Johnathan')
        self.assertEqual(self.general_user.last_name, 'Alexander Doe')

    def test_profile_change_password(self):
        """Changing password from profile updates user password securely."""
        self.client.force_login(self.general_user)
        res = self.client.post('/accounts/profile/', {
            'action': 'change_password',
            'old_password': 'password123',
            'new_password1': 'NewStrongPassword456!',
            'new_password2': 'NewStrongPassword456!',
        }, follow=True)
        self.assertEqual(res.status_code, 200)
        self.general_user.refresh_from_db()
        self.assertTrue(self.general_user.check_password('NewStrongPassword456!'))

    def test_admin_users_view_excludes_admins(self):
        """User management list must only show General Users & Reviewers, excluding Admins & Superusers."""
        self.client.force_login(self.admin_user)
        res = self.client.get('/accounts/admin-panel/users/')
        self.assertEqual(res.status_code, 200)
        page_users = res.context['page_obj'].object_list
        self.assertIn(self.general_user, page_users)
        self.assertIn(self.reviewer_user, page_users)
        self.assertNotIn(self.admin_user, page_users)

    def test_admin_add_user_success(self):
        """Admin can create a new Reviewer or General User account."""
        self.client.force_login(self.admin_user)
        res = self.client.post('/accounts/admin-panel/users/add/', {
            'full_name': 'New Reviewer Agent',
            'email': 'new_reviewer@example.com',
            'role': UserProfile.ROLE_REVIEWER,
            'password': 'SecurePassword789!',
        }, follow=True)
        self.assertEqual(res.status_code, 200)

        created = User.objects.get(email='new_reviewer@example.com')
        self.assertEqual(created.first_name, 'New')
        self.assertEqual(created.last_name, 'Reviewer Agent')
        self.assertEqual(created.profile.role, UserProfile.ROLE_REVIEWER)
        self.assertTrue(created.is_active)
        self.assertTrue(created.check_password('SecurePassword789!'))

    def test_admin_edit_user_success_no_role_swap(self):
        """Admin can edit user details, active status, and password, but the role remains unchanged."""
        self.client.force_login(self.admin_user)
        original_role = self.general_user.profile.role

        res = self.client.post(f'/accounts/admin-panel/users/{self.general_user.id}/edit/', {
            'full_name': 'Updated General User',
            'email': 'updated_gen@example.com',
            'new_password': 'BrandNewPassword123!',
        }, follow=True)
        self.assertEqual(res.status_code, 200)

        self.general_user.refresh_from_db()
        self.assertEqual(self.general_user.first_name, 'Updated')
        self.assertEqual(self.general_user.last_name, 'General User')
        self.assertEqual(self.general_user.email, 'updated_gen@example.com')
        self.assertTrue(self.general_user.is_active)
        self.assertTrue(self.general_user.check_password('BrandNewPassword123!'))
        # Ensure role was not swapped/altered
        self.assertEqual(self.general_user.profile.role, original_role)

    def test_admin_delete_user_success(self):
        """Admin can delete a General User or Reviewer account."""
        self.client.force_login(self.admin_user)
        user_to_delete = User.objects.create_user(
            username='temp_delete@example.com',
            email='temp_delete@example.com',
            password='password123'
        )
        user_id = user_to_delete.id
        res = self.client.post(f'/accounts/admin-panel/users/{user_id}/delete/', follow=True)
        self.assertEqual(res.status_code, 200)
        self.assertFalse(User.objects.filter(id=user_id).exists())

    def test_admin_cannot_delete_admin_account(self):
        """Administrator accounts cannot be deleted through this interface."""
        self.client.force_login(self.admin_user)
        res = self.client.post(f'/accounts/admin-panel/users/{self.admin_user.id}/delete/', follow=True)
        self.assertEqual(res.status_code, 200)
        self.assertTrue(User.objects.filter(id=self.admin_user.id).exists())

