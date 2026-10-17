from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from accounts.models import UserProfile
from products.models import Product
from reviews.models import Review


class ReviewActionTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.reviewer_user = User.objects.create_user(
            username='reviewer1',
            email='reviewer1@example.com',
            password='Password123!'
        )
        self.reviewer_user.profile.role = UserProfile.ROLE_REVIEWER
        self.reviewer_user.profile.save()

        self.other_user = User.objects.create_user(
            username='reviewer2',
            email='reviewer2@example.com',
            password='Password123!'
        )
        self.other_user.profile.role = UserProfile.ROLE_REVIEWER
        self.other_user.profile.save()

        self.product = Product.objects.create(
            name='Test Monitor Wireless Mouse',
            description='Ergonomic wireless mouse with rechargeable battery.',
            category='Electronics',
            platform='Amazon',
            created_by=self.reviewer_user
        )

        self.review = Review.objects.create(
            product=self.product,
            user=self.reviewer_user,
            text='Great mouse, very comfortable to hold during long work days.',
            rating=5,
            final_score=95.0,
            label='Genuine'
        )

    def test_reviewer_dashboard_renders_clean_layout(self):
        self.client.login(username='reviewer1', password='Password123!')
        response = self.client.get(reverse('reviewer_dashboard'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Quick Actions')
        self.assertContains(response, 'Add Review')
        self.assertContains(response, 'Reviewed Products')
        self.assertContains(response, 'Recent System Alerts')

    def test_reviewer_my_reviews_renders_dropdowns_and_dashboard_table(self):
        self.client.login(username='reviewer1', password='Password123!')
        response = self.client.get(reverse('reviewer_my_reviews'))
        self.assertEqual(response.status_code, 200)
        # Check dropdown filters
        self.assertContains(response, 'All Classifications')
        self.assertContains(response, 'All Ratings')
        self.assertContains(response, 'Newest First')
        # Check table headers matching dashboard
        self.assertContains(response, 'Product Title')
        self.assertContains(response, 'Review Content')
        self.assertContains(response, 'Rating')
        self.assertContains(response, 'Classification')
        self.assertContains(response, 'Final Score')
        self.assertContains(response, 'Action')
        self.assertContains(response, 'Edit')
        self.assertContains(response, 'Delete')

    def test_reviewer_my_reviews_filters_by_rating_and_label(self):
        self.client.login(username='reviewer1', password='Password123!')
        response = self.client.get(reverse('reviewer_my_reviews') + '?rating=5&label=Genuine')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Great mouse')

        # Filter for non-matching rating
        response_empty = self.client.get(reverse('reviewer_my_reviews') + '?rating=1')
        self.assertEqual(response_empty.status_code, 200)
        self.assertContains(response_empty, 'No Reviews Found')

    def test_edit_review_by_owner(self):
        self.client.login(username='reviewer1', password='Password123!')
        edit_url = reverse('edit_review', args=[self.review.pk])
        response = self.client.post(edit_url, {
            'text': 'Updated: Solid build quality, battery life lasts over two weeks.',
            'rating': 4
        })
        self.assertRedirects(response, reverse('reviewer_dashboard'))
        self.review.refresh_from_db()
        self.assertEqual(self.review.rating, 4)
        self.assertIn('Updated: Solid build quality', self.review.text)

    def test_edit_review_by_unauthorized_user(self):
        self.client.login(username='reviewer2', password='Password123!')
        edit_url = reverse('edit_review', args=[self.review.pk])
        response = self.client.post(edit_url, {
            'text': 'Malicious overwrite attempt',
            'rating': 1
        })
        self.assertRedirects(response, reverse('reviewer_dashboard'))
        self.review.refresh_from_db()
        self.assertEqual(self.review.rating, 5)
        self.assertNotIn('Malicious', self.review.text)

    def test_delete_review_by_owner(self):
        self.client.login(username='reviewer1', password='Password123!')
        delete_url = reverse('delete_review', args=[self.review.pk])
        response = self.client.post(delete_url)
        self.assertRedirects(response, reverse('reviewer_dashboard'))
        self.assertFalse(Review.objects.filter(pk=self.review.pk).exists())

    def test_delete_review_by_unauthorized_user(self):
        self.client.login(username='reviewer2', password='Password123!')
        delete_url = reverse('delete_review', args=[self.review.pk])
        response = self.client.post(delete_url)
        self.assertRedirects(response, reverse('reviewer_dashboard'))
        self.assertTrue(Review.objects.filter(pk=self.review.pk).exists())
