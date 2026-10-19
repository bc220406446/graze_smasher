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


# ==============================================================================
# MACHINE LEARNING ENGINE & INFERENCE PIPELINE TESTS
# ==============================================================================

from ML_Model.feature_extraction import extract_numeric_features, compute_sentiment
from ML_Model.config import J48_FEATURE_NAMES
from ML_Model.service import analyze_review, compute_authenticity


class FeatureExtractionTestCase(TestCase):
    """Verifies that behavioral, structural, and sentiment features are computed accurately."""

    def test_feature_names_count(self):
        """Feature names must contain exactly the 8 numerical features used by J-48."""
        self.assertEqual(len(J48_FEATURE_NAMES), 8)
        self.assertIn("review_length", J48_FEATURE_NAMES)
        self.assertIn("word_count", J48_FEATURE_NAMES)
        self.assertIn("exclamation_count", J48_FEATURE_NAMES)
        self.assertIn("uppercase_ratio", J48_FEATURE_NAMES)
        self.assertIn("repeated_word_ratio", J48_FEATURE_NAMES)
        self.assertIn("sentiment_polarity", J48_FEATURE_NAMES)
        self.assertIn("sentiment_subjectivity", J48_FEATURE_NAMES)
        self.assertIn("rating_vs_sentiment_mismatch", J48_FEATURE_NAMES)

    def test_feature_extraction_values(self):
        """Extract features from structured text and verify bounded mathematical ranges."""
        text = "AMAZING product!!!! I LOVE IT SO MUCH!"
        feats = extract_numeric_features(text, rating=5)

        self.assertEqual(len(feats), 8)
        self.assertGreater(feats["review_length"], 0)
        self.assertGreater(feats["word_count"], 0)
        self.assertEqual(feats["exclamation_count"], 5.0)
        self.assertGreater(feats["uppercase_ratio"], 0.3)
        self.assertGreaterEqual(feats["repeated_word_ratio"], 0.0)
        self.assertGreater(feats["sentiment_polarity"], 0.0)
        self.assertGreater(feats["sentiment_subjectivity"], 0.0)
        self.assertLessEqual(feats["rating_vs_sentiment_mismatch"], 2.0)

    def test_rating_mismatch_boundary(self):
        """High rating with negative sentiment should produce a high mismatch score."""
        negative_text = "Terrible, horrible, worst disaster ever, completely broke."
        mismatch_5_star = extract_numeric_features(negative_text, rating=5)["rating_vs_sentiment_mismatch"]
        mismatch_1_star = extract_numeric_features(negative_text, rating=1)["rating_vs_sentiment_mismatch"]
        self.assertGreater(mismatch_5_star, mismatch_1_star)


class TextBlobSentimentEngineTestCase(TestCase):
    """Verifies TextBlob sentiment polarity analysis and rating mismatch bounds."""

    def test_positive_sentiment_computation(self):
        """Positive text must yield positive polarity."""
        polarity, subjectivity = compute_sentiment("Outstanding quality and wonderful experience.")
        self.assertGreater(polarity, 0.0)
        self.assertGreaterEqual(subjectivity, 0.0)
        self.assertLessEqual(subjectivity, 1.0)

    def test_negative_sentiment_computation(self):
        """Negative text must yield negative polarity."""
        polarity, subjectivity = compute_sentiment("Disappointing, broken, completely useless garbage.")
        self.assertLess(polarity, 0.0)


class CompositeScoringTestCase(TestCase):
    """Tests the configurable weighted formula synthesizing SVM, J-48, and TextBlob Sentiment."""

    def test_high_authenticity_genuine_review(self):
        """Low fake probabilities should yield a high score labeled Genuine."""
        score, label = compute_authenticity(
            svm_fake_prob=0.05,
            j48_fake_prob=0.10,
            mismatch_score=0.1
        )
        self.assertGreaterEqual(score, 50.0)
        self.assertEqual(label, "Genuine")

    def test_low_authenticity_fake_review(self):
        """High fake probabilities should yield a low score labeled Fake."""
        score, label = compute_authenticity(
            svm_fake_prob=0.92,
            j48_fake_prob=0.88,
            mismatch_score=1.5
        )
        self.assertLess(score, 50.0)
        self.assertEqual(label, "Fake")

    def test_binary_classification_boundary(self):
        """Boundary testing at 50.0% threshold: >= 50 is Genuine, < 50 is Fake."""
        score_gen, label_gen = compute_authenticity(
            svm_fake_prob=0.40,
            j48_fake_prob=0.40,
            mismatch_score=0.1
        )
        self.assertGreaterEqual(score_gen, 50.0)
        self.assertEqual(label_gen, "Genuine")

        score_fake, label_fake = compute_authenticity(
            svm_fake_prob=0.70,
            j48_fake_prob=0.70,
            mismatch_score=0.1
        )
        self.assertLess(score_fake, 50.0)
        self.assertEqual(label_fake, "Fake")


class AnalyzeReviewServiceTestCase(TestCase):
    """Verifies end-to-end analyze_review() service cached inference and schema."""

    def test_analyze_review_output_keys(self):
        """analyze_review() must return all required keys with valid data types."""
        res = analyze_review("Excellent wireless headphones with deep bass.", rating=5)

        required_keys = ['svm_score', 'j48_score', 'sentiment_polarity', 'sentiment_subjectivity', 'final_score', 'label', 'cleaned_text']
        for k in required_keys:
            self.assertIn(k, res)

        self.assertIsInstance(res['svm_score'], float)
        self.assertIsInstance(res['j48_score'], float)
        self.assertIsInstance(res['final_score'], float)
        self.assertIn(res['label'], ['Genuine', 'Fake'])
        self.assertGreaterEqual(res['final_score'], 0.0)
        self.assertLessEqual(res['final_score'], 100.0)

    def test_analyze_review_handles_none_rating_and_ip(self):
        """Gracefully handle None for rating and IP without exceptions."""
        res = analyze_review("Just a neutral review text without rating.", rating=None, ip=None)
        self.assertIsNotNone(res)
        self.assertIn('final_score', res)
