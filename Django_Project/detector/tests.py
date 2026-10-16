"""
Unit Tests for Machine Learning Service & Heuristic Detectors
CS619 Final-Year Project: Graze Smasher Product Hunting Tool
"""

import time
from django.test import TestCase
from ML_Model.feature_extraction import extract_numeric_features, compute_sentiment
from ML_Model.config import J48_FEATURE_NAMES
from ML_Model.ip_detector import IPVelocityTracker, generate_simulated_ips
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


class IPVelocityDetectorTestCase(TestCase):
    """Verifies sliding-window IP velocity tracking and simulated IP generator."""

    def test_generate_simulated_ips(self):
        """Generated IPs must be well-formed RFC 5737 test IP strings."""
        ips = generate_simulated_ips(10)
        self.assertEqual(len(ips), 10)
        for ip in ips:
            self.assertTrue(ip.startswith("192.0.2.") or ip.startswith("198.51.100.") or ip.startswith("203.0.113."))

    def test_ip_velocity_trigger(self):
        """Exceeding max_reviews within the window must trigger ip_flag."""
        tracker = IPVelocityTracker(max_reviews=3, window_minutes=5)
        test_ip = "192.0.2.99"

        # 1st, 2nd, 3rd submission: OK
        self.assertFalse(tracker.record_submission(test_ip))
        self.assertFalse(tracker.record_submission(test_ip))
        self.assertFalse(tracker.record_submission(test_ip))

        # 4th submission within window: Flagged!
        self.assertTrue(tracker.record_submission(test_ip))

    def test_ip_window_expiration(self):
        """Submissions outside the time window must not trigger velocity flag."""
        tracker = IPVelocityTracker(max_reviews=2, window_minutes=0.001)
        test_ip = "192.0.2.100"

        tracker.record_submission(test_ip)
        tracker.record_submission(test_ip)
        time.sleep(0.1)
        flagged = tracker.record_submission(test_ip)
        self.assertFalse(flagged)


class CompositeScoringTestCase(TestCase):
    """Tests the configurable weighted formula synthesizing SVM, J-48, and heuristic risk."""

    def test_high_authenticity_genuine_review(self):
        """Low fake probabilities with clean IP should yield a high score labeled Genuine."""
        score, label = compute_authenticity(
            svm_fake_prob=0.05,
            j48_fake_prob=0.10,
            ip_flag=False,
            mismatch_score=0.1
        )
        self.assertGreaterEqual(score, 50.0)
        self.assertEqual(label, "Genuine")

    def test_low_authenticity_fake_review(self):
        """High fake probabilities with an IP velocity flag should yield a low score labeled Fake."""
        score, label = compute_authenticity(
            svm_fake_prob=0.92,
            j48_fake_prob=0.88,
            ip_flag=True,
            mismatch_score=1.5
        )
        self.assertLess(score, 50.0)
        self.assertEqual(label, "Fake")

    def test_binary_classification_boundary(self):
        """Boundary testing at 50.0% threshold: >= 50 is Genuine, < 50 is Fake."""
        score_gen, label_gen = compute_authenticity(
            svm_fake_prob=0.40,
            j48_fake_prob=0.40,
            ip_flag=False,
            mismatch_score=0.1
        )
        self.assertGreaterEqual(score_gen, 50.0)
        self.assertEqual(label_gen, "Genuine")

        score_fake, label_fake = compute_authenticity(
            svm_fake_prob=0.70,
            j48_fake_prob=0.70,
            ip_flag=False,
            mismatch_score=0.1
        )
        self.assertLess(score_fake, 50.0)
        self.assertEqual(label_fake, "Fake")


class AnalyzeReviewServiceTestCase(TestCase):
    """Verifies end-to-end analyze_review() service cached inference and schema."""

    def test_analyze_review_output_keys(self):
        """analyze_review() must return all required keys with valid data types."""
        res = analyze_review("Excellent wireless headphones with deep bass.", rating=5, ip="198.51.100.5", record_ip=False)

        required_keys = ['svm_score', 'j48_score', 'sentiment_polarity', 'sentiment_subjectivity', 'ip_flag', 'final_score', 'label', 'cleaned_text']
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
        res = analyze_review("Just a neutral review text without rating.", rating=None, ip=None, record_ip=False)
        self.assertIsNotNone(res)
        self.assertIn('final_score', res)
