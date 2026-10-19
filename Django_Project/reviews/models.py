from django.db import models
from django.contrib.auth.models import User
from products.models import Product


class Review(models.Model):
    """
    Stores an individual customer review associated with a Product.
    Populated with all multi-tier ML scores and heuristic flags from analyze_review().
    """
    LABEL_GENUINE = 'Genuine'
    LABEL_FAKE = 'Fake'

    LABEL_CHOICES = [
        (LABEL_GENUINE, 'Genuine'),
        (LABEL_FAKE, 'Fake'),
    ]

    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='reviews')
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='reviews')
    text = models.TextField()
    rating = models.IntegerField(default=5)
    
    # ML Multi-Tier Inference Scores
    svm_score = models.FloatField(default=0.0, help_text="Calibrated SVM fake probability [0.0 - 1.0]")
    j48_score = models.FloatField(default=0.0, help_text="J-48 decision tree fake probability [0.0 - 1.0]")
    sentiment = models.FloatField(default=0.0, help_text="TextBlob sentiment polarity [-1.0 - 1.0]")
    final_score = models.FloatField(default=0.0, help_text="Composite authenticity score [0.0 - 100.0, higher = genuine]")
    label = models.CharField(max_length=20, choices=LABEL_CHOICES, default=LABEL_GENUINE)
    
    source = models.CharField(max_length=50, default='platform', help_text="e.g. 'platform', 'dataset_seed', 'api'")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'reviews_with_analysis'
        ordering = ['-created_at']

    def __str__(self):
        return f"Review #{self.id} on {self.product.name} ({self.label} - {self.final_score}%)"

    @property
    def is_fake(self):
        return self.label == self.LABEL_FAKE

    @property
    def is_genuine(self):
        return self.label == self.LABEL_GENUINE


class ReviewerReviewedProduct(models.Model):
    """
    Minimalist table tracking products that a specific reviewer has evaluated.
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='reviewed_products_rel')
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='reviewed_by_reviewers')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'reviewer_reviewed_products'
        unique_together = ('user', 'product')
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user.username} reviewed {self.product.name}"

