from django.db import models
from django.contrib.auth.models import User
from django.db.models import Avg, Count, Q


class Product(models.Model):
    """
    Represents an e-commerce product tracked by the Graze Smasher tool.
    Can be created natively by users/reviewers or ingested via external platforms/APIs.
    """
    PLATFORM_CHOICES = [
        ('Amazon', 'Amazon'),
        ('eBay', 'eBay'),
        ('Walmart', 'Walmart'),
        ('Shopify', 'Shopify'),
        ('AliExpress', 'AliExpress'),
        ('Other', 'Other Platform'),
    ]

    CATEGORY_CHOICES = [
        ('Electronics', 'Electronics'),
        ('Home_and_Kitchen', 'Home & Kitchen'),
        ('Sports_and_Outdoors', 'Sports & Outdoors'),
        ('Clothing_and_Apparel', 'Clothing & Apparel'),
        ('Beauty_and_Personal_Care', 'Beauty & Personal Care'),
        ('Books', 'Books'),
        ('General', 'General Products'),
    ]

    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, default='')
    category = models.CharField(max_length=100, choices=CATEGORY_CHOICES, default='General')
    platform = models.CharField(max_length=50, choices=PLATFORM_CHOICES, default='Amazon')
    source_url = models.URLField(blank=True, null=True)
    image = models.ImageField(upload_to='product_images/', blank=True, null=True)
    image_url = models.URLField(blank=True, null=True, help_text="Direct image URL for demo/external products")
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='products')
    is_external = models.BooleanField(default=False, help_text="True if imported via third-party platform API")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} ({self.platform})"

    @property
    def display_image(self):
        if self.image:
            return self.image.url
        if self.image_url:
            return self.image_url
        return "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=600&auto=format&fit=crop&q=80"

    def review_count(self) -> int:
        return self.reviews.count()

    def average_rating(self) -> float:
        avg = self.reviews.aggregate(avg_r=Avg('rating'))['avg_r']
        return round(float(avg), 1) if avg is not None else 0.0

    def overall_authenticity_score(self) -> float:
        """Average composite authenticity score across all reviews for this product."""
        avg_score = self.reviews.aggregate(avg_s=Avg('final_score'))['avg_s']
        return round(float(avg_score), 1) if avg_score is not None else 100.0

    def authenticity_label(self) -> str:
        score = self.overall_authenticity_score()
        if score >= 70.0:
            return "Genuine"
        elif score >= 40.0:
            return "Suspicious"
        else:
            return "Fake"

    def genuine_reviews_count(self) -> int:
        return self.reviews.filter(label='Genuine').count()

    def fake_reviews_count(self) -> int:
        return self.reviews.filter(label='Fake').count()

    def suspicious_reviews_count(self) -> int:
        return self.reviews.filter(label='Suspicious').count()

    def community_trust_rating(self) -> float:
        """Trust rating computed exclusively from verified genuine customer reviews."""
        genuine_avg = self.reviews.filter(label='Genuine').aggregate(avg_r=Avg('rating'))['avg_r']
        return round(float(genuine_avg), 1) if genuine_avg is not None else self.average_rating()


class SavedProduct(models.Model):
    """
    Monitored product saved in a general user or reviewer watchlist.
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='saved_products')
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='saved_by')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'product')
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user.username} saved {self.product.name}"

