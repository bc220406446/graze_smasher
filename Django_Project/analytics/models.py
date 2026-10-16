from django.db import models
from django.contrib.auth.models import User
from products.models import Product


class Alert(models.Model):
    """
    Automated notification triggered when abnormal fake review surges,
    suspicious IP velocity bursts, or critical drops in authenticity occur.
    """
    ALERT_TYPES = [
        ('fake_spike', 'Fake Review Spike'),
        ('ip_burst', 'Suspicious IP Burst'),
        ('low_authenticity', 'Low Authenticity Warning'),
        ('sentiment_mismatch', 'Extreme Rating Discrepancy'),
    ]

    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='alerts')
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='alerts',
                             help_text="Recipient user or null for public/system alerts")
    type = models.CharField(max_length=50, choices=ALERT_TYPES, default='fake_spike')
    message = models.TextField()
    is_read = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"[{self.get_type_display()}] {self.product.name}: {self.message[:40]}"


class SystemConfiguration(models.Model):
    """
    Singleton model providing runtime-editable detection weights, thresholds,
    and velocity rules without requiring application restarts.
    """
    weight_svm = models.FloatField(default=0.50, help_text="Weight for SVM fake probability (default: 0.50)")
    weight_j48 = models.FloatField(default=0.30, help_text="Weight for J-48 decision tree fake probability (default: 0.30)")
    weight_risk = models.FloatField(default=0.20, help_text="Weight for heuristic/IP risk (default: 0.20)")
    
    genuine_threshold = models.FloatField(default=50.0, help_text="Score >= threshold is classified as Genuine (default: 50.0)")
    suspicious_threshold = models.FloatField(default=50.0, help_text="Binary threshold (default: 50.0)")
    
    ip_max_reviews = models.IntegerField(default=5, help_text="Maximum allowed reviews per IP within velocity window (default: 5)")
    ip_window_minutes = models.IntegerField(default=60, help_text="Sliding time window for IP velocity in minutes (default: 60)")
    
    alert_fake_ratio_threshold = models.FloatField(default=0.40, help_text="Ratio of fake reviews triggering an alert (default: 0.40)")
    alert_fake_spike_threshold = models.FloatField(default=0.25, help_text="Surge in fake ratio triggering a spike alert (default: 0.25)")
    
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"System Detection Configuration (Updated: {self.updated_at.strftime('%Y-%m-%d %H:%M')})"

    @classmethod
    def get_solo(cls):
        """Retrieves or initializes the single system configuration instance."""
        config, _ = cls.objects.get_or_create(id=1)
        return config
