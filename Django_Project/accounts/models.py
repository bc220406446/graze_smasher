from django.db import models
from django.contrib.auth.models import User


class UserProfile(models.Model):
    """
    Extends standard Django User with Role-Based Access Control (RBAC).
    Roles:
      - General User: Browses products, views authenticity scores, saves favorites.
      - Reviewer: General User + can submit product reviews for ML authenticity analysis.
      - Admin: Full administrative privileges, manage flags, models, and catalog.
    """
    ROLE_GENERAL = 'general'
    ROLE_REVIEWER = 'reviewer'
    ROLE_ADMIN = 'admin'

    ROLE_CHOICES = [
        (ROLE_GENERAL, 'General User'),
        (ROLE_REVIEWER, 'Reviewer'),
        (ROLE_ADMIN, 'Admin'),
    ]

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default=ROLE_GENERAL)
    bio = models.TextField(blank=True, default='')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} ({self.get_role_display()})"

    @property
    def is_reviewer(self):
        return self.role in [self.ROLE_REVIEWER, self.ROLE_ADMIN] or self.user.is_staff or self.user.is_superuser

    @property
    def is_admin(self):
        return self.role == self.ROLE_ADMIN or self.user.is_staff or self.user.is_superuser

