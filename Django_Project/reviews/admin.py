from django.contrib import admin
from .models import Review


@admin.register(Review)
class ReviewAdmin(admin.ModelAdmin):
    list_display = ('id', 'product', 'user', 'rating', 'label', 'final_score', 'ip_address', 'ip_flag', 'created_at')
    list_filter = ('label', 'ip_flag', 'rating', 'created_at', 'source')
    search_fields = ('text', 'product__name', 'ip_address', 'user__username')
    readonly_fields = ('svm_score', 'j48_score', 'sentiment', 'ip_flag', 'final_score', 'label', 'created_at')

