from django.contrib import admin
from .models import Review, ReviewerReviewedProduct


@admin.register(Review)
class ReviewAdmin(admin.ModelAdmin):
    list_display = ('id', 'product', 'user', 'rating', 'label', 'final_score', 'created_at')
    list_filter = ('label', 'rating', 'created_at', 'source')
    search_fields = ('text', 'product__name', 'user__username')
    readonly_fields = ('svm_score', 'j48_score', 'sentiment', 'final_score', 'label', 'created_at')


@admin.register(ReviewerReviewedProduct)
class ReviewerReviewedProductAdmin(admin.ModelAdmin):
    list_display = ('user', 'product', 'created_at')
    list_filter = ('created_at',)

