from django.contrib import admin
from .models import Product, SavedProduct


@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = ('name', 'category', 'platform', 'is_external', 'created_at')
    list_filter = ('category', 'platform', 'is_external', 'created_at')
    search_fields = ('name', 'description')


@admin.register(SavedProduct)
class SavedProductAdmin(admin.ModelAdmin):
    list_display = ('user', 'product', 'created_at')
    list_filter = ('created_at',)

