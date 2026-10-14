from django.urls import path
from . import views

urlpatterns = [
    path('products/', views.ProductListCreateAPIView.as_view(), name='product_list_create'),
    path('products/<int:pk>/', views.ProductDetailAPIView.as_view(), name='api_product_detail'),
    path('products/<int:pk>/reviews/', views.ProductReviewsAPIView.as_view(), name='product_reviews'),
    path('reviews/', views.SingleReviewCreateAPIView.as_view(), name='review_create'),
]
