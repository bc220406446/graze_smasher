from django.urls import path
from . import views

urlpatterns = [
    path('', views.product_list, name='product_list'),
    path('products/<int:pk>/', views.product_detail, name='product_detail'),
    path('products/create/', views.create_product, name='product_create'),
    path('products/<int:pk>/save/', views.toggle_save_product, name='toggle_save_product'),
    path('privacy/', views.privacy_policy_view, name='privacy_policy'),
]

