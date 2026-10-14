from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_view, name='home'),
    path('explore/', views.product_list, name='product_list'),
    path('home/', views.home_view, name='home_alt'),
    path('products/<int:pk>/', views.product_detail, name='product_detail'),
    path('products/create/', views.create_product, name='product_create'),
    path('products/<int:pk>/save/', views.toggle_save_product, name='toggle_save_product'),
    path('products/<int:pk>/flag/', views.flag_product_view, name='flag_product'),
    path('privacy/', views.privacy_policy_view, name='privacy_policy'),
]
