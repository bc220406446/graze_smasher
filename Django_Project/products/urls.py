from django.urls import path
from django.views.generic import RedirectView
from . import views

urlpatterns = [
    # 1. Public URLs (Guests & Unauthenticated Users)
    path('', views.home_view, name='home'),
    path('explore/', views.product_list, name='product_list'),
    path('product/<int:pk>/', views.product_detail, name='product_detail'),
    path('policy/', views.privacy_policy_view, name='privacy_policy'),

    # Backward compatibility aliases
    path('privacy/', views.privacy_policy_view),
    path('home/', views.home_view, name='home_alt'),
    path('products/<int:pk>/', RedirectView.as_view(url='/product/%(pk)s/', permanent=False)),
]
