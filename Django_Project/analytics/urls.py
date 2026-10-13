from django.urls import path
from . import views

urlpatterns = [
    path('dashboard/', views.user_dashboard, name='user_dashboard'),
    path('analytics/', views.analytics_overview, name='analytics_overview'),
    path('suspicious-products/', views.suspicious_products_discovery, name='suspicious_discovery'),
    path('alerts/<int:alert_id>/read/', views.mark_alert_read, name='mark_alert_read'),
]

