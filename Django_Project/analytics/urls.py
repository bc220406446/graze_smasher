from django.urls import path
from . import views

urlpatterns = [
    # General User Dashboard & Subpages
    path('dashboard/', views.user_dashboard, name='user_dashboard'),
    path('saved-products/', views.saved_products_view, name='saved_products'),
    path('monitored-products/', views.monitored_products_view, name='monitored_products'),
    path('alerts/', views.user_alerts_view, name='user_alerts'),
    path('my-analytics/', views.user_personal_analytics, name='user_personal_analytics'),
    
    # Reviewer Dashboard & Subpages
    path('reviewer/dashboard/', views.reviewer_dashboard, name='reviewer_dashboard'),
    path('reviewer/my-reviews/', views.reviewer_my_reviews, name='reviewer_my_reviews'),
    path('reviewer/reviewed-products/', views.reviewer_reviewed_products, name='reviewer_reviewed_products'),
    
    # Global Analytics & Alerts
    path('analytics/', views.analytics_overview, name='analytics_overview'),
    path('suspicious-products/', views.suspicious_products_discovery, name='suspicious_discovery'),
    path('alerts/<int:alert_id>/read/', views.mark_alert_read, name='mark_alert_read'),
    path('alerts/<int:alert_id>/delete/', views.delete_alert, name='delete_alert'),
]
