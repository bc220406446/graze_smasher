from django.urls import path
from django.contrib.auth import views as auth_views
from . import views
from . import admin_views

urlpatterns = [
    # User Authentication
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('profile/', views.profile_view, name='profile'),
    
    # Password Reset
    path('password_reset/', auth_views.PasswordResetView.as_view(
        template_name='auth/password_reset.html'
    ), name='password_reset'),
    path('password_reset/done/', auth_views.PasswordResetDoneView.as_view(
        template_name='auth/password_reset_done.html'
    ), name='password_reset_done'),
    path('reset/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(
        template_name='auth/password_reset_confirm.html'
    ), name='password_reset_confirm'),
    path('reset/done/', auth_views.PasswordResetCompleteView.as_view(
        template_name='auth/password_reset_complete.html'
    ), name='password_reset_complete'),

    # Custom Admin Control Panel
    path('admin-panel/', admin_views.admin_control_panel, name='admin_control_panel'),
    path('admin-panel/dashboard/', admin_views.admin_control_panel, name='admin_dashboard'),
    
    # Product Management
    path('admin-panel/products/', admin_views.admin_products_list, name='admin_products'),
    path('admin-panel/products/<int:product_id>/edit/', admin_views.admin_product_edit, name='admin_product_edit'),
    path('admin-panel/products/<int:product_id>/delete/', admin_views.admin_product_delete, name='admin_product_delete'),
    path('admin-panel/products/<int:product_id>/toggle-hide/', admin_views.admin_toggle_hide_product, name='admin_toggle_hide_product'),

    # User Management
    path('admin-panel/users/', admin_views.admin_users_view, name='admin_users'),
    path('admin-panel/users/<int:user_id>/role/', admin_views.admin_update_user_role, name='admin_update_user_role'),

    # Flagged Content Moderation
    path('admin-panel/flagged/', admin_views.admin_flagged_content_view, name='admin_flagged_content'),
    path('admin-panel/flags/<int:flag_id>/resolve/', admin_views.admin_resolve_product_flag, name='admin_resolve_product_flag'),
    path('admin-panel/reviews/<int:review_id>/override/', admin_views.admin_override_review_label, name='admin_override_review_label'),

    # System Settings & ML Model Performance
    path('admin-panel/settings/', admin_views.admin_settings_view, name='admin_settings'),
    path('admin-panel/models/', admin_views.admin_model_metrics_view, name='admin_model_metrics'),

    # Analytics & Reports
    path('admin-panel/analytics/', admin_views.admin_analytics_reports, name='admin_analytics_reports'),

    # API Integration
    path('admin-panel/api/', admin_views.admin_api_integration, name='admin_api_integration'),
]
