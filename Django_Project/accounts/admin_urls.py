from django.urls import path
from . import admin_views
from . import views as account_views
import analytics.views as analytics_views

urlpatterns = [
    # 5. Administrator URLs (/administrator/)
    path('dashboard/', admin_views.admin_control_panel, name='admin_dashboard'),
    path('products/', admin_views.admin_products_list, name='admin_products'),
    path('product/<int:pk>/edit/', admin_views.admin_product_edit, name='admin_product_edit'),
    path('product/<int:pk>/delete/', admin_views.admin_product_delete, name='admin_product_delete'),
    path('product/<int:pk>/toggle-hide/', admin_views.admin_toggle_hide_product, name='admin_toggle_hide_product'),
    path('users/', admin_views.admin_users_view, name='admin_users'),
    path('users/add/', admin_views.admin_add_user, name='admin_add_user'),
    path('users/<int:id>/edit/', admin_views.admin_edit_user, name='admin_edit_user'),
    path('users/<int:id>/delete/', admin_views.admin_delete_user, name='admin_delete_user'),
    path('reports/', admin_views.admin_flagged_content_view, name='admin_flagged_content'),
    path('reports/<int:id>/resolve/', admin_views.admin_resolve_product_flag, name='admin_resolve_product_flag'),
    path('reviews/<int:id>/override/', admin_views.admin_override_review_label, name='admin_override_review_label'),
    path('settings/', admin_views.admin_settings_view, name='admin_settings'),
    path('model-config/', admin_views.admin_model_metrics_view, name='admin_model_metrics'),
    path('alerts/', analytics_views.user_alerts_view, name='admin_alerts'),
    path('alerts/<int:id>/read/', analytics_views.mark_alert_read, name='admin_mark_alert_read'),
    path('alerts/<int:id>/delete/', analytics_views.delete_alert, name='admin_delete_alert'),
    path('profile/', account_views.profile_view, name='admin_profile'),

    # Named aliases for backward compatibility
    path('', admin_views.admin_control_panel, name='admin_control_panel'),
]
