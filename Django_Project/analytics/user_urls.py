from django.urls import path
from . import views as analytics_views
import accounts.views as account_views
import products.views as product_views

urlpatterns = [
    # 3. General User URLs (/user/)
    path('dashboard/', analytics_views.user_dashboard, name='user_dashboard'),
    path('explore/', product_views.product_list, name='user_explore'),
    path('saved-products/', analytics_views.saved_products_view, name='saved_products'),
    path('monitored-products/', analytics_views.monitored_products_view, name='monitored_products'),
    path('my-reports/', analytics_views.user_my_reports, name='user_my_reports'),
    path('alerts/', analytics_views.user_alerts_view, name='user_alerts'),
    path('alerts/<int:id>/read/', analytics_views.mark_alert_read, name='mark_alert_read'),
    path('alerts/<int:id>/delete/', analytics_views.delete_alert, name='delete_alert'),
    path('profile/', account_views.profile_view, name='profile'),
    path('product/<int:pk>/', product_views.product_detail, name='user_product_detail'),
    path('product/<int:pk>/monitor-toggle/', product_views.toggle_save_product, name='toggle_save_product'),
    path('product/<int:pk>/report/', product_views.flag_product_view, name='flag_product'),
    path('add-product/', product_views.create_product, name='product_create'),
    path('edit-product/<int:pk>/', product_views.edit_product, name='product_edit'),
    path('delete-product/<int:pk>/', product_views.delete_product, name='product_delete'),
]

