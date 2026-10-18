from django.urls import path
from . import views as analytics_views
import accounts.views as account_views
import products.views as product_views
import reviews.views as review_views

urlpatterns = [
    # 4. Reviewer URLs (/reviewer/)
    path('dashboard/', analytics_views.reviewer_dashboard, name='reviewer_dashboard'),
    path('explore/', product_views.product_list, name='reviewer_explore'),
    path('my-reviews/', analytics_views.reviewer_my_reviews, name='reviewer_my_reviews'),
    path('reviewed-products/', analytics_views.reviewer_reviewed_products, name='reviewer_reviewed_products'),
    path('alerts/', analytics_views.user_alerts_view, name='reviewer_alerts'),
    path('alerts/<int:id>/read/', analytics_views.mark_alert_read, name='reviewer_mark_alert_read'),
    path('alerts/<int:id>/delete/', analytics_views.delete_alert, name='reviewer_delete_alert'),
    path('profile/', account_views.profile_view, name='reviewer_profile'),
    path('product/<int:pk>/', product_views.product_detail, name='reviewer_product_detail'),
    path('product/<int:pk>/review/', review_views.submit_review, name='submit_review'),
    path('review/<int:id>/edit/', review_views.edit_review, name='edit_review'),
    path('review/<int:id>/delete/', review_views.delete_review, name='delete_review'),
]

