from django.urls import path
from . import views

urlpatterns = [
    path('products/<int:product_id>/review/', views.submit_review, name='submit_review'),
    path('reviews/<int:review_id>/edit/', views.edit_review, name='edit_review'),
    path('reviews/<int:review_id>/delete/', views.delete_review, name='delete_review'),
]

