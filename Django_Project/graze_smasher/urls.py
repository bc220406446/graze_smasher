"""
URL Configuration for Graze Smasher.
CS619 Final-Year Project
"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import RedirectView

urlpatterns = [
    path('admin/dashboard/', RedirectView.as_view(url='/accounts/admin-panel/', permanent=False), name='admin_dashboard_redirect'),
    path('admin/', admin.site.urls),
    path('admin-panel/', RedirectView.as_view(url='/accounts/admin-panel/', permanent=False)),
    path('accounts/', include('accounts.urls')),
    path('', include('products.urls')),
    path('', include('reviews.urls')),
    path('', include('analytics.urls')),
    path('detector/', include('detector.urls')),
    path('api/', include('api.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
