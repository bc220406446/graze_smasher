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
    # 1. Public URLs (Guests & Unauthenticated Users)
    path('', include('products.urls')),

    # 2. Authentication URLs (/account/)
    path('accounts/', include('accounts.urls')),
    path('account/', include('accounts.urls')),

    # 3. General User URLs (/user/)
    path('user/', include('analytics.user_urls')),

    # 4. Reviewer URLs (/reviewer/)
    path('reviewer/', include('analytics.reviewer_urls')),

    # 5. Administrator URLs (/administrator/)
    path('administrator/', include('accounts.admin_urls')),

    # 6. Django Core Admin (/admin/)
    path('admin/', admin.site.urls),

    # 7. Programmatic APIs & ML Inference (/api/ & /detector/)
    path('api/', include('api.urls')),
    path('detector/', include('detector.urls')),

    # Backward compatibility redirects
    path('admin/dashboard/', RedirectView.as_view(url='/administrator/dashboard/', permanent=False), name='admin_dashboard_redirect'),
    path('admin-panel/', RedirectView.as_view(url='/administrator/dashboard/', permanent=False)),
    path('accounts/admin-panel/', RedirectView.as_view(url='/administrator/dashboard/', permanent=False)),
    path('accounts/admin-panel/dashboard/', RedirectView.as_view(url='/administrator/dashboard/', permanent=False)),
    path('accounts/admin-panel/products/', RedirectView.as_view(url='/administrator/products/', permanent=False)),
    path('accounts/admin-panel/users/', RedirectView.as_view(url='/administrator/users/', permanent=False)),
    path('accounts/admin-panel/flagged/', RedirectView.as_view(url='/administrator/reports/', permanent=False)),
    path('accounts/admin-panel/settings/', RedirectView.as_view(url='/administrator/settings/', permanent=False)),
    path('accounts/admin-panel/models/', RedirectView.as_view(url='/administrator/model-config/', permanent=False)),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
