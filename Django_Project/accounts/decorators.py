from functools import wraps
from django.shortcuts import redirect
from django.contrib import messages
from django.contrib.auth.mixins import AccessMixin


def get_role_dashboard_name(user):
    """Returns the named URL pattern of the appropriate dashboard for the user's role."""
    if not user or not user.is_authenticated:
        return 'login'
    profile = getattr(user, 'profile', None)
    if not profile:
        return 'user_dashboard'
    if profile.is_admin or user.is_superuser or user.is_staff:
        return 'admin_control_panel'
    elif profile.is_reviewer:
        return 'reviewer_dashboard'
    return 'user_dashboard'


def product_creator_required(view_func):
    """
    Decorator ensuring current user is permitted to create products.
    General Users and Administrators CAN create products.
    Reviewers CANNOT create products.
    """
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        if not request.user.is_authenticated:
            messages.warning(request, "Please log in to submit a new product.")
            return redirect('login')
        profile = getattr(request.user, 'profile', None)
        # Reviewers cannot create products per functional requirements
        if profile and profile.role == 'reviewer' and not (request.user.is_superuser or profile.is_admin):
            messages.error(request, "Access restricted: Reviewers evaluate existing products and cannot create new products.")
            return redirect('reviewer_dashboard')
        return view_func(request, *args, **kwargs)
    return _wrapped_view


def reviewer_required(view_func):
    """Decorator ensuring current user is a Reviewer (or Admin/Superuser)."""
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        if not request.user.is_authenticated:
            messages.warning(request, "Please log in to access the reviewer area.")
            return redirect('login')
        profile = getattr(request.user, 'profile', None)
        if not (profile and profile.is_reviewer):
            messages.error(request, "Permission denied: Reviewer privileges required.")
            return redirect('product_list')
        return view_func(request, *args, **kwargs)
    return _wrapped_view


def admin_required(view_func):
    """Decorator ensuring current user is an Admin or Superuser."""
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        if not request.user.is_authenticated:
            messages.warning(request, "Please log in with administrator privileges.")
            return redirect('login')
        profile = getattr(request.user, 'profile', None)
        if not (profile and profile.is_admin):
            messages.error(request, "Permission denied: Administrator privileges required.")
            return redirect('product_list')
        return view_func(request, *args, **kwargs)
    return _wrapped_view


class ReviewerRequiredMixin(AccessMixin):
    """CBV Mixin for reviewer authorization."""
    def dispatch(self, request, *args, **kwargs):
        if not request.user.is_authenticated:
            return self.handle_no_permission()
        profile = getattr(request.user, 'profile', None)
        if not (profile and profile.is_reviewer):
            messages.error(request, "Reviewer permissions required.")
            return redirect('product_list')
        return super().dispatch(request, *args, **kwargs)


class AdminRequiredMixin(AccessMixin):
    """CBV Mixin for administrator authorization."""
    def dispatch(self, request, *args, **kwargs):
        if not request.user.is_authenticated:
            return self.handle_no_permission()
        profile = getattr(request.user, 'profile', None)
        if not (profile and profile.is_admin):
            messages.error(request, "Administrator permissions required.")
            return redirect('product_list')
        return super().dispatch(request, *args, **kwargs)
