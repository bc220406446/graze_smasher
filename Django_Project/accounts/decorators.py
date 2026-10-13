from functools import wraps
from django.shortcuts import redirect
from django.contrib import messages
from django.contrib.auth.mixins import AccessMixin


def reviewer_required(view_func):
    """Decorator ensuring current user is at least a Reviewer (or Admin/Superuser)."""
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        if not request.user.is_authenticated:
            messages.warning(request, "Please log in to perform this action.")
            return redirect('login')
        profile = getattr(request.user, 'profile', None)
        if not (profile and profile.is_reviewer):
            messages.error(request, "Permission denied: Only Reviewers or Admins can submit reviews.")
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

