from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout, update_session_auth_hash
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import (
    UserRegistrationForm,
    UserProfileUpdateForm,
    EmailLoginForm,
    UserPasswordChangeForm,
)
from .models import UserProfile


from .decorators import get_role_dashboard_name


def register_view(request):
    if request.user.is_authenticated:
        return redirect(get_role_dashboard_name(request.user))
        
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            display_name = user.get_full_name() or user.email
            messages.success(request, f"Account created successfully for {display_name}! You can now log in.")
            return redirect('login')
    else:
        form = UserRegistrationForm()
        
    return render(request, 'auth/register.html', {'form': form})


def login_view(request):
    if request.user.is_authenticated:
        return redirect(get_role_dashboard_name(request.user))
        
    if request.method == 'POST':
        form = EmailLoginForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data.get('email')
            password = form.cleaned_data.get('password')
            user = authenticate(request, username=email, password=password)
            if user is not None:
                login(request, user)
                display_name = user.get_full_name() or user.email
                messages.success(request, f"Welcome back, {display_name}!")
                next_url = request.GET.get('next')
                if next_url and next_url not in ['/', '/accounts/login/', '/accounts/register/']:
                    return redirect(next_url)
                return redirect(get_role_dashboard_name(user))
            else:
                form.add_error(None, "Invalid email address or password. Please try again.")
        else:
            messages.error(request, "Please enter a valid email address and password.")
    else:
        form = EmailLoginForm()
        
    return render(request, 'auth/login.html', {'form': form})


def logout_view(request):
    logout(request)
    messages.info(request, "You have been logged out successfully.")
    return redirect('product_list')


@login_required
def profile_view(request):
    user = request.user
    profile = user.profile
    
    password_form = UserPasswordChangeForm(user=user)
    
    if request.method == 'POST':
        if 'change_password' in request.POST or request.POST.get('action') == 'change_password':
            password_form = UserPasswordChangeForm(user=user, data=request.POST)
            initial = {
                'full_name': user.get_full_name() or user.first_name or '',
            }
            form = UserProfileUpdateForm(instance=profile, initial=initial)
            if password_form.is_valid():
                user = password_form.save()
                update_session_auth_hash(request, user)
                messages.success(request, "Your password has been changed successfully.")
                return redirect('profile')
            else:
                for field, errors in password_form.errors.items():
                    for error in errors:
                        messages.error(request, error)
        else:
            form = UserProfileUpdateForm(request.POST, instance=profile)
            if form.is_valid():
                full_name = form.cleaned_data['full_name'].strip()
                name_parts = full_name.split(' ', 1)
                user.first_name = name_parts[0]
                user.last_name = name_parts[1] if len(name_parts) > 1 else ''
                user.save()
                form.save()
                messages.success(request, "Your profile has been updated.")
                return redirect('profile')
    else:
        initial = {
            'full_name': user.get_full_name() or user.first_name or '',
        }
        form = UserProfileUpdateForm(instance=profile, initial=initial)
        
    template_name = 'includes/profile.html'

    return render(request, template_name, {
        'form': form,
        'password_form': password_form,
        'profile': profile,
        'is_admin': user.is_superuser or profile.is_admin,
        'is_reviewer': profile.is_reviewer,
    })

