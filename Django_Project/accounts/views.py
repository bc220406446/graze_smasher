from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import UserRegistrationForm, UserProfileUpdateForm
from .models import UserProfile


from .decorators import get_role_dashboard_name


def register_view(request):
    if request.user.is_authenticated:
        return redirect(get_role_dashboard_name(request.user))
        
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data['password'])
            user.save()
            
            # Update role on profile (created via post_save signal)
            profile = user.profile
            profile.role = form.cleaned_data.get('role', UserProfile.ROLE_GENERAL)
            profile.save()
            
            messages.success(request, f"Account created successfully for {user.username}! You can now log in.")
            return redirect('login')
    else:
        form = UserRegistrationForm()
        
    return render(request, 'auth/register.html', {'form': form})


def login_view(request):
    if request.user.is_authenticated:
        return redirect(get_role_dashboard_name(request.user))
        
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f"Welcome back, {user.username}!")
                next_url = request.GET.get('next')
                if next_url and next_url not in ['/', '/accounts/login/', '/accounts/register/']:
                    return redirect(next_url)
                return redirect(get_role_dashboard_name(user))
        else:
            messages.error(request, "Invalid username or password. Please try again.")
    else:
        form = AuthenticationForm()
        
    return render(request, 'auth/login.html', {'form': form})


def logout_view(request):
    logout(request)
    messages.info(request, "You have been logged out successfully.")
    return redirect('product_list')


@login_required
def profile_view(request):
    user = request.user
    profile = user.profile
    
    if request.method == 'POST':
        form = UserProfileUpdateForm(request.POST, instance=profile)
        if form.is_valid():
            user.first_name = form.cleaned_data['first_name']
            user.last_name = form.cleaned_data['last_name']
            user.email = form.cleaned_data['email']
            user.save()
            form.save()
            messages.success(request, "Your profile has been updated.")
            return redirect('profile')
    else:
        initial = {
            'first_name': user.first_name,
            'last_name': user.last_name,
            'email': user.email
        }
        form = UserProfileUpdateForm(instance=profile, initial=initial)
        
    template_name = 'users/general/profile.html'
    if user.is_superuser or profile.is_admin:
        template_name = 'users/admin/profile.html'
    elif profile.is_reviewer:
        template_name = 'users/reviewer/profile.html'

    return render(request, template_name, {
        'form': form,
        'profile': profile
    })

