from django import forms
from django.contrib.auth.models import User
from .models import UserProfile


class UserRegistrationForm(forms.ModelForm):
    full_name = forms.CharField(
        max_length=150,
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your full name',
            'id': 'id_full_name'
        }),
        label="Full Name"
    )
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your email address',
            'id': 'id_email'
        }),
        label="Email Address"
    )
    role = forms.ChoiceField(
        choices=[
            (UserProfile.ROLE_GENERAL, 'General User (Search & Browse)'),
            (UserProfile.ROLE_REVIEWER, 'Reviewer (Write Reviews & Test Detection)'),
        ],
        widget=forms.Select(attrs={
            'class': 'form-select',
            'id': 'id_role'
        }),
        label="Account Role"
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control password-input',
            'placeholder': 'Create password',
            'id': 'id_password'
        }),
        label="Password",
        min_length=6
    )
    confirm_password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control password-input',
            'placeholder': 'Confirm password',
            'id': 'id_confirm_password'
        }),
        label="Confirm Password"
    )

    class Meta:
        model = User
        fields = ['full_name', 'email', 'role', 'password', 'confirm_password']

    def clean_email(self):
        email = self.cleaned_data.get('email', '').strip().lower()
        if User.objects.filter(email__iexact=email).exists() or User.objects.filter(username__iexact=email).exists():
            raise forms.ValidationError("An account with this email address already exists. Please log in.")
        return email

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get("password")
        confirm_password = cleaned_data.get("confirm_password")

        if password and confirm_password and password != confirm_password:
            self.add_error('confirm_password', "Passwords do not match.")
        return cleaned_data

    def save(self, commit=True):
        user = super().save(commit=False)
        email = self.cleaned_data['email'].strip().lower()
        full_name = self.cleaned_data['full_name'].strip()

        # Split full name into first and last name
        name_parts = full_name.split(' ', 1)
        user.first_name = name_parts[0]
        user.last_name = name_parts[1] if len(name_parts) > 1 else ''

        user.email = email
        user.username = email  # Map username internally to email
        user.set_password(self.cleaned_data['password'])

        if commit:
            user.save()
            profile = user.profile
            profile.role = self.cleaned_data.get('role', UserProfile.ROLE_GENERAL)
            profile.save()

        return user


class EmailLoginForm(forms.Form):
    email = forms.EmailField(
        label="Email Address",
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your email address',
            'id': 'id_email',
            'autocomplete': 'email',
            'autofocus': True,
        })
    )
    password = forms.CharField(
        label="Password",
        widget=forms.PasswordInput(attrs={
            'class': 'form-control password-input',
            'placeholder': 'Enter your password',
            'id': 'id_password',
            'autocomplete': 'current-password',
        })
    )


class UserProfileUpdateForm(forms.ModelForm):
    full_name = forms.CharField(
        max_length=150,
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your full name',
            'id': 'id_full_name'
        }),
        label="Full Name"
    )

    class Meta:
        model = UserProfile
        fields = []

