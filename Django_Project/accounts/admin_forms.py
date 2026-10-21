from django import forms
from django.contrib.auth.models import User
from analytics.models import SystemConfiguration
from .models import UserProfile


class AdminAddUserForm(forms.Form):
    full_name = forms.CharField(
        max_length=150,
        required=True,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g. Alex Morgan'}),
        label="Full Name"
    )
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={'class': 'form-control', 'placeholder': 'name@example.com'}),
        label="Email Address"
    )
    role = forms.ChoiceField(
        choices=[
            (UserProfile.ROLE_GENERAL, 'General User'),
            (UserProfile.ROLE_REVIEWER, 'Reviewer'),
        ],
        required=True,
        widget=forms.Select(attrs={'class': 'form-select'}),
        label="Account Role"
    )
    password = forms.CharField(
        min_length=6,
        required=True,
        widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Minimum 6 characters'}),
        label="Initial Password"
    )

    def clean_email(self):
        email = self.cleaned_data.get('email', '').strip().lower()
        if User.objects.filter(email__iexact=email).exists() or User.objects.filter(username__iexact=email).exists():
            raise forms.ValidationError("An account with this email address already exists.")
        return email

    def save(self):
        email = self.cleaned_data['email'].strip().lower()
        full_name = self.cleaned_data['full_name'].strip()
        role = self.cleaned_data['role']
        password = self.cleaned_data['password']

        parts = full_name.split(' ', 1)
        first_name = parts[0]
        last_name = parts[1] if len(parts) > 1 else ''

        user = User.objects.create_user(
            username=email,
            email=email,
            password=password,
            first_name=first_name,
            last_name=last_name,
            is_active=True
        )
        profile = user.profile
        profile.role = role
        profile.save()
        return user


class AdminEditUserForm(forms.Form):
    full_name = forms.CharField(
        max_length=150,
        required=True,
        widget=forms.TextInput(attrs={'class': 'form-control'}),
        label="Full Name"
    )
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={'class': 'form-control'}),
        label="Email Address"
    )
    new_password = forms.CharField(
        min_length=6,
        required=False,
        widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Leave blank to keep existing password'}),
        label="Reset Password (Optional)"
    )

    def __init__(self, *args, target_user=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_user = target_user
        if target_user and not self.is_bound:
            self.fields['full_name'].initial = target_user.get_full_name() or target_user.first_name
            self.fields['email'].initial = target_user.email

    def clean_email(self):
        email = self.cleaned_data.get('email', '').strip().lower()
        if self.target_user:
            qs = User.objects.filter(email__iexact=email).exclude(pk=self.target_user.pk)
            if qs.exists():
                raise forms.ValidationError("Another account with this email address already exists.")
        return email

    def save(self):
        if not self.target_user:
            return None
        email = self.cleaned_data['email'].strip().lower()
        full_name = self.cleaned_data['full_name'].strip()
        parts = full_name.split(' ', 1)
        self.target_user.first_name = parts[0]
        self.target_user.last_name = parts[1] if len(parts) > 1 else ''
        self.target_user.email = email
        self.target_user.username = email

        new_password = self.cleaned_data.get('new_password')
        if new_password and new_password.strip():
            self.target_user.set_password(new_password.strip())

        self.target_user.save()
        return self.target_user


class SystemConfigurationForm(forms.ModelForm):
    class Meta:
        model = SystemConfiguration
        fields = [
            'weight_svm',
            'weight_j48',
            'weight_risk',
            'genuine_threshold',
            'ip_max_reviews',
        ]
        widgets = {
            'weight_svm': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.05', 'min': '0', 'max': '1'}),
            'weight_j48': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.05', 'min': '0', 'max': '1'}),
            'weight_risk': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.05', 'min': '0', 'max': '1'}),
            'genuine_threshold': forms.NumberInput(attrs={'class': 'form-control', 'step': '1.0', 'min': '10', 'max': '90'}),
            'ip_max_reviews': forms.NumberInput(attrs={'class': 'form-control', 'min': '1', 'max': '3'}),
        }
        labels = {
            'weight_svm': 'LinearSVC Weight (Textual TF-IDF)',
            'weight_j48': 'J-48 Weight (Behavioral / Structural)',
            'weight_risk': 'Sentiment Mismatch Weight',
            'genuine_threshold': 'Genuine Classification Cutoff (%)',
            'ip_max_reviews': 'Max Reviews Per IP Per Minute (Rate Limit)',
        }

    def clean(self):
        cleaned_data = super().clean()
        w_svm = cleaned_data.get('weight_svm', 0)
        w_j48 = cleaned_data.get('weight_j48', 0)
        w_risk = cleaned_data.get('weight_risk', 0)
        total_w = round(w_svm + w_j48 + w_risk, 2)
        if total_w != 1.0:
            raise forms.ValidationError(
                f"Model weights must sum exactly to 1.0 (Current sum: {total_w}). Please adjust weights."
            )
        return cleaned_data


