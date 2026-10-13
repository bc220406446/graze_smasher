from django import forms
from analytics.models import SystemConfiguration
from .models import UserProfile


class UserRoleUpdateForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ['role']
        widgets = {
            'role': forms.Select(attrs={'class': 'form-select form-select-sm'}),
        }


class SystemConfigurationForm(forms.ModelForm):
    class Meta:
        model = SystemConfiguration
        fields = [
            'weight_svm',
            'weight_j48',
            'weight_risk',
            'genuine_threshold',
            'suspicious_threshold',
            'ip_max_reviews',
            'ip_window_minutes',
            'alert_fake_ratio_threshold',
            'alert_fake_spike_threshold',
        ]
        widgets = {
            'weight_svm': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.05', 'min': '0', 'max': '1'}),
            'weight_j48': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.05', 'min': '0', 'max': '1'}),
            'weight_risk': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.05', 'min': '0', 'max': '1'}),
            'genuine_threshold': forms.NumberInput(attrs={'class': 'form-control', 'step': '1', 'min': '0', 'max': '100'}),
            'suspicious_threshold': forms.NumberInput(attrs={'class': 'form-control', 'step': '1', 'min': '0', 'max': '100'}),
            'ip_max_reviews': forms.NumberInput(attrs={'class': 'form-control', 'min': '1'}),
            'ip_window_minutes': forms.NumberInput(attrs={'class': 'form-control', 'min': '1'}),
            'alert_fake_ratio_threshold': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.05', 'min': '0', 'max': '1'}),
            'alert_fake_spike_threshold': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.05', 'min': '0', 'max': '1'}),
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

