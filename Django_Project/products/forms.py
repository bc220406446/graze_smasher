from django import forms
from .models import Product, ProductFlag


class ProductForm(forms.ModelForm):
    class Meta:
        model = Product
        fields = ['name', 'description', 'category', 'platform', 'source_url', 'image_url', 'image']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g. Sony WH-1000XM5 Wireless Headphones', 'required': True}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 4, 'placeholder': 'Product specifications, technical details, and features...'}),
            'category': forms.Select(attrs={'class': 'form-select'}),
            'platform': forms.Select(attrs={'class': 'form-select'}),
            'source_url': forms.URLInput(attrs={'class': 'form-control', 'placeholder': 'https://www.amazon.com/dp/B09XS7JWHH'}),
            'image_url': forms.URLInput(attrs={'class': 'form-control', 'placeholder': 'https://images.unsplash.com/... or direct image link'}),
            'image': forms.FileInput(attrs={'class': 'form-control'}),
        }


class ProductAdminForm(forms.ModelForm):
    class Meta:
        model = Product
        fields = ['name', 'description', 'category', 'platform', 'source_url', 'image_url', 'image', 'is_hidden', 'is_external']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 4}),
            'category': forms.Select(attrs={'class': 'form-select'}),
            'platform': forms.Select(attrs={'class': 'form-select'}),
            'source_url': forms.URLInput(attrs={'class': 'form-control'}),
            'image_url': forms.URLInput(attrs={'class': 'form-control'}),
            'image': forms.FileInput(attrs={'class': 'form-control'}),
            'is_hidden': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'is_external': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }


class ProductFlagForm(forms.ModelForm):
    class Meta:
        model = ProductFlag
        fields = ['category', 'details']
        widgets = {
            'category': forms.Select(attrs={'class': 'form-select'}),
            'details': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Provide specific details explaining why this product violates content guidelines (adult content, hate speech, counterfeit goods, or fraud)...',
                'required': True
            }),
        }
