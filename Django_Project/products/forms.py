from django import forms
from .models import Product


class ProductForm(forms.ModelForm):
    class Meta:
        model = Product
        fields = ['name', 'description', 'category', 'platform', 'source_url', 'image_url', 'image']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g. Sony WH-1000XM5 Wireless Headphones'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 4, 'placeholder': 'Product specifications and details...'}),
            'category': forms.Select(attrs={'class': 'form-select'}),
            'platform': forms.Select(attrs={'class': 'form-select'}),
            'source_url': forms.URLInput(attrs={'class': 'form-control', 'placeholder': 'https://www.amazon.com/dp/...'}),
            'image_url': forms.URLInput(attrs={'class': 'form-control', 'placeholder': 'https://example.com/image.jpg'}),
            'image': forms.FileInput(attrs={'class': 'form-control'}),
        }

