from django import forms
from .models import Review


class ReviewSubmissionForm(forms.ModelForm):
    RATING_CHOICES = [
        (5, '⭐⭐⭐⭐⭐ 5 Stars'),
        (4, '⭐⭐⭐⭐ 4 Stars'),
        (3, '⭐⭐⭐ 3 Stars'),
        (2, '⭐⭐ 2 Stars'),
        (1, '⭐ 1 Star'),
    ]

    rating = forms.ChoiceField(
        choices=RATING_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select'}),
        initial=5
    )

    class Meta:
        model = Review
        fields = ['rating', 'text']
        widgets = {
            'text': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Share your detailed experience with this product...'
            }),
        }

