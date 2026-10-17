from django.shortcuts import get_object_or_404, redirect
from django.contrib import messages
from django.views.decorators.http import require_POST
from products.models import Product
from analytics.models import Alert
from accounts.decorators import reviewer_required
from detector.utils import analyze_review
from .models import Review
from .forms import ReviewSubmissionForm


def get_client_ip(request):
    """Safely extracts real client IP address from HTTP request headers."""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0].strip()
    else:
        ip = request.META.get('REMOTE_ADDR', '127.0.0.1')
    return ip


@require_POST
@reviewer_required
def submit_review(request, product_id):
    """
    Handles reviewer review submission:
    1. Enforces single review per product limit for Reviewers
    2. Zero IP address storage for user privacy compliance
    3. Calls unified 100% text-based ML analyze_review() pipeline (SVM + J-48 + Sentiment)
    4. Persists Review record with multi-tier ML scores and labels
    5. Automatically generates an Alert if suspicious activity or fake content is detected
    """
    product = get_object_or_404(Product, pk=product_id)

    # Reviewers are strictly limited to one review per product
    profile = getattr(request.user, 'profile', None)
    if profile and profile.is_reviewer:
        if Review.objects.filter(product=product, user=request.user).exists():
            messages.warning(
                request,
                "You have already reviewed this product. Each reviewer is limited to one review per product."
            )
            return redirect('product_detail', pk=product.pk)

    form = ReviewSubmissionForm(request.POST)

    if form.is_valid():
        text = form.cleaned_data['text']
        rating = int(form.cleaned_data['rating'])

        # Execute unified ML analysis service (100% text-based scoring, no IP retention)
        analysis = analyze_review(text=text, rating=rating, ip=None, record_ip=False)

        review = Review.objects.create(
            product=product,
            user=request.user,
            text=text,
            rating=rating,
            ip_address=None,  # Zero IP storage
            svm_score=analysis['svm_score'],
            j48_score=analysis['j48_score'],
            sentiment=analysis['sentiment_polarity'],
            ip_flag=analysis['ip_flag'],
            final_score=analysis['final_score'],
            label=analysis['label'],
            source='user_submission'
        )

        # Evaluate and trigger product alerts (rolling window, spike, and low score)
        from analytics.services import evaluate_product_alerts
        evaluate_product_alerts(product=product, trigger_review=review)

        messages.success(
            request,
            f"Review analyzed and published! Classification: {analysis['label']} "
            f"(Authenticity Score: {analysis['final_score']}%)"
        )
    else:
        messages.error(request, "Please enter a valid review text and star rating.")

    return redirect('product_detail', pk=product.pk)

