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
    Handles user review submission:
    1. Captures client IP address
    2. Calls unified analyze_review() pipeline (Calibrated SVM + J-48 + Sentiment + IP velocity)
    3. Persists Review record with all multi-tier ML scores and labels
    4. Automatically generates an Alert if suspicious activity or fake content is detected
    """
    product = get_object_or_404(Product, pk=product_id)
    form = ReviewSubmissionForm(request.POST)

    if form.is_valid():
        text = form.cleaned_data['text']
        rating = int(form.cleaned_data['rating'])
        client_ip = get_client_ip(request)

        # Execute unified ML analysis service
        analysis = analyze_review(text=text, rating=rating, ip=client_ip, record_ip=True)

        review = Review.objects.create(
            product=product,
            user=request.user,
            text=text,
            rating=rating,
            ip_address=client_ip,
            svm_score=analysis['svm_score'],
            j48_score=analysis['j48_score'],
            sentiment=analysis['sentiment_polarity'],
            ip_flag=analysis['ip_flag'],
            final_score=analysis['final_score'],
            label=analysis['label'],
            source='user_submission'
        )

        # Evaluate and trigger product alerts (rolling window, spike, low score, and IP burst)
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

