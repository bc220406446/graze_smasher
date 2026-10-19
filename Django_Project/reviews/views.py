from django.shortcuts import get_object_or_404, redirect, render
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST
from products.models import Product
from analytics.models import Alert
from accounts.decorators import reviewer_required
from ML_Model.service import analyze_review
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
def submit_review(request, product_id=None, pk=None):
    """
    Handles reviewer review submission:
    1. Enforces single review per product limit for Reviewers
    2. Zero IP address storage for user privacy compliance
    3. Calls unified 100% text-based ML analyze_review() pipeline (SVM + J-48 + Sentiment)
    4. Persists Review record with multi-tier ML scores and labels
    5. Automatically generates an Alert if suspicious activity or fake content is detected
    """
    pid = pk if pk is not None else product_id
    product = get_object_or_404(Product, pk=pid)

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


@login_required
def edit_review(request, review_id=None, id=None):
    """Allows review author or admin to update a review and recalculates ML scores."""
    rid = id if id is not None else review_id
    review = get_object_or_404(Review, pk=rid)
    profile = getattr(request.user, 'profile', None)
    is_admin = request.user.is_superuser or request.user.is_staff or (profile and profile.is_admin)

    if review.user != request.user and not is_admin:
        messages.error(request, "Permission denied: You can only edit your own reviews.")
        next_url = request.POST.get('next') or request.GET.get('next')
        if not next_url:
            if profile and profile.is_reviewer:
                next_url = 'reviewer_dashboard'
            elif is_admin:
                next_url = 'admin_dashboard'
            else:
                next_url = 'user_dashboard'
        return redirect(next_url)

    if request.method == 'POST':
        form = ReviewSubmissionForm(request.POST, instance=review)
        if form.is_valid():
            text = form.cleaned_data['text']
            rating = int(form.cleaned_data['rating'])

            # Run ML detector pipeline on updated review text and rating
            analysis = analyze_review(text=text, rating=rating, ip=None, record_ip=False)

            review.text = text
            review.rating = rating
            review.svm_score = analysis['svm_score']
            review.j48_score = analysis['j48_score']
            review.sentiment = analysis['sentiment_polarity']
            review.ip_flag = analysis['ip_flag']
            review.final_score = analysis['final_score']
            review.label = analysis['label']
            review.save()

            # Re-evaluate product alerts
            from analytics.services import evaluate_product_alerts
            evaluate_product_alerts(product=review.product, trigger_review=review)

            messages.success(
                request,
                f"Review updated and re-analyzed! Classification: {analysis['label']} "
                f"(Authenticity Score: {analysis['final_score']}%)"
            )
            next_url = request.POST.get('next') or request.GET.get('next')
            if not next_url:
                if profile and profile.is_reviewer:
                    next_url = 'reviewer_dashboard'
                elif is_admin:
                    next_url = 'admin_dashboard'
                else:
                    next_url = 'user_dashboard'
            return redirect(next_url)
        else:
            messages.error(request, "Please provide valid review details.")
            next_url = request.POST.get('next') or request.GET.get('next')
            if not next_url:
                if profile and profile.is_reviewer:
                    next_url = 'reviewer_dashboard'
                elif is_admin:
                    next_url = 'admin_dashboard'
                else:
                    next_url = 'user_dashboard'
            return redirect(next_url)

    form = ReviewSubmissionForm(instance=review)
    return render(request, 'includes/edit_review.html', {'form': form, 'review': review})


@require_POST
@login_required
def delete_review(request, review_id=None, id=None):
    """Allows review author or admin to delete a review."""
    rid = id if id is not None else review_id
    review = get_object_or_404(Review, pk=rid)
    profile = getattr(request.user, 'profile', None)
    is_admin = request.user.is_superuser or request.user.is_staff or (profile and profile.is_admin)

    if review.user != request.user and not is_admin:
        messages.error(request, "Permission denied: You can only delete your own reviews.")
        next_url = request.POST.get('next') or request.GET.get('next')
        if not next_url:
            if profile and profile.is_reviewer:
                next_url = 'reviewer_dashboard'
            elif is_admin:
                next_url = 'admin_dashboard'
            else:
                next_url = 'user_dashboard'
        return redirect(next_url)

    product_name = review.product.name
    review.delete()
    messages.success(request, f"Review for '{product_name}' has been deleted.")
    next_url = request.POST.get('next') or request.GET.get('next')
    if not next_url:
        if profile and profile.is_reviewer:
            next_url = 'reviewer_dashboard'
        elif is_admin:
            next_url = 'admin_dashboard'
        else:
            next_url = 'user_dashboard'
    return redirect(next_url)



