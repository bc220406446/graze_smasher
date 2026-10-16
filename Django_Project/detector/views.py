from django.shortcuts import render, redirect
from .utils import analyze_review

# Pre-defined 10 validation reviews for the demonstration table
DEMO_REVIEWS = [
    {
        "id": 1,
        "text": "the sound is amazing.  The bass is great and you can hear the highs and lows through the speaker.  The mic is nice, but the",
        "expected": "Fake",
        "rating": 5
    },
    {
        "id": 2,
        "text": "Pro it seems comfortable. Cons; first, it doesn't feel comfortable. Second, the handle has a slight clunk. Third, the lid is a bit too short. The lid is still very comfortable. I can't imagine how this could be improved. I'm not a fan of the sticky feel of plastic and it is a little uncomfortable. So I will probably not buy this product again. However, I would not have given it five stars if it were not for the fact that the handle is a little short.",
        "expected": "Fake",
        "rating": 3
    },
    {
        "id": 3,
        "text": "the product did not work, it was just a static image.  If you have a larger screen, you have to open the screen protector and then the screen protector",
        "expected": "Fake",
        "rating": 1
    },
    {
        "id": 4,
        "text": "I am very happy to say that I am very pleased with this product.  I have used it for two years and it",
        "expected": "Fake",
        "rating": 5
    },
    {
        "id": 5,
        "text": "Great product! I highly recommend this set!I love these containers.  They work well and look great. ",
        "expected": "Fake",
        "rating": 5
    },
    {
        "id": 6,
        "text": "I purchased this for new furniture that will be delivered later this week.  I am used to leather but this time purchased microfiber. Was worried about the dog getting on the recliner and getting it dirty.  Will post a later review on how it works out once furniture is delivered.",
        "expected": "Genuine",
        "rating": 5
    },
    {
        "id": 7,
        "text": "This tripod for the price is good. It is short but I knew that when I bought it and then I just bought 3 pieces of  half inch steel gas pipe, 7 feet in length for about $20 from Lowes and slide them over the original tripod legs and now I have a nice tall tripod. Works great.",
        "expected": "Genuine",
        "rating": 4
    },
    {
        "id": 8,
        "text": "I love this pitcher! Perfect for tea and it doesn't drip all over everything!",
        "expected": "Genuine",
        "rating": 5
    },
    {
        "id": 9,
        "text": "I didn't get to try this knife before buying but a friend recommended it so I decided to give it a try. I was worried that it might be too heavy or big for my small hands. It's actually very light but feels solid and comfortable. I've had no trouble cutting all sorts of vegetables, fruits, and meat. A good purchase.",
        "expected": "Genuine",
        "rating": 4
    },
    {
        "id": 10,
        "text": "Quality rack, and installed in minutes.  Delivery was faster then expected, so overall a great product with quick customer service.",
        "expected": "Genuine",
        "rating": 5
    }
]

def get_client_ip(request):
    """Extracts client IP address safely from request headers."""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0].strip()
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

def get_demo_predictions():
    """
    Predicts multi-tier metrics for all 10 demo reviews dynamically.
    """
    demo_list = []
    for item in DEMO_REVIEWS:
        analysis = analyze_review(item["text"], rating=item.get("rating"), ip=None, record_ip=False)
        pred = analysis["label"]
        demo_list.append({
            "id": item["id"],
            "text": item["text"],
            "rating": item.get("rating", 5),
            "expected": item["expected"],
            "predicted": pred,
            "is_correct": item["expected"] == pred,
            "final_score": analysis["final_score"],
            "svm_score": analysis["svm_score"],
            "j48_score": analysis["j48_score"],
            "sentiment_polarity": analysis["sentiment_polarity"],
        })
    return demo_list

def index(request):
    """Legacy public ML sandbox redirected to product catalog."""
    return redirect('product_list')

def predict(request):
    """Legacy public ML sandbox predict redirected to product catalog."""
    return redirect('product_list')
