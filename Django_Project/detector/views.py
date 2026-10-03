from django.shortcuts import render
from .utils import predict_review

# Pre-defined 10 validation reviews for the demonstration table
DEMO_REVIEWS = [
    {
        "id": 1,
        "text": "the sound is amazing.  The bass is great and you can hear the highs and lows through the speaker.  The mic is nice, but the",
        "expected": "Fake"
    },
    {
        "id": 2,
        "text": "Pro it seems comfortable. Cons; first, it doesn't feel comfortable. Second, the handle has a slight clunk. Third, the lid is a bit too short. The lid is still very comfortable. I can't imagine how this could be improved. I'm not a fan of the sticky feel of plastic and it is a little uncomfortable. So I will probably not buy this product again. However, I would not have given it five stars if it were not for the fact that the handle is a little short.",
        "expected": "Fake"
    },
    {
        "id": 3,
        "text": "the product did not work, it was just a static image.  If you have a larger screen, you have to open the screen protector and then the screen protector",
        "expected": "Fake"
    },
    {
        "id": 4,
        "text": "I am very happy to say that I am very pleased with this product.  I have used it for two years and it",
        "expected": "Fake"
    },
    {
        "id": 5,
        "text": "Great product! I highly recommend this set!I love these containers.  They work well and look great. ",
        "expected": "Fake"
    },
    {
        "id": 6,
        "text": "I purchased this for new furniture that will be delivered later this week.  I am used to leather but this time purchased microfiber. Was worried about the dog getting on the recliner and getting it dirty.  Will post a later review on how it works out once furniture is delivered.",
        "expected": "Genuine"
    },
    {
        "id": 7,
        "text": "This tripod for the price is good. It is short but I knew that when I bought it and then I just bought 3 pieces of  half inch steel gas pipe, 7 feet in length for about $20 from Lowes and slide them over the original tripod legs and now I have a nice tall tripod. Works great.",
        "expected": "Genuine"
    },
    {
        "id": 8,
        "text": "I love this pitcher! Perfect for tea and it doesn't drip all over everything!",
        "expected": "Genuine"
    },
    {
        "id": 9,
        "text": "I didn't get to try this knife before buying but a friend recommended it so I decided to give it a try. I was worried that it might be too heavy or big for my small hands. It's actually very light but feels solid and comfortable. I've had no trouble cutting all sorts of vegetables, fruits, and meat. A good purchase.",
        "expected": "Genuine"
    },
    {
        "id": 10,
        "text": "Quality rack, and installed in minutes.  Delivery was faster then expected, so overall a great product with quick customer service.",
        "expected": "Genuine"
    }
]

def get_demo_predictions():
    """
    Predicts labels for all 10 demo reviews dynamically to ensure accuracy.
    """
    demo_list = []
    for item in DEMO_REVIEWS:
        pred, _ = predict_review(item["text"])
        demo_list.append({
            "id": item["id"],
            "text": item["text"],
            "expected": item["expected"],
            "predicted": pred,
            "is_correct": item["expected"] == pred
        })
    return demo_list

def index(request):
    # Populate predictions for the demonstration table
    demo_data = get_demo_predictions()
    context = {
        "demo_reviews": demo_data
    }
    return render(request, 'detector/index.html', context)

def predict(request):
    if request.method == 'POST':
        review_text = request.POST.get('review_text', '').strip()
        
        if review_text:
            # Predict review label
            predicted_label, cleaned_text = predict_review(review_text)
            
            # Prepare result dictionary
            result = {
                "original_text": review_text,
                "cleaned_text": cleaned_text,
                "label": predicted_label
            }
        else:
            result = {
                "error": "Please enter a valid review text."
            }
            
        demo_data = get_demo_predictions()
        context = {
            "result": result,
            "demo_reviews": demo_data,
            "review_text": review_text
        }
        return render(request, 'detector/index.html', context)
        
    return index(request)
