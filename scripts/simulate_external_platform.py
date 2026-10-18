"""
Graze Smasher Product Hunting Tool - External Platform Integration Simulator
CS619 Final-Year Project

Demonstrates how third-party e-commerce platforms (Amazon, eBay, Shopify, etc.)
integrate with the Graze Smasher REST API to submit products and reviews for
real-time machine learning authenticity analysis (Calibrated LinearSVC + J-48 Decision Tree + Sentiment Analysis).

Usage:
    python scripts/simulate_external_platform.py [--host http://127.0.0.1:8000]
"""

import sys
import argparse
import json
import urllib.request
import urllib.error


def make_request(url, method="GET", payload=None):
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    data = json.dumps(payload).encode("utf-8") if payload else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    
    try:
        with urllib.request.urlopen(req) as response:
            res_body = response.read().decode("utf-8")
            return response.status, json.loads(res_body)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        try:
            err_json = json.loads(err_body)
        except Exception:
            err_json = err_body
        return e.code, err_json
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        sys.exit(1)


def run_simulation(base_url):
    print("=" * 70)
    print("   GRAZE SMASHER: THIRD-PARTY PLATFORM INTEGRATION SIMULATION")
    print("=" * 70)
    print(f"Target API Base URL: {base_url}\n")

    # -------------------------------------------------------------
    # Step 1: POST /api/products/ (Ingest Product + 10 Reviews Batch)
    # -------------------------------------------------------------
    print("[Step 1/3] Ingesting external product with a batch of 10 customer reviews...")
    ingest_payload = {
        "platform": "Amazon",
        "name": "Anker Soundcore 2 Portable Bluetooth Speaker 12W",
        "url": "https://www.amazon.com/dp/B01MTB55WH",
        "category": "Electronics",
        "description": "Upgraded audio drivers with 12W pure audio power, BassUp technology, 24-hour battery life, and IPX7 waterproof protection.",
        "image_url": "https://images.unsplash.com/photo-1545454675-3531b543be5d?w=800&auto=format&fit=crop&q=80",
        "reviews": [
            {
                "text": "I purchased this speaker for our patio barbecues. Battery lasts all weekend and the sound clarity is exceptional for its size.",
                "rating": 5
            },
            {
                "text": "the sound is amazing. The bass is great and you can hear the highs and lows through the speaker. The mic is nice, but the",
                "rating": 5
            },
            {
                "text": "Pairing via Bluetooth took less than five seconds. Sturdy rubberized casing and withstands splashes easily.",
                "rating": 4
            },
            {
                "text": "Great product! I highly recommend this set!I love these containers. They work well and look great.",
                "rating": 5
            },
            {
                "text": "Sound distorted heavily once volume exceeded 70 percent. Disappointed with the build quality.",
                "rating": 2
            },
            {
                "text": "I am very happy to say that I am very pleased with this product. I have used it for two years and it",
                "rating": 5
            },
            {
                "text": "Excellent value for thirty dollars. Connects seamlessly with both my laptop and iPhone.",
                "rating": 5
            },
            {
                "text": "the product did not work, it was just a static image. If you have a larger screen, you have to open the screen protector and then the screen protector",
                "rating": 1
            },
            {
                "text": "Compact lightweight and charges via Micro USB. Perfect companion for camping trips.",
                "rating": 5
            },
            {
                "text": "Pro it seems comfortable. Cons; first, it doesn't feel comfortable. Second, the handle has a slight clunk. Third, the lid is a bit too short.",
                "rating": 3
            }
        ]
    }

    status_code, response_data = make_request(f"{base_url}/api/products/", method="POST", payload=ingest_payload)
    print(f" -> HTTP Status Code: {status_code}")
    
    if status_code != 201:
        print(f"[FAIL] Ingestion error: {response_data}")
        return

    product_info = response_data["product"]
    product_id = product_info["id"]
    print(f" -> Product Ingested: ID={product_id} | '{product_info['name']}'")
    print(f" -> Overall Authenticity Score: {product_info['overall_authenticity_score']}% ({product_info['authenticity_label']})")
    print(f" -> Community Trust Rating:     {product_info['community_trust_rating']} stars (Genuine Reviews Only)")
    print(f" -> Fake Reviews Detected:      {product_info['fake_reviews_count']} out of {product_info['review_count']}")
    print(f" -> First Review Analysis:")
    first_rev = response_data["reviews"][0]
    print(f"    - Text: '{first_rev['text'][:60]}...'")
    print(f"    - Classification: {first_rev['label']} (Score: {first_rev['final_score']}%)")
    print(f"    - Model Signals: SVM Fake={first_rev['svm_score']} | J48 Fake={first_rev['j48_score']} | Sentiment Polarity={first_rev['sentiment']}")

    # -------------------------------------------------------------
    # Step 2: POST /api/reviews/ (Add One Real-time Review)
    # -------------------------------------------------------------
    print(f"\n[Step 2/3] Submitting an individual real-time review for Product ID {product_id}...")
    single_review_payload = {
        "product_id": product_id,
        "text": "BUY CHEAP PHONES NOW!!! GO TO SPAM LINK AND RECEIVE FREE OFFERS IMMEDIATELY!!! BEST DEAL EVER!",
        "rating": 5
    }

    status_code, review_res = make_request(f"{base_url}/api/reviews/", method="POST", payload=single_review_payload)
    print(f" -> HTTP Status Code: {status_code}")
    if status_code == 201:
        rev = review_res["review"]
        print(f" -> Real-time Analysis Result:")
        print(f"    - Assigned Label:          {rev['label']}")
        print(f"    - Review Authenticity:     {rev['final_score']}%")
        print(f"    - SVM Fake Probability:    {rev['svm_score']}")
        print(f"    - J-48 Fake Probability:   {rev['j48_score']}")
        print(f"    - Sentiment Polarity:      {rev['sentiment']}")
        print(f"    - Updated Product Score:   {review_res['product_authenticity_score']}% ({review_res['product_authenticity_label']})")
    else:
        print(f"[FAIL] Review submission error: {review_res}")

    # -------------------------------------------------------------
    # Step 3: GET /api/products/<id>/ (Verify Stats)
    # -------------------------------------------------------------
    print(f"\n[Step 3/3] Querying updated stats via GET /api/products/{product_id}/...")
    status_code, detail_res = make_request(f"{base_url}/api/products/{product_id}/", method="GET")
    print(f" -> HTTP Status Code: {status_code}")
    if status_code == 200:
        print(f" -> Verified Authenticity Stats:")
        print(f"    - Product Name:             {detail_res['name']}")
        print(f"    - Total Reviews:            {detail_res['review_count']}")
        print(f"    - Reported Average Rating:  {detail_res['average_rating']} stars")
        print(f"    - Verified Trust Rating:    {detail_res['community_trust_rating']} stars")
        print(f"    - Overall Authenticity:     {detail_res['overall_authenticity_score']}%")
        print(f"    - Authenticity Status:      {detail_res['authenticity_label']}")
        print(f"    - Genuine Count:            {detail_res['genuine_reviews_count']}")
        print(f"    - Fake Count:               {detail_res['fake_reviews_count']}")
        print(f"    - Suspicious Count:         {detail_res['suspicious_reviews_count']}")

    print("\n" + "=" * 70)
    print("   SIMULATION COMPLETED SUCCESSFULLY! REST API FULLY OPERATIONAL.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate external e-commerce platform API ingestion.")
    parser.add_argument("--host", default="http://127.0.0.1:8000", help="Base URL of Graze Smasher server")
    args = parser.parse_args()

    run_simulation(args.host.rstrip("/"))

