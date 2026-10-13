"""
Management command to seed demo products, user accounts, and reviews
from the Salminen et al. fake reviews dataset.
"""

import os
import pandas as pd
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from django.conf import settings
from accounts.models import UserProfile
from products.models import Product, SavedProduct
from reviews.models import Review
from analytics.models import Alert
from detector.utils import analyze_review
from ML_Model.config import DATASET_PATH


class Command(BaseCommand):
    help = "Seeds demo users, sample products, and reviews analyzed via the ML pipeline."

    def add_arguments(self, parser):
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Clear existing products, reviews, and demo users before seeding',
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.NOTICE("=== Starting Graze Smasher Demo Seeding ==="))

        if options['clear']:
            self.stdout.write("Clearing existing database records...")
            Review.objects.all().delete()
            Alert.objects.all().delete()
            SavedProduct.objects.all().delete()
            Product.objects.all().delete()
            User.objects.filter(username__in=['admin', 'reviewer', 'user']).delete()
            self.stdout.write(self.style.SUCCESS("Existing demo records cleared."))

        # 1. Create Demo Users
        self.stdout.write("Creating demo user accounts...")
        users = {}
        demo_user_configs = [
            ('admin', 'admin@grazesmasher.com', 'admin123', UserProfile.ROLE_ADMIN, True, True),
            ('reviewer', 'reviewer@grazesmasher.com', 'reviewer123', UserProfile.ROLE_REVIEWER, False, False),
            ('user', 'user@grazesmasher.com', 'user123', UserProfile.ROLE_GENERAL, False, False),
        ]

        for username, email, password, role, is_staff, is_super in demo_user_configs:
            u, created = User.objects.get_or_create(username=username, defaults={'email': email})
            u.set_password(password)
            u.is_staff = is_staff
            u.is_superuser = is_super
            u.save()

            profile, _ = UserProfile.objects.get_or_create(user=u)
            profile.role = role
            profile.save()
            users[username] = u
            status_text = "created" if created else "updated"
            self.stdout.write(f" - User '{username}' ({role}): {status_text} (Password: {password})")

        # 2. Define Sample Products
        demo_products_data = [
            {
                "name": "Sony WH-1000XM5 Wireless Noise Canceling Headphones",
                "description": "Industry-leading noise canceling with Auto NC Optimizer, crystal clear hands-free calling with 4 beamforming microphones, and 30-hour battery life.",
                "category": "Electronics",
                "platform": "Amazon",
                "source_url": "https://www.amazon.com/dp/B09XS7JWHH",
                "image_url": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=800&auto=format&fit=crop&q=80",
                "dataset_cat": "Electronics_5",
            },
            {
                "name": "Apple AirPods Pro (2nd Gen) with MagSafe Case USB-C",
                "description": "Up to 2x more Active Noise Cancellation, Transparency mode, Adaptive Audio, and Personalized Spatial Audio with dynamic head tracking.",
                "category": "Electronics",
                "platform": "eBay",
                "source_url": "https://www.ebay.com/itm/apple-airpods-pro-2",
                "image_url": "https://images.unsplash.com/photo-1600294037681-c80b4cb5b434?w=800&auto=format&fit=crop&q=80",
                "dataset_cat": "Electronics_5",
            },
            {
                "name": "Ninja AF101 Air Fryer 4-Quart Capacity",
                "description": "Crisp and roast with up to 75% less fat than traditional frying methods. Features wide temperature range from 105 to 400 degrees Fahrenheit.",
                "category": "Home_and_Kitchen",
                "platform": "Amazon",
                "source_url": "https://www.amazon.com/dp/B07FDJMC9Q",
                "image_url": "https://images.unsplash.com/photo-1584269600464-37b1b58a9fe7?w=800&auto=format&fit=crop&q=80",
                "dataset_cat": "Home_and_Kitchen_5",
            },
            {
                "name": "Instant Pot Duo 7-in-1 Electric Pressure Cooker 6-Qt",
                "description": "Combines 7 kitchen appliances in 1: pressure cooker, slow cooker, rice cooker, steamer, sauté pan, yogurt maker, and food warmer.",
                "category": "Home_and_Kitchen",
                "platform": "Walmart",
                "source_url": "https://www.walmart.com/ip/instant-pot-duo",
                "image_url": "https://images.unsplash.com/photo-1544233726-9f1d2b27be8b?w=800&auto=format&fit=crop&q=80",
                "dataset_cat": "Home_and_Kitchen_5",
            },
            {
                "name": "Fitbit Charge 6 Advanced Fitness Tracker",
                "description": "Fitness tracking band with Google apps, Heart Rate on gym equipment, 40+ exercise modes, built-in GPS, and 7-day battery life.",
                "category": "Sports_and_Outdoors",
                "platform": "Amazon",
                "source_url": "https://www.amazon.com/dp/B0CC625X35",
                "image_url": "https://images.unsplash.com/photo-1576243345690-4e4b79b63288?w=800&auto=format&fit=crop&q=80",
                "dataset_cat": "Sports_and_Outdoors_5",
            },
            {
                "name": "Hydro Flask 32 oz Wide Mouth Stainless Steel Bottle",
                "description": "TempShield double-wall vacuum insulation keeps drinks cold up to 24 hours and hot up to 12 hours. Pure taste with 18/8 pro-grade stainless steel.",
                "category": "Sports_and_Outdoors",
                "platform": "Amazon",
                "source_url": "https://www.amazon.com/dp/B07NVV3L7H",
                "image_url": "https://images.unsplash.com/photo-1602143407151-7111542de6e8?w=800&auto=format&fit=crop&q=80",
                "dataset_cat": "Sports_and_Outdoors_5",
            },
        ]

        # 3. Load dataset
        self.stdout.write(f"Reading reviews from dataset at: {DATASET_PATH}")
        df = pd.read_csv(DATASET_PATH)

        # 4. Create products and populate reviews
        products = []
        for p_data in demo_products_data:
            cat_key = p_data.pop("dataset_cat")
            product, _ = Product.objects.get_or_create(
                name=p_data["name"],
                defaults={
                    **p_data,
                    "created_by": users["admin"]
                }
            )
            products.append((product, cat_key))
            self.stdout.write(f" - Configured product: {product.name} (ID: {product.id})")

        # Select representative reviews per category (balanced genuine and fake)
        total_seeded_reviews = 0
        self.stdout.write("\nAnalyzing and inserting reviews via ML pipeline...")

        for product, cat_key in products:
            # Filter dataset reviews for this category
            cat_df = df[df['category'] == cat_key]
            if len(cat_df) < 15:
                cat_df = df # fallback
                
            # Sample 8 Fake (CG) and 7 Genuine (OR) reviews
            cg_sample = cat_df[cat_df['label'] == 'CG'].head(8)
            or_sample = cat_df[cat_df['label'] == 'OR'].head(7)
            sample_df = pd.concat([cg_sample, or_sample]).sample(frac=1, random_state=42)

            for _, row in sample_df.iterrows():
                review_text = str(row['text_']).strip()
                rating_val = int(float(row.get('rating', 5.0)))
                ip_val = str(row.get('simulated_ip', '203.0.113.1'))

                # Run through unified analyze_review
                analysis = analyze_review(
                    text=review_text,
                    rating=rating_val,
                    ip=ip_val,
                    record_ip=False
                )

                Review.objects.create(
                    product=product,
                    user=users['reviewer'] if total_seeded_reviews % 2 == 0 else None,
                    text=review_text,
                    rating=rating_val,
                    ip_address=ip_val,
                    svm_score=analysis['svm_score'],
                    j48_score=analysis['j48_score'],
                    sentiment=analysis['sentiment_polarity'],
                    ip_flag=analysis['ip_flag'],
                    final_score=analysis['final_score'],
                    label=analysis['label'],
                    source='dataset_seed'
                )
                total_seeded_reviews += 1

            # Trigger sample alert for products with fake reviews
            if product.fake_reviews_count() > 0:
                Alert.objects.create(
                    product=product,
                    type='fake_spike',
                    message=f"Authenticity warning: {product.fake_reviews_count()} computer-generated reviews detected on '{product.name}'.",
                    user=users['admin']
                )

            self.stdout.write(
                f"   -> Seeded {product.reviews.count()} reviews for '{product.name}' "
                f"(Authenticity Score: {product.overall_authenticity_score()}%, Label: {product.authenticity_label()})"
            )

        # 5. Save product into demo user's watchlist
        SavedProduct.objects.get_or_create(user=users['user'], product=products[0][0])
        SavedProduct.objects.get_or_create(user=users['user'], product=products[2][0])
        self.stdout.write(f"\nAdded 2 sample products to '{users['user'].username}' watchlist.")

        self.stdout.write(self.style.SUCCESS(
            f"\n=== Demo seeding complete! Seeded {len(products)} products with {total_seeded_reviews} reviews. ==="
        ))

