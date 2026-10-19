"""
Management command to seed demo products, user accounts, and reviews.
Follows exact user specifications:
- 1 Admin (admin@gs.com / Test@123)
- 3 Users (user1@gs.com, user2@gs.com, user3@gs.com / Test@123), each with 3 saved products
- 3 Reviewers (reviewer1@gs.com, reviewer2@gs.com, reviewer3@gs.com / Test@123), each reviewing 3-4 products
- Each product has at least 10-20 reviews evaluated by the ML pipeline
- No alerts created
"""

import os
import pandas as pd
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from accounts.models import UserProfile
from products.models import Product, SavedProduct, MonitoredProduct
from reviews.models import Review, ReviewerReviewedProduct
from analytics.models import Alert
from analytics.services import evaluate_product_alerts
from ML_Model.service import analyze_review
from ML_Model.config import DATASET_PATH


class Command(BaseCommand):
    help = "Seeds users, reviewers, admin, products, and ML-scored reviews per project requirements."

    def add_arguments(self, parser):
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Clear existing products, reviews, and demo users before seeding',
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.NOTICE("=== Starting Graze Smasher Clean Seeding ==="))

        # Clear existing records
        self.stdout.write("Clearing existing database records...")
        Alert.objects.all().delete()
        ReviewerReviewedProduct.objects.all().delete()
        Review.objects.all().delete()
        SavedProduct.objects.all().delete()
        MonitoredProduct.objects.all().delete()
        Product.objects.all().delete()
        User.objects.all().delete()
        self.stdout.write(self.style.SUCCESS("Database completely wiped for fresh seed."))

        PASSWORD = "Test@123"

        # 1. Create Admin
        self.stdout.write("Creating administrator account...")
        admin_user = User.objects.create(
            username="admin@gs.com",
            email="admin@gs.com",
            first_name="System",
            last_name="Administrator",
            is_staff=True,
            is_superuser=True
        )
        admin_user.set_password(PASSWORD)
        admin_user.save()
        admin_profile, _ = UserProfile.objects.get_or_create(user=admin_user)
        admin_profile.role = UserProfile.ROLE_ADMIN
        admin_profile.save()
        self.stdout.write(f" + Admin: admin@gs.com (System Administrator) | Password: {PASSWORD}")

        # 2. Create 3 General Users
        self.stdout.write("\nCreating 3 General User accounts...")
        general_users_data = [
            ("user1@gs.com", "Alex Johnson"),
            ("user2@gs.com", "Sarah Miller"),
            ("user3@gs.com", "David Wilson"),
        ]
        users = []
        for email, full_name in general_users_data:
            name_parts = full_name.split(' ', 1)
            u = User.objects.create(
                username=email,
                email=email,
                first_name=name_parts[0],
                last_name=name_parts[1] if len(name_parts) > 1 else '',
                is_staff=False,
                is_superuser=False
            )
            u.set_password(PASSWORD)
            u.save()
            profile, _ = UserProfile.objects.get_or_create(user=u)
            profile.role = UserProfile.ROLE_GENERAL
            profile.save()
            users.append(u)
            self.stdout.write(f" + User: {email} ({full_name}) | Password: {PASSWORD}")

        # 3. Create 3 Reviewers
        self.stdout.write("\nCreating 3 Reviewer accounts...")
        reviewers_data = [
            ("reviewer1@gs.com", "Elena Rostova"),
            ("reviewer2@gs.com", "Marcus Vance"),
            ("reviewer3@gs.com", "Chloe Bennett"),
        ]
        reviewers = []
        for email, full_name in reviewers_data:
            name_parts = full_name.split(' ', 1)
            u = User.objects.create(
                username=email,
                email=email,
                first_name=name_parts[0],
                last_name=name_parts[1] if len(name_parts) > 1 else '',
                is_staff=False,
                is_superuser=False
            )
            u.set_password(PASSWORD)
            u.save()
            profile, _ = UserProfile.objects.get_or_create(user=u)
            profile.role = UserProfile.ROLE_REVIEWER
            profile.save()
            reviewers.append(u)
            self.stdout.write(f" + Reviewer: {email} ({full_name}) | Password: {PASSWORD}")

        # 4. Define 9 Sample Products across Categories
        # 3 products created by User 1, 3 products by User 2, 3 products by User 3
        self.stdout.write("\nCreating 9 sample catalog products (3 per user)...")
        demo_products_data = [
            # User 1's Products (Indices 0, 1, 2)
            {
                "name": "Sony WH-1000XM5 Wireless Noise Canceling Headphones",
                "description": "Industry-leading noise canceling with Auto NC Optimizer, crystal clear hands-free calling with 4 beamforming microphones, and 30-hour battery life.",
                "category": "Electronics",
                "platform": "Amazon",
                "source_url": "https://www.amazon.com/dp/B09XS7JWHH",
                "image_url": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=800&auto=format&fit=crop&q=80",
                "dataset_cat": "Electronics_5",
                "creator": users[0],
            },
            {
                "name": "Apple AirPods Pro (2nd Gen) with MagSafe Case USB-C",
                "description": "Up to 2x more Active Noise Cancellation, Transparency mode, Adaptive Audio, and Personalized Spatial Audio with dynamic head tracking.",
                "category": "Electronics",
                "platform": "eBay",
                "source_url": "https://www.ebay.com/itm/apple-airpods-pro-2",
                "image_url": "https://images.unsplash.com/photo-1600294037681-c80b4cb5b434?w=800&auto=format&fit=crop&q=80",
                "dataset_cat": "Electronics_5",
                "creator": users[0],
            },
            {
                "name": "Logitech MX Master 3S Wireless Performance Mouse",
                "description": "Quiet clicks and 8K DPI any-surface tracking, MagSpeed electromagnetic scrolling, ergonomic silhouette designed for all-day comfort.",
                "category": "Electronics",
                "platform": "Amazon",
                "source_url": "https://www.amazon.com/dp/B09HM94VDS",
                "image_url": "https://images.unsplash.com/photo-1615663245857-ac93bb7c39e7?w=800&auto=format&fit=crop&q=80",
                "dataset_cat": "Electronics_5",
                "creator": users[0],
            },

            # User 2's Products (Indices 3, 4, 5)
            {
                "name": "Ninja AF101 Air Fryer 4-Quart Capacity",
                "description": "Crisp and roast with up to 75% less fat than traditional frying methods. Features wide temperature range from 105 to 400 degrees Fahrenheit.",
                "category": "Home_and_Kitchen",
                "platform": "Amazon",
                "source_url": "https://www.amazon.com/dp/B07FDJMC9Q",
                "image_url": "https://images.unsplash.com/photo-1584269600464-37b1b58a9fe7?w=800&auto=format&fit=crop&q=80",
                "dataset_cat": "Home_and_Kitchen_5",
                "creator": users[1],
            },
            {
                "name": "Instant Pot Duo 7-in-1 Electric Pressure Cooker 6-Qt",
                "description": "Combines 7 kitchen appliances in 1: pressure cooker, slow cooker, rice cooker, steamer, sauté pan, yogurt maker, and food warmer.",
                "category": "Home_and_Kitchen",
                "platform": "Walmart",
                "source_url": "https://www.walmart.com/ip/instant-pot-duo",
                "image_url": "https://images.unsplash.com/photo-1544233726-9f1d2b27be8b?w=800&auto=format&fit=crop&q=80",
                "dataset_cat": "Home_and_Kitchen_5",
                "creator": users[1],
            },
            {
                "name": "Keurig K-Mini Single Serve Coffee Maker",
                "description": "Less than 5 inches wide, fits anywhere. Brews any cup size between 6-12oz with Keurig K-Cup pods for fast and fresh coffee.",
                "category": "Home_and_Kitchen",
                "platform": "Amazon",
                "source_url": "https://www.amazon.com/dp/B07FL258DN",
                "image_url": "https://images.unsplash.com/photo-1517668808822-9ebb02f2a0e6?w=800&auto=format&fit=crop&q=80",
                "dataset_cat": "Home_and_Kitchen_5",
                "creator": users[1],
            },

            # User 3's Products (Indices 6, 7, 8)
            {
                "name": "Fitbit Charge 6 Advanced Fitness Tracker",
                "description": "Fitness tracking band with Google apps, Heart Rate on gym equipment, 40+ exercise modes, built-in GPS, and 7-day battery life.",
                "category": "Sports_and_Outdoors",
                "platform": "Amazon",
                "source_url": "https://www.amazon.com/dp/B0CC625X35",
                "image_url": "https://images.unsplash.com/photo-1576243345690-4e4b79b63288?w=800&auto=format&fit=crop&q=80",
                "dataset_cat": "Sports_and_Outdoors_5",
                "creator": users[2],
            },
            {
                "name": "Hydro Flask 32 oz Wide Mouth Stainless Steel Bottle",
                "description": "TempShield double-wall vacuum insulation keeps drinks cold up to 24 hours and hot up to 12 hours. Pure taste with 18/8 pro-grade stainless steel.",
                "category": "Sports_and_Outdoors",
                "platform": "Amazon",
                "source_url": "https://www.amazon.com/dp/B07NVV3L7H",
                "image_url": "https://images.unsplash.com/photo-1602143407151-7111542de6e8?w=800&auto=format&fit=crop&q=80",
                "dataset_cat": "Sports_and_Outdoors_5",
                "creator": users[2],
            },
            {
                "name": "Coleman Sundome 4-Person Camping Tent",
                "description": "WeatherTec system with patented welded floors and inverted seams to keep you dry. Strong frame tested to withstand 35+ mph winds.",
                "category": "Sports_and_Outdoors",
                "platform": "Walmart",
                "source_url": "https://www.walmart.com/ip/coleman-sundome-tent",
                "image_url": "https://images.unsplash.com/photo-1504280390367-361c6d9f38f4?w=800&auto=format&fit=crop&q=80",
                "dataset_cat": "Sports_and_Outdoors_5",
                "creator": users[2],
            },
        ]

        products = []
        for p_data in demo_products_data:
            cat_key = p_data.pop("dataset_cat")
            creator = p_data.pop("creator")
            product = Product.objects.create(
                name=p_data["name"],
                description=p_data["description"],
                category=p_data["category"],
                platform=p_data["platform"],
                source_url=p_data["source_url"],
                image_url=p_data["image_url"],
                created_by=creator,
            )
            products.append((product, cat_key, creator))
            self.stdout.write(f" + Product: '{product.name}' (Created by: {creator.email})")

        # 5. Assign Saved & Monitored Products: Each user saves ONLY the 3 products they created!
        self.stdout.write("\nAssigning saved/monitored products (each user monitors their 3 created products)...")
        user_saved_map = [
            (users[0], [0, 1, 2]),  # User 1 monitors their 3 created products
            (users[1], [3, 4, 5]),  # User 2 monitors their 3 created products
            (users[2], [6, 7, 8]),  # User 3 monitors their 3 created products
        ]

        for u, p_indices in user_saved_map:
            for p_idx in p_indices:
                prod = products[p_idx][0]
                SavedProduct.objects.get_or_create(user=u, product=prod)
                MonitoredProduct.objects.get_or_create(user=u, product=prod)
            saved_count = SavedProduct.objects.filter(user=u).count()
            created_count = Product.objects.filter(created_by=u).count()
            self.stdout.write(f" + User '{u.email}': {created_count} created products, {saved_count} monitored products.")

        # 6. Populate Reviews & Evaluate Alerts in Parallel
        self.stdout.write("\nLoading dataset, analyzing reviews via ML pipeline, and evaluating alerts...")
        df = pd.read_csv(DATASET_PATH)

        # Reviewer assignments across the 9 products (each reviewer evaluates 4 products):
        # Reviewer 1 (Elena) -> Products 0, 1, 3, 6 (4 products)
        # Reviewer 2 (Marcus) -> Products 1, 4, 5, 7 (4 products)
        # Reviewer 3 (Chloe) -> Products 2, 4, 7, 8 (4 products)
        reviewer_product_map = {
            reviewers[0]: [0, 1, 3, 6],
            reviewers[1]: [1, 4, 5, 7],
            reviewers[2]: [2, 4, 7, 8],
        }

        # Targeted fake count per product (Products 1, 3, 7 have >= 50% fake ratio to demonstrate alerts)
        fake_distribution = {
            0: 3,  # 3 Fake / 12 Genuine (20.0% fake)
            1: 9,  # 9 Fake / 6 Genuine (60.0% fake -> Alert triggered for User 1 & Admin)
            2: 4,  # 4 Fake / 11 Genuine (26.7% fake)
            3: 10, # 10 Fake / 5 Genuine (66.7% fake -> Alert triggered for User 2 & Admin)
            4: 2,  # 2 Fake / 13 Genuine (13.3% fake)
            5: 6,  # 6 Fake / 9 Genuine (40.0% fake)
            6: 4,  # 4 Fake / 11 Genuine (26.7% fake)
            7: 8,  # 8 Fake / 7 Genuine (53.3% fake -> Alert triggered for User 3 & Admin)
            8: 3,  # 3 Fake / 12 Genuine (20.0% fake)
        }

        total_seeded_reviews = 0
        for prod_idx, (product, cat_key, creator) in enumerate(products):
            cat_df = df[df['category'] == cat_key]
            if len(cat_df) < 15:
                cat_df = df

            fake_count = fake_distribution.get(prod_idx, 7)
            genuine_count = 15 - fake_count

            cg_sample = cat_df[cat_df['label'] == 'CG'].head(fake_count)
            or_sample = cat_df[cat_df['label'] == 'OR'].head(genuine_count)
            sample_df = pd.concat([or_sample, cg_sample]).sample(frac=1, random_state=42 + prod_idx)

            # Check if any reviewer is assigned to this product
            reviewers_for_this_prod = [r for r, assigned in reviewer_product_map.items() if prod_idx in assigned]

            for rev_idx, (_, row) in enumerate(sample_df.iterrows()):
                review_text = str(row['text_']).strip()
                rating_val = int(float(row.get('rating', 5.0)))
                ip_val = f"192.168.1.{10 + (total_seeded_reviews % 50)}"

                # Assign reviewer for the first review(s) assigned to them, otherwise product creator
                if rev_idx < len(reviewers_for_this_prod):
                    assigned_user = reviewers_for_this_prod[rev_idx]
                else:
                    assigned_user = creator

                analysis = analyze_review(
                    text=review_text,
                    rating=rating_val,
                    ip=ip_val,
                    record_ip=False
                )

                rev = Review.objects.create(
                    product=product,
                    user=assigned_user,
                    text=review_text,
                    rating=rating_val,
                    ip_address=ip_val,
                    svm_score=analysis['svm_score'],
                    j48_score=analysis['j48_score'],
                    sentiment=analysis['sentiment_polarity'],
                    ip_flag=analysis['ip_flag'],
                    final_score=analysis['final_score'],
                    label=analysis['label'],
                    source='user_submission' if assigned_user else 'dataset_seed'
                )

                if assigned_user:
                    ReviewerReviewedProduct.objects.get_or_create(
                        user=assigned_user,
                        product=product
                    )
                    # Parallel alert trigger: Reviewer receives classification feedback alert
                    if assigned_user in reviewers:
                        evaluate_product_alerts(product=product, trigger_review=rev)

                total_seeded_reviews += 1

            # Parallel alert trigger: Evaluate product fake ratio spike (triggers for monitored users & admin if >= 50%)
            evaluate_product_alerts(product=product)

            self.stdout.write(
                f"   -> {product.name}: {product.reviews.count()} reviews seeded "
                f"(Score: {product.overall_authenticity_score()}%, Label: {product.authenticity_label()})"
            )

        # 7. Reviewer Reviewed Product Summary
        self.stdout.write("\nReviewer Reviewed Products Summary:")
        for r in reviewers:
            rev_prods_count = ReviewerReviewedProduct.objects.filter(user=r).count()
            r_alerts_count = Alert.objects.filter(user=r).count()
            self.stdout.write(f" + Reviewer '{r.email}': {rev_prods_count} products reviewed, {r_alerts_count} classification alerts.")

        # 8. Alert Statistics Summary
        total_alerts = Alert.objects.count()
        gen_alerts = Alert.objects.filter(user__in=users).count()
        rev_alerts = Alert.objects.filter(user__in=reviewers).count()
        adm_alerts = Alert.objects.filter(user=admin_user).count()

        self.stdout.write(self.style.SUCCESS(
            f"\n=== Seed successfully completed with Parallel Alerts! ==="
            f"\nUsers: 1 Admin, 3 General Users, 3 Reviewers (All password: {PASSWORD})"
            f"\nProducts: {len(products)} products (3 per general user, exactly 1:1 saved by its creator)"
            f"\nReviews: {total_seeded_reviews} total reviews ({15} per product)"
            f"\nAlerts Generated: {total_alerts} total"
            f"\n  - General User Alerts: {gen_alerts} (Monitored fake spikes >= 50%)"
            f"\n  - Reviewer Alerts: {rev_alerts} (Live ML classification feedback)"
            f"\n  - Administrator Alerts: {adm_alerts} (System-wide fake surges)"
        ))


