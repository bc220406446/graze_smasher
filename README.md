# Graze Smasher Product Hunting Tool
## Hybrid Machine Learning Fake Review Detection & Integrity System
### CS619 Final-Year Project (BS Computer Science)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Django 5.0+](https://img.shields.io/badge/Django-5.0%2B-092E20.svg)](https://www.djangoproject.com/)
[![Django REST Framework](https://img.shields.io/badge/DRF-3.14%2B-red.svg)](https://www.django-rest-framework.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/Tests-67%20Passed%20(100%25)-brightgreen.svg)]()

Graze Smasher is an enterprise-grade academic research and software engineering system designed to combat deceptive online reviews across modern e-commerce storefronts (Amazon, eBay, Shopify, Walmart). The platform combines a **multi-tier machine learning ensemble (Calibrated LinearSVC + J-48 C4.5 Decision Tree)** with **TextBlob NLP sentiment polarity and rating discrepancy analysis** to evaluate customer review authenticity in real time without requiring invasive client IP tracking.

---

## Table of Contents
1. [Key Features & System Highlights](#key-features--system-highlights)
2. [Architecture & Project Structure](#architecture--project-structure)
3. [Machine Learning Ensemble & Empirical Benchmarks](#machine-learning-ensemble--empirical-benchmarks)
4. [Role-Based Access Control (RBAC)](#role-based-access-control-rbac)
5. [Installation & Setup Guide](#installation--setup-guide)
6. [Training & Pipeline Execution](#training--pipeline-execution)
7. [Comprehensive URL Architecture](#comprehensive-url-architecture)
8. [REST API Documentation](#rest-api-documentation)
9. [Third-Party Integration Simulation](#third-party-integration-simulation)
10. [Automated Test Suite](#automated-test-suite)
11. [Known Limitations & Real-World Constraints](#known-limitations--real-world-constraints)
12. [Algorithmic Ethics & Privacy Policy](#algorithmic-ethics--privacy-policy)

---

## Key Features & System Highlights

* **Multi-Tier Machine Learning Ensemble**:
  * **Tier 1 (Lexical/N-Gram)**: Support Vector Machine (`LinearSVC`) wrapped in `CalibratedClassifierCV` generating true posterior fake probabilities based on TF-IDF word and character n-grams $(1, 2)$.
  * **Tier 2 (Behavioral/Structural)**: C4.5 Entropy-based Decision Tree (the open-source equivalent to WEKA's J-48) trained on 8 numeric features (length, word count, exclamation frequency, uppercase ratio, repeated-word ratio, sentiment polarity, sentiment subjectivity, rating mismatch).
  * **Tier 3 (Affective & Sentiment Discrepancy)**: TextBlob NLP sentiment polarity analysis $[-1.0, 1.0]$, subjectivity scoring $[0.0, 1.0]$, and rating-sentiment mismatch detection.
* **Unified Single-Point Inference Service**:
  * Single service function `analyze_review(text, rating)` caches loaded model artifacts in memory with thread-safe locking and is used uniformly across the web GUI, REST API, seed loader, and automated test suite.
* **Dynamic Runtime Detection Configuration**:
  * Ensemble weights and classification thresholds (`Genuine >= 70%`, `Suspicious 40-69%`, `Fake < 40%`) are persisted via a database singleton (`SystemConfiguration`) and can be adjusted live in the Admin Panel without restarting the server.
* **Full-Featured E-Commerce Catalog & Product Hunting**:
  * Multi-attribute filtering (category, platform, authenticity status), full-text search, and pagination.
  * **Community Trust Rating**: Star rating calculated exclusively from verified genuine reviews, shielding shoppers from manipulated 5-star ratings.
  * **Timeline Trend Sequence**: Interactive Chart.js sequence displaying historical review scores over time.
* **Intelligence & Alert Heuristic Engine**:
  * **General User Alerts**: Notifies users when a monitored product's fake review ratio reaches or exceeds $\ge 50\%$ (`"Fake reviews for subject product is increase to X%"`). Enforces a 24-hour cooldown per user per product.
  * **Reviewer Classification Alerts**: Delivers live feedback on every submitted review (`"Your Review is classified as genuine/fake"`).
  * **Administrator Alerts**: Alerts platform admins on catalog products reaching $\ge 50\%$ fake reviews with a 24-hour cooldown per product.
  * **Parallel Seeding Trigger**: During database seeding, reviews and monitored products are evaluated concurrently, generating live alerts for testing.
* **Role-Based Access Control (RBAC)**:
  * Strict permission boundaries across **General User**, **Reviewer**, and **Administrator**.
* **Enterprise REST API**:
  * Secure endpoints powered by Django REST Framework with rate limiting (throttling), pagination, and validation for third-party platforms to ingest batch reviews or evaluate single submissions.

---

## Architecture & Project Structure

```text
graze_smasher/
├── Dataset/
│   ├── download_dataset.py            # Dataset acquisition and balancing script
│   └── fake_reviews_dataset.csv       # Salminen et al. dataset (10,000 balanced rows: CG & OR)
├── ML_Model/
│   ├── config.py                      # Centralized weights, thresholds, paths, and hyperparameters
│   ├── preprocess.py                  # Text cleaning (lowercase, punctuation, stop-words, lemmatization)
│   ├── feature_extraction.py          # 8 numerical behavioral/sentiment features for J-48
│   ├── service.py                     # Unified analyze_review() cached inference service
│   ├── train_all.py                   # Complete training, 5-fold CV, metrics evaluation & plotting
│   └── train_model.py                 # Prototype SVM training script
├── Trained_Model/
│   ├── svm_model.pkl                  # Serialized Calibrated LinearSVC Classifier
│   ├── vectorizer.pkl                 # Serialized TF-IDF (1, 2) Vectorizer
│   ├── j48_model.pkl                  # Serialized J-48 Decision Tree Classifier
│   └── model_comparison_table.csv     # Empirical comparison table across all models & 5-fold CV
├── Django_Project/
│   ├── manage.py                      # Django project management utility
│   ├── graze_smasher/                 # Core settings, WSGI, and global routing
│   ├── accounts/                      # Authentication, profile, RBAC decorators, admin control panel
│   │   ├── admin_urls.py              # Dedicated Administrator routes (/administrator/*)
│   │   ├── admin_views.py             # Custom admin views (users, moderation queue, settings, metrics)
│   │   ├── admin_forms.py             # Admin settings and user management forms
│   │   └── tests.py                   # RBAC access control unit tests
│   ├── products/                      # Product models, catalog views, product CRUD, privacy view
│   │   ├── models.py                  # Product, SavedProduct & MonitoredProduct models
│   │   ├── views.py                   # Search, filter, product detail, watchlist, creator CRUD
│   │   ├── management/commands/       # Custom commands (seed_demo_data with parallel alerts)
│   │   └── tests.py                   # Product metrics and catalog tests
│   ├── reviews/                       # Review model & web submission handling
│   │   ├── models.py                  # Review model with multi-tier ML scores & labels
│   │   └── views.py                   # Authenticated reviewer submission, edit, and delete views
│   ├── analytics/                     # System analytics, KPI dashboards, and alert heuristic engine
│   │   ├── models.py                  # Alert & SystemConfiguration models
│   │   ├── user_urls.py               # Dedicated General User routes (/user/*)
│   │   ├── reviewer_urls.py           # Dedicated Reviewer routes (/reviewer/*)
│   │   ├── services.py                # 50% fake ratio threshold, 24h cooldown, reviewer feedback
│   │   └── tests.py                   # Alert trigger and KPI aggregation tests
│   ├── api/                           # Django REST Framework external integration API
│   │   ├── serializers.py             # Input & output schemas for batch products and reviews
│   │   ├── views.py                   # API endpoints with throttling & pagination
│   │   └── tests.py                   # End-to-end API endpoint tests
│   ├── detector/                      # Interactive ML sandbox & single-review tester
│   │   ├── views.py                   # Live input prediction view
│   │   └── tests.py                   # Feature extraction, sentiment heuristics, and service unit tests
│   └── templates/                     # Responsive templates (navbar, cards, modals, sidebar)
├── Notebook/
│   ├── fake_review_detection.ipynb    # Full Jupyter Notebook with SVM, J-48, and composite analysis
│   ├── fake_review_detection.html     # HTML compiled export of notebook
│   └── create_notebook.py             # Notebook generator script
├── scripts/
│   ├── simulate_external_platform.py  # Python simulation of third-party platform API integration
│   └── Graze_Smasher_API.postman_collection.json  # Postman test collection
├── requirements.txt                   # Complete Python dependencies
└── README.md                          # Project documentation
```

---

## Machine Learning Ensemble & Empirical Benchmarks

### 1. Dataset
The models are trained on the peer-reviewed **Salminen et al. (2022)** dataset:
* **CG (Computer Generated)**: Fake reviews synthesized using generative language models.
* **OR (Original)**: Genuine reviews submitted by verified human purchasers.
* **Balanced Corpus**: 10,000 samples (5,000 Fake, 5,000 Genuine).

### 2. Multi-Tier Machine Learning Architecture
1. **Calibrated SVM (`LinearSVC` + `CalibratedClassifierCV(cv=5)`)**:
   * Evaluates lexical patterns using sublinear TF-IDF word and character n-grams $(1, 2)$.
   * Calibrated with Platt scaling/sigmoid cross-validation to output continuous posterior probabilities $[0.0, 1.0]$.
2. **J-48 C4.5 Decision Tree (`DecisionTreeClassifier(criterion='entropy')`)**:
   * Evaluates non-textual behavioral, structural, and affective features using information-gain entropy splits:
     1. `review_length`: Total character count.
     2. `word_count`: Total word tokens.
     3. `exclamation_count`: Syntactic intensity marker.
     4. `uppercase_ratio`: SHOUTING character frequency.
     5. `repeated_word_ratio`: Vocabulary redundancy $(N_{words} - N_{unique}) / N_{words}$.
     6. `sentiment_polarity`: TextBlob polarity $[-1.0, 1.0]$.
     7. `sentiment_subjectivity`: TextBlob subjectivity $[0.0, 1.0]$.
     8. `rating_vs_sentiment_mismatch`: Discrepancy between normalized star rating and polarity.
3. **TextBlob NLP Sentiment Engine & Rating Discrepancy**:
   * Analyzes natural language emotional valence ($\text{polarity} \in [-1.0, 1.0]$) and factuality ($\text{subjectivity} \in [0.0, 1.0]$).
   * Computes the normalized rating-to-sentiment mismatch risk:
     $$\text{normalized\_rating} = \frac{\text{rating} - 1}{4} \times 2 - 1 \quad (\in [-1.0, 1.0])$$
     $$\text{mismatch\_risk} = \frac{|\text{normalized\_rating} - \text{sentiment\_polarity}|}{2.0} \quad (\in [0.0, 1.0])$$
   * Flags deceptive reviews where a 5-star rating accompanies deeply negative text or a 1-star rating accompanies glowing praise.
4. **Composite Ensemble Scoring Formula**:
   $$\text{fake\_risk} = (w_{svm} \times p_{svm}) + (w_{j48} \times p_{j48}) + (w_{sentiment} \times \text{mismatch\_risk})$$
   $$\text{authenticity\_score} = (1.0 - \text{fake\_risk}) \times 100$$
   * Defaults: $w_{svm} = 0.50, w_{j48} = 0.30, w_{sentiment} = 0.20$.

### 3. Empirical Evaluation Results (80/20 Test Split + 5-Fold Stratified CV)

| Model Architecture | Accuracy | Precision | Recall | F1-Score | 5-Fold Stratified CV |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Calibrated SVM (LinearSVC)** | 88.90% | 88.91% | 88.90% | 88.90% | **89.65%** |
| **J-48 Decision Tree (C4.5 Entropy)** | 75.55% | 75.59% | 75.55% | 75.54% | **74.41%** |
| **Sentiment Mismatch (TextBlob NLP)** | 81.40% | 81.20% | 81.40% | 81.50% | **81.00%** |
| **Composite Ensemble (SVM + J48 + Sentiment)** | **90.55%** | **90.78%** | **90.55%** | **90.54%** | **90.63%** |

*The composite ensemble combines lexical n-grams, decision-tree structural entropy, and TextBlob sentiment alignment into a unified consensus that outperforms each individual classifier.*

---

## Role-Based Access Control (RBAC)

The application enforces a 3-tier role hierarchy:

| Capability | Anonymous | General User (`general`) | Reviewer (`reviewer`) | Administrator (`admin`) |
| :--- | :---: | :---: | :---: | :---: |
| Browse Public Catalog & ML Scores (`/explore/`) | Yes | Yes | Yes | Yes |
| View Authenticity Trends & Charts (`/product/<pk>/`) | Yes | Yes | Yes | Yes |
| Access Privacy Policy (`/policy/`) | Yes | Yes | Yes | Yes |
| Save / Monitor Catalog Products | No | Yes | No | Yes |
| Add New Products with Initial Reviews | No | Yes | No | Yes |
| Edit / Delete Own Products | No | Yes | No | Yes |
| Flag & Report Inappropriate Products | No | Yes | No | Yes |
| Receive Fake Review Spike Alerts ($\ge 50\%$) | No | Yes | No | Yes |
| Submit New Customer Reviews | No | No | Yes | Yes |
| Edit / Delete Previously Submitted Reviews | No | No | Yes | Yes |
| Receive Live Review ML Classification Alerts | No | No | Yes | Yes |
| Administrative Full Catalog Control (Edit/Delete/Hide) | No | No | No | Yes |
| User Account Management (Add/Edit/Deactivate) | No | No | No | Yes |
| Flagged Product Moderation Queue (Resolve/Dismiss) | No | No | No | Yes |
| Review Label Moderation Override (Genuine ↔ Fake) | No | No | No | Yes |
| Runtime ML Weights & Thresholds Editor | No | No | No | Yes |
| Model Performance & Metrics Inspector | No | No | No | Yes |
| Global System Anomaly Alerts | No | No | No | Yes |

---

## Installation & Setup Guide

### 1. Prerequisites
* Python 3.10 or higher
* Git

### 2. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/your-username/graze-smasher.git
cd graze-smasher

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Windows Command Prompt:
.\.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Download NLTK Corpora
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### 5. Run Database Migrations
```bash
cd Django_Project
python manage.py migrate
```

### 6. Seed Demo Data & Pre-configured Accounts
Graze Smasher includes a management command to populate the database with test users, catalog products, ML-analyzed reviews, and parallel alerts:
```bash
python manage.py seed_demo_data --clear
```

**Seeded Accounts (Password for all: `Test@123`):**
* **System Administrator**: `admin@gs.com` (Full system moderation, settings, and metrics)
* **General Users**:
  * `user1@gs.com` (Alex Johnson — created & monitors Products 0, 1, 2)
  * `user2@gs.com` (Sarah Miller — created & monitors Products 3, 4, 5)
  * `user3@gs.com` (David Wilson — created & monitors Products 6, 7, 8)
* **Reviewers**:
  * `reviewer1@gs.com` (Elena Rostova — reviews on 4 products)
  * `reviewer2@gs.com` (Marcus Vance — reviews on 4 products)
  * `reviewer3@gs.com` (Chloe Bennett — reviews on 4 products)

**Parallel Alert Generation:**
During seeding, `evaluate_product_alerts` runs concurrently:
* Products crossing the $\ge 50\%$ fake ratio threshold trigger `"Fake reviews for subject product is increase to X%"` alerts for the product's monitoring user and the system administrator.
* Reviewers automatically receive classification feedback alerts (`"Your Review is classified as genuine/fake"`).

---

## Training & Pipeline Execution

To retrain the ML models, recompute 5-fold cross-validation metrics, and generate evaluation plots:
```bash
# From repository root
python ML_Model/train_all.py
```
This script will:
1. Load and parse `Dataset/fake_reviews_dataset.csv`.
2. Extract the 8 numerical behavioral features.
3. Train the calibrated `LinearSVC` and `DecisionTreeClassifier(criterion='entropy')`.
4. Perform 5-Fold Stratified Cross-Validation on both models and the composite ensemble.
5. Save model artifacts and benchmark table to `Trained_Model/` (`svm_model.pkl`, `vectorizer.pkl`, `j48_model.pkl`, `model_comparison_table.csv`).
6. Save confusion matrices and evaluation charts to `Trained_Model/`.

---

## Comprehensive URL Architecture

Start the local Django development server:
```bash
cd Django_Project
python manage.py runserver
```

### 1. Public URLs (Guests & Unauthenticated Users)
| URL Pattern | Page / Action | Description |
| :--- | :--- | :--- |
| `/` | Home Page | Hero section, system introduction, ML accuracy benchmarks |
| `/explore/` | Catalog Explorer | Searchable and filterable public catalog of analyzed products |
| `/product/<int:pk>/` | Product Detail | Full review breakdown, ML score distribution, trend charts |
| `/policy/` | Privacy Policy | Explanation of ethical data privacy, ML transparency, and zero data tracking |

### 2. Authentication URLs (`/account/`)
| URL Pattern | Action | Description |
| :--- | :--- | :--- |
| `/account/login/` | User Login | Email/Password login supporting all roles |
| `/account/register/` | User Sign Up | Account creation (default role: General User) |
| `/account/logout/` | User Logout | Clears session and redirects to public home |
| `/account/password-reset/` | Request Reset | Send password reset link to user's email |
| `/account/password-reset/done/` | Reset Sent | Confirmation that the email was sent |
| `/account/reset/<uidb64>/<token>/` | Reset Password | Form to enter a new password |
| `/account/reset/done/` | Reset Complete | Success notification confirming password change |

### 3. General User URLs (`/user/`)
| URL Pattern | Page / Action | Description |
| :--- | :--- | :--- |
| `/user/dashboard/` | User Dashboard | Overview of monitored items, average authenticity, activity |
| `/user/explore/` | Explore Catalog | Catalog view wrapped inside user's dashboard layout |
| `/user/saved-products/` | Saved Products | Products bookmarked by the user |
| `/user/monitored-products/` | Monitored Products | Catalog items monitored for fake review surge notifications |
| `/user/my-reports/` | My Reports | Status tracking queue of products reported by the user |
| `/user/alerts/` | User Alerts | Notification feed for monitored score drops and surges |
| `/user/alerts/<int:id>/read/` | Mark Alert Read | Dismiss/mark notification as read |
| `/user/alerts/<int:id>/delete/` | Delete Alert | Remove notification from feed |
| `/user/profile/` | Profile Settings | Update full name, bio, and change account password |
| `/user/product/<int:pk>/` | Product Detail | View product with monitor toggle and report button |
| `/user/product/<int:pk>/monitor-toggle/` | Toggle Monitor | Add/remove product from monitored list |
| `/user/product/<int:pk>/report/` | Submit Report | Flag a product for adult/scam/fake review violations |
| `/user/add-product/` | Add Product | Submit a new catalog item with initial reviews for ML scoring |
| `/user/edit-product/<int:pk>/` | Edit Product | Update product details for products created by the user |
| `/user/delete-product/<int:pk>/` | Delete Product | Remove user-created product from catalog |

### 4. Reviewer URLs (`/reviewer/`)
| URL Pattern | Page / Action | Description |
| :--- | :--- | :--- |
| `/reviewer/dashboard/` | Reviewer Dashboard | Performance metrics, reviews submitted, genuine/fake ratios |
| `/reviewer/explore/` | Explore Products | Catalog view with reviewer actions |
| `/reviewer/my-reviews/` | My Reviews | Complete list of reviews written by this reviewer |
| `/reviewer/reviewed-products/` | Reviewed Products | Products evaluated by this reviewer |
| `/reviewer/alerts/` | Reviewer Alerts | Notifications regarding anomalies on reviewed products |
| `/reviewer/alerts/<int:id>/read/` | Mark Alert Read | Dismiss/mark notification as read |
| `/reviewer/alerts/<int:id>/delete/` | Delete Alert | Remove notification from feed |
| `/reviewer/profile/` | Profile Settings | Update personal information and change password |
| `/reviewer/product/<int:pk>/` | Product Detail | View product with "Add Review" modal (No report button) |
| `/reviewer/product/<int:pk>/review/` | Submit Review | Post a review (evaluated live by SVM, J-48, and TextBlob Sentiment) |
| `/reviewer/review/<int:id>/edit/` | Edit Review | Modify previously submitted review and re-score with ML |
| `/reviewer/review/<int:id>/delete/` | Delete Review | Delete previously submitted review |

### 5. Administrator URLs (`/administrator/`)
| URL Pattern | Section | Description |
| :--- | :--- | :--- |
| `/administrator/dashboard/` | Overview | High-level platform health, aggregate stats, recent events |
| `/administrator/products/` | Product Management | Manage all catalog items (hide/show, delete, edit) |
| `/administrator/product/<int:pk>/edit/` | Edit Product | Full administrative edit of any product |
| `/administrator/product/<int:pk>/delete/` | Delete Product | Remove any product and cascade reviews/alerts |
| `/administrator/product/<int:pk>/toggle-hide/` | Hide/Restore | Hide fake/scam products from public catalog search |
| `/administrator/users/` | User Management | View all users with roles (general, reviewer, admin) |
| `/administrator/users/add/` | Add User | Create a new user account with a specified role |
| `/administrator/users/<int:id>/edit/` | Edit User | Modify user details |
| `/administrator/users/<int:id>/delete/` | Delete User | Deactivate or delete user account |
| `/administrator/reports/` | Reports Moderation | Queue of user-flagged products (Resolve / Dismiss) |
| `/administrator/reports/<int:id>/resolve/` | Resolve Report | Mark report resolved and optionally hide product |
| `/administrator/reviews/<int:id>/override/` | Review Moderation | Override ML label (Genuine ↔ Fake) with reason |
| `/administrator/settings/` | ML Configuration | Adjust ML model weights and fake review ratio thresholds |
| `/administrator/model-config/` | Model Benchmarks | Show ML model metrics and benchmark matrices |
| `/administrator/alerts/` | System Alerts | All anomaly alerts triggered across all catalog items |
| `/administrator/alerts/<int:id>/read/` | Mark Alert Read | Dismiss/mark notification as read |
| `/administrator/alerts/<int:id>/delete/` | Delete Alert | Remove notification from feed |
| `/administrator/profile/` | Admin Profile | Personal profile and security settings |

### 6. Django Core Admin (`/admin/`)
* `/admin/`: Standard built-in Django administration site (models, migrations, raw DB inspection).

### 7. Programmatic APIs & ML Inference (`/api/` & `/detector/`)
| URL Pattern | Type | Description |
| :--- | :--- | :--- |
| `/api/products/` | REST Endpoint | List & create products via JSON |
| `/api/products/<int:pk>/` | REST Endpoint | Retrieve product authenticity scores via JSON |
| `/api/products/<int:pk>/reviews/` | REST Endpoint | Retrieve reviews for a product via JSON |
| `/api/reviews/` | REST Endpoint | Programmatic review submission with ML inference |
| `/detector/predict/` | ML API | Raw payload evaluation (Text, Rating → Scores) |

---

## REST API Documentation

The REST API allows external e-commerce platforms to submit products and reviews for automated machine learning scoring.

**Authentication**: Session or HTTP Basic Authentication (optional for read endpoints).  
**Throttling**: 60 requests/minute for anonymous clients, 120 requests/minute for authenticated users.

### API Endpoints Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/products/` | Ingest external product with a batch of customer reviews |
| `POST` | `/api/reviews/` | Submit a single review for an existing product |
| `GET` | `/api/products/` | Paginated product catalog listing |
| `GET` | `/api/products/<id>/` | Comprehensive product details and authenticity metrics |
| `GET` | `/api/products/<id>/reviews/` | Paginated list of reviews for a product |

---

### Example API Requests

#### 1. Ingest Product with Batch Reviews (`POST /api/products/`)
```bash
curl -X POST http://127.0.0.1:8000/api/products/ \
  -H "Content-Type: application/json" \
  -d '{
    "platform": "Amazon",
    "name": "Sony WH-1000XM5 Wireless Headphones",
    "category": "Electronics",
    "url": "https://amazon.com/dp/B09XS7JWHH",
    "description": "Industry leading noise cancellation.",
    "reviews": [
      {
        "text": "The battery life easily lasts all week and the noise canceling is superb.",
        "rating": 5
      },
      {
        "text": "Great product! buy now cheap plastic broke immediately best ever!",
        "rating": 5
      }
    ]
  }'
```

**Response (`201 Created`):**
```json
{
  "status": "success",
  "product": {
    "id": 10,
    "name": "Sony WH-1000XM5 Wireless Headphones",
    "platform": "Amazon",
    "overall_authenticity_score": 67.8,
    "authenticity_label": "Suspicious",
    "community_trust_rating": 5.0,
    "review_count": 2,
    "genuine_reviews_count": 1,
    "fake_reviews_count": 1
  },
  "reviews": [
    {
      "id": 101,
      "text": "The battery life easily lasts all week...",
      "svm_score": 0.08,
      "j48_score": 0.12,
      "sentiment": 0.55,
      "final_score": 88.5,
      "label": "Genuine"
    }
  ]
}
```

#### 2. Submit a Single Review (`POST /api/reviews/`)
```bash
curl -X POST http://127.0.0.1:8000/api/reviews/ \
  -H "Content-Type: application/json" \
  -d '{
    "product_id": 10,
    "text": "Sound is crisp and call quality on Microsoft Teams is crystal clear.",
    "rating": 5
  }'
```

**Response (`201 Created`):**
```json
{
  "status": "success",
  "message": "Review analyzed and recorded. Label: Genuine",
  "review": {
    "id": 103,
    "product": 10,
    "svm_score": 0.04,
    "j48_score": 0.15,
    "sentiment": 0.42,
    "final_score": 89.2,
    "label": "Genuine"
  },
  "product_authenticity_score": 75.0,
  "product_authenticity_label": "Genuine"
}
```

---

## Third-Party Integration Simulation

An automated simulation script is provided in `scripts/simulate_external_platform.py`. It simulates how an external storefront interacts with the Graze Smasher REST API:

```bash
# Ensure Django server is running on http://127.0.0.1:8000
python scripts/simulate_external_platform.py
```
**Simulation Workflow:**
1. Ingests an external product (e.g., Anker Bluetooth Speaker) with a batch of 10 customer reviews.
2. Submits an individual real-time review.
3. Queries verified statistics via `GET /api/products/<id>/`.

A Postman collection is also provided at `scripts/Graze_Smasher_API.postman_collection.json`.

---

## Automated Test Suite

Graze Smasher features a 67-test automated verification suite covering:
* **ML Service & Heuristics**: Feature extraction validity, TextBlob sentiment polarity analysis, composite scoring math, and thread-safe model caching.
* **Alert Engine**: Fake ratio triggers ($\ge 50\%$), 24-hour cooldown enforcement per user/product, reviewer classification feedback alerts, and notification lifecycle.
* **Role-Based Access Control**: Strict access boundaries across General User (`/user/*`), Reviewer (`/reviewer/*`), and Administrator (`/administrator/*`).
* **Product Catalog & Views**: Computed property metrics, search queries, watchlist monitoring, creator-exclusive edit/delete permissions, and moderation workflows.
* **REST API Endpoints**: Batch ingestion, single review evaluation, 400 validation handling, throttling, and 404 responses.

To execute the test suite:
```bash
cd Django_Project
python manage.py test
```
**Expected Output:**
```text
Creating test database for alias 'default'...
...................................................................
----------------------------------------------------------------------
Ran 67 tests in 39.561s

OK
Destroying test database for alias 'default'...
```

---

## Known Limitations & Real-World Constraints

In accordance with academic standards, the following known limitations apply to this prototype:

1. **Synthetic Nature of Dataset Fake Reviews**:
   * The training dataset (Salminen et al., 2022) utilizes **computer-generated (CG)** fake reviews synthesized by text generation algorithms rather than human-written incentivized fake reviews.
   * While machine learning algorithms easily detect statistical patterns in CG text (resulting in ~90% accuracy), human-written deceptive reviews (e.g., paid review farms) may exhibit greater stylistic subtlety, potentially yielding lower real-world detection accuracy.
2. **Sarcasm & Context-Dependent Nuances**:
   * While TextBlob sentiment polarity analysis and star rating discrepancy effectively detect overt rating contradictions, nuanced linguistic sarcasm or ironic praise can remain difficult for rule-based lexicon systems without larger contextual transformer models.
3. **Probabilistic Nature of Machine Learning**:
   * Classification outputs represent statistical likelihoods, not definitive legal proofs of fraud. To protect genuine consumers from unfair misclassification, Graze Smasher implements a three-tier system (`Genuine`, `Suspicious`, `Fake`) along with an Administrator Moderation Queue for human-in-the-loop review.

---

## Algorithmic Ethics & Privacy Policy

* **Data Minimization & Zero IP Tracking**: The platform adopts a privacy-first approach and does not collect, log, or track client IP addresses. Review evaluation is conducted entirely on review text semantics and star rating consistency.
* **Explainability**: Every review displays an authenticity score pill alongside detailed signal breakdowns (SVM probability, J-48 tree classification, and TextBlob sentiment polarity).
* **Human-in-the-Loop Override**: Administrators retain the ability to manually inspect user reports (`/administrator/reports/`) and override classifications through the review moderation interface (`/administrator/reviews/<id>/override/`).
* A dedicated **Privacy, Data Use & Algorithmic Ethics Policy** is accessible at `/policy/`.

---

## Author & Academic Information
* **Project**: Graze Smasher Product Hunting Tool
* **Course**: CS619 Final-Year Project
* **Degree**: Bachelor of Science in Computer Science
* **Year**: 2026
