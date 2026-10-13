# Graze Smasher Product Hunting Tool
## Hybrid Machine Learning Fake Review Detection & Integrity System
### CS619 Final-Year Project (BS Computer Science)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Django 5.0+](https://img.shields.io/badge/Django-5.0%2B-092E20.svg)](https://www.djangoproject.com/)
[![Django REST Framework](https://img.shields.io/badge/DRF-3.14%2B-red.svg)](https://www.django-rest-framework.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/Tests-40%20Passed%20(100%25)-brightgreen.svg)]()

Graze Smasher is an enterprise-grade academic research and software engineering system designed to combat deceptive online reviews across modern e-commerce storefronts (Amazon, eBay, Shopify, Walmart). The platform combines a **dual-tier machine learning ensemble (Calibrated LinearSVC + J-48 C4.5 Decision Tree)** with **TextBlob sentiment polarity analysis**, **rating discrepancy detection**, and a **sliding-window IP velocity heuristic engine** to evaluate customer review authenticity in real time.

---

## Table of Contents
1. [Key Features & System Highlights](#key-features--system-highlights)
2. [Architecture & Project Structure](#architecture--project-structure)
3. [Machine Learning Ensemble & Empirical Benchmarks](#machine-learning-ensemble--empirical-benchmarks)
4. [Role-Based Access Control (RBAC)](#role-based-access-control-rbac)
5. [Installation & Setup Guide](#installation--setup-guide)
6. [Training & Pipeline Execution](#training--pipeline-execution)
7. [Running the Web Application](#running-the-web-application)
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
  * **Tier 3 (Heuristic Signals)**: Real-time sliding-window IP velocity tracking and absolute discrepancy between star rating and natural language sentiment.
* **Unified Single-Point Inference Service**:
  * Single service function `analyze_review(text, rating, ip)` caches loaded model artifacts in memory with thread-safe locking and is used uniformly across the web GUI, REST API, seed loader, and automated test suite.
* **Dynamic Runtime Detection Configuration**:
  * Ensemble weights, classification thresholds (`Genuine >= 70%`, `Suspicious 40-69%`, `Fake < 40%`), and IP velocity thresholds are persisted via a database singleton (`SystemConfiguration`) and can be adjusted live in the Admin Panel without restarting the server.
* **Full-Featured E-Commerce Catalog & Product Hunting**:
  * Multi-attribute filtering (category, platform, authenticity status), full-text search, and pagination.
  * **Community Trust Rating**: Star rating calculated exclusively from verified genuine reviews, shielding shoppers from manipulated 5-star ratings.
  * **Timeline Trend Sequence**: Interactive Chart.js sequence displaying historical review scores over time.
* **Intelligence & Alert Engine**:
  * Automated detection of fake review ratios crossing configurable thresholds (default: $\ge 40\%$), sudden spikes in fake submissions ($\ge +25\%$), critical score drops ($<40\%$), and bot bursts.
  * In-app user notifications and console-based email dispatching.
  * **Deception Radar (`/suspicious-products/`)**: Discovery page filtering high-risk catalog products.
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
│   ├── ip_detector.py                 # Sliding-window IP velocity tracker & simulated IP generator
│   ├── service.py                     # Unified analyze_review() cached inference service
│   ├── train_all.py                   # Complete training, 5-fold CV, metrics evaluation & plotting
│   └── train_model.py                 # Prototype SVM training script
├── Trained_Model/
│   ├── svm_model.pkl                  # Serialized Calibrated LinearSVC Classifier
│   ├── vectorizer.pkl                 # Serialized TF-IDF (1, 2) Vectorizer
│   └── j48_model.pkl                  # Serialized J-48 Decision Tree Classifier
├── Django_Project/
│   ├── manage.py                      # Django project management utility
│   ├── graze_smasher/                 # Core settings, WSGI, and global routing
│   ├── accounts/                      # Authentication, profile, RBAC decorators, admin control panel
│   │   ├── admin_views.py             # Custom admin pages (users, flagged queue, settings, metrics)
│   │   ├── admin_forms.py             # Admin settings and role modification forms
│   │   └── tests.py                   # RBAC access control unit tests
│   ├── products/                      # Product models, catalog views, product CRUD, privacy view
│   │   ├── models.py                  # Product & SavedProduct models with computed metrics
│   │   ├── views.py                   # Search, filter, product detail, watchlist, privacy policy
│   │   └── tests.py                   # Product metrics and catalog tests
│   ├── reviews/                       # Review model & web submission handling
│   │   ├── models.py                  # Review model with multi-tier ML scores & labels
│   │   └── views.py                   # Authenticated review submission view
│   ├── analytics/                     # System analytics, KPI dashboards, and alert heuristic engine
│   │   ├── models.py                  # Alert & SystemConfiguration models
│   │   ├── services.py                # Rolling window alert detection & email dispatching
│   │   ├── views.py                   # User dashboard, overview charts, suspicious radar
│   │   └── tests.py                   # Alert trigger and KPI aggregation tests
│   ├── api/                           # Django REST Framework external integration API
│   │   ├── serializers.py             # Input & output schemas for batch products and reviews
│   │   ├── views.py                   # API endpoints with throttling & pagination
│   │   └── tests.py                   # End-to-end API endpoint tests
│   ├── detector/                      # Interactive ML sandbox & single-review tester
│   │   ├── views.py                   # Live input prediction view
│   │   └── tests.py                   # Feature extraction, IP rules, and service unit tests
│   └── templates/                     # Bootstrap 5 responsive templates (navbar, cards, modals)
├── Notebook/
│   ├── fake_review_detection.ipynb    # Full Jupyter Notebook with SVM, J-48, and composite analysis
│   ├── fake_review_detection.html     # HTML compiled export of notebook
│   └── create_notebook.py             # Notebook generator script
├── Screenshots/
│   ├── model_comparison_table.csv     # 80/20 & 5-fold CV empirical comparison table
│   ├── model_metrics.png              # Precision, Recall, F1 comparison bar chart
│   └── confusion_matrices.png         # Confusion matrices for SVM, J-48, and Composite
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

### 2. Multi-Model Architecture
1. **Calibrated SVM (`LinearSVC` + `CalibratedClassifierCV(cv=5)`)**:
   * Evaluates lexical patterns using sublinear TF-IDF word and character n-grams $(1, 2)$.
   * Calibrated with Platt scaling/sigmoid cross-validation to output continuous posterior probabilities $[0.0, 1.0]$.
2. **J-48 C4.5 Decision Tree (`DecisionTreeClassifier(criterion='entropy')`)**:
   * Evaluates 8 non-textual behavioral, structural, and affective features:
     1. `review_length`: Total character count.
     2. `word_count`: Total word tokens.
     3. `exclamation_count`: Syntactic intensity marker.
     4. `uppercase_ratio`: SHOUTING character frequency.
     5. `repeated_word_ratio`: Vocabulary redundancy $(N_{words} - N_{unique}) / N_{words}$.
     6. `sentiment_polarity`: TextBlob polarity $[-1.0, 1.0]$.
     7. `sentiment_subjectivity`: TextBlob subjectivity $[0.0, 1.0]$.
     8. `rating_vs_sentiment_mismatch`: Absolute discrepancy between normalized star rating and sentiment.
3. **Composite Scoring Formula**:
   $$\text{risk\_score} = (0.6 \times \text{ip\_flag}) + (0.4 \times \text{mismatch\_risk})$$
   $$\text{fake\_risk} = (w_{svm} \times p_{svm}) + (w_{j48} \times p_{j48}) + (w_{risk} \times \text{risk\_score})$$
   $$\text{authenticity\_score} = (1.0 - \text{fake\_risk}) \times 100$$
   * Defaults: $w_{svm} = 0.50, w_{j48} = 0.30, w_{risk} = 0.20$.

### 3. Empirical Evaluation Results (80/20 Test Split + 5-Fold Stratified CV)

| Model Architecture | Accuracy | Precision | Recall | F1-Score | 5-Fold Stratified CV |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Calibrated SVM (LinearSVC)** | 88.90% | 88.91% | 88.90% | 88.90% | **89.65%** |
| **J-48 Decision Tree (C4.5)** | 75.55% | 75.59% | 75.55% | 75.54% | **74.41%** |
| **Composite Ensemble (SVM + J48 + Risk)** | **90.55%** | **90.78%** | **90.55%** | **90.54%** | **90.63%** |

*The composite ensemble achieves superior generalization over individual models, minimizing false positives while detecting subtle rating-sentiment anomalies and bot bursts.*

---

## Role-Based Access Control (RBAC)

The application enforces a 3-tier role hierarchy:

| Capability | Anonymous | General User (`general`) | Reviewer (`reviewer`) | Administrator (`admin`) |
| :--- | :---: | :---: | :---: | :---: |
| Browse Product Catalog & Search | Yes | Yes | Yes | Yes |
| View Authenticity Trends & Analytics | Yes | Yes | Yes | Yes |
| Access Interactive ML Sandbox | Yes | Yes | Yes | Yes |
| Save Products to Monitored Watchlist | No | Yes | Yes | Yes |
| Personal Activity Dashboard | No | Yes | Yes | Yes |
| Add New Products to Catalog | No | No | Yes | Yes |
| Post Customer Product Reviews | No | No | Yes | Yes |
| Manage User Roles (Promote/Demote) | No | No | No | Yes |
| Flagged Review Moderation (Override) | No | No | No | Yes |
| Dynamic Runtime ML Settings Editor | No | No | No | Yes |
| Inspect Model Artifacts & CV Metrics | No | No | No | Yes |

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
Graze Smasher includes a management command to populate the database with test users, catalog products, and 90 ML-analyzed reviews from the dataset:
```bash
python manage.py seed_demo_data --clear
```

**Default Demo Credentials:**
* **Administrator**: Username `admin` | Password `admin123`
* **Reviewer**: Username `reviewer` | Password `reviewer123`
* **General User**: Username `user` | Password `user123`

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
5. Save model artifacts to `Trained_Model/` (`svm_model.pkl`, `vectorizer.pkl`, `j48_model.pkl`).
6. Save confusion matrices and evaluation charts to `Screenshots/`.

---

## Running the Web Application

Start the local Django development server:
```bash
cd Django_Project
python manage.py runserver
```
Visit `http://127.0.0.1:8000/` in your web browser.

### Key Pages to Explore:
* **Product Catalog (`/`)**: Search and filter products by category, e-commerce platform, or authenticity status.
* **Product Detail (`/products/<id>/`)**: Visual breakdown of reviews, star distribution, timeline trend chart, and submission form.
* **Suspicious Radar (`/suspicious-products/`)**: Threat detection hub isolating products with low authenticity scores.
* **System Analytics (`/analytics/`)**: Chart.js graphs for model cross-validation, catalog distribution, and platform comparisons.
* **ML Sandbox (`/detector/`)**: Interactive review tester to experiment with arbitrary text, star ratings, and client IPs.
* **Personal Dashboard (`/dashboard/`)**: Saved products watchlist, review history, and unread alerts.
* **Admin Control Panel (`/admin-panel/`)**:
  * User Role Management: `/accounts/admin-panel/users/`
  * Flagged Review Moderation Queue: `/accounts/admin-panel/flagged/`
  * Runtime Detection Settings: `/accounts/admin-panel/settings/`
  * ML Diagnostic Inspector: `/accounts/admin-panel/models/`
* **Privacy & Algorithmic Ethics Policy (`/privacy/`)**: Explains data use, IP velocity tracking, and user rights.

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
        "rating": 5,
        "ip": "203.0.113.10"
      },
      {
        "text": "Great product! buy now cheap plastic broke immediately best ever!",
        "rating": 5,
        "ip": "198.51.100.11"
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
      "ip_flag": false,
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
    "rating": 5,
    "ip": "203.0.113.44"
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
    "ip_flag": false,
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

Graze Smasher features a 40-test automated verification suite covering:
* **ML Service & Heuristics**: Feature extraction validity, sliding-window IP velocity tracking, composite scoring math, and model caching.
* **Alert Engine**: Fake ratio triggers, surge calculations, authenticity drops, and notification deduplication.
* **Role-Based Access Control**: Decorators and access boundaries across General, Reviewer, and Admin users.
* **Product Catalog & Views**: Computed property metrics, search queries, watchlist saving, and review submission.
* **REST API Endpoints**: Batch ingestion, single review evaluation, 400 validation handling, and 404 responses.

To execute the test suite:
```bash
cd Django_Project
python manage.py test
```
**Expected Output:**
```text
Creating test database for alias 'default'...
........................................
----------------------------------------------------------------------
Ran 40 tests in 16.261s

OK
Destroying test database for alias 'default'...
```

---

## Known Limitations & Real-World Constraints

In accordance with academic standards, the following known limitations apply to this prototype:

1. **Synthetic Nature of Dataset Fake Reviews**:
   * The training dataset (Salminen et al., 2022) utilizes **computer-generated (CG)** fake reviews synthesized by text generation algorithms rather than human-written incentivized fake reviews.
   * While machine learning algorithms easily detect statistical patterns in CG text (resulting in ~90% accuracy), human-written deceptive reviews (e.g., paid review farms) may exhibit greater stylistic subtlety, potentially yielding lower real-world detection accuracy.
2. **Simulated IP Addresses in Demonstration**:
   * Public e-commerce datasets do not include consumer IP addresses due to privacy constraints.
   * To demonstrate sliding-window network velocity tracking, the demo seeding and test scripts utilize clearly designated **simulated IPs** drawn from the RFC 5737 documentation blocks (`192.0.2.x`, `198.51.100.x`, `203.0.113.x`).
3. **Probabilistic Nature of Machine Learning**:
   * Classification outputs represent statistical likelihoods, not definitive legal proofs of fraud. To protect genuine consumers from unfair misclassification, Graze Smasher implements a three-tier system (`Genuine`, `Suspicious`, `Fake`) along with an Administrator Moderation Queue for human-in-the-loop review.

---

## Algorithmic Ethics & Privacy Policy

* **Data Minimization**: Client IP addresses are processed strictly to mitigate automated submission bot bursts and are never sold or shared.
* **Explainability**: Every review displays an authenticity score pill alongside detailed signal breakdowns (SVM probability, J-48 tree classification, sentiment polarity, and IP status).
* **Human-in-the-Loop Override**: Administrators retain the ability to manually inspect flagged content and override labels through the moderation queue (`/accounts/admin-panel/flagged/`).
* A dedicated **Privacy, Data Use & Algorithmic Ethics Policy** is accessible at `/privacy/`.

---

## Author & Academic Information
* **Project**: Graze Smasher Product Hunting Tool
* **Course**: CS619 Final-Year Project
* **Degree**: Bachelor of Science in Computer Science
* **Year**: 2026
