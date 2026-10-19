"""
Django settings for graze_smasher project.
CS619 Final-Year Project
"""

import os
import sys
from pathlib import Path

# Build paths inside the project: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = BASE_DIR.parent

# Ensure workspace and ML_Model are on sys.path
for p in [str(WORKSPACE_ROOT), str(WORKSPACE_ROOT / "ML_Model")]:
    if p not in sys.path:
        sys.path.insert(0, p)

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-%w#e-ig(80ix$l24gk*f)0c7zi3b-ifn=s@y2gvf4qnny%z4)o'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['*']


# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    
    # Third-Party Libraries
    'rest_framework',
    
    # Graze Smasher Domain Apps
    'accounts.apps.AccountsConfig',
    'products.apps.ProductsConfig',
    'reviews.apps.ReviewsConfig',
    'analytics.apps.AnalyticsConfig',
    'api.apps.ApiConfig',
]

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication',
        'rest_framework.authentication.BasicAuthentication',
    ],
    'DEFAULT_THROTTLE_CLASSES': [
        'rest_framework.throttling.AnonRateThrottle',
        'rest_framework.throttling.UserRateThrottle',
    ],
    'DEFAULT_THROTTLE_RATES': {
        'anon': '60/minute',
        'user': '120/minute',
    },
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 10,
}

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'graze_smasher.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'graze_smasher.wsgi.application'


# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}


# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True


# Static and Media Files
STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
STATIC_ROOT = BASE_DIR / 'staticfiles'

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Authentication Configuration & Backends
LOGIN_URL = '/account/login/'
LOGIN_REDIRECT_URL = 'product_list'
LOGOUT_REDIRECT_URL = 'product_list'

AUTHENTICATION_BACKENDS = [
    'accounts.backends.EmailOrUsernameModelBackend',
    'django.contrib.auth.backends.ModelBackend',
]

# Email Backend (Console for testing password reset and alerts)
EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Import ML Detection Configuration
try:
    from ML_Model import config as ml_config
    WEIGHT_SVM = ml_config.WEIGHT_SVM
    WEIGHT_J48 = ml_config.WEIGHT_J48
    WEIGHT_RISK = ml_config.WEIGHT_RISK
    AUTHENTICITY_GENUINE_THRESHOLD = ml_config.AUTHENTICITY_GENUINE_THRESHOLD
    WEIGHT_SENTIMENT = getattr(ml_config, 'WEIGHT_SENTIMENT', 0.20)
    IP_MAX_REVIEWS_PER_WINDOW = getattr(ml_config, 'IP_MAX_REVIEWS_PER_WINDOW', 3)
    IP_TIME_WINDOW_MINUTES = getattr(ml_config, 'IP_TIME_WINDOW_MINUTES', 1)
    ALERT_FAKE_RATIO_THRESHOLD = ml_config.ALERT_FAKE_RATIO_THRESHOLD
    ALERT_FAKE_SPIKE_THRESHOLD = ml_config.ALERT_FAKE_SPIKE_THRESHOLD
    ALERT_WINDOW_DAYS = ml_config.ALERT_WINDOW_DAYS
    ALERT_WINDOW_COUNT = ml_config.ALERT_WINDOW_COUNT
    ALERT_EMAIL_NOTIFICATIONS = ml_config.ALERT_EMAIL_NOTIFICATIONS
except ImportError:
    WEIGHT_SVM = 0.50
    WEIGHT_J48 = 0.30
    WEIGHT_RISK = 0.20
    AUTHENTICITY_GENUINE_THRESHOLD = 70.0
    AUTHENTICITY_SUSPICIOUS_THRESHOLD = 40.0
    IP_MAX_REVIEWS_PER_WINDOW = 3
    IP_TIME_WINDOW_MINUTES = 1
    ALERT_FAKE_RATIO_THRESHOLD = 0.50
    ALERT_FAKE_SPIKE_THRESHOLD = 0.25
    ALERT_WINDOW_DAYS = 7
    ALERT_WINDOW_COUNT = 10
    ALERT_EMAIL_NOTIFICATIONS = True
DEFAULT_FROM_EMAIL = 'alerts@grazesmasher.com'
