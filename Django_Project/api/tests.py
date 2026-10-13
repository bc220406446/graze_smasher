"""
Unit Tests for REST API Endpoints
CS619 Final-Year Project: Graze Smasher Product Hunting Tool
"""

from rest_framework.test import APITestCase
from products.models import Product
from reviews.models import Review


class RestApiEndpointsTestCase(APITestCase):
    """Verifies all external integration REST API endpoints and data schemas."""

    def setUp(self):
        self.product = Product.objects.create(
            name="Wireless Mouse M310",
            category="Electronics",
            platform="Amazon",
            source_url="https://amazon.com/dp/B003NR57BY"
        )
        self.review = Review.objects.create(
            product=self.product,
            text="Smooth scrolling and very durable battery life.",
            rating=5,
            svm_score=0.1,
            j48_score=0.2,
            final_score=85.0,
            label=Review.LABEL_GENUINE
        )

    def test_get_products_list(self):
        """GET /api/products/ returns a paginated list of catalog products."""
        res = self.client.get('/api/products/')
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn('results', data)
        self.assertGreaterEqual(len(data['results']), 1)

    def test_get_product_detail(self):
        """GET /api/products/<id>/ returns comprehensive authenticity statistics."""
        res = self.client.get(f'/api/products/{self.product.id}/')
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data['id'], self.product.id)
        self.assertEqual(data['name'], "Wireless Mouse M310")
        self.assertIn('overall_authenticity_score', data)
        self.assertIn('community_trust_rating', data)
        self.assertEqual(data['genuine_reviews_count'], 1)

    def test_get_product_not_found(self):
        """GET /api/products/999999/ returns 404 Not Found."""
        res = self.client.get('/api/products/999999/')
        self.assertEqual(res.status_code, 404)

    def test_get_product_reviews(self):
        """GET /api/products/<id>/reviews/ returns paginated reviews with ML scores."""
        res = self.client.get(f'/api/products/{self.product.id}/reviews/')
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn('results', data)
        self.assertEqual(len(data['results']), 1)
        self.assertEqual(data['results'][0]['label'], 'Genuine')

    def test_post_external_product_with_reviews_batch(self):
        """POST /api/products/ ingests third-party product and evaluates review batch."""
        payload = {
            "platform": "eBay",
            "name": "Mechanical Keyboard RGB",
            "category": "Electronics",
            "url": "https://ebay.com/itm/123456",
            "description": "Tactile blue switches with customizable backlight.",
            "reviews": [
                {
                    "text": "Sturdy aluminum top plate and clicky switches feel awesome.",
                    "rating": 5,
                    "ip": "198.51.100.20"
                },
                {
                    "text": "Keys stopped working within two hours of unboxing, terrible.",
                    "rating": 1,
                    "ip": "198.51.100.21"
                }
            ]
        }
        res = self.client.post('/api/products/', payload, format='json')
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertIn('product', data)
        self.assertEqual(data['product']['platform'], "eBay")
        self.assertEqual(data['product']['review_count'], 2)
        self.assertEqual(len(data['reviews']), 2)

        # Confirm persisted in DB
        created_prod = Product.objects.get(id=data['product']['id'])
        self.assertTrue(created_prod.is_external)
        self.assertEqual(created_prod.reviews.count(), 2)

    def test_post_external_product_validation_failure(self):
        """POST /api/products/ returns 400 Bad Request when required fields are missing."""
        payload = {
            "platform": "",  # Empty platform
            "name": ""       # Empty name
        }
        res = self.client.post('/api/products/', payload, format='json')
        self.assertEqual(res.status_code, 400)

    def test_post_single_review(self):
        """POST /api/reviews/ submits a single review to an existing product with ML scoring."""
        payload = {
            "product_id": self.product.id,
            "text": "I love using this mouse for my daily software development tasks.",
            "rating": 5,
            "ip": "203.0.113.88"
        }
        res = self.client.post('/api/reviews/', payload, format='json')
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertIn('review', data)
        self.assertIn('product_authenticity_score', data)
        self.assertIn('product_authenticity_label', data)
        self.assertIn(data['review']['label'], ['Genuine', 'Suspicious', 'Fake'])

        # Verify product review count incremented
        self.assertEqual(self.product.reviews.count(), 2)

    def test_post_single_review_invalid_product(self):
        """POST /api/reviews/ returns 404 if product does not exist."""
        payload = {
            "product_id": 999999,
            "text": "Valid text but invalid product target.",
            "rating": 5
        }
        res = self.client.post('/api/reviews/', payload, format='json')
        self.assertEqual(res.status_code, 404)
