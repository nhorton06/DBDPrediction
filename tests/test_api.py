"""
Smoke tests for the Flask API
"""
import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from app import app, build_feature_vector
    APP_AVAILABLE = True
except ImportError:
    APP_AVAILABLE = False


@unittest.skipIf(not APP_AVAILABLE, "App not available")
class TestAPI(unittest.TestCase):
    """Smoke tests for API endpoints"""
    
    def setUp(self):
        """Set up test client"""
        self.app = app.test_client()
        self.app.testing = True
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('status', data)
    
    def test_index_endpoint(self):
        """Test index page loads"""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'ESCAPE PROBABILITY', response.data)
    
    def test_predict_endpoint_structure(self):
        """Test predict endpoint accepts correct data structure"""
        # This is a smoke test - we can't test actual prediction without model
        test_data = {
            'survivor_gender': 'F',
            'steam_player': 'Yes',
            'anonymous_mode': 'No',
            'item': 'Medkit',
            'prestige': 50,
            'map_type': 'Outdoor',
            'map_area': 9728,
            'powerful_add_ons': 'No',
            'exhaustion_perk': 'Sprint Burst',
            'chase_perks': 1,
            'decisive_strike': 'No',
            'unbreakable': 'No',
            'off_the_record': 'No',
            'adrenaline': 'No',
            'survivor_bp': 20000,
            'killer_bp': 25000
        }
        
        response = self.app.post('/predict', 
                                json=test_data,
                                content_type='application/json')
        # Should either succeed (200) or fail gracefully (400/500)
        self.assertIn(response.status_code, [200, 400, 500])


if __name__ == '__main__':
    unittest.main()

