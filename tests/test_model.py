"""
Unit tests for the DBD prediction model
"""
import unittest
import sys
import os
import torch
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from train_model import DBDModel, train_model


class TestModel(unittest.TestCase):
    """Test model architecture and training"""
    
    def test_model_initialization(self):
        """Test that model can be initialized with correct input size"""
        input_size = 30
        hidden_size = 64
        model = DBDModel(input_size=input_size, hidden_size=hidden_size)
        
        # Test forward pass
        x = torch.randn(1, input_size)
        output = model(x)
        
        self.assertEqual(output.shape, (1, 1))
        self.assertFalse(torch.isnan(output).any())
    
    def test_model_architecture(self):
        """Test model architecture components"""
        model = DBDModel(input_size=30, hidden_size=64)
        
        # Check all layers exist
        self.assertTrue(hasattr(model, 'l1'))
        self.assertTrue(hasattr(model, 'l2'))
        self.assertTrue(hasattr(model, 'out'))
        self.assertTrue(hasattr(model, 'batchnorm1'))
        self.assertTrue(hasattr(model, 'batchnorm2'))
        self.assertTrue(hasattr(model, 'dropout'))


class TestDataProcessing(unittest.TestCase):
    """Test data preprocessing functions"""
    
    def test_csv_loading(self):
        """Test that CSV can be loaded"""
        csv_path = 'DBDData.csv'
        if os.path.exists(csv_path):
            import pandas as pd
            df = pd.read_csv(csv_path, keep_default_na=False, na_values=[])
            self.assertGreater(len(df), 0)
            self.assertIn('Result', df.columns)


if __name__ == '__main__':
    unittest.main()

