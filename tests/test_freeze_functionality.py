#!/usr/bin/env python3
"""
Unit tests for freezing functionality in GA, RHC, and SA neural networks.
"""

import unittest
import torch
import numpy as np

from pyperch.neural.ga_nn import GAModule
from pyperch.neural.rhc_nn import RHCModule
from pyperch.neural.sa_nn import SAModule


class TestFreezeFunctionality(unittest.TestCase):
    """Test cases for neural network freezing functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.layer_sizes = [10, 20, 15, 5]  # 4 layers: input, hidden1, hidden2, output
        self.trainable_layers = 2  # Keep last 2 layers trainable - hidden1,2
        self.freeze_seed = 42
        self.random_seed = 123
        
    def test_ga_module_freezing(self):
        """Test GA module freezing functionality."""
        ga_module = GAModule(
            layer_sizes=self.layer_sizes,
            trainable_layers=self.trainable_layers,
            freeze_seed=self.freeze_seed,
            random_seed=self.random_seed
        )
        
        # assert that first layer is frozen
        self.assertFalse(ga_module.layers[0].weight.requires_grad)
        self.assertFalse(ga_module.layers[0].bias.requires_grad)
        
        # assert that last 2 layers are trainable
        self.assertTrue(ga_module.layers[1].weight.requires_grad)
        self.assertTrue(ga_module.layers[1].bias.requires_grad)
        self.assertTrue(ga_module.layers[2].weight.requires_grad)
        self.assertTrue(ga_module.layers[2].bias.requires_grad)
        
    def test_rhc_module_freezing(self):
        """Test RHC module freezing functionality."""
        rhc_module = RHCModule(
            layer_sizes=self.layer_sizes,
            trainable_layers=self.trainable_layers,
            freeze_seed=self.freeze_seed,
            random_seed=self.random_seed
        )
        
        # Check that first layer is frozen
        self.assertFalse(rhc_module.layers[0].weight.requires_grad)
        self.assertFalse(rhc_module.layers[0].bias.requires_grad)
        
        # Check that last 2 layers are trainable
        self.assertTrue(rhc_module.layers[1].weight.requires_grad)
        self.assertTrue(rhc_module.layers[1].bias.requires_grad)
        self.assertTrue(rhc_module.layers[2].weight.requires_grad)
        self.assertTrue(rhc_module.layers[2].bias.requires_grad)
        
    def test_sa_module_freezing(self):
        """Test SA module freezing functionality."""
        sa_module = SAModule(
            layer_sizes=self.layer_sizes,
            trainable_layers=self.trainable_layers,
            freeze_seed=self.freeze_seed,
            random_seed=self.random_seed
        )
        
        # Check that first layer is frozen
        self.assertFalse(sa_module.layers[0].weight.requires_grad)
        self.assertFalse(sa_module.layers[0].bias.requires_grad)
        
        # Check that last 2 layers are trainable
        self.assertTrue(sa_module.layers[1].weight.requires_grad)
        self.assertTrue(sa_module.layers[1].bias.requires_grad)
        self.assertTrue(sa_module.layers[2].weight.requires_grad)
        self.assertTrue(sa_module.layers[2].bias.requires_grad)
        
    def test_no_freezing_all_trainable(self):
        """Test that no freezing keeps all layers trainable."""
        ga_module = GAModule(
            layer_sizes=self.layer_sizes,
            trainable_layers=None,  # No freezing
            random_seed=self.random_seed
        )
        
        # All layers should be trainable
        for i, layer in enumerate(ga_module.layers):
            self.assertTrue(layer.weight.requires_grad, f"Layer {i} weight should be trainable")
            self.assertTrue(layer.bias.requires_grad, f"Layer {i} bias should be trainable")
            
    def test_freeze_all_layers(self):
        """Test edge case: trainable_layers=0 raises ValueError."""
        with self.assertRaises(ValueError) as context:
            GAModule(
                layer_sizes=self.layer_sizes,
                trainable_layers=0,  # Should raise ValueError
                random_seed=self.random_seed
            )
        
        self.assertIn("trainable_layers must be > 0", str(context.exception))
        self.assertIn("Use trainable_layers=None to train all layers", str(context.exception))
        
    def test_freeze_negative_layers(self):
        """Test edge case: trainable_layers < 0 raises ValueError."""
        with self.assertRaises(ValueError) as context:
            GAModule(
                layer_sizes=self.layer_sizes,
                trainable_layers=-1,  # Should raise ValueError
                random_seed=self.random_seed
            )
        
        self.assertIn("trainable_layers must be > 0", str(context.exception))
        
    def test_trainable_layers_error_all_modules(self):
        """Test that all three modules raise ValueError for trainable_layers <= 0."""
        modules = [GAModule, RHCModule, SAModule]
        
        for module_class in modules:
            with self.subTest(module=module_class.__name__):
                with self.assertRaises(ValueError) as context:
                    module_class(
                        layer_sizes=self.layer_sizes,
                        trainable_layers=0,
                        random_seed=self.random_seed
                    )
                
                self.assertIn("trainable_layers must be > 0", str(context.exception))
            
    def test_trainable_layers_greater_than_total(self):
        """Test edge case: trainable_layers >= total layers (all trainable)."""
        ga_module = GAModule(
            layer_sizes=self.layer_sizes,
            trainable_layers=len(self.layer_sizes),  # More than total layers
            random_seed=self.random_seed
        )
        
        # All layers should be trainable
        for i, layer in enumerate(ga_module.layers):
            self.assertTrue(layer.weight.requires_grad, f"Layer {i} weight should be trainable")
            self.assertTrue(layer.bias.requires_grad, f"Layer {i} bias should be trainable")
            
    def test_parameter_count_reduction(self):
        """Test that freezing reduces the number of trainable parameters."""
        # Large network to test parameter reduction
        large_layer_sizes = [100, 200, 150, 50]
        trainable_layers = 2
        
        # Create modules with and without freezing
        ga_no_freeze = GAModule(layer_sizes=large_layer_sizes, trainable_layers=None)
        ga_with_freeze = GAModule(layer_sizes=large_layer_sizes, trainable_layers=trainable_layers)
        
        # Count trainable parameters
        def count_trainable_params(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        trainable_no_freeze = count_trainable_params(ga_no_freeze)
        trainable_with_freeze = count_trainable_params(ga_with_freeze)
        
        # With freezing should have fewer trainable parameters
        self.assertLess(trainable_with_freeze, trainable_no_freeze)
        
        # Should be under 50k trainable parameters as required
        self.assertLessEqual(trainable_with_freeze, 50000)
        
    def test_freeze_seed_reproducibility(self):
        """Test that freeze_seed produces reproducible results."""
        # Create two modules with same freeze_seed
        ga1 = GAModule(
            layer_sizes=self.layer_sizes,
            trainable_layers=self.trainable_layers,
            freeze_seed=self.freeze_seed,
            random_seed=self.random_seed
        )
        
        ga2 = GAModule(
            layer_sizes=self.layer_sizes,
            trainable_layers=self.trainable_layers,
            freeze_seed=self.freeze_seed,
            random_seed=self.random_seed
        )
        
        # Both should have identical freezing pattern
        for i, (layer1, layer2) in enumerate(zip(ga1.layers, ga2.layers)):
            self.assertEqual(layer1.weight.requires_grad, layer2.weight.requires_grad,
                           f"Layer {i} weight requires_grad should be identical")
            self.assertEqual(layer1.bias.requires_grad, layer2.bias.requires_grad,
                           f"Layer {i} bias requires_grad should be identical")
            
    def test_different_trainable_layers_values(self):
        """Test different trainable_layers values."""
        test_cases = [
            (1, 1),  # Keep 1 layer trainable
            (2, 2),  # Keep 2 layers trainable  
            (3, 3),  # Keep 3 layers trainable
        ]
        
        for trainable_layers, expected_trainable in test_cases:
            with self.subTest(trainable_layers=trainable_layers):
                ga_module = GAModule(
                    layer_sizes=self.layer_sizes,
                    trainable_layers=trainable_layers,
                    random_seed=self.random_seed
                )
                
                # Count trainable layers
                trainable_count = sum(1 for layer in ga_module.layers 
                                    if any(param.requires_grad for param in layer.parameters()))
                
                self.assertEqual(trainable_count, expected_trainable,
                               f"Expected {expected_trainable} trainable layers, got {trainable_count}")
                
    def test_layer_requires_grad_property(self):
        """Test that layer parameters require_grad works correctly."""
        ga_module = GAModule(
            layer_sizes=self.layer_sizes,
            trainable_layers=self.trainable_layers,
            random_seed=self.random_seed
        )
        
        # First layer should not require grad
        self.assertFalse(any(param.requires_grad for param in ga_module.layers[0].parameters()))
        
        # Last two layers should require grad
        self.assertTrue(any(param.requires_grad for param in ga_module.layers[1].parameters()))
        self.assertTrue(any(param.requires_grad for param in ga_module.layers[2].parameters()))
        
    def test_all_modules_consistency(self):
        """Test that all three module types behave consistently."""
        modules = [
            GAModule(layer_sizes=self.layer_sizes, trainable_layers=self.trainable_layers, random_seed=self.random_seed),
            RHCModule(layer_sizes=self.layer_sizes, trainable_layers=self.trainable_layers, random_seed=self.random_seed),
            SAModule(layer_sizes=self.layer_sizes, trainable_layers=self.trainable_layers, random_seed=self.random_seed)
        ]
        
        # All modules should have identical freezing pattern
        for i in range(len(self.layer_sizes) - 1):
            layer_requires_grad = [any(param.requires_grad for param in module.layers[i].parameters()) for module in modules]
            
            # All modules should have same requires_grad for this layer
            self.assertTrue(all(x == layer_requires_grad[0] for x in layer_requires_grad),
                           f"Layer {i} requires_grad should be consistent across all modules")


class TestFreezeFunctionalityIntegration(unittest.TestCase):
    """Integration tests for freezing functionality."""
    
    def test_forward_pass_with_frozen_layers(self):
        """Test that forward pass works correctly with frozen layers."""
        layer_sizes = [5, 10, 3]
        trainable_layers = 1
        
        # Test all three module types
        modules = [
            GAModule(layer_sizes=layer_sizes, trainable_layers=trainable_layers),
            RHCModule(layer_sizes=layer_sizes, trainable_layers=trainable_layers),
            SAModule(layer_sizes=layer_sizes, trainable_layers=trainable_layers)
        ]
        
        # Create test input
        test_input = torch.randn(2, layer_sizes[0])
        
        for module in modules:
            with self.subTest(module_type=type(module).__name__):
                # Forward pass should work without errors
                output = module(test_input)
                
                # Output should have correct shape
                self.assertEqual(output.shape, (2, layer_sizes[-1]))
                
                # Output should be valid (no NaN or Inf)
                self.assertTrue(torch.isfinite(output).all())
                
    def test_parameter_modification_with_frozen_layers(self):
        """Test that frozen parameters are properly marked as non-trainable."""
        layer_sizes = [5, 10, 3]
        trainable_layers = 1
        
        ga_module = GAModule(layer_sizes=layer_sizes, trainable_layers=trainable_layers)
        
        # Check that frozen layer parameters are marked as non-trainable
        self.assertFalse(ga_module.layers[0].weight.requires_grad)
        self.assertFalse(ga_module.layers[0].bias.requires_grad)
        
        # Check that trainable layer parameters are marked as trainable
        self.assertTrue(ga_module.layers[1].weight.requires_grad)
        self.assertTrue(ga_module.layers[1].bias.requires_grad)
        
        # Test that we can still modify frozen parameters (they're just not trainable)
        original_weight = ga_module.layers[0].weight.clone()
        with torch.no_grad():
            ga_module.layers[0].weight += 1.0
        # Weight can be modified, but it won't participate in gradient computation
        self.assertFalse(torch.equal(ga_module.layers[0].weight, original_weight))


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
