#!/usr/bin/env python3
"""
Test script to reproduce the Python 3.13 multiprocessing error in OrdinaryKriging.
"""
import numpy as np
from skgstat import Variogram, OrdinaryKriging

def test_reproduce_error():
    """Reproduce the multiprocessing error from issue #201."""
    print("Testing OrdinaryKriging with n_jobs > 1 in Python 3.13...")
    print(f"Python version: {__import__('sys').version}")
    
    # Create test data
    np.random.seed(42)
    coords = np.random.gamma(10, 4, size=(50, 2))
    values = np.random.normal(10, 2, size=50)
    
    print(f"Created {len(coords)} test points")
    
    # Create variogram
    V = Variogram(coords, values, model='gaussian', normalize=False)
    print(f"Created variogram: {V}")
    
    try:
        # Test with n_jobs > 1 (should fail on Python 3.13)
        print("\n--- Testing with n_jobs=2 (parallel) ---")
        ok = OrdinaryKriging(V, min_points=5, max_points=20, n_jobs=2)
        print("OrdinaryKriging instance created successfully")
        
        # This should trigger the error
        result = ok.transform(coords[:, 0], coords[:, 1])
        print(f"SUCCESS: Parallel kriging completed. Result shape: {result.shape}")
        return True
        
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        return False

def test_serial_baseline():
    """Test serial execution as baseline."""
    print("\n--- Testing with n_jobs=1 (serial baseline) ---")
    
    # Create test data
    np.random.seed(42)
    coords = np.random.gamma(10, 4, size=(50, 2))
    values = np.random.normal(10, 2, size=50)
    
    # Create variogram
    V = Variogram(coords, values, model='gaussian', normalize=False)
    
    try:
        ok = OrdinaryKriging(V, min_points=5, max_points=20, n_jobs=1)
        result = ok.transform(coords[:, 0], coords[:, 1])
        print(f"SUCCESS: Serial kriging completed. Result shape: {result.shape}")
        return result
    except Exception as e:
        print(f"ERROR in serial execution: {type(e).__name__}: {e}")
        return None

if __name__ == "__main__":
    print("=" * 60)
    print("Reproducing Python 3.13 multiprocessing issue #201")
    print("=" * 60)
    
    # Test serial baseline
    serial_result = test_serial_baseline()
    
    # Test parallel (should fail)
    parallel_success = test_reproduce_error()
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Serial execution: {'SUCCESS' if serial_result is not None else 'FAILED'}")
    print(f"Parallel execution: {'SUCCESS' if parallel_success else 'FAILED'}")
    
    if not parallel_success:
        print("\nThis confirms the Python 3.13 multiprocessing issue.")
    else:
        print("\nIssue appears to be resolved or not reproducible.")