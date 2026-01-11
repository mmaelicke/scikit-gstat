#!/usr/bin/env python3
"""
Debug script to understand the pickling error.
"""
import numpy as np
import pickle
from skgstat import Variogram, OrdinaryKriging

def test_pickling():
    """Test what can and cannot be pickled."""
    print("Testing pickling of various components...")

    # Create test data
    np.random.seed(42)
    coords = np.random.gamma(10, 4, size=(10, 2))
    values = np.random.normal(10, 2, size=10)
    V = Variogram(coords, values, model='gaussian', normalize=False)
    ok = OrdinaryKriging(V, min_points=5, max_points=20, n_jobs=2)

    print("1. Testing OrdinaryKriging instance pickling...")
    try:
        pickle.dumps(ok)
        print("   SUCCESS: OrdinaryKriging instance can be pickled")
    except Exception as e:
        print(f"   FAILED: {e}")

    print("2. Testing _estimator method pickling...")
    try:
        pickle.dumps(ok._estimator)
        print("   SUCCESS: _estimator method can be pickled")
    except Exception as e:
        print(f"   FAILED: {e}")

    print("3. Testing _kriging_estimator_wrapper function pickling...")
    try:
        from skgstat.Kriging import _kriging_estimator_wrapper
        pickle.dumps(_kriging_estimator_wrapper)
        print("   SUCCESS: _kriging_estimator_wrapper can be pickled")
    except Exception as e:
        print(f"   FAILED: {e}")

    print("4. Testing bound method pickling...")
    try:
        bound_method = _kriging_estimator_wrapper.__get__(ok, type(ok))
        pickle.dumps(bound_method)
        print("   SUCCESS: bound method can be pickled")
    except Exception as e:
        print(f"   FAILED: {e}")

if __name__ == "__main__":
    test_pickling()
