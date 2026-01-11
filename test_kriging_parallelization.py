#!/usr/bin/env python3
"""
Comprehensive test for OrdinaryKriging parallelization functionality.
Tests Python 3.13+ compatibility and result consistency.
"""
import numpy as np
import warnings
from skgstat import Variogram, OrdinaryKriging

def test_serial_vs_parallel_consistency():
    """Test that serial and parallel execution give identical results."""
    print("Testing serial vs parallel consistency...")

    # Create test data
    np.random.seed(42)
    coords = np.random.gamma(10, 4, size=(30, 2))
    values = np.random.normal(10, 2, size=30)
    V = Variogram(coords, values, model='gaussian', normalize=False)

    # Serial execution
    ok_serial = OrdinaryKriging(V, min_points=5, max_points=15, n_jobs=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Ignore kriging warnings for this test
        result_serial = ok_serial.transform(coords[:, 0], coords[:, 1])

    # Parallel execution
    ok_parallel = OrdinaryKriging(V, min_points=5, max_points=15, n_jobs=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Ignore warnings for this test
        result_parallel = ok_parallel.transform(coords[:, 0], coords[:, 1])

    # Compare results
    nan_mask_serial = np.isnan(result_serial)
    nan_mask_parallel = np.isnan(result_parallel)

    # Check that NaN positions match
    if not np.array_equal(nan_mask_serial, nan_mask_parallel):
        print("  FAIL: NaN positions don't match between serial and parallel")
        return False

    # Check that non-NaN values are close
    valid_mask = ~nan_mask_serial
    if not np.allclose(result_serial[valid_mask], result_parallel[valid_mask], rtol=1e-10):
        print("  FAIL: Results differ between serial and parallel execution")
        print(f"    Serial max diff: {np.max(np.abs(result_serial[valid_mask] - result_parallel[valid_mask]))}")
        return False

    print("  PASS: Serial and parallel results are consistent")
    return True

def test_different_n_jobs_values():
    """Test with different n_jobs values."""
    print("Testing different n_jobs values...")

    np.random.seed(42)
    coords = np.random.gamma(10, 4, size=(20, 2))
    values = np.random.normal(10, 2, size=20)
    V = Variogram(coords, values, model='exponential', normalize=False)

    n_jobs_values = [1, 2, 4]
    results = {}

    for n_jobs in n_jobs_values:
        print(f"  Testing n_jobs={n_jobs}...")
        ok = OrdinaryKriging(V, min_points=5, max_points=15, n_jobs=n_jobs)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ok.transform(coords[:, 0], coords[:, 1])

        results[n_jobs] = result

    # Compare all results
    for i in range(len(n_jobs_values)):
        for j in range(i + 1, len(n_jobs_values)):
            n1, n2 = n_jobs_values[i], n_jobs_values[j]
            r1, r2 = results[n1], results[n2]

            # Check NaN consistency
            nan_mask_1 = np.isnan(r1)
            nan_mask_2 = np.isnan(r2)

            if not np.array_equal(nan_mask_1, nan_mask_2):
                print(f"  FAIL: NaN positions differ between n_jobs={n1} and n_jobs={n2}")
                return False

            # Check value consistency
            valid_mask = ~nan_mask_1
            if not np.allclose(r1[valid_mask], r2[valid_mask], rtol=1e-10):
                print(f"  FAIL: Results differ between n_jobs={n1} and n_jobs={n2}")
                return False

    print("  PASS: All n_jobs values give consistent results")
    return True

def test_python_version_compatibility():
    """Test Python version specific behavior."""
    print("Testing Python version compatibility...")

    import sys
    py_version = sys.version_info

    print(f"  Python version: {py_version.major}.{py_version.minor}.{py_version.micro}")

    if py_version >= (3, 13):
        print("  Testing Python 3.13+ warning behavior...")

        np.random.seed(42)
        coords = np.random.gamma(10, 4, size=(15, 2))
        values = np.random.normal(10, 2, size=15)
        V = Variogram(coords, values, model='spherical', normalize=False)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ok = OrdinaryKriging(V, min_points=5, max_points=15, n_jobs=2)
            result = ok.transform(coords[:, 0], coords[:, 1])

            # Check if warning was issued for Python 3.13+
            warning_found = any("multiprocessing" in str(warning.message).lower() for warning in w)

            if warning_found:
                print("  PASS: Appropriate warning issued for Python 3.13+")
            else:
                print("  INFO: No multiprocessing warning (may be fixed)")

    else:
        print("  Testing older Python version multiprocessing...")

        np.random.seed(42)
        coords = np.random.gamma(10, 4, size=(15, 2))
        values = np.random.normal(10, 2, size=15)
        V = Variogram(coords, values, model='cubic', normalize=False)

        try:
            ok = OrdinaryKriging(V, min_points=5, max_points=15, n_jobs=2)
            result = ok.transform(coords[:, 0], coords[:, 1])
            print("  PASS: Multiprocessing works on older Python versions")
        except Exception as e:
            print(f"  FAIL: Multiprocessing failed on Python {py_version.major}.{py_version.minor}: {e}")
            return False

    return True

def test_different_variogram_models():
    """Test with different variogram models."""
    print("Testing different variogram models...")

    models = ['spherical', 'exponential', 'gaussian', 'cubic']

    for model in models:
        print(f"  Testing {model} model...")

        np.random.seed(42)
        coords = np.random.gamma(10, 4, size=(20, 2))
        values = np.random.normal(10, 2, size=20)
        V = Variogram(coords, values, model=model, normalize=False)

        try:
            # Test serial
            ok_serial = OrdinaryKriging(V, min_points=5, max_points=15, n_jobs=1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result_serial = ok_serial.transform(coords[:, 0], coords[:, 1])

            # Test parallel
            ok_parallel = OrdinaryKriging(V, min_points=5, max_points=15, n_jobs=2)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result_parallel = ok_parallel.transform(coords[:, 0], coords[:, 1])

            # Check consistency
            nan_mask = np.isnan(result_serial) | np.isnan(result_parallel)
            valid_mask = ~nan_mask

            if np.allclose(result_serial[valid_mask], result_parallel[valid_mask], rtol=1e-10):
                print(f"    PASS: {model} model gives consistent results")
            else:
                print(f"    FAIL: {model} model gives inconsistent results")
                return False

        except Exception as e:
            print(f"    FAIL: {model} model failed: {e}")
            return False

    return True

def run_all_tests():
    """Run all parallelization tests."""
    print("=" * 60)
    print("COMPREHENSIVE KRIGING PARALLELIZATION TESTS")
    print("=" * 60)

    tests = [
        test_serial_vs_parallel_consistency,
        test_different_n_jobs_values,
        test_python_version_compatibility,
        test_different_variogram_models,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"  ERROR: Test failed with exception: {e}")
            print()

    print("=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("ALL TESTS PASSED! ✅")
        return True
    else:
        print("SOME TESTS FAILED! ❌")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
