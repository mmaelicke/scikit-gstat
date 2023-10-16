import pytest
import numpy as np
import skgstat as skg
import sys

from skgstat.interfaces.gstatsim_mod import Grid, prediction_grid

# Test data generation
coords, vals = skg.data.pancake(N=60).get('sample')
variogram = skg.Variogram(coords, vals, maxlag=0.6, n_lags=12)

# Test cases
@pytest.mark.skipif(sys.version_info < (3, 8), reason="requires Python >= 3.8 or higher")
def test_grid_initialization_resolution():
    # Initialize grid with resolution
    grid = Grid(variogram, resolution=0.1)

    assert grid.resolution == 0.1
    assert grid.rows == 4661
    assert grid.cols == 4731


@pytest.mark.skipif(sys.version_info < (3, 8), reason="requires Python >= 3.8 or higher")
def test_grid_initialization_infer_resolution():
    # Initialize grid without rows/cols
    # Set resolution
    grid = Grid(variogram, rows=74, cols=66)

    assert np.abs(grid.resolution - 6.3) < 0.01


@pytest.mark.skipif(sys.version_info < (3, 8), reason="requires Python >= 3.8 or higher")
def test_grid_initialization_no_resolution_rows_cols():
    # Initialize grid without resolution or rows/cols (expecting an error)
    with pytest.raises(AttributeError):
        grid = Grid(variogram, resolution=None, rows=None, cols=None)


@pytest.mark.skipif(sys.version_info < (3, 8), reason="requires Python >= 3.8 or higher")
def test_grid_resolution_setting():
    # Set resolution
    grid = Grid(variogram, rows=100, cols=100)

    assert np.abs(grid.resolution - 4.66) < 0.01

    # Set resolution
    grid.resolution = 5
    assert grid.rows == 94
    assert grid.cols == 96


@pytest.mark.skipif(sys.version_info < (3, 8), reason="requires Python >= 3.8 or higher")
def test_grid_resolution_from_bbox():
    # get the bbox from the variogram
    bbox = [variogram.coordinates[:, 0].min(), variogram.coordinates[:, 0].max(), variogram.coordinates[:, 1].min(), variogram.coordinates[:, 1].max()]

    grid = Grid(bbox=bbox, resolution = 5)

    assert grid.rows == 94
    assert grid.cols == 96


@pytest.mark.skipif(sys.version_info < (3, 8), reason="requires Python >= 3.8 or higher")
def test_grid_init_symetric_bbox():
    bbox = [0, 100, 0, 100]
    grid = Grid(bbox=bbox, resolution=5)

    assert grid.rows == 21
    assert grid.cols == 21


@pytest.mark.skipif(sys.version_info < (3, 8), reason="requires Python >= 3.8 or higher")
def test_grid_init_bbox_row_and_resolution():
    bbox = [0, 100, 0, 100]
    grid = Grid(bbox=bbox, rows=21, resolution=5)

    assert grid.shape == (21, 21)


@pytest.mark.skipif(sys.version_info < (3, 8), reason="requires Python >= 3.8 or higher")
def test_grid_init_bbox_col_and_resolution():
    bbox = [0, 100, 0, 100]
    grid = Grid(bbox=bbox, cols=21, resolution=5)

    assert grid.shape == (21, 21)


@pytest.mark.skipif(sys.version_info < (3, 8), reason="requires Python >= 3.8 or higher")
def test_grid_rows_setting():
    grid = Grid(variogram, resolution=5)

    # Set rows
    grid.rows = 50

    assert grid.cols == 52
    assert grid.rows == 50
    assert np.abs(grid.resolution - 9.32) < 0.01


@pytest.mark.skipif(sys.version_info < (3, 8), reason="requires Python >= 3.8 or higher")
def test_grid_cols_setting():
    grid = Grid(variogram, resolution=5)

    # Set cols
    grid.cols = 100

    assert grid.cols == 100
    assert grid.rows == 100
    assert np.abs(grid.resolution - 4.73) < 0.01


@pytest.mark.skipif(sys.version_info < (3, 8), reason="requires Python >= 3.8 or higher")
def test_grid_prediction_grid():
    # Test prediction grid generation
    grid = Grid(variogram, resolution=0.1)
    prediction_grid = grid.prediction_grid

    assert isinstance(prediction_grid, np.ndarray)
    assert prediction_grid.shape == (grid.shape[0] * grid.shape[1], 2)


@pytest.mark.skipif(sys.version_info < (3, 8), reason="requires Python >= 3.8 or higher")
def test_grid_call_operator():
    # Test calling the Grid instance
    grid = Grid(variogram, resolution=0.1)
    prediction_grid = grid()

    assert isinstance(prediction_grid, np.ndarray)
    assert prediction_grid.shape == (grid.shape[0] * grid.shape[1], 2)


@pytest.mark.skipif(sys.version_info < (3, 8), reason="requires Python >= 3.8 or higher")
def test_grid_str_representation():
    # Test string representation
    grid = Grid(variogram, resolution=0.1)
    assert str(grid) == f'<Grid with {grid.rows} rows and {grid.cols} cols at {grid.resolution} resolution>'


@pytest.mark.skipif(sys.version_info < (3, 8), reason="requires Python >= 3.8 or higher")
def test_prediction_grid_resolution():
    grid = prediction_grid(variogram, resolution=1, as_numpy=False)

    assert isinstance(grid, Grid)
    assert grid.rows == 467
    assert grid.cols == 474


@pytest.mark.skipif(sys.version_info < (3, 8), reason="requires Python >= 3.8 or higher")
def test_prediction_grid_cols_rows():
    grid = prediction_grid(variogram, cols=50, rows=50, as_numpy=True)

    assert isinstance(grid, np.ndarray)
    assert grid.shape == (2652 , 2)


@pytest.mark.skipif(sys.version_info < (3, 8), reason="requires Python >= 3.8 or higher")
def test_prediction_grid_interface():
    coords, vals = skg.data.pancake(N=60, seed=42).get('sample')
    vario = skg.Variogram(coords, vals, maxlag=0.6, n_lags=12)

    # create the grid
    grid = vario.gstatsim_prediction_grid(resolution=5)

    assert isinstance(grid, Grid)
    assert grid.rows == 94
    assert grid.cols == 96


@pytest.mark.skipif(sys.version_info < (3, 8), reason="requires Python >= 3.8 or higher")
def test_prediction_grid_interface_as_numpy():
    coords, vals = skg.data.pancake(N=60).get('sample')
    vario = skg.Variogram(coords, vals, maxlag=0.6, n_lags=12)

    # create the grid
    grid = vario.gstatsim_prediction_grid(resolution=5, as_numpy=True)

    assert isinstance(grid, np.ndarray)
    assert grid.shape == (96 * 95, 2)


# Run the tests
if __name__ == '__main__':
    pytest.main()
