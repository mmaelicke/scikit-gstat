"""
Interface to GStatSim.

You need to install GStatSim separately, as it is not a dependency of SciKit-GStat.

```bash
pip install gstatsim
```
"""
from typing import Any, overload, Optional, Union, Tuple, List
from typing_extensions import Literal
import warnings
import inspect
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
from skgstat.Variogram import Variogram
from skgstat.DirectionalVariogram import DirectionalVariogram

try:
    import gstatsim as gss
    GSTATSIM_AVAILABLE = True
    HAS_VERBOSE = 'verbose' in inspect.signature(gss.Interpolation.okrige_sgs).parameters
except ImportError:
    GSTATSIM_AVAILABLE = False
    HAS_VERBOSE = False



# type the bounding box of a 2D grid
BBOX = Union[Tuple[int, int, int, int], Tuple[float, float, float, float]]

class Grid:
    """
    Representation of a regular grid, represented by a :class:`numpy.ndarray`,
    with some additional meta-data about the grid.
    """
    @overload
    def __init__(self, bbox: Variogram, resolution: int) -> None:
        ...
    @overload
    def __init__(self, bbox: Variogram, resolution=..., rows: int=..., cols: int=...) -> None:
        ...
    @overload
    def __init__(self, bbox: BBOX, resolution: int) -> None:
        ...
    @overload
    def __init__(self, bbox: BBOX, resolution=..., rows: int=..., cols: int=...) -> None:
        ...
    def __init__(self, bbox: Union[BBOX, Variogram], resolution: Optional[int] = None, rows: Optional[int] = None, cols: Optional[int] = None) -> None:
        # check if gstatsim is available
        if not self.__check_gstatsim_available():
            raise ImportError('GStatSim is not available. Please install it with `pip install gstatsim`')
        
        # check the resolution and rows/cols:
        if resolution is None and rows is None and cols is None:
            raise AttributeError('Either resolution or rows/cols must be set')

        # get the resolution and rows/cols
        if resolution is not None:
            self._resolution = resolution
            self._rows = None
            self._cols = None

        # finally infer the bounding box from the variogram
        self._infer_bbox(bbox)

        # infer the resolution from the bounding box
        self._infer_resolution()

    def __check_gstatsim_available(self) -> bool:
        """
        Check if GStatSim is available.

        Returns
        -------
        bool
            True if GStatSim is available, False otherwise.

        """
        if GSTATSIM_AVAILABLE:
            return True
        else:
            return False

    def _infer_bbox(self, bbox: Union[BBOX, Variogram]) -> None:
        """
        Infer the bounding box from the variogram.
        """
        # check the type of the bbox
        if isinstance(bbox, Variogram):            
            # get the bounding box
            self._xmax = bbox.coordinates[:, 0].max()
            self._xmin = bbox.coordinates[:, 0].min()
            self._ymax = bbox.coordinates[:, 1].max()
            self._ymin = bbox.coordinates[:, 1].min()
        else:
            self._xmin, self._xmax, self._ymin, self._ymax = bbox
    
    def _infer_resolution(self) -> None:
        """
        Infer the resolution from the bounding box.
        """
        # if resolution is set, infer cols and rows
        if self._resolution is not None:
            self._rows = int(np.rint((self._ymax - self._ymin + self._resolution) / self._resolution))
            self._cols = int(np.rint((self._xmax - self._xmin + self._resolution) / self._resolution))
        
        # if rows and cols are set, infer resolution
        elif self._rows is not None and self._cols is not None:
            xres = (self._xmax - self._xmin) / self._cols
            yres = (self._ymax - self._ymin) / self._rows
            
            # check if the resolution is the same in both directions
            if xres == yres:
                self._resolution = xres
            else:
                warnings.warn('The resolution is not the same in both directions. Adjusting the rows/cols setting')
                self._resolution = min(xres, yres)
                self._rows = None
                self._cols = None
                self._infer_resolution()
            
    @property
    def resolution(self) -> Union[int, float]:
        return self._resolution
    
    @resolution.setter
    def resolution(self, resolution: Union[int, float]) -> None:
        # set resolution
        self._resolution = resolution
        
        # recalculate the rows and cols
        self._rows = None
        self._cols = None
        self._infer_resolution()
    
    @property
    def rows(self) -> int:
        return self._rows
    
    @rows.setter
    def rows(self, rows: int) -> None:
        # set rows
        self._rows = rows
        
        # recalculate the resolution
        self._resolution = None
        self._infer_resolution()

    @property
    def cols(self) -> int:
        return self._cols
    
    @cols.setter
    def cols(self, cols: int) -> None:
        # set cols
        self._cols = cols
        
        # recalculate the resolution
        self._resolution = None
        self._infer_resolution()

    @property
    def prediction_grid(self) -> np.ndarray:
        grid: np.ndarray = gss.Gridding.prediction_grid(self._xmin, self._xmax, self._ymin, self._ymax, self._resolution)
        
        return grid

    @property
    def shape(self) -> Tuple[int, int]:
        if self._rows is None or self._cols is None:
            self._infer_resolution()
        return self._rows, self._cols
        
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.prediction_grid
    
    def __str__(self) -> str:
        return f'<Grid with {self._rows} rows and {self._cols} cols at {self._resolution} resolution>'


@overload
def prediction_grid(bbox: Variogram, resolution: Optional[int], rows: Optional[int], cols: Optional[int], as_numpy: Literal[False] = False) -> Grid:
    ...
@overload
def prediction_grid(bbox: Variogram, resolution: Optional[int], rows: Optional[int], cols: Optional[int], as_numpy: Literal[True]) -> np.ndarray:
    ...
@overload
def prediction_grid(bbox: BBOX, resolution: Optional[int], rows: Optional[int], cols: Optional[int], as_numpy: Literal[False] = False) -> Grid:
    ...
@overload
def prediction_grid(bbox: BBOX, resolution: Optional[int], rows: Optional[int], cols: Optional[int], as_numpy: Literal[True]) -> np.ndarray:
    ...
def prediction_grid(bbox: Union[BBOX, Variogram], resolution: Optional[int] = None, rows: Optional[int] = None, cols: Optional[int] = None, as_numpy: bool = False) -> Union[Grid, np.ndarray]:
    if resolution is not None:
        grid = Grid(bbox, resolution=resolution)
    elif rows is not None and cols is not None:
        grid = Grid(bbox, rows=rows, cols=cols)
    else:
        raise AttributeError('Either resolution or rows/cols must be set')

    if as_numpy:
        return grid.prediction_grid
    else:
        return grid


def simulation_params(
        variogram: Variogram,
        grid: Optional[Union[Grid, np.ndarray, Union[int, float], Tuple[int, int]]] = None,
        minor_range: Optional[Union[int, float]] = None,
) -> Tuple[Union[Grid, np.ndarray], pd.DataFrame, list]:
    # the simulation needs the condition data as pd.DataFrame
    data = np.concatenate((variogram.coordinates, variogram.values.reshape(-1, 1)), axis=1)
    df = pd.DataFrame(data, columns=['x', 'y', 'v'])
    
    # build the grid
    if isinstance(grid, (int, float)):
        # resolution is given
        grid = Grid(variogram, resolution=grid)
    elif isinstance(grid, (tuple, list)):
        # rows / cols are given
        grid = Grid(variogram, rows=grid[0], cols=grid[1])
    elif grid is None:
        # we infer the resolution
        grid = Grid(variogram, resolution=1)
        new_res = min((grid._xmax - grid._xmin) / 100., (grid._ymax - grid._ymin) / 100.)
        grid.resolution = new_res
    
    # now grid should be a Grid object or a numpy.ndarray
    if not isinstance(grid, (Grid, np.ndarray)):
        raise AttributeError('grid must be either a Grid object, a resolution or a tuple/list of rows and cols')

    # get the variogram parameters
    major = variogram.parameters[0]
    nugget = variogram.parameters[-1]
    sill = variogram.parameters[1]
    vtype = variogram.model.__name__

        # extract the azimuth
    if isinstance(variogram, DirectionalVariogram):
        azimuth = variogram.azimuth
        if minor_range is None:
            raise AttributeError('minor_range must be set for directional variograms')
    else:
        azimuth = 0
        minor_range = major

    # build the params
    params = [azimuth, nugget, major, minor_range, sill, vtype]

    return grid, df, params


def run_simulation(
    grid: Union[Grid, np.ndarray], 
    cond_data: pd.DataFrame, 
    vario_params: list, 
    num_points: int = 20, 
    radius: Optional[Union[int, float]] = None,
    method: Union[Literal['simple'], Literal['ordinary']] = 'simple',
    verbose: bool = False
) -> np.ndarray:
    # get the radius
    if radius is None:
        radius = vario_params[2] * 3

    # get the right interpolation method
    if method.lower() == 'simple':
        sim_func = gss.Interpolation.skrige_sgs
    elif method.lower() == 'ordinary':
        sim_func = gss.Interpolation.okrige_sgs
    else:
        raise AttributeError('method must be either "simple" or "ordinary"')

    # get an prediction grid
    if isinstance(grid, Grid):
        pred_grid = grid.prediction_grid
    else:
        pred_grid = grid

    # run the simulation
    if HAS_VERBOSE:
        field: np.ndarray = sim_func(pred_grid, cond_data, 'x', 'y', 'v', num_points, vario_params, radius, verbose)
    else:
        field: np.ndarray = sim_func(pred_grid, cond_data, 'x', 'y', 'v', num_points, vario_params, radius)

    if isinstance(grid, Grid):
        return field.reshape(grid.shape)
    else:
        return field


def simulate(
    variogram: Variogram,
    grid: Optional[Union[Grid, np.ndarray, Union[int, float], Tuple[int, int]]] = None,
    num_points: int = 20, 
    radius: Optional[Union[int, float]] = None,
    method: Union[Literal['simple'], Literal['ordinary']] = 'simple',
    verbose: bool = False,
    n_jobs: int = 1,
    size: int  = 1,
    **kwargs,
) -> List[np.ndarray]:
    # extract minor_range
    minor_range = kwargs.get('minor_range', None)

    # get the simulation parameters
    grid, cond_data, vario_params = simulation_params(variogram, grid, minor_range)

    # multiprocessing?
    if n_jobs > 1 and size > 1:
        # build th pool
        pool = Parallel(n_jobs=n_jobs, verbose=0 if not verbose else 10)
        
        # wrapper
        gen = (delayed(run_simulation)(grid, cond_data, vario_params, num_points, radius, method, verbose) for _ in range(size))

        # run the simulation
        fields = pool(gen)
        return fields
    else:
        field = run_simulation(grid, cond_data, vario_params, num_points, radius, method, verbose)
        return [field]
