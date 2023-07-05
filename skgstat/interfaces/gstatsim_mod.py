"""
Interface to GStatSim.

You need to install GStatSim separately, as it is not a dependency of SciKit-GStat.

```bash
pip install gstatsim
```
"""
from typing import Any, overload, Optional, Union, Tuple, List, TYPE_CHECKING
from typing_extensions import Literal
import warnings
import inspect
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
if TYPE_CHECKING:
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
    def __init__(self, bbox: 'Variogram', resolution: int) -> None:
        ...
    @overload
    def __init__(self, bbox: 'Variogram', resolution=..., rows: int=..., cols: int=...) -> None:
        ...
    @overload
    def __init__(self, bbox: BBOX, resolution: int) -> None:
        ...
    @overload
    def __init__(self, bbox: BBOX, resolution=..., rows: int=..., cols: int=...) -> None:
        ...
    def __init__(self, bbox: Union[BBOX, 'Variogram'], resolution: Optional[int] = None, rows: Optional[int] = None, cols: Optional[int] = None) -> None:
        """
        Initialize a new Grid instance.

        Parameters
        ----------
        bbox : Union[BBOX, Variogram]
            The bounding box or variogram to use for the grid.
        resolution : Optional[int], optional
            The resolution of the grid, by default None.
        rows : Optional[int], optional
            The number of rows in the grid, by default None.
        cols : Optional[int], optional
            The number of columns in the grid, by default None.

        Raises
        ------
        ImportError
            If the `gstatsim` package is not available.
        AttributeError
            If neither `resolution` nor `rows`/`cols` are set.

        Notes
        -----
        If `resolution` is set, it will be used as cell size and `rows` and `cols` are ignored.
        If `rows` and `cols` are set, the grid will have `rows` rows and `cols` columns and
        `resolution` has to be set to `None`.

        """
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

    def __check_gstatsim_available(self) -> bool: # pragma: no cover
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

    def _infer_bbox(self, bbox: Union[BBOX, 'Variogram']) -> None:
        """
        Infer the bounding box from the variogram or bounding box.

        Parameters
        ----------
        bbox : Union[BBOX, Variogram]
            The bounding box or variogram to infer the bounding box from.

        Raises
        ------
        TypeError
            If `bbox` is not a `BBOX` or `Variogram` instance.

        Notes
        -----
        If `bbox` is a `Variogram` instance, the bounding box is inferred from the coordinates of the variogram.
        If `bbox` is a `BBOX` instance, the bounding box is set to the values of the instance.
        """
        # import the Variogram class only here to avoid circular imports
        from skgstat import Variogram
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
        If `resolution` is set, the number of rows and columns are inferred from the bounding box.
        If `rows` and `cols` are set, the resolution is inferred from the number of rows and columns.
        If neither `resolution` nor `rows`/`cols` are set, a warning is issued and the 
        resolution is set to the minimum of the x and y resolutions.

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
def prediction_grid(bbox: 'Variogram', resolution: Optional[int], rows: Optional[int], cols: Optional[int], as_numpy = False) -> Grid:
    ...
@overload
def prediction_grid(bbox: 'Variogram', resolution: Optional[int], rows: Optional[int], cols: Optional[int], as_numpy = True) -> np.ndarray:
    ...
@overload
def prediction_grid(bbox: BBOX, resolution: Optional[int], rows: Optional[int], cols: Optional[int], as_numpy = False) -> Grid:
    ...
@overload
def prediction_grid(bbox: BBOX, resolution: Optional[int], rows: Optional[int], cols: Optional[int], as_numpy = True) -> np.ndarray:
    ...
def prediction_grid(bbox: Union[BBOX, 'Variogram'], resolution: Optional[int] = None, rows: Optional[int] = None, cols: Optional[int] = None, as_numpy: bool = False) -> Union[Grid, np.ndarray]:
    """
    Generate a prediction grid as used by `gstatsim.Interpolation` methods.

    Parameters
    ----------
    bbox : Union[BBOX, Variogram]
        The bounding box defining the spatial extent of the prediction grid. It can be either a BBOX object or a Variogram object.
    resolution : Optional[int], optional
        The resolution of the prediction grid. The number of cells along each axis.
        Either `resolution` or `rows` and `cols` should be set. Default is None.
    rows : Optional[int], optional
        The number of rows in the prediction grid. Required only when `resolution` is not provided. Default is None.
    cols : Optional[int], optional
        The number of columns in the prediction grid. Required only when `resolution` is not provided. Default is None.
    as_numpy : bool, optional
        If True, return the prediction grid as a numpy array. If False, return as a Grid object. Default is False.

    Returns
    -------
    Union[Grid, np.ndarray]
        The prediction grid either as a Grid object or a numpy array, based on the value of `as_numpy`.

    Raises
    ------
    AttributeError
        If neither `resolution` nor `rows` and `cols` are set.

    """
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
    variogram: 'Variogram',
    grid: Optional[Union[Grid, np.ndarray, Union[int, float], Tuple[int, int]]] = None, 
    minor_range: Optional[Union[int, float]] = None
) -> Tuple[Union[Grid, np.ndarray], pd.DataFrame, list]:
    """
    Generate simulation parameters for the `Interpolation.skrige_sgs` and
    `Interpolation.okrige_sgs`methods of GStatSim.

    Parameters
    ----------
    variogram : 'Variogram'
        The variogram object used for simulation.
    grid : Optional[Union[Grid, np.ndarray, Union[int, float], Tuple[int, int]]], optional
        The grid object or array defining the simulation grid. 
        It can be either a :class:`Grid <skgstat.interface.gstatsim_mod.Grid>` object, 
        a :class:`numpy array <numpy.ndarray>`, a resolution value (int or float), 
        or a tuple/list of rows and columns. If None, the resolution is inferred from the variogram.
    minor_range : Optional[Union[int, float]], optional
        The minor range for directional variograms. Required only for directional variograms. 
        Default is None.

    Returns
    -------
    Tuple[Union[Grid, np.ndarray], pd.DataFrame, list]
        A tuple containing the simulation grid, the condition data as a pandas DataFrame, and a list of simulation parameters.

    Raises
    ------
    AttributeError
        If grid is not a Grid object, a numpy array, a resolution value, or a tuple/list of rows and columns.
    AttributeError
        If minor_range is not set for directional variograms.

    """
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

    # due to circular imports we need to import it here
    from skgstat import DirectionalVariogram
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
    """
    Run a sequential gaussian simulation using GStatSim.

    Parameters
    ----------
    grid : Union[Grid, np.ndarray]
        The grid object or array representing the simulation grid.
    cond_data : pd.DataFrame
        The condition data as a pandas DataFrame containing the coordinates and values.
    vario_params : list
        A list of variogram parameters used for simulation.
    num_points : int, optional
        The number of neighboring points used for interpolation. Default is 20.
    radius : Optional[Union[int, float]], optional
        The search radius for neighboring points. If not provided, it is calculated as 
        3 times the major range from the variogram parameters.
    method : Union[Literal['simple'], Literal['ordinary']], optional
        The interpolation method to use. Either 'simple' for simple kriging 
        or 'ordinary' for ordinary kriging. Default is 'simple'.
    verbose : bool, optional
        If True, enable verbose output during the simulation. Default is False.

    Returns
    -------
    np.ndarray
        The simulated field as a numpy array.

    Raises
    ------
    AttributeError
        If the provided method is neither 'simple' nor 'ordinary'.

    """
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
    variogram: 'Variogram',
    grid: Optional[Union[Grid, np.ndarray, Union[int, float], Tuple[int, int]]] = None,
    num_points: int = 20, 
    radius: Optional[Union[int, float]] = None,
    method: Union[Literal['simple'], Literal['ordinary']] = 'simple',
    verbose: bool = False,
    n_jobs: int = 1,
    size: int  = 1,
    **kwargs,
) -> List[np.ndarray]:
    """
    Perform spatial simulation using GStatSim. The GStatSim simulation is
    can be run in parallel using joblib. Note that this will enable the
    parallel execution of **multiple** simulations, it does not parallelize
    the simulation itself.

    Parameters
    ----------
    variogram : 'Variogram'
        The variogram object used for simulation.
    grid : Optional[Union[Grid, np.ndarray, Union[int, float], Tuple[int, int]]], optional
        The grid object or array representing the simulation grid. 
        It can be either a Grid object, a numpy array, a resolution value (int or float),
        or a tuple/list of rows and columns. If None, the resolution is inferred from the variogram.
    num_points : int, optional
        The number of neighboring points used for interpolation. Default is 20.
    radius : Optional[Union[int, float]], optional
        The search radius for neighboring points. If not provided, it is calculated based on the major 
        range from the variogram parameters.
    method : Union[Literal['simple'], Literal['ordinary']], optional
        The interpolation method to use. Either 'simple' for simple kriging or 'ordinary' for ordinary kriging. Default is 'simple'.
    verbose : bool, optional
        If True, enable verbose output during the simulation. Default is False.
    n_jobs : int, optional
        The number of parallel jobs to run. Default is 1 (no parallelization).
    size : int, optional
        The number of simulation realizations to generate. Default is 1.
    **kwargs : optional keyword arguments
        Additional arguments to pass to the simulation_params function.

    Returns
    -------
    List[np.ndarray]
        A list of simulated fields, each represented as a numpy array.

    """
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
