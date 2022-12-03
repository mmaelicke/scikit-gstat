from typing import Union, Tuple, List
import os
from glob import glob
import numpy as np
import pandas as pd

# for python 3.6 compability
try:
    from imageio.v3 import imread
except ImportError:
    from imageio import imread

PATH = os.path.abspath(os.path.dirname(__file__))


def field_names() -> List[str]:
    """
    Get all available fields
    """
    fnames = glob(os.path.join(PATH, 'rf', '*.png'))
    basenames = [os.path.basename(name) for name in fnames]
    return [os.path.splitext(name)[0] for name in basenames]


def field(fname: str, band: Union[int, str] = 0) -> np.ndarray:
    """
    Return one of the fields stored in the ``rf`` folder
    and return as numpy array.

    Parameters
    ----------
    fname : str
        The filename (not path) of the field. The file
        extenstion can be omitted.
    band : int, str
        The band to use, can either be an integer or the
        literal ``'mean'``, which will average all bands

    Returns
    -------
    field : numpy.ndarray
        The ndarray representing the requested band

    """
    # append a png file extension if needed
    if not fname.endswith('.png'):
        fname += '.png'

    # read image
    # TODO: with imageio v3 this can be switched to imageio.imread - the result should be the same here
    img = imread(os.path.join(PATH, 'rf', fname))

    # switch band
    if isinstance(band, int):
        if len(img.shape) > 2:
            return img[:, :, band]
        elif len(img.shape) == 2:
            return img
    elif band.lower() == 'mean':
        return np.mean(img, axis=2)

    raise AttributeError('band parameter is invalid')


def get_sample(
        fname: str,
        N: int = 100,
        seed: int = None,
        band: Union[int, str] = 0
) -> Tuple[np.ndarray]:
    """
    Sample one of the fields. The filename and band are passed down to
    :func:`field`. To return reproducible results the random Generator
    can be seeded.

    Parameters
    ----------
    fname : str
        The filename (not path) of the field. The file
        extenstion can be omitted.
    N : int
        Sample size
    seed : int
        seed to use for the random generator.
    band : int, str
        The band to use, can either be an integer or the
        literal ``'mean'``, which will average all bands

    Returns
    -------
    coordinates : numpy.ndarray
        Coordinate array of shape ``(N, 2)``.
    values : numpy.ndarray
        1D array of the values at the coordinates

    """
    # first get the image
    img = field(fname, band)

    # randomly sample points
    rng = np.random.default_rng(seed)

    # sample at random flattened indices without replace
    idx = rng.choice(np.multiply(*img.shape), replace=False, size=N)

    # build a meshgrid over the image
    _x, _y = np.meshgrid(*[range(dim) for dim in img.shape])
    x = _x.flatten()
    y = _y.flatten()

    # get the coordinates and values
    coordinates = np.asarray([[x[i], y[i]] for i in idx])
    values = np.asarray([img[c[0], c[1]] for c in coordinates])

    return coordinates, values


def read_sample_file(fname) -> pd.DataFrame:
    """
    Return a sample from a sample-file as a
    pandas DataFrame

    Returns
    -------
    df : pandas.DataFrame
        The file content

    """
    # build the path
    path = os.path.join(PATH, 'samples', fname)
    return pd.read_csv(path)


def sample_to_df(coordinates: np.ndarray, values: np.ndarray) -> pd.DataFrame:
    """
    Turn the coordinates and values array into a pandas DataFrame.
    The columns will be called x[,y[,z]],v

    Parameters
    ----------
    coordinates : numpy.ndarray
        Coordinate array of shape ``(N, 1), (N, 2)`` or ``(N, 3)``.
    values : numpy.ndarray
        1D array of the values at the coordinates

    Returns
    -------
    df : pandas.DataFrame
        The data as a pandas DataFrame

    """
    # check that the array sizes match
    if len(coordinates) != len(values):
        raise AttributeError('coordinates and values must have the same length')

    # check dimensions
    col_names = ['x']
    if coordinates.ndim >= 2:
        col_names.append('y')
    if coordinates.ndim == 3:
        col_names.append('z')

    # build the dataframe
    df = pd.DataFrame(coordinates, columns=col_names)
    df['v'] = values
    return df
