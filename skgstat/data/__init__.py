import pandas as pd

from skgstat.data import _loader
from skgstat.data._loader import field_names, sample_to_df

# define all names
names = field_names()

origins = dict(
    pancake="""Image of a pancake with apparent spatial structure.
    Copyright Mirko Mälicke, 2020. If you use this data,
    cite SciKit-GStat: https://doi.org/10.5281/zenodo.1345584
    """,
    aniso="""Random field greyscale image with geometric anisotropy.
    The anisotropy in North-East direction has a factor of 3. The random
    field was generated using gstools.
    Copyright Mirko Mälicke, 2020. If you use this data,
    cite SciKit-GStat: https://doi.org/10.5281/zenodo.1345584
    """,
    meuse=""""Sample dataset of real measurements of heavy metal pollutions
    in the topsoil on a 15x15 meter plot along the river Meuse.
    The data is distributed along with the R-package sp.
    IMPORTANT: If you use this data, cite Pebesma and Bivand (2005)
    and Bivand et al (2013):

      Pebesma EJ, Bivand RS (2005). “Classes and methods for spatial
      data in R.” R News, 5(2), 9–13. https://CRAN.R-project.org/doc/Rnews/.

      Bivand RS, Pebesma E, Gomez-Rubio V (2013). Applied spatial data
      analysis with R, Second edition. Springer, NY. https://asdar-book.org/.

    """
)


# define the functions
def pancake(N=500, band=0, seed=42, as_dataframe=False):
    """
    Sample of the :func:`pancake_field <skgstat.data.pancake_field>`.
    By default, the Red band is sampled at 500 random
    location without replacement.

    Parameters
    ----------
    N : int
        Number of sample points to use.
    band : int
        can be 0 (Red), 1 (Green), 2 (Blue) or ``'mean'``, which
        will average all three RGB bands
    seed : int
        By default a seed is set to always return the same
        sample for same N and band input
    as_dataframe : bool
        If True, the data is returned as pandas.Dataframe.
        Default is False.

    Returns
    -------
    result : dict
        Dictionary of the sample and a citation information.
        The sample is a tuple of two numpy arrays.

    See Also
    --------
    :func:`get_sample <skgstat.data._loader.get_sample>`
    :func:`pancake_field <skgstat.data.pancake_field>`

    Notes
    -----
    The image originates from a photograph of an actual pancake.
    The image was cropped to an 500x500 pixel extent keeping the
    center of the original photograph.
    If you use this example somewhere else, please cite
    SciKit-GStat [501]_, as it is distributed with the library.

    References
    ----------
    .. [501] Mirko Mälicke, Helge David Schneider, Sebastian Müller,
        & Egil Möller. (2021, April 20). mmaelicke/scikit-gstat: A scipy
        flavoured geostatistical variogram analysis toolbox (Version v0.5.0).
        Zenodo. http://doi.org/10.5281/zenodo.4704356

    """
    sample = _loader.get_sample('pancake', N=N, seed=seed, band=band)

    if as_dataframe:
        sample = sample_to_df(*sample)

    return dict(
        sample=sample,
        origin=origins.get('pancake')
    )


def pancake_field(band=0):
    """
    Image of a pancake with apparent spatial structure.
    The pankcake has three RGB bands.

    Parameters
    ----------
    band : int
        can be 0 (Red), 1 (Green), 2 (Blue) or ``'mean'``, which
        will average all three RGB bands

    Returns
    -------
    result : dict
        Dictionary of the sample and a citation information.
        The sample is 2D numpy array of the field.

    See Also
    --------
    skgstat.data._loader.field
    skgstat.data.pancake

    Notes
    -----
    The image originates from a photograph of an actual pancake.
    The image was cropped to an 500x500 pixel extent keeping the
    center of the original photograph.
    If you use this example somewhere else, please cite
    SciKit-GStat [501]_, as it is distributed with the library.

    References
    ----------
    .. [501] Mirko Mälicke, Helge David Schneider, Sebastian Müller,
        & Egil Möller. (2021, April 20). mmaelicke/scikit-gstat: A scipy
        flavoured geostatistical variogram analysis toolbox (Version v0.5.0).
        Zenodo. http://doi.org/10.5281/zenodo.4704356

    """
    sample = _loader.field('pancake', band)

    return dict(
        sample=sample,
        origin=origins.get('pancake')
    )


def aniso(N=500, seed=42, as_dataframe=False):
    """
    Sample of the :func:`ansio_field <skgstat.data.aniso_field>`.
    By default the greyscale image is sampled
    at 500 random locations.

    Parameters
    ----------
    N : int
        Number of sample points to use.
    seed : int
        By default a seed is set to always return the same
        sample for same N and band input
    as_dataframe : bool
        If True, the data is returned as pandas.Dataframe.
        Default is False.

    Returns
    -------
    result : dict
        Dictionary of the sample and a citation information.
        The sample is a tuple of two numpy arrays.

    See Also
    --------
    skgstat.data._loader.field : field loader
    aniso_field : Return the full field

    Notes
    -----
    This image was created using :any:`gstools.SRF`.
    The spatial random field was created using a Gaussian model
    and has a size of 500x500 pixel. The created field
    was normalized and rescaled to the value range of a
    :any:`uint8 <numpy.uint8>`.
    The spatial model includes a small nugget (~ 1/25 of the value range).
    If you use this example somewhere else, please cite
    SciKit-GStat [501]_, as it is distributed with the library.

    References
    ----------
    .. [501] Mirko Mälicke, Helge David Schneider, Sebastian Müller,
        & Egil Möller. (2021, April 20). mmaelicke/scikit-gstat: A scipy
        flavoured geostatistical variogram analysis toolbox (Version v0.5.0).
        Zenodo. http://doi.org/10.5281/zenodo.4704356

    """
    sample = _loader.get_sample('aniso', N=N, seed=seed)

    if as_dataframe:
        sample = sample_to_df(*sample)

    return dict(
        sample=sample,
        origin=origins.get('aniso')
    )


def aniso_field():
    """
    Image of a greyscale image with geometric anisotropy.
    The anisotropy has a North-Easth orientation and has
    a approx. 3 times larger correlation length than in
    the perpendicular orientation.

    Returns
    -------
    result : dict
        Dictionary of the sample and a citation information.
        The sample a numpy array repesenting the image.

    See Also
    --------
    skgstat.data._loader.field : field loader
    aniso : Return a sample

    Notes
    -----
    This image was created using :any:`gstools.SRF`.
    The spatial random field was created using a Gaussian model
    and has a size of 500x500 pixel. The created field
    was normalized and rescaled to the value range of a
    :any:`uint8 <numpy.uint8>`.
    The spatial model includes a small nugget (~ 1/25 of the value range).
    If you use this example somewhere else, please cite
    SciKit-GStat [501]_, as it is distributed with the library.

    References
    ----------
    .. [501] Mirko Mälicke, Helge David Schneider, Sebastian Müller,
        & Egil Möller. (2021, April 20). mmaelicke/scikit-gstat: A scipy
        flavoured geostatistical variogram analysis toolbox (Version v0.5.0).
        Zenodo. http://doi.org/10.5281/zenodo.4704356

    """
    sample = _loader.field('aniso')

    return dict(
        sample=sample,
        origin=origins.get('aniso')
    )


def meuse(variable='lead', as_dataframe=False):
    """
    Returns one of the samples of the well-known Meuse dataset.
    You can specify which heave metal data you want to load.

    Parameters
    ----------
    variable : str
        Name of the variable to be extracted from the dataset.
        Can be one of ['cadmium', 'copper', 'lead', 'zinc'].
        Default is 'lead'.
    as_dataframe : bool
        If True, the data is returned as pandas.Dataframe.
        Default is False.

    Returns
    -------
    result : dict
        Dictionary of the sample and a citation information.

    Notes
    -----
    The example data was taken from the R package 'sp'
    as published on CRAN: https://cran.r-project.org/package=sp
    The package is licensed under GPL-3, which applies
    to the sample if used somewhere else.
    If you use this sample, please cite the original sources
    [502]_, [503]_ and not SciKit-GStat.

    References
    ----------
    .. [502] Pebesma EJ, Bivand RS (2005). “Classes and methods for spatial
      data in R.” R News, 5(2), 9–13. https://CRAN.R-project.org/doc/Rnews/.

    .. [503] Bivand RS, Pebesma E, Gomez-Rubio V (2013). Applied spatial data
      analysis with R, Second edition. Springer, NY. https://asdar-book.org/.

    """
    # check variable
    if variable not in ('cadmium', 'copper', 'lead', 'zinc'):
        raise AttributeError(
            "variable has to be in ['cadmium', 'copper', 'lead', 'zinc']"
        )

    # get the data
    df = _loader.read_sample_file('meuse.txt')

    # get the coordinates
    coords = df[['x', 'y']].values

    # get the correct variable
    values = df[[variable]].values

    # create sample
    if as_dataframe:
        sample = sample_to_df(coords, values)
    else:
        sample = (coords, values, )
    # return
    return dict(
        sample=sample,
        origin=origins.get('meuse')
    )
