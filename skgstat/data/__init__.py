from skgstat.data import _loader
from skgstat.data._loader import field_names

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
    """
)


# define the functions
def pancake(N=500, band=0, seed=42):
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


def aniso(N=500, seed=42):
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
