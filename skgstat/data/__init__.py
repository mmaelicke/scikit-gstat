from skgstat.data import _loader
from skgstat.data._loader import field_names

# define all names
names = field_names()

origins = dict(
    pancake="""Image of a pancake with apparent spatial structure.
    If you use this data, cite SciKit-GStat: https://doi.org/10.5281/zenodo.1345584
    Copyright Mirko Mälicke, 2020.""",
    aniso="""Random field greyscale image with geometric anisotropy.
    The anisotropy in North-East direction has a factor of 3. The random
    field was generated using gstools.
    Copyright Mirko Mälicke, 2021"""
)


# define the functions
def pancake(N=500, band=0, seed=42):
    sample = _loader.get_sample('pancake', N=N, seed=seed, band=band)

    return dict(
        sample=sample,
        origin=origins.get('pancake')
    )


def pancake_field(band=0):
    sample = _loader.field('pancake', band)

    return dict(
        sample=sample,
        origin=origins.get('pancake')
    )


def aniso(N=500, seed=42):
    sample = _loader.get_sample('aniso', N=N, seed=seed)

    return dict(
        sample=sample,
        origin=origins.get('aniso')
    )


def aniso_field():
    sample = _loader.field('aniso')

    return dict(
        sample=sample,
        origin=origins.get('aniso')
    )
