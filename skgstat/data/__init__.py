from skgstat.data import _loader
from skgstat.data._loader import field_names

# define all names
names = field_names()


# define the functions
def pancake(N=500, band=0, seed=42):
    return _loader.get_sample('pancake', N=N, seed=seed, band=band)


def pancake_field(band=0):
    return _loader.field('pancake', band)
