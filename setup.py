from setuptools import setup, find_packages


def readme():
    with open('README.rst') as f:
        return f.read().strip()


def version():
    with open('skgstat/__version__.py') as f:
        loc = dict()
        exec(f.read(), loc, loc)
        return loc['__version__']


def requirements():
    with open('requirements.txt') as f:
        return f.read().strip().split('\n')


def classifiers():
    with open('classifiers.txt') as f:
        return f.read().strip().split('\n')


setup(name='scikit-gstat',
      license='MIT License',
      version=version(),
      author='Mirko Maelicke',
      author_email='mirko.maelicke@kit.edu',
      description='Geostatistical expansion in the scipy style',
      long_description=readme(),
      long_description_content_type='text/x-rst',
      project_urls={
          "Documentation": "https://scikit-gstat.readthedocs.io",
          "Source": "https://github.com/scikit-gstat/scikit-gstat",
          "Tracker": "https://github.com/scikit-gstat/scikit-gstat/issues",
      },
      url="https://github.com/scikit-gstat/scikit-gstat",
      classifiers=classifiers(),
      install_requires=requirements(),
      test_suite='nose.collector',
      # test_require=['nose'],
      extras_require={"gstools": ["gstools>=1.3"]},
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False
)
