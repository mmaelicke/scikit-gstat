from setuptools import setup, find_packages


def readme():
    with open('README.rst') as f:
        return f.read().strip()


def version():
    with open('VERSION') as f:
        return f.read().strip()


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
      classifiers=classifiers(),
      install_requires=requirements(),
      test_suite='nose.collector',
      # test_require=['nose'],
      extras_require={"gstools": ["gstools>=1.3"]},
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False
)
