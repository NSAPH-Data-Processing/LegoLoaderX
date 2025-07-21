from setuptools import setup, find_packages

setup(
    name='legoloaderx',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'pandas',
        'pyarrow',
        'duckdb',
        'hydra-core',
    ],
    author='NSAPH Team',
    description='Data loader and preprocessor for Lego datasets',
    url='https://github.com/NSAPH-Data-Processing/LegoLoaderX',
    python_requires='>=3.10',
)
