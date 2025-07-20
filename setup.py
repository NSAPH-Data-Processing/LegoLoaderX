from setuptools import setup, find_packages

setup(
    name='legoloaderx',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'pandas==2.2.2',
        'pyarrow==11.0.0',
        'duckdb==0.9.2',
        'hydra-core==1.3.2',
        'snakemake==8.16',
        'tqdm'
    ],
    author='NSAPH Team',
    description='Data loader and preprocessor for Lego datasets',
    url='https://github.com/NSAPH-Data-Processing/LegoLoaderX',
    python_requires='>=3.11',
)
