from setuptools import setup

setup(
    name='ChemSPX',
    version='1.0',
    author='Ignas Pakamore',
    description='Chemical Space Explorer',
    long_description='',
    url='https://github.com/ignaspakamore/ChemSPX',
    python_requires='>=3.7, <4',
    install_requires=[
        'numpy',
        'sklearn',
        'matplotlib',
        'pandas',
        'scipy', 
        'smt', 
        'scikit-optimize', 
        'geneticalgorithm2'
    ],
)