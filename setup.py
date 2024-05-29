from setuptools import setup

setup(
    name='ChemSPX',
    version='1.0',
    author='Ignas Pakamore',
    description='Chemical Space Explorer',
    long_description='Algorithm for drawing spamples from a multi dimensional parameter space and its assessment.',
    maintainer='Ignas Pakamore',
    url='https://github.com/ignaspakamore/ChemSPX',
    python_requires='>=3.7, <4',
    license='MIT',
    keywords=['parameter space', 'chemical space', 'reaction space'],
    install_requires=[
        'numpy',
        'scikit-learn',
        'matplotlib',
        'pandas',
        'scipy', 
        'smt', 
        'scikit-optimize', 
        'geneticalgorithm2==6.8.6'
    ],
    packages=['ChemSPX']
)