from setuptools import setup

setup(
    name='ChemSPX',
    version='1.0',
    author='Ignas Pakamore',
    description='Chemical Space Explorer',
    long_description='A much longer explanation of the project and helpful resources',
    url='https://github.com/BenjaminFranline',
    keywords='development, setup, setuptools',
    python_requires='>=3.7, <4',
    install_requires=[
        'PyYAML',
        'pandas==0.23.3',
        'numpy>=1.14.5',
        'matplotlib>=2.2.0',
        'jupyter'
    ],
    package_data={
        'sample': ['sample_data.csv'],
    },
    entry_points={
        'runners': [
            'sample=sample:main',
        ]
    }
)