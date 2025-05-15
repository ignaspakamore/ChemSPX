from setuptools import setup

setup(
    name="ChemSPX",
    version="1.0",
    author="Ignas Pakamore",
    description="Chemical Space Explorer",
    long_description="Algorithm for drawing spamples from a multi dimensional parameter space and its assessment.",
    maintainer="Ignas Pakamore",
    url="https://github.com/ignaspakamore/ChemSPX",
    python_requires=">=3.7, <4",
    license="MIT",
    keywords=["parameter space", "chemical space", "reaction space"],
    install_requires=[
        "numpy",
        "scikit-learn==1.3.2",
        "matplotlib==3.9.2",
        "pandas==2.2.3",
        "scipy==1.14.1",
        "smt==2.7.0",
        "scikit-optimize==0.9",
        "geneticalgorithm2==6.9.2",
        "ipython==9.2.0",
        "ipykernel==6.29.5"
    ],
    packages=["ChemSPX"],
    entry_points={
        "console_scripts": [
            "chemspx = ChemSPX.run:program",
        ],
    },
)
