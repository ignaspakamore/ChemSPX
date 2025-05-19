
# ChemSPX 

A toolkit for sampling and analysis of reaction and formulation design parameter spaces. 

## Features

- **Parameter Space Sampling**: 
  - Utilises an inverse distance function to perform efficient sampling within the parameter space, ensuring diverse and representative point selection.

- **Undersampled Region Detection**: 
  - Automatically identifies regions within the parameter space that are undersampled, guiding further exploration and refinement for optimal coverage.

- **N-Dimensional Sample Distribution Analysis**:
  - Analyses the spatial distribution of sample points across the parameter space to assess both coverage and clustering characteristics.

- **Exploration Coverage Calculation**:
  - Includes an algorithm to compute the percentage of parameter space explored, offering quantitative insights into the completeness of parameter space exploration.


# Instalation

ChemSPX requires **python 3.12**. 

Tested on **macOS Sonoma Version 14.0** and **Ubuntu 22.04.4 LTS**. Conda version 24.7.1.

1. Clone the repository.

```
git clone https://github.com/ignaspakamore/ChemSPX.git
```

2. Go into cloned directory.

```
cd ChemSPX
```

3. Set up conda environmnet.

```
conda env create -f environment.yaml
```

4. Activate conda environment.

```
conda activate chemspx
```

5. Install dependencies.

```
pip install .
```

# Getting started 

After instalation the code can be execute by calling the following command in the terminal. 

The input file contains program execution instruction. 
Example input can be found in *examples/directory*.

```
chemspx input.txt
```

Alternatively, jupyter notebooks can be used to execute the code. 
