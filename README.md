# Granular Traffic Data Imputation using PARATUCK2

This repository contains the source code and data for the paper:
> [**A Method for Granular Traffic Data Imputation Based on PARATUCK2**](https://doi.org/10.1177/03611981221089298)\
> Mina Nouri, Mostafa Reisi-Gahrooei, and Mohammad Ilbeigi, 2022.


## Overview

Imputing missing data is essential for reliable traffic monitoring and forecasting in intelligent transportation systems.  
This project implements a **PARATUCK2 decomposition-based imputation method** that models the data matrix as the product of three factor matrices: a row factor matrix representing spatial clusters, a column factor matrix representing temporal clusters, and a latent interaction matrix capturing relationships between them. The factors are learned through an **Alternating Least Squares (ALS)** optimization with regularization that minimizes reconstruction error over observed entries to accurately estimate missing values in large, complex urban networks.


## Repository Contents

- `main.py`: The main Python script containing the implementation of the `PARATUCK2_imputer` function and a usage example that loads data, performs imputation, and evaluates error.
- `data/`: Folder containing input data:
  - `masked_matrix.csv`: Traffic data matrix with missing values (`NaN`) representing unobserved entries.
  - `original_matrix.csv`: Ground-truth traffic data matrix used for evaluation.


## Requirements

This code was developed and tested with Python 3.8.  
Required Python libraries:  
- `numpy`  
- `pandas`  
- `pathlib`


## Usage

Run the [`main.py`](main.py) script to perform missing value imputation and evaluate median relative error on missing entries:

```bash
python main.py
```


The script execution workflow:

1. Loads the masked (incomplete) and original (complete) traffic matrices from the `data/` folder.
2. Calls the `PARATUCK2_imputer` function with:
   - The masked matrix *`X`*
   - Number of latent components `(p, q) = (5, 7)`
   - Regularization parameter `200`
3. Computes the median relative error only on the missing entries to evaluate imputation performance.
4. Prints the imputed matrix and error metric.



