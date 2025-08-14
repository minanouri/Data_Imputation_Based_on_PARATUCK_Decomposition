# Granular Traffic Data Imputation using PARATUCK2

This repository contains the source code and data for the paper:
> [**A Method for Granular Traffic Data Imputation Based on PARATUCK2**](https://doi.org/10.1177/03611981221089298)\
> Mina Nouri, Mostafa Reisi-Gahrooei, and Mohammad Ilbeigi, 2022.


## Overview

Imputing missing traffic data is crucial for reliable monitoring and prediction in intelligent transportation systems.  
This repository implements a **PARATUCK2 decomposition-based imputation method** that captures both spatial and temporal clusters in traffic data, allowing accurate recovery of missing values even in large and complex urban road networks.

The PARATUCK2 model decomposes the data matrix into three factor matrices:

\[
X \approx A \, R \, B^T
\]

- **A**: Row (spatial) latent factor matrix  
- **R**: Latent interaction matrix  
- **B**: Column (temporal) latent factor matrix  

The method is optimized using an **Alternating Least Squares (ALS)** approach with regularization to ensure stability.


## Repository Contents

- [`main.py`](main.py): The main Python script containing the implementation of the `PARATUCK2_imputer` function and a usage example that loads data, performs imputation, and evaluates error.
- `data/`: Folder containing input data:
  - [`masked_matrix.csv`](data/masked_matrix.csv): Traffic data matrix with missing values (`NaN`) representing unobserved entries.
  - [`original_matrix.csv`](data/original_matrix.csv): Ground-truth traffic data matrix used for evaluation.


## Requirements

This code was developed and tested with **Python 3.8**.  
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

1. Loads the masked (incomplete) and original (complete) traffic matrices from the `data` folder.
2. Calls the `PARATUCK2_imputer` function with:
   - The masked matrix `X`
   - Number of latent components `(p, q) = (5, 7)`
   - Regularization parameter `200`
3. Computes the median relative error only on the originally missing entries to evaluate imputation performance.
4. Prints the imputed matrix and error metric.



