# A Data-Centric Investigation on the Challenges of Machine Learning Methods for Bridging Life Cycle Inventory Data Gaps

This repository contains the code and data for the paper:

> **"A Data-Centric Investigation on the Challenges of Machine Learning Methods for Bridging Life Cycle Inventory Data Gaps"**  
> Bu Zhao, Jitong Jiang, Ming Xu, and Qingshi Tu. *Journal of Industrial Ecology*, 2025. [DOI: 10.1111/jiec.70022](https://doi.org/10.1111/jiec.70022)

## Overview
This study investigates the instability of machine learning model performance when applied to life cycle inventory (LCI) data, focusing on the effects of data imbalance, magnitude variations, and train–test split randomness. A similarity-based framework was used to explore these challenges and propose directions for future improvement.

## Repository Structure

### Data Preparation and Visualization
- `flow_freq.m` — Normalizes flow data, calculates the frequency of flow appearances across processes, and generates the bar plot for Figure 1.
- `flow_magnitude.m` — Calculates the maximum magnitude for each flow, identifying top and bottom flows, and generates the plots for Figure 3.
- `flow_median.m` — Computes the median of non-zero values for each flow and generates the bar plot for Supporting Information Figure S2.
- `isic.m` — Matches processes to ISIC classifications and generates a bar plot showing the distribution across ISIC categories (Figure 2).

### Parameter Tuning for Training Model
- `training_big_range.m` — Performs a coarse grid search over a wide q range (0.001–10) and k range (1–20) to explore model performance, generating the heatmap for Supporting Information Figure S1(a).
- `training_small_range.m` — Conducts a fine-scale grid search within a narrow q range (0.065–0.085) and k range (1–5) to identify optimal parameters, generating the heatmap for Supporting Information Figure S1(b).  
  After running, users should manually save the results as `data_training_small_range.mat` for later use (e.g., using the `save` function).

### Performance on Testing Model
- `testing.m` — Evaluates the model on an independent test set using the optimal (q, k) parameters from `data_training_small_range.mat` and visualizes the test performance.

### Performance Variability Analysis
- `randomsplit_training.m`: Finds optimal parameters for 20 different train–test splits and records model performance (Figure 4a).
- `randomsplit_testing.m`: Evaluates model performance on 50 random sub-test sets with fixed parameters (Figure 4b).
- `figure4.R` — Creates two scatter plots to show how model performance varies under different train–test split randomness with 5% missing data. (Input files: `figure4a.csv` and `figure4b.csv`.)
- `figure5.R` — Creates two scatter plots to visualize how model performance varies with flow frequency and value magnitude under 5% missing data. (Input files: `Frequency_5%_new.csv` and `5%missing_new.csv`.)

### U.S. LCI Database (USLCI) Analysis
- `USLCIdata.m` — Prepares the USLCI dataset and generates `USLCIdata.mat` (not not publicly available; available upon request.).
- `USLCI_performance.m` — Analyzes USLCI model performance and generates the heatmap for Supporting Information Figure S3.
  
## LCI Data Information
- `Flowinfo.xlsx` — Basic information for flow names.
- `Processinfo.csv` — Basic information for process names.
- `activity_overview_for_users_3.1_default.xlsx` — ISIC classification information.

## Data Availability
Due to licensing and data sharing restrictions, `rawdata.csv`(i.e. Ecoinvent raw data) and `USLCIdata.mat` cannot be made publicly available. Please contact the authors to request access to these files.

## How to Cite
If you use this code or data, please cite:

> Zhao, B., Jiang, J., Xu, M., & Tu, Q. (2025). A data-centric investigation on the challenges of machine learning methods for bridging life cycle inventory data gaps. *Journal of Industrial Ecology*, 1–12. [DOI: 10.1111/jiec.70022](https://doi.org/10.1111/jiec.70022)

