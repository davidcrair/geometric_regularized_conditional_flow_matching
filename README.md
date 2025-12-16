# Geometric Regularized Autoencoder for Conditional Flow Matching

This repo contains the implementation of a Geometric Regularized Autoencoder for Conditional Flow Matching, as described in our paper.

## Repository Structure
`ae_model.py`: Pytorch Implementation of the Autoencoder architecture used for the single cell experiment
`neural_flow_model.py`: Pytorch Implementation of the Conditional Flow Matching architecture
`train_ae.ipynb`: Notebook to train the autoencoder on single cell data
`train_cfm.ipynb`: Notebook to train the Conditional Flow Matching model using the trained autoencoder
`datasets.py`: Data loading and preprocessing utilities for scRNA-seq

## Usage

### Installation and Requirements

We have uses `uv` for package management, so if you are familiar with it, you can create a virtual environment using the provided `pyproject.toml` file.

If you prefer using `pip`, you can install the required packages using the following command:

```bash
pip install -r requirements.txt
```

### Data

The data needed to run the single-cell experiments is included in the `data/` directory. You may need a GPU to efficiently train the models, but theoretically, they can be trained on a CPU as well.

### Toy Example
To visualize the manifold straightening effect on the S-curve toy dataset, run the `toy_example.ipynb` notebook.
If you want to turn off the geometric regularization of the autoencoder, set the `REG_LAMBDA` constant to 0 at the top of the notebook.

#### Single-Cell Data (subset of Parse-PBMC)
We evaluate the model on the C14 Monocyte population from the Parse-PBMC dataset. The model is trained to generalize to held-out perturbations (cytokines) for specific donors. We report Wasserstein distances between generated and real perturbed cells on differentially expressed genes (DEGs) to measure model performance.