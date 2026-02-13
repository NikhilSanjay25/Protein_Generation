#Molecule Generation with WGAN-GP
This project implements a generative model to create novel small molecules using a Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP). The models are trained on a subset of the ChEMBL dataset, focusing on small molecule candidates with 9 atoms or less.

Project Structure
The repository contains the following key components:


app.py: A Streamlit-based web application for visualizing and comparing generated molecules, physicochemical properties, and chemical space (PCA).

SimpleGAN.py: The main implementation of the GAN architecture and training loop for graph-based molecule representation.

Simplified_bond.py: A variation of the model focusing on binary bond representations (exists/does not exist) to simplify the learning task.

baseline_WGAN.ipynb: Jupyter notebook containing the training logic and initial results for the standard WGAN model.

ChemBLwork.ipynb: Notebook dedicated to data preprocessing, including loading ChEMBL representations and UniProt mapping.

main5.ipynb: A core training notebook showing execution logs, loss curves, and epoch-by-epoch generation results.

requirments.txt: List of Python dependencies required to run the project.

Features
Graph-based Generation: Molecules are represented as graphs where nodes are atom types and edges are bond types.

WGAN-GP Training: Uses Wasserstein loss with a gradient penalty to stabilize training and improve generation quality.

ChEMBL Dataset Integration: Connects to a SQLite version of the ChEMBL 35 database for training data.


Evaluation Metrics: Analyzes generated molecules based on validity, uniqueness, novelty, and scaffold diversity.


Visualization Dashboard: Compare model versions side-by-side using the Streamlit UI, including PCA of chemical space and distribution of properties like HBA and HBD.

Prerequisites
Ensure you have a Python environment (e.g., venv or conda) set up. The project requires CUDA-compatible hardware for efficient training.

Key Dependencies
torch (2.1.0)

rdkit-pypi

streamlit

pandas, numpy, matplotlib, seaborn

torch-geometric

Getting Started
Install Dependencies:

Dataset Preparation:
The code expects the ChEMBL database at the following path:


DL_ENDSEM__DATASET/chembl_35/chembl_35_sqlite/chembl_35.db.

Training:
Run the training script or notebooks (SimpleGAN.py or main5.ipynb) to start training. Models are saved every 10 epochs by default to the specified MODEL_DIR.

Running the Dashboard:
To visualize results, launch the Streamlit app:

Model Configurations
The models are tuned for small graphs with the following parameters:

Max Atoms: 9.

Atom List: C, N, O, F, S, Cl.

Bond Types: Single, Double, Triple, Aromatic (or binary in simplified version).

Latent Dimension: 128.

Optimizer: Adam with a learning rate of 1e-4 for both Generator and Discriminator.
