import streamlit as st
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import Descriptors, QED, Draw, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
import sqlite3
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from IPython.display import display

# =========================================
# SETUP & CONFIGURATION
# =========================================

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

# --- Try to import SA Score ---
# (This assumes the sascorer.py file is accessible in the specified path)
SA_SCORE_ENABLED = False
try:
    from rdkit import RDConfig
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer
    SA_SCORE_ENABLED = True
    print("SA Score module loaded successfully.")
except ImportError:
    print("SA Score module not found. Skipping SA_Score calculation in UI.")


# --- Main Configuration Class ---
# This class holds all paths and hyperparameters.
# !! UPDATE THESE PATHS to match your system !!
class Config:
    # --- System & Data Paths ---
    # Update this path to your ChEMBL DB
    CHEMBL_DB_PATH = 'DL_ENDSEM__DATASET/chembl_35/chembl_35_sqlite/chembl_35.db'
    
    # Directory where models and analysis files are stored
    MODEL_DIR = 'models_simplified_v3' 
    
    # Log file from training
    LOG_FILE = 'training_log.csv'
    
    # Generated molecules file (from your best epoch)
    GENERATED_FILE = 'final_molecules_ep460_v3.csv'
    
    # --- Model Hyperparameters (from your script) ---
    MAX_ATOMS = 9
    ATOM_LIST = ['C', 'N', 'O', 'F', 'S', 'Cl'] 
    NUM_ATOM_TYPES = len(ATOM_LIST) + 1 
    
    BOND_TYPES = [Chem.rdchem.BondType.ZERO,
                  Chem.rdchem.BondType.SINGLE,
                  Chem.rdchem.BondType.DOUBLE,
                  Chem.rdchem.BondType.TRIPLE,
                  Chem.rdchem.BondType.AROMATIC]
    NUM_BOND_TYPES = len(BOND_TYPES)

    LATENT_DIM = 128
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- Analysis Config ---
    BEST_EPOCH = 460 # The epoch you want to use for generation
    
    # --- File Paths (resolved) ---
    @staticmethod
    def get_model_path():
        return os.path.join(Config.MODEL_DIR, f'g_{Config.BEST_EPOCH}.pth')

    @staticmethod
    def get_generated_csv_path():
        return os.path.join(Config.MODEL_DIR, Config.GENERATED_FILE)

    @staticmethod
    def get_log_path():
        return os.path.join(Config.MODEL_DIR, Config.LOG_FILE)


# =========================================
# CORE MODEL & UTILITY DEFINITIONS
# (Copied directly from your script)
# =========================================

class GraphTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, adj_channels):
        super().__init__()
        self.embed_dim, self.num_heads, self.head_dim = embed_dim, num_heads, embed_dim // num_heads
        self.q_proj, self.k_proj, self.v_proj = nn.Linear(embed_dim, embed_dim), nn.Linear(embed_dim, embed_dim), nn.Linear(embed_dim, embed_dim)
        self.out_proj, self.adj_proj = nn.Linear(embed_dim, embed_dim), nn.Linear(adj_channels, num_heads)
    
    def forward(self, x, adj):
        B, N, _ = x.shape
        Q, K, V = [proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2) for proj in (self.q_proj, self.k_proj, self.v_proj)]
        attn = (Q @ K.transpose(-2, -1)) / np.sqrt(self.head_dim) + self.adj_proj(adj).permute(0, 3, 1, 2)
        return self.out_proj((F.softmax(attn, dim=-1) @ V).transpose(1, 2).reshape(B, N, self.embed_dim)) + x

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(Config.LATENT_DIM, 128)
        self.fc_nodes = nn.Linear(128, Config.MAX_ATOMS * Config.NUM_ATOM_TYPES)
        self.fc_adj = nn.Linear(128, Config.MAX_ATOMS * Config.MAX_ATOMS * Config.NUM_BOND_TYPES)
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.normal_(m.weight, 0, 0.02)

    def forward(self, z, temperature=1.0):
        h = F.leaky_relu(self.fc(z), 0.2)
        nodes = F.gumbel_softmax(self.fc_nodes(h).view(-1, Config.MAX_ATOMS, Config.NUM_ATOM_TYPES), tau=temperature, hard=False, dim=-1)
        adj = self.fc_adj(h).view(-1, Config.MAX_ATOMS, Config.MAX_ATOMS, Config.NUM_BOND_TYPES)
        return nodes, F.gumbel_softmax((adj + adj.permute(0, 2, 1, 3)) / 2.0, tau=temperature, hard=False, dim=-1)

def graphs_to_mols(node_X, adj_A, hard=True):
    mols = []
    if hard:
        if isinstance(node_X, torch.Tensor): node_X = torch.argmax(node_X, dim=-1).detach().cpu().numpy()
        if isinstance(adj_A, torch.Tensor): adj_A = torch.argmax(adj_A, dim=-1).detach().cpu().numpy()

    for b in range(node_X.shape[0]):
        mol = Chem.RWMol()
        atom_indices = []
        for i in range(Config.MAX_ATOMS):
            atom_type = node_X[b, i]
            if atom_type == len(Config.ATOM_LIST): continue
            atom_indices.append(mol.AddAtom(Chem.Atom(Config.ATOM_LIST[atom_type])))
        
        # Map original atom indices to new RWMol indices
        idx_map = {orig_i: new_i for new_i, orig_i in enumerate(atom_indices)}
        
        # Use a set to avoid adding duplicate bonds
        added_bonds = set()

        for i_orig in range(Config.MAX_ATOMS):
            for j_orig in range(i_orig + 1, Config.MAX_ATOMS):
                if i_orig not in idx_map or j_orig not in idx_map:
                    continue
                
                bond_idx = adj_A[b, i_orig, j_orig]
                if bond_idx != 0:
                    try:
                        mol.AddBond(idx_map[i_orig], idx_map[j_orig], Config.BOND_TYPES[bond_idx])
                        added_bonds.add(tuple(sorted((idx_map[i_orig], idx_map[j_orig]))))
                    except Exception:
                        pass
        try:
            mol = mol.GetMol()
            Chem.SanitizeMol(mol)
            mols.append(mol)
        except Exception:
            mols.append(None)
    return mols


# =========================================
# STREAMLIT CACHED FUNCTIONS
# =========================================

# --- Generation Tab Functions ---

@st.cache_resource
def load_generator(epoch):
    """Loads the generator model into memory. Cached."""
    model_path = Config.get_model_path()
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please check `Config.MODEL_DIR` and `Config.BEST_EPOCH`.")
        return None
    
    gen = Generator().to(Config.DEVICE)
    try:
        gen.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
        gen.eval()
        return gen
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def generate_single_molecule(generator, temperature):
    """Generates a single valid molecule."""
    mol = None
    smiles = None
    while mol is None:
        with torch.no_grad():
            z = torch.randn(1, Config.LATENT_DIM, device=Config.DEVICE)
            nodes, adj = generator(z, temperature=temperature)
            mol_list = graphs_to_mols(nodes, adj, hard=True)
            if mol_list and mol_list[0]:
                try:
                    smi = Chem.MolToSmiles(mol_list[0])
                    # Ensure it's a single fragment
                    if len(smi) > 0 and '.' not in smi:
                        mol = mol_list[0]
                        smiles = smi
                except Exception:
                    pass
    return mol, smiles

def get_mol_properties(mol):
    """Calculates key properties for a single RDKit molecule."""
    if mol is None:
        return {}
    props = {}
    props['MW'] = Descriptors.MolWt(mol)
    props['LogP'] = Descriptors.MolLogP(mol)
    props['QED'] = QED.qed(mol)
    props['TPSA'] = Descriptors.TPSA(mol)
    props['HBD'] = Descriptors.NumHDonors(mol)
    props['HBA'] = Descriptors.NumHAcceptors(mol)
    if SA_SCORE_ENABLED:
        props['SA_Score'] = sascorer.calculateScore(mol)
    else:
        props['SA_Score'] = "N/A"
    return props

# --- Analysis Tab Functions ---

@st.cache_data
def analysis_plot_loss():
    """Plots training loss curves from the log file."""
    log_path = Config.get_log_path()
    if not os.path.exists(log_path):
        return None
    
    df = pd.read_csv(log_path)
    window = max(50, len(df) // 100) 
    df['D_smooth'] = df['D_loss'].rolling(window=window).mean()
    df['G_smooth'] = df['G_loss'].rolling(window=window).mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Discriminator
    axes[0].set_title(f'Critic (Discriminator) Loss\n(Rolling window: {window} iters)')
    axes[0].plot(df['D_loss'], color='blue', alpha=0.15, label='Raw')
    axes[0].plot(df['D_smooth'], color='blue', linewidth=2, label='Smoothed')
    axes[0].set_xlabel('Training Iterations')
    axes[0].set_ylabel('Wasserstein Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Generator
    axes[1].set_title(f'Generator Loss\n(Rolling window: {window} iters)')
    axes[1].plot(df['G_loss'], color='orange', alpha=0.15, label='Raw')
    axes[1].plot(df['G_smooth'], color='darkorange', linewidth=2, label='Smoothed')
    axes[1].set_xlabel('Training Iterations')
    axes[1].set_ylabel('Wasserstein Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

@st.cache_data
def analysis_physchem():
    """Calculates and plots physicochemical properties."""
    csv_path = Config.get_generated_csv_path()
    if not os.path.exists(csv_path):
        return None, None

    df = pd.read_csv(csv_path)
    df['mol'] = df['smiles'].apply(Chem.MolFromSmiles)
    df_clean = df.dropna(subset=['mol']).copy()
    
    if len(df_clean) == 0:
        return pd.DataFrame(), None

    df_clean['MW'] = df_clean['mol'].apply(Descriptors.MolWt)
    df_clean['LogP'] = df_clean['mol'].apply(Descriptors.MolLogP)
    df_clean['QED'] = df_clean['mol'].apply(QED.qed)
    df_clean['TPSA'] = df_clean['mol'].apply(Descriptors.TPSA)
    if SA_SCORE_ENABLED:
        df_clean['SA_Score'] = df_clean['mol'].apply(sascorer.calculateScore)

    plot_rows = 3 if SA_SCORE_ENABLED else 2
    fig, axes = plt.subplots(plot_rows, 2, figsize=(12, 5 * plot_rows))
    ax_flat = axes.flatten()

    df_clean['MW'].hist(ax=ax_flat[0], bins=20, color='skyblue', edgecolor='black')
    ax_flat[0].set_title('Molecular Weight (MW)')
    
    df_clean['LogP'].hist(ax=ax_flat[1], bins=20, color='lightgreen', edgecolor='black')
    ax_flat[1].set_title('LogP (Lipophilicity)')

    df_clean['QED'].hist(ax=ax_flat[2], bins=20, color='salmon', edgecolor='black')
    ax_flat[2].set_title('QED (Drug-likeness)'); ax_flat[2].set_xlim(0, 1)

    df_clean['TPSA'].hist(ax=ax_flat[3], bins=20, color='gold', edgecolor='black')
    ax_flat[3].set_title('TPSA (Polar Surface Area)')
    
    if SA_SCORE_ENABLED:
        df_clean['SA_Score'].hist(ax=ax_flat[4], bins=20, color='orchid', edgecolor='black')
        ax_flat[4].set_title('SA Score (Synthesizability)'); ax_flat[4].set_xlim(1, 10)
        ax_flat[5].axis('off')
    elif plot_rows == 2:
        ax_flat[4].axis('off')
        ax_flat[5].axis('off')


    plt.tight_layout()
    return df_clean.drop(columns=['mol']).describe(), fig

@st.cache_data
def analysis_novelty():
    """Checks novelty against the ChEMBL training set."""
    if not os.path.exists(Config.CHEMBL_DB_PATH):
        return "DB_NOT_FOUND", 0, 0, []
    
    gen_path = Config.get_generated_csv_path()
    if not os.path.exists(gen_path):
        return "GEN_FILE_NOT_FOUND", 0, 0, []

    try:
        conn = sqlite3.connect(Config.CHEMBL_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT canonical_smiles FROM compound_structures WHERE length(canonical_smiles) < 50 LIMIT 200000")
        training_smiles = set([row[0] for row in cursor.fetchall()])
        conn.close()
    except Exception as e:
        return f"DB_ERROR: {e}", 0, 0, []
    
    df_gen = pd.read_csv(gen_path)
    generated_smiles = set(df_gen['smiles'].tolist())
    
    novel_molecules = generated_smiles - training_smiles
    novelty_score = (len(novel_molecules) / len(generated_smiles)) * 100
    sorted_novel = sorted(list(novel_molecules), key=len, reverse=True)
    
    return novelty_score, len(novel_molecules), len(generated_smiles), sorted_novel[:5]

@st.cache_data
def analysis_diversity():
    """Analyzes internal diversity using Tanimoto similarity."""
    gen_path = Config.get_generated_csv_path()
    if not os.path.exists(gen_path):
        return None, None, None

    df = pd.read_csv(gen_path)
    mols = [Chem.MolFromSmiles(s) for s in df['smiles']]
    mols = [m for m in mols if m is not None]
    
    if len(mols) < 2:
        return 0, 0, None

    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2014) for m in mols]
    
    similarities = []
    num_mols = len(fps)
    for i in range(num_mols):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:])
        similarities.extend(sims)

    similarities = np.array(similarities)
    if len(similarities) == 0:
        return 0, 0, None
        
    avg_sim = np.mean(similarities)
    internal_diversity = 1.0 - avg_sim

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(similarities, bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax.axvline(x=avg_sim, color='red', linestyle='--', linewidth=2, label=f'Mean Sim: {avg_sim:.2f}')
    ax.set_title('Distribution of Pairwise Tanimoto Similarities')
    ax.set_xlabel('Tanimoto Similarity (Higher = More Similar)')
    ax.set_ylabel('Count of Pairs')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return internal_diversity, avg_sim, fig

@st.cache_data
def _get_fingerprints(smiles_list):
    """Helper for PCA: gets fingerprints."""
    fps = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                arr = np.zeros((0,), dtype=np.int8)
                DataStructs.ConvertToNumpyArray(fp, arr)
                fps.append(arr)
        except: continue
    return np.array(fps)

@st.cache_data
def analysis_pca():
    """Generates PCA plot of chemical space."""
    if not os.path.exists(Config.CHEMBL_DB_PATH):
        return "DB_NOT_FOUND"
    
    gen_path = Config.get_generated_csv_path()
    if not os.path.exists(gen_path):
        return "GEN_FILE_NOT_FOUND"
    
    try:
        conn = sqlite3.connect(Config.CHEMBL_DB_PATH)
        query = "SELECT canonical_smiles FROM compound_structures WHERE length(canonical_smiles) < 50 ORDER BY RANDOM() LIMIT 2000"
        real_smiles = [row[0] for row in conn.execute(query).fetchall()]
        conn.close()
    except Exception as e:
        return f"DB_ERROR: {e}"

    gen_smiles = pd.read_csv(gen_path)['smiles'].tolist()
    
    real_fps = _get_fingerprints(real_smiles)
    gen_fps = _get_fingerprints(gen_smiles)
    
    if len(real_fps) == 0 or len(gen_fps) == 0:
        return "FP_ERROR"

    all_fps = np.concatenate([real_fps, gen_fps], axis=0)
    pca = PCA(n_components=2)
    all_pcs = pca.fit_transform(all_fps)
    
    real_pcs = all_pcs[:len(real_fps)]
    gen_pcs = all_pcs[len(real_fps):]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(real_pcs[:, 0], real_pcs[:, 1], c='blue', alpha=0.3, label=f'Real ChEMBL ({len(real_pcs)})', s=15)
    ax.scatter(gen_pcs[:, 0], gen_pcs[:, 1], c='red', alpha=0.7, label=f'Generated ({len(gen_pcs)})', s=30, marker='^')
    ax.set_title('Chemical Space Visualization (PCA of Morgan Fingerprints)')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

@st.cache_data
def analysis_property_comparison():
    """Compares generated vs. real property distributions."""
    if not os.path.exists(Config.CHEMBL_DB_PATH):
        return "DB_NOT_FOUND", None
    
    gen_path = Config.get_generated_csv_path()
    if not os.path.exists(gen_path):
        return "GEN_FILE_NOT_FOUND", None

    gen_smiles = pd.read_csv(gen_path)['smiles'].tolist()

    try:
        conn = sqlite3.connect(Config.CHEMBL_DB_PATH)
        query = "SELECT canonical_smiles FROM compound_structures WHERE length(canonical_smiles) < 50 ORDER BY RANDOM() LIMIT 2000"
        real_smiles = [row[0] for row in conn.execute(query).fetchall()]
        conn.close()
    except Exception as e:
        return f"DB_ERROR: {e}", None

    def get_props(smiles_list):
        props = {'MW': [], 'LogP': [], 'QED': []}
        for smi in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    props['MW'].append(Descriptors.MolWt(mol))
                    props['LogP'].append(Descriptors.MolLogP(mol))
                    props['QED'].append(QED.qed(mol))
            except: continue
        return pd.DataFrame(props)

    df_real = get_props(real_smiles)
    df_gen = get_props(gen_smiles)
    
    df_real['Source'] = 'Real (Baseline)'
    df_gen['Source'] = 'Generated'
    df_combined = pd.concat([df_real, df_gen])
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.kdeplot(data=df_combined, x='MW', hue='Source', ax=axes[0], fill=True, common_norm=False)
    axes[0].set_title('Molecular Weight Distribution')

    sns.kdeplot(data=df_combined, x='LogP', hue='Source', ax=axes[1], fill=True, common_norm=False)
    axes[1].set_title('LogP (Lipophilicity) Distribution')

    sns.kdeplot(data=df_combined, x='QED', hue='Source', ax=axes[2], fill=True, common_norm=False)
    axes[2].set_title('QED (Drug-likeness) Distribution'); axes[2].set_xlim(0, 1)

    plt.tight_layout()
    return fig, df_combined.groupby('Source').mean()
@st.cache_data
def analysis_plot_loss_combined():
    """Plots training loss curves (G and D) on a single axis."""
    log_path = Config.get_log_path()
    if not os.path.exists(log_path):
        return None
    
    df = pd.read_csv(log_path)
    
    # Smoothing window (larger window = smoother lines)
    window = max(50, len(df) // 50) 
    
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot smoothed Critic Loss
    ax.plot(df['D_loss'].rolling(window).mean(), 
             color='tab:blue', linewidth=2.5, label='Critic Loss (D)')

    # Plot smoothed Generator Loss
    ax.plot(df['G_loss'].rolling(window).mean(), 
             color='tab:orange', linewidth=2.5, label='Generator Loss (G)')

    # Formatting
    ax.set_title(f'WGAN-GP Training Dynamics (Smoothed, Window={window})', fontsize=16)
    ax.set_xlabel('Training Iterations (Batches)', fontsize=12)
    ax.set_ylabel('Wasserstein Loss', fontsize=14)
    ax.legend(fontsize=12, frameon=True, facecolor='white', framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return fig

@st.cache_data
def analysis_scaffolds():
    """Analyzes scaffold novelty."""
    if not os.path.exists(Config.CHEMBL_DB_PATH):
        return "DB_NOT_FOUND", 0, 0
    
    gen_path = Config.get_generated_csv_path()
    if not os.path.exists(gen_path):
        return "GEN_FILE_NOT_FOUND", 0, 0

    def get_scaffolds(smiles_list):
        scaffolds = set()
        for s in smiles_list:
            try:
                mol = Chem.MolFromSmiles(s)
                if mol:
                    core = MurckoScaffold.GetScaffoldForMol(mol)
                    scaffolds.add(Chem.MolToSmiles(core))
            except: continue
        return scaffolds

    try:
        conn = sqlite3.connect(Config.CHEMBL_DB_PATH)
        query = "SELECT canonical_smiles FROM compound_structures WHERE length(canonical_smiles) < 50 LIMIT 50000"
        train_smiles = [row[0] for row in conn.execute(query).fetchall()]
        conn.close()
    except Exception as e:
        return f"DB_ERROR: {e}", 0, 0
    
    train_scaffolds = get_scaffolds(train_smiles)
    
    gen_smiles = pd.read_csv(gen_path)['smiles'].tolist()
    gen_scaffolds = get_scaffolds(gen_smiles)

    if len(gen_scaffolds) == 0:
        return 0, 0, 0

    novel_scaffolds = gen_scaffolds - train_scaffolds
    novelty_rate = (len(novel_scaffolds) / len(gen_scaffolds)) * 100
    
    return novelty_rate, len(novel_scaffolds), len(gen_scaffolds)

# =========================================
# STREAMLIT UI LAYOUT
# (Updated for better UI/UX)
# =========================================

st.set_page_config(
    page_title="WGAN-GP Molecule Generator",
    page_icon="🧬",
    layout="wide"
)

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("🔬 Generation Controls")
    temp_slider = st.slider(
        "Generation Temperature", 
        0.1, 2.0, 0.8, 0.1,
        help="Higher values = More diverse/random molecules"
    )
    
    st.markdown("---")
    st.header("ℹ️ Model Info")
    st.info(f"**Epoch:** {Config.BEST_EPOCH}\n\n**Device:** {Config.DEVICE}\n\n**SA Score:** {'Enabled' if SA_SCORE_ENABLED else 'Disabled'}")
    

# --- Main Page Title ---
st.title("WGAN-GP Molecule Generator")
st.markdown(f"An interactive dashboard to generate and analyze molecules from a WGAN-GP model.")


# --- Define Tabs ---
tab1, tab2 = st.tabs(["🧪 Generate Molecules", "📊 Analyze Model"])

# --- TAB 1: MOLECULE GENERATION ---
with tab1:
    st.header("Generate New Molecules")
    st.write("Click the button to generate a new, valid molecule using the trained WGAN-GP.")
    
    if st.button("Run Generator", type="primary", use_container_width=True):
        generator = load_generator(Config.BEST_EPOCH)
        if generator:
            with st.spinner("Generating molecule..."):
                mol, smiles = generate_single_molecule(generator, temp_slider)
                props = get_mol_properties(mol)
            
            st.subheader("Generated SMILES")
            st.code(smiles, language="smiles")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Visualization")
                try:
                    img = Draw.MolToImage(mol, size=(350, 350))
                    st.image(img)
                except Exception as e:
                    st.error(f"Error visualizing molecule: {e}")
            
            with col2:
                st.subheader("Calculated Properties")
                c1, c2 = st.columns(2)
                c1.metric("QED (Drug-likeness)", f"{props.get('QED', 0):.3f}")
                c2.metric("Molecular Weight (Da)", f"{props.get('MW', 0):.2f}")
                c1.metric("LogP (Lipophilicity)", f"{props.get('LogP', 0):.2f}")
                
                sa_score = props.get('SA_Score', "N/A")
                if isinstance(sa_score, float):
                    sa_score = f"{sa_score:.2f}"
                c2.metric("SA Score (Synthesizability)", sa_score)
                
                c1.metric("TPSA", f"{props.get('TPSA', 0):.2f}")
                c2.metric("H-Bond Acceptors", f"{props.get('HBA', 0)}")
                c1.metric("H-Bond Donors", f"{props.get('HBD', 0)}")


# --- TAB 2: MODEL ANALYSIS ---
# --- TAB 2: MODEL ANALYSIS ---
with tab2:
    st.header(f"Analysis of Model (Epoch {Config.BEST_EPOCH})")
    st.info(f"These analyses are based on the pre-generated file: `{Config.GENERATED_FILE}`")

    # --- THIS IS THE NEW EXPANDER ---
    with st.expander("📈 Training Loss Dynamics (Combined Plot)", expanded=True):
        combined_loss_fig = analysis_plot_loss_combined()
        if combined_loss_fig:
            st.pyplot(combined_loss_fig)
        else:
            st.warning(f"Could not find log file at {Config.get_log_path()}.")
    # --- END OF NEW EXPANDER ---

    with st.expander("📈 Training Loss Dynamics (Separate Plots)"):
        loss_fig = analysis_plot_loss()
        if loss_fig:
            st.pyplot(loss_fig)
        else:
            st.warning(f"Could not find log file at {Config.get_log_path()}.")

    with st.expander("🔬 Physicochemical Properties (of generated set)"):
        stats_df, props_fig = analysis_physchem()
        if props_fig:
            st.pyplot(props_fig)
            st.subheader("Summary Statistics")
            st.dataframe(stats_df.round(3), use_container_width=True)
        else:
            st.warning(f"Could not find generated file at {Config.get_generated_csv_path()}.")

    with st.expander("⚖️ Comparative Property Distributions (Generated vs. Real)"):
        comp_fig, comp_stats = analysis_property_comparison()
        if isinstance(comp_fig, str):
            st.warning(f"Error: {comp_fig}")
        elif comp_fig:
            st.pyplot(comp_fig)
            st.subheader("Mean Properties Comparison")
            st.dataframe(comp_stats.round(3))
        else:
            st.warning("Could not generate comparative plot.")
            
    with st.expander("🌌 Chemical Space Visualization (PCA)"):
        pca_fig = analysis_pca()
        if isinstance(pca_fig, str):
            st.warning(f"Error: {pca_fig}")
        elif pca_fig:
            st.pyplot(pca_fig)
        else:
            st.warning("Could not generate PCA plot.")

    with st.expander("✨ Novelty, Diversity & Scaffolds"):
        st.subheader("Key Performance Metrics")
        
        # --- Top-line metrics ---
        col1, col2, col3 = st.columns(3)
        with col1:
            score, novel_count, total_gen, samples = analysis_novelty()
            if isinstance(score, str):
                st.warning(f"Novelty Error: {score}")
            else:
                st.metric("Molecule Novelty", f"{score:.2f}%", f"{novel_count} / {total_gen} novel")
                
        with col2:
            diversity, avg_sim, div_fig = analysis_diversity()
            if diversity is not None:
                st.metric("Internal Diversity (1 - Sim)", f"{diversity:.4f}", help="0=all same, 1=all different")
                st.metric("Avg. Tanimoto Similarity", f"{avg_sim:.4f}")
            else:
                st.warning("Could not calculate diversity.")
                
        with col3:
            rate, novel_scaff_count, total_scaffs = analysis_scaffolds()
            if isinstance(rate, str):
                st.warning(f"Scaffold Error: {rate}")
            else:
                st.metric("Scaffold Novelty", f"{rate:.1f}%", f"{novel_scaff_count} / {total_scaffs} novel")
        
        st.markdown("---")
        
        # --- Plots and Details ---
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("Diversity Distribution")
            if div_fig:
                st.pyplot(div_fig, use_container_width=True)
        
        with c2:
            st.subheader("Novelty Details")
            if samples:
                st.write("Sample Novel Molecules (sorted by length, desc):")
                st.code("\n".join(samples))
            else:
                st.info("No novel molecules found or error in calculation.")