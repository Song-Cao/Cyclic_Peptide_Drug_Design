import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from Bio.SeqUtils import ProtParam
from sklearn.preprocessing import StandardScaler
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Define constants
DATA_SOURCE = 'cyclic_pep_101023.csv'
OUTPUT_CLEANED = 'cleaned_cyclic_peptide_data.csv'
PROTEIN_FEATURES = [
    'MolecularWeight', 'Aromaticity', 'InstabilityIndex',
    'IsoelectricPoint', 'Hydrophobicity', 'ChargeAtPH7'
]
SMILES_FEATURES = [
    'MolecularWeight', 'LogP', 'NumHAcceptors', 'NumHDonors',
    'TPSA', 'NumRotatableBonds', 'NumAromaticRings'
]

def load_dataset(file_path):
    """
    Load dataset from a CSV file.
    """
    df = pd.read_csv(file_path)
    logging.info(f"Dataset loaded with {len(df)} rows.")
    return df

def validate_sequence(sequence):
    """
    Validate protein sequence.
    """
    valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    return set(sequence.upper()).issubset(valid_amino_acids)

def validate_smiles(smiles):
    """
    Validate molecular structure SMILES.
    """
    return Chem.MolFromSmiles(smiles) is not None

def clean_dataset(df):
    """
    Clean dataset by validating sequences and structures.
    """
    logging.info("Cleaning dataset...")
    original_count = len(df)
    
    # Process protein sequences
    if 'Sequence' in df.columns:
        df['Valid_Sequence'] = df['Sequence'].apply(lambda x: validate_sequence(x) if pd.notnull(x) else False)
    
    # Process SMILES strings
    if 'SMILES' in df.columns:
        df['Valid_SMILES'] = df['SMILES'].apply(lambda x: validate_smiles(x) if pd.notnull(x) else False)
    
    # Retain valid rows
    if 'Valid_Sequence' in df.columns and 'Valid_SMILES' in df.columns:
        df = df[df['Valid_Sequence'] | df['Valid_SMILES']]
        df = df.drop(columns=['Valid_Sequence', 'Valid_SMILES'])
    elif 'Valid_Sequence' in df.columns:
        df = df[df['Valid_Sequence']].drop(columns=['Valid_Sequence'])
    elif 'Valid_SMILES' in df.columns:
        df = df[df['Valid_SMILES']].drop(columns=['Valid_SMILES'])

    cleaned_count = len(df)
    logging.info(f"Removed {original_count - cleaned_count} invalid rows.")
    return df

def compute_protein_features(sequence):
    """
    Compute features for protein sequences.
    """
    try:
        analyzer = ProtParam.ProteinAnalysis(sequence)
        features = {
            'MolecularWeight': analyzer.molecular_weight(),
            'Aromaticity': analyzer.aromaticity(),
            'InstabilityIndex': analyzer.instability_index(),
            'IsoelectricPoint': analyzer.isoelectric_point(),
            'Hydrophobicity': np.mean(analyzer.protein_scale(ProtParam.kd, len(sequence))),
            'ChargeAtPH7': analyzer.charge_at_pH(7.0)
        }
        return features
    except Exception as e:
        logging.error(f"Error computing features for sequence {sequence}: {e}")
        return {key: np.nan for key in PROTEIN_FEATURES}

def compute_smiles_features(smiles):
    """
    Compute molecular features for SMILES strings.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {key: np.nan for key in SMILES_FEATURES}
        features = {
            'MolecularWeight': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'NumAromaticRings': Descriptors.NumAromaticRings(mol)
        }
        return features
    except Exception as e:
        logging.error(f"Error computing features for SMILES {smiles}: {e}")
        return {key: np.nan for key in SMILES_FEATURES}

def process_dataset(df):
    """
    Compute features for both sequences and SMILES strings.
    """
    logging.info("Processing dataset to compute features...")
    feature_data = []
    for _, row in df.iterrows():
        if pd.notnull(row.get('Sequence')):
            feature_data.append(compute_protein_features(row['Sequence']))
        elif pd.notnull(row.get('SMILES')):
            feature_data.append(compute_smiles_features(row['SMILES']))
    feature_df = pd.DataFrame(feature_data)
    return pd.concat([df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)

def normalize_data(df, columns):
    """
    Normalize specified columns in the dataframe.
    """
    logging.info("Normalizing features...")
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

def save_dataset(df, output_path):
    """
    Save the processed dataset to a CSV file.
    """
    logging.info(f"Saving processed dataset to {output_path}...")
    df.to_csv(output_path, index=False)
    logging.info("Dataset saved successfully.")

def main():
    df = load_dataset(DATA_SOURCE)
    df = clean_dataset(df)
    df = process_dataset(df)
    all_features = PROTEIN_FEATURES + SMILES_FEATURES
    df = normalize_data(df, [col for col in all_features if col in df.columns])
    save_dataset(df, OUTPUT_CLEANED)

if __name__ == "__main__":
    main()
