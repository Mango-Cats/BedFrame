"""
Drugsberter

This module generates sentence embeddings for pharmaceutical products by processing
drug descriptions and ATC (Anatomical Therapeutic Chemical) codes using a BioBERT-based
sentence transformer model. The embeddings are stored in HDF5 format for efficient
retrieval and similarity computations.

Pipeline stages:
1. Load configuration from TOML file
2. Load and validate drug data from CSV
3. Initialize pre-trained BioBERT model
4. Generate embeddings for descriptions and ATC codes separately
5. Save embeddings to HDF5 with metadata

Author: Zhean Robby Ganituen (zrygan)
Date: October 2025
"""

import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from sentence_transformers import SentenceTransformer
import torch
import tomllib
import pandas as pd
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


def load_configuration(config_path: Path) -> dict:
    """
    Load configuration parameters from TOML file.

    Args:
        config_path: Path to the configuration TOML file

    Returns:
        Dictionary containing configuration parameters including:
        - model.name: HuggingFace model identifier
        - data.csv_file: Path to input CSV file
        - data column mappings
        - output.hdf5_file: Path for output HDF5 file

    Raises:
        FileNotFoundError: If configuration file doesn't exist
        tomllib.TOMLDecodeError: If TOML file is malformed
    """
    print(f"Loading configuration from {config_path}...")
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    return config


def extract_config_parameters(config: dict) -> Tuple[str, str, str, str, str, str, str, bool, bool]:
    """
    Extract and validate required parameters from configuration dictionary.

    Args:
        config: Configuration dictionary loaded from TOML

    Returns:
        Tuple of (model_name, device, csv_file, generic_name_col, drug_name_col, desc_col, atc_col, hdf5_file, enable_atc)

    Raises:
        KeyError: If required configuration keys are missing
    """
    model_name = config["model"]["name"]
    device = config["model"]["device"]
    csv_file = config["data"]["csv_file"]
    hdf5_file = config["output"]["hdf5_file"]
    enable_atc = config.get("embeddings", {}).get("atc", True)
    generic_name_col = "Generic Name"
    drug_name_col = "Drug Name"
    desc_col = "Description"
    atc_col = "ATC Codes"

    print(f"Model: {model_name}")
    print(f"CSV File: {csv_file}")
    print(f"Output HDF5: {hdf5_file}")
    print(f"Generate ATC embeddings: {enable_atc}")

    return model_name, device, csv_file, generic_name_col, drug_name_col, desc_col, atc_col, hdf5_file, enable_atc


def load_and_process_data(
    csv_file: str, generic_col: str, drug_col: str, desc_col: str, atc_col: str
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Load CSV file and create mappings from drug names to descriptions and ATC codes.

    Args:
        csv_file: Path to CSV file containing drug data
        generic_col: Name of the column containing generic names
        drug_col: Name of the column containing drug names (brand names)
        desc_col: Name of the column containing drug descriptions
        atc_col: Name of the column containing ATC classification codes

    Returns:
        Tuple of two dictionaries:
        - drug_to_description: Maps drug names to their descriptions
        - drug_to_atc: Maps drug names to their ATC codes

    Notes:
        Only non-empty descriptions and ATC codes are included in the mappings.
        This allows handling of incomplete data where some drugs may lack
        descriptions or ATC classifications.
        If brand name is empty, "unbranded", or "none", the generic name is used instead.
    """
    print(f"\nLoading drug data from {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows from CSV")

    drug_to_description = {}
    drug_to_atc = {}

    for idx, row in df.iterrows():
        generic_name = str(row[generic_col]) if pd.notna(row[generic_col]) else ""
        drug_name = str(row[drug_col]) if pd.notna(row[drug_col]) else ""
        description = str(row[desc_col]) if pd.notna(row[desc_col]) else ""
        atc_codes = str(row[atc_col]) if (atc_col in df.columns and pd.notna(row[atc_col])) else ""

        # Default to generic name if brand name is empty, "unbranded", or "none"
        if drug_name.strip().lower() in ["", "unbranded", "none", "none reformulated"]:
            if not generic_name.strip():
                raise ValueError(
                    f"Error at row {idx}: Brand name is invalid ('{row[drug_col]}') "
                    f"and generic name is empty. Cannot process this drug entry."
                )
            drug_name = generic_name

        # Create a unique key combining generic name and drug name to handle
        # cases where the same brand name is used for different drugs
        # (e.g., "healthcare" is used for amoxicillin, cotrimoxazole, etc.)
        unique_key = f"{generic_name.strip().lower()}|{drug_name.strip().lower()}"

        # Store non-empty descriptions
        if description.strip():
            drug_to_description[unique_key] = description.strip()

        # Store non-empty ATC codes
        if atc_codes.strip():
            drug_to_atc[unique_key] = atc_codes.strip()

    print(f"Drugs with descriptions: {len(drug_to_description)}")
    print(f"Drugs with ATC codes: {len(drug_to_atc)}")

    return drug_to_description, drug_to_atc


def initialize_model(model_name: str, device: str = "") -> Tuple[SentenceTransformer, str]:
    """
    Load pre-trained sentence transformer model and determine compute device.

    Args:
        model_name: HuggingFace model identifier
        device: Device, optional
    Returns:
        loaded_model

    Notes:
        The model is expected to be a BioBERT variant fine-tuned on medical/scientific
        text for generating semantically meaningful embeddings of pharmaceutical descriptions.
        CUDA will be used if available for faster embedding generation.
    """
    print(f"\nLoading model: {model_name}...")
    if torch.cuda.is_available():
        selected_device = "cuda"
    elif device:
        selected_device = device
    else:
        selected_device = "cpu"
    model = SentenceTransformer(model_name, selected_device)
    print(f"Model loaded on device: {device}")
    return model


def encode_texts(
    texts: List[str], model: SentenceTransformer, show_progress: bool = True
) -> torch.Tensor:
    """
    Generate embeddings for a list of text strings.

    Args:
        texts: List of text strings to encode
        model: Loaded SentenceTransformer model
        show_progress: Whether to display encoding progress bar

    Returns:
        Tensor of shape (n_texts, embedding_dim) containing dense vector representations

    Notes:
        Embeddings are returned as PyTorch tensors for efficient GPU operations.
        They can be converted to numpy arrays using .cpu().numpy() if needed.
        The embedding dimension is determined by the model architecture (typically 768 for BERT).
    """
    return model.encode(texts, convert_to_tensor=True, show_progress_bar=show_progress)


def generate_description_embeddings(
    drug_to_description: Dict[str, str], model: SentenceTransformer
) -> Tuple[List[str], torch.Tensor]:
    """
    Generate embeddings for drug descriptions.

    Args:
        drug_to_description: Dictionary mapping drug names to description text
        model: Loaded SentenceTransformer model

    Returns:
        Tuple of (drug_names, embeddings)
        - drug_names: List of drug names in the same order as embeddings
        - embeddings: Tensor of shape (n_drugs, embedding_dim)

    Notes:
        Descriptions are encoded in batch for efficiency. The order of drugs
        in the returned list corresponds to the row order in the embedding tensor.
    """
    print("\n=== Processing Description Embeddings ===")
    drug_names = list(drug_to_description.keys())
    descriptions = [drug_to_description[name] for name in drug_names]

    print(f"Generating embeddings for {len(descriptions)} drug descriptions...")
    embeddings = encode_texts(descriptions, model, show_progress=True)
    print(
        f"Generated {len(embeddings)} description embeddings with shape: {embeddings.shape}"
    )

    return drug_names, embeddings


def generate_atc_embeddings(
    drug_to_atc: Dict[str, str], model: SentenceTransformer
) -> Tuple[List[str], torch.Tensor]:
    """
    Generate embeddings for ATC (Anatomical Therapeutic Chemical) codes.

    Args:
        drug_to_atc: Dictionary mapping drug names to ATC code strings
        model: Loaded SentenceTransformer model

    Returns:
        Tuple of (drug_names, embeddings)
        - drug_names: List of drug names in the same order as embeddings
        - embeddings: Tensor of shape (n_drugs, embedding_dim)

    Notes:
        ATC codes are hierarchical classification codes (e.g., "N02BE01; N02BE51").
        Multiple codes are separated by semicolons and encoded as a single string
        to capture their combined semantic meaning.
    """
    print("\n=== Processing ATC Code Embeddings ===")
    drug_names = list(drug_to_atc.keys())
    atc_codes = [drug_to_atc[name] for name in drug_names]

    print(f"Generating embeddings for {len(atc_codes)} drug ATC codes...")
    embeddings = encode_texts(atc_codes, model, show_progress=True)
    print(f"Generated {len(embeddings)} ATC embeddings with shape: {embeddings.shape}")

    return drug_names, embeddings


def build_embedding_databases(
    desc_drug_names: List[str],
    desc_embeddings: torch.Tensor,
    atc_drug_names: List[str] = None,
    atc_embeddings: torch.Tensor = None,
):
    """
    Create lookup dictionaries mapping drug names to their embeddings.

    Args:
        desc_drug_names: List of drug names with descriptions
        desc_embeddings: Tensor of description embeddings
        atc_drug_names: List of drug names with ATC codes (optional)
        atc_embeddings: Tensor of ATC embeddings (optional)

    Returns:
        Tuple of (description_db, atc_db)
        - description_db: Dictionary mapping drug_name -> embedding_tensor
        - atc_db: Dictionary mapping drug_name -> embedding_tensor, or None if not provided

    Notes:
        These databases enable O(1) lookup of embeddings by drug name,
        useful for similarity searches and retrieval operations.
    """
    print("\nBuilding drug embedding databases...")

    # Build description embeddings database
    description_db = {
        drug_name: desc_embeddings[i] for i, drug_name in enumerate(desc_drug_names)
    }

    print(f"Description embeddings: {len(description_db)} drugs")

    # Build ATC embeddings database if provided
    if atc_drug_names is not None and atc_embeddings is not None and len(atc_drug_names) > 0:
        atc_db = {
            drug_name: atc_embeddings[i] for i, drug_name in enumerate(atc_drug_names)
        }
        print(f"ATC embeddings: {len(atc_db)} drugs")
    else:
        atc_db = None
        print("ATC embeddings: skipped")

    return description_db, atc_db


def prepare_embedding_arrays(
    embedding_db,
):
    """
    Convert embedding dictionary to numpy arrays for HDF5 storage.

    Args:
        embedding_db: Dictionary mapping drug names to embedding tensors, or None

    Returns:
        Tuple of (drug_names_list, embeddings_matrix) or (None, None) if input is None
        - drug_names_list: Ordered list of drug names
        - embeddings_matrix: 2D numpy array of shape (n_drugs, embedding_dim)

    Notes:
        Ensures all embeddings are float32 dtype for storage efficiency
        and compatibility with downstream similarity computation libraries.
    """
    if embedding_db is None:
        return None, None
    
    drug_names = []
    embedding_arrays = []

    for drug_name, embedding in embedding_db.items():
        drug_names.append(str(drug_name))
        emb_array = embedding.cpu().numpy()
        # Ensure consistent dtype
        if emb_array.dtype == object:
            emb_array = np.array(emb_array, dtype=np.float32)
        embedding_arrays.append(emb_array)

    # Stack into single matrix
    embeddings_matrix = np.stack(embedding_arrays).astype(np.float32)

    return drug_names, embeddings_matrix


def save_embeddings_hdf5(
    hdf5_file: str,
    model_name: str,
    description_db: Dict[str, torch.Tensor],
    atc_db=None,
) -> None:
    """
    Save drug embeddings to HDF5 file with compression and metadata.

    Args:
        hdf5_file: Path to output HDF5 file
        model_name: Name of the model used to generate embeddings
        description_db: Dictionary of drug name -> description embedding
        atc_db: Dictionary of drug name -> ATC code embedding (optional)

    File Structure:
        /description/
            drug_names: Variable-length string dataset
            embeddings: 2D float32 array (n_drugs, embedding_dim)
            @num_drugs: Number of drugs with descriptions
            @embedding_dim: Dimensionality of embeddings
        /atc/ (only if atc_db is provided)
            drug_names: Variable-length string dataset
            embeddings: 2D float32 array (n_drugs, embedding_dim)
            @num_drugs: Number of drugs with ATC codes
            @embedding_dim: Dimensionality of embeddings
        Global attributes:
            @model_name: Model identifier
            @description: File description
            @total_drugs_with_descriptions: Count
            @total_drugs_with_atc: Count (0 if not provided)

    Notes:
        Uses gzip compression level 9 to minimize file size while maintaining
        reasonable decompression speed. Variable-length strings used for drug names
        to handle names of varying lengths efficiently.
    """
    print(f"\nSaving embeddings to HDF5 format: {hdf5_file}...")

    # Prepare description embeddings
    desc_drug_names, desc_embeddings_matrix = prepare_embedding_arrays(description_db)
    desc_embedding_dim = desc_embeddings_matrix.shape[1]

    # Prepare ATC embeddings (optional)
    atc_drug_names, atc_embeddings_matrix = prepare_embedding_arrays(atc_db)

    # Save to HDF5
    with h5py.File(hdf5_file, "w") as f:
        # Variable-length string dtype for drug names
        dt = h5py.special_dtype(vlen=str)

        # Create description group
        desc_group = f.create_group("description")
        desc_names_ds = desc_group.create_dataset(
            "drug_names", (len(desc_drug_names),), dtype=dt
        )
        desc_names_ds[:] = desc_drug_names
        desc_emb_ds = desc_group.create_dataset(
            "embeddings",
            data=desc_embeddings_matrix,
            dtype=np.float32,
            compression="gzip",
            compression_opts=9,
        )
        desc_group.attrs["num_drugs"] = len(desc_drug_names)
        desc_group.attrs["embedding_dim"] = desc_embedding_dim

        # Create ATC group (only if ATC embeddings were provided)
        if atc_embeddings_matrix is not None:
            atc_embedding_dim = atc_embeddings_matrix.shape[1]
            atc_group = f.create_group("atc")
            atc_names_ds = atc_group.create_dataset(
                "drug_names", (len(atc_drug_names),), dtype=dt
            )
            atc_names_ds[:] = atc_drug_names
            atc_emb_ds = atc_group.create_dataset(
                "embeddings",
                data=atc_embeddings_matrix,
                dtype=np.float32,
                compression="gzip",
                compression_opts=9,
            )
            atc_group.attrs["num_drugs"] = len(atc_drug_names)
            atc_group.attrs["embedding_dim"] = atc_embedding_dim
            num_atc = len(atc_drug_names)
        else:
            num_atc = 0

        # Add global metadata
        f.attrs["model_name"] = model_name
        f.attrs["description"] = (
            "Drug embeddings: separate mappings for descriptions and ATC codes"
        )
        f.attrs["total_drugs_with_descriptions"] = len(desc_drug_names)
        f.attrs["total_drugs_with_atc"] = num_atc

    file_size = Path(hdf5_file).stat().st_size / 1024 / 1024
    print(f"  Successfully created {hdf5_file}")
    print(
        f"  Description embeddings: {len(desc_drug_names)} drugs, dim={desc_embedding_dim}"
    )
    if atc_embeddings_matrix is not None:
        print(f"  ATC embeddings: {len(atc_drug_names)} drugs, dim={atc_embedding_dim}")
    else:
        print(f"  ATC embeddings: skipped")
    print(f"  File size: {file_size:.2f} MB")

def main():
    """
    Main execution pipeline for generating drug embeddings.

    Pipeline stages:
    1. Load configuration from TOML
    2. Load and process drug data from CSV
    3. Initialize BioBERT sentence transformer model
    4. Generate embeddings for descriptions
    5. Generate embeddings for ATC codes
    6. Build lookup databases
    7. Save embeddings to HDF5 with compression

    Raises:
        FileNotFoundError: If configuration or CSV file not found
        KeyError: If required configuration keys are missing
        Exception: For other errors during processing
    """
    # Load configuration
    config_path = Path(__file__).parent / "config.toml"
    config = load_configuration(config_path)

    # Extract parameters
    model_name, device, csv_file, generic_name_col, drug_name_col, desc_col, atc_col, hdf5_file, enable_atc = (
        extract_config_parameters(config)
    )

    # Load and process data
    try:
        drug_to_description, drug_to_atc = load_and_process_data(
            csv_file, generic_name_col, drug_name_col, desc_col, atc_col
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    # Initialize model
    model = initialize_model(model_name, device)

    # Generate description embeddings
    desc_drug_names, desc_embeddings = generate_description_embeddings(
        drug_to_description, model
    )

    # Generate ATC embeddings only if enabled in config and we have ATC data
    atc_drug_names = None
    atc_embeddings = None
    if enable_atc and len(drug_to_atc) > 0:
        atc_drug_names, atc_embeddings = generate_atc_embeddings(drug_to_atc, model)
    elif not enable_atc:
        print("\n=== ATC Embeddings Disabled ===\nSkipping ATC code processing (disabled in config).")
    else:
        print("\n=== Skipping ATC Embeddings ===\nNo ATC codes found in input data.")

    # Build embedding databases
    description_db, atc_db = build_embedding_databases(
        desc_drug_names, desc_embeddings,
        atc_drug_names, atc_embeddings
    )

    # Save to HDF5
    try:
        save_embeddings_hdf5(hdf5_file, model_name, description_db, atc_db)
    except Exception as e:
        print(f"Error saving HDF5 file: {e}")
        raise


if __name__ == "__main__":
    main()
