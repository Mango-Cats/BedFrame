"""
Drug Embedding Evaluation Script

This script evaluates drug embeddings by computing cosine similarities between drug vectors.
It analyzes similarity distributions, identifies top/middle/bottom similar pairs, and generates
per-drug similarity reports.

Usage:
    python main.py --hdf5 <path_to_embeddings> --input <csv_files>...
    python main.py --hdf5 drugsberted.h5 --input data/amoxicillin_desc.csv data/nsaid_desc.csv
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import h5py
import sys


def load_embeddings(hdf5_path, embedding_type="description"):
    """
    Load drug embeddings from an HDF5 file.

    Args:
        hdf5_path (str): Path to the HDF5 file containing embeddings
        embedding_type (str): Type of embeddings to load (default: "description")

    Returns:
        tuple: (drug_names, embeddings) where drug_names is a list of drug names
               and embeddings is a numpy array of shape (n_drugs, embedding_dim)

    Raises:
        FileNotFoundError: If the HDF5 file doesn't exist
        KeyError: If the embedding_type doesn't exist in the file
    """
    """
    Load drug embeddings from an HDF5 file.

    Args:
        hdf5_path (str): Path to the HDF5 file containing embeddings
        embedding_type (str): Type of embeddings to load (default: "description")

    Returns:
        tuple: (drug_names, embeddings) where drug_names is a list of drug names
               and embeddings is a numpy array of shape (n_drugs, embedding_dim)

    Raises:
        FileNotFoundError: If the HDF5 file doesn't exist
        KeyError: If the embedding_type doesn't exist in the file
    """
    if not Path(hdf5_path).exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
    
    with h5py.File(hdf5_path, "r") as f:
        if embedding_type not in f:
            raise KeyError(f"Embedding type '{embedding_type}' not found in {hdf5_path}")
        
        group = f[embedding_type]

        drug_names = [name.decode("utf-8") for name in group["drug_names"][:]]
        embeddings = group["embeddings"][:]

        print(f"Loaded {len(drug_names)} {embedding_type} embeddings")
        print(f"Embedding shape: {embeddings.shape}")

    return drug_names, embeddings


def get_embeddings(drug_names_to_find, all_drug_names, all_embeddings):
    """
    Extract embeddings for specific drugs from a larger embedding dataset.

    Args:
        drug_names_to_find (list): List of drug names to find embeddings for
        all_drug_names (list): List of all available drug names
        all_embeddings (np.ndarray): Array of all embeddings

    Returns:
        tuple: (found_embeddings, found_names) containing the embeddings and names
               of drugs that were successfully found
    """
    """
    Extract embeddings for specific drugs from a larger embedding dataset.

    Args:
        drug_names_to_find (list): List of drug names to find embeddings for
        all_drug_names (list): List of all available drug names
        all_embeddings (np.ndarray): Array of all embeddings

    Returns:
        tuple: (found_embeddings, found_names) containing the embeddings and names
               of drugs that were successfully found
    """
    name_to_idx = {name: idx for idx, name in enumerate(all_drug_names)}

    found_embeddings = []
    found_names = []

    for drug_name in drug_names_to_find:
        if drug_name in name_to_idx:
            idx = name_to_idx[drug_name]
            found_embeddings.append(all_embeddings[idx])
            found_names.append(drug_name)
        else:
            print(f"Warning: {drug_name} not found in embeddings")

    return np.array(found_embeddings), found_names


def pair_cosine_similarity(vectors):
    """
    Compute pairwise cosine similarity between vectors.

    Args:
        vectors (np.ndarray): Array of vectors of shape (n_vectors, vector_dim)

    Returns:
        np.ndarray: Similarity matrix of shape (n_vectors, n_vectors) with values in [0, 1]
    """
    """
    Compute pairwise cosine similarity between vectors.

    Args:
        vectors (np.ndarray): Array of vectors of shape (n_vectors, vector_dim)

    Returns:
        np.ndarray: Similarity matrix of shape (n_vectors, n_vectors) with values in [0, 1]
    """
    similarity_matrix = cosine_similarity(vectors)
    similarity_matrix = np.clip(similarity_matrix, 0, 1)

    return similarity_matrix


# to cross check between the generic names in the csv and the provided predictions
def accuracy_checker(predictions, csv_name):
    """
    Compare predicted drug names against actual values in a CSV file.

    Args:
        predictions (list): List of predicted drug names
        csv_name (str): Name of CSV file containing actual drug names (in data/ directory)

    Returns:
        float: Accuracy as a percentage (0-1)
    """
    """
    Compare predicted drug names against actual values in a CSV file.

    Args:
        predictions (list): List of predicted drug names
        csv_name (str): Name of CSV file containing actual drug names (in data/ directory)

    Returns:
        float: Accuracy as a percentage (0-1)
    """
    df = pd.read_csv(f"data/{csv_name}", dtype=str, header=None).fillna("")
    actual_drugs = df.iloc[:, 0]
    correct = 0

    for i in range(len(predictions)):
        if predictions[i] == actual_drugs.iloc[i]:
            correct += 1

    accuracy = correct / len(predictions)
    return accuracy


# get the top n pairs, middle n pairs, bottom n pairs
def levels(vectors, n, drug_names):
    """
    Analyze and display similarity statistics for top, middle, and bottom drug pairs.

    This function computes pairwise similarities between all drugs and identifies
    the most similar, moderately similar, and least similar pairs.

    Args:
        vectors (np.ndarray): Drug embedding vectors of shape (n_drugs, embedding_dim)
        n (int): Number of drugs/pairs to analyze
        drug_names (list): List of drug names corresponding to vectors

    Prints:
        Statistics about top, middle, and bottom similarity pairs including
        individual pair similarities and mean cosine similarities
    """
    similarity_matrix = cosine_similarity(vectors)
    similarity_matrix = np.clip(similarity_matrix, 0, 1)

    upper_triangle_indices = np.triu_indices_from(similarity_matrix, k=1)
    similarities = similarity_matrix[upper_triangle_indices]

    pairs = []
    for idx, sim in enumerate(similarities):
        i, j = upper_triangle_indices[0][idx], upper_triangle_indices[1][idx]
        pairs.append((drug_names[i], drug_names[j], sim))

    sorted_sims = sorted(pairs, key=lambda x: x[2], reverse=True)

    top_n = sorted_sims[:n]
    bott_n = sorted_sims[-n:]
    middle_start = len(similarities) // 2
    mid_n = sorted_sims[middle_start : middle_start + n]

    print(f"Top, Middle, and Bottom Pairs of the Top {n} Pairs:")
    print(f"Top pair: {top_n[0][0]} <-> {top_n[0][1]} = {top_n[0][2]}")
    print(f"Middle pair: {top_n[n//2][0]} <-> {top_n[n//2][1]} = {top_n[n//2][2]}")
    print(f"Bottom pair: {top_n[n-1][0]} <-> {top_n[n-1][1]} = {top_n[n-1][2]}")
    # for i, (drug1, drug2, sim) in enumerate(islice(top_n, 3)):
    #     print(f"{i+1:2d}. {drug1} <-> {drug2} = {sim:.4f}")
    print("\n")
    print(f"Top, Middle, and Bottom Pairs of the Middle {n} Pairs:")
    print(f"Top pair: {mid_n[0][0]} <-> {mid_n[0][1]} = {mid_n[0][2]}")
    print(f"Middle pair: {mid_n[n//2][0]} <-> {mid_n[n//2][1]} = {mid_n[n//2][2]}")
    print(f"Bottom pair: {mid_n[n-1][0]} <-> {mid_n[n-1][1]} = {mid_n[n-1][2]}")
    # for i, (drug1, drug2, sim) in enumerate(islice(mid_n, 3)):
    #     print(f"{i+1:2d}. {drug1} <-> {drug2} = {sim:.4f}")
    print("\n")
    print(f"Top, Middle, and Bottom Pairs of the Bottom {n} Pairs:")
    print(f"Top pair: {bott_n[0][0]} <-> {bott_n[0][1]} = {bott_n[0][2]}")
    print(f"Middle pair: {bott_n[n//2][0]} <-> {bott_n[n//2][1]} = {bott_n[n//2][2]}")
    print(f"Bottom pair: {bott_n[n-1][0]} <-> {bott_n[n-1][1]} = {bott_n[n-1][2]}")
    # for i, (drug1, drug2, sim) in enumerate(islice(bott_n[::-1], 3)):
    #     print(f"{i+1:2d}. {drug1} <-> {drug2} = {sim:.4f}")
    print("\n")

    top_n_cs = np.mean([pair[2] for pair in top_n])
    mid_n_cs = np.mean([pair[2] for pair in mid_n])
    bott_n_cs = np.mean([pair[2] for pair in bott_n])

    print(f"Mean of the Cosine Similarities of the TOP {n} pairs: {top_n_cs:.4f}")
    print(f"Mean of the Cosine Similarities of the MIDDLE {n} pairs: {mid_n_cs:.4f}")
    print(f"Mean of the Cosine Similarities of the BOTTOM {n} pairs: {bott_n_cs:.4f}")
    print("\n")


# get mean, std dev, min, and max of vector pairs in the sim matrix
def get_vector_analysis(vectors):
    """
    Compute and display statistical analysis of cosine similarities between vectors.

    Args:
        vectors (np.ndarray): Drug embedding vectors of shape (n_drugs, embedding_dim)

    Prints:
        Mean, standard deviation, minimum, and maximum cosine similarity values
        for all unique drug pairs (excluding self-similarities)
    """
    similarity_matrix = cosine_similarity(vectors)
    similarity_matrix = np.clip(similarity_matrix, 0, 1)

    upper_triangle_indices = np.triu_indices_from(similarity_matrix, k=1)
    non_self_similarities = similarity_matrix[upper_triangle_indices]

    print(f"\nCS matrix shape: {similarity_matrix.shape}")
    print(f"CS range: [{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]")

    print(f"\nMean: {np.nanmean(non_self_similarities):.4f}")
    print(f"Std Dev:  {np.nanstd(non_self_similarities):.4f}")
    print(f"Min:  {np.nanmin(non_self_similarities):.4f}")
    print(f"Max:  {np.nanmax(non_self_similarities):.4f}")
    print("\n")


def sim_per_drug(vectors, drug_names, n, name):
    """
    Calculate average similarity for each drug and save results to CSV.

    For each drug, computes its average cosine similarity with all other drugs
    and saves the results sorted by similarity.

    Args:
        vectors (np.ndarray): Drug embedding vectors of shape (n_drugs, embedding_dim)
        drug_names (list): List of drug names corresponding to vectors
        n (int): Number of drugs
        name (str): Base filename for output CSV (saved in sbert_results/)

    Outputs:
        CSV file: sbert_results/{name}_sims.csv with columns [Drug, Avg_Similarity]
    """
    similarity_matrix = cosine_similarity(vectors)
    similarity_matrix = np.clip(similarity_matrix, 0, 1)

    upper_triangle_indices = np.triu_indices_from(similarity_matrix, k=1)
    similarities = similarity_matrix[upper_triangle_indices]

    drug_sims = []
    for i in range(n):
        similarities = []
        for j in range(n):
            if i != j:
                similarities.append(similarity_matrix[i, j])

        if similarities:
            avg = np.mean(similarities)

        drug_sims.append(avg)

    drug_sim_df = pd.DataFrame({"Drug": drug_names, "Avg_Similarity": drug_sims})
    drug_sim_df = drug_sim_df.sort_values("Avg_Similarity", ascending=False)

    file_path = Path("sbert_results")
    file_path.mkdir(parents=True, exist_ok=True)
    drug_sim_df.to_csv(f"sbert_results/{name}_sims.csv", index=False)


def drug_tests(csv_name, hdf5_path, embedding_type="description"):
    """
    Perform comprehensive similarity analysis for drugs in a CSV file.

    Loads drug data from CSV, retrieves corresponding embeddings, and performs
    multiple analyses including similarity statistics and pair analysis.

    Args:
        csv_name (str): Path to CSV file containing drug data with columns:
                       [Generic Name, Drug Name, Description, ATC Codes]
        hdf5_path (str): Path to HDF5 file containing drug embeddings
        embedding_type (str): Type of embeddings to use (default: "description")

    Prints:
        Comprehensive analysis including:
        - Vector analysis statistics
        - Top/middle/bottom similarity pairs
        - Per-drug similarity scores (saved to CSV)
    """
    # Extract test name from filename
    csv_path = Path(csv_name)
    test_drug_name = csv_path.stem.split("_")[0]
    
    drug_df = pd.read_csv(csv_name, dtype=str, header=0).fillna("")

    # Rename columns to standard format (CSV has header: generic_name, drug_name, description)
    drug_df.columns = ["Generic Name", "Drug Name", "Description"]

    # Filter and fix brand names
    processed_rows = []
    for idx, row in drug_df.iterrows():
        generic_name = str(row["Generic Name"])
        drug_name = str(row["Drug Name"])
        
        # Default to generic name if brand name is empty, "unbranded", or "none"
        if drug_name.strip().lower() in ["", "unbranded", "none", "none reformulated"]:
            if not generic_name.strip():
                print(f"Warning: Skipping row {idx} - both brand and generic names are invalid")
                continue
            drug_name = generic_name
            row["Drug Name"] = drug_name
        
        processed_rows.append(row)
    
    drug_df = pd.DataFrame(processed_rows).reset_index(drop=True)

    drug_df["combined_drug"] = drug_df["Generic Name"] + "/" + drug_df["Drug Name"]
    drug_df["combined_text"] = drug_df["Drug Name"] + " " + drug_df["Description"]
    
    # Create unique keys matching the format used in drugsberter for embedding storage
    # This handles cases where the same brand name is used for different drugs
    drug_df["unique_key"] = drug_df["Generic Name"].str.strip().str.lower() + "|" + drug_df["Drug Name"].str.strip().str.lower()

    all_drug_names, all_embeddings = load_embeddings(hdf5_path, embedding_type)

    # Use unique keys for lookup, but keep combined_drug for display
    test_drug_keys = drug_df["unique_key"].tolist()
    display_names = drug_df["combined_drug"].tolist()

    drug_vec, found_keys = get_embeddings(
        test_drug_keys, all_drug_names, all_embeddings
    )
    
    # Map found keys back to display names for output
    key_to_display = dict(zip(test_drug_keys, display_names))
    found_display_names = [key_to_display.get(key, key) for key in found_keys]

    print("-" * 67)
    print(f"BERT TEST: {test_drug_name.upper()}")
    get_vector_analysis(drug_vec)
    levels(drug_vec, len(drug_vec), found_display_names)
    sim_per_drug(drug_vec, found_display_names, len(drug_vec), f"{test_drug_name}_bert")


def parse_arguments():
    """
    Parse command-line arguments for the evaluation script.

    Returns:
        argparse.Namespace: Parsed arguments containing:
            - hdf5: Path to HDF5 embeddings file
            - input: List of CSV files to process
            - embedding_type: Type of embeddings to use
    """
    parser = argparse.ArgumentParser(
        description="Evaluate drug embeddings by computing cosine similarities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single CSV file
  python main.py --hdf5 drugsberted.h5 --input data/amoxicillin_desc.csv

  # Analyze multiple CSV files
  python main.py --hdf5 drugsberted.h5 --input data/amoxicillin_desc.csv data/nsaid_desc.csv

  # Use custom embedding type
  python main.py --hdf5 drugsberted.h5 --input data/amoxicillin_desc.csv --embedding-type custom
        """
    )
    
    parser.add_argument(
        "--hdf5",
        type=str,
        required=True,
        help="Path to the HDF5 file containing drug embeddings"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        required=True,
        help="One or more CSV files to analyze (can be paths or filenames in data/)"
    )
    
    parser.add_argument(
        "--embedding-type",
        type=str,
        default="description",
        help="Type of embeddings to use from HDF5 file (default: description)"
    )
    
    return parser.parse_args()


def main():
    """
    Main execution function for drug embedding evaluation.

    Parses command-line arguments and runs similarity analysis on specified
    CSV files using the provided HDF5 embeddings file.
    """
    args = parse_arguments()
    
    # Validate HDF5 file exists
    if not Path(args.hdf5).exists():
        print(f"Error: HDF5 file not found: {args.hdf5}", file=sys.stderr)
        sys.exit(1)
    
    # Process each input CSV file
    for csv_file in args.input:
        csv_path = Path(csv_file)
        
        # If file doesn't exist as-is, try looking in data/ directory
        if not csv_path.exists():
            csv_path = Path("data") / csv_file
            if not csv_path.exists():
                print(f"Warning: CSV file not found: {csv_file}", file=sys.stderr)
                continue
        
        print(f"\n{'='*67}")
        print(f"Processing: {csv_path}")
        print(f"{'='*67}\n")
        
        try:
            drug_tests(str(csv_path), args.hdf5, args.embedding_type)
        except Exception as e:
            print(f"Error processing {csv_path}: {e}", file=sys.stderr)
            continue


if __name__ == "__main__":
    main()
