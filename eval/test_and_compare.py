"""
Test and Compare Script

This script runs both BERT embeddings evaluation and TF-IDF analysis on the same
drug CSV files and compares their outputs.
"""

# CSV files to process
csv_files = [
    "amoxicillin.csv",
    "adapalene.csv",
    "nsaid.csv",
    "corticosteroid.csv",
    "asthma.csv"
]

# Sample sizes for each CSV file (None = use all drugs, int = random sample)
samples = [
    None,
    None,
    None,
    None,
    None
]

assert len(csv_files) == len(samples), "The number of samples and csv_files must be the same."

import subprocess
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path


def run_evaluation_script(csv_files, hdf5_path="../artifact/drugsberted.h5"):
    """
    Run the evaluation.py script on the specified CSV files.
    
    Args:
        csv_files (list): List of CSV filenames to process
        hdf5_path (str): Path to the HDF5 embeddings file
    """
    cmd = [sys.executable, "evaluation.py", "--hdf5", hdf5_path, "--input"] + csv_files
    
    try:
        subprocess.run(cmd, check=True, capture_output=False, text=True)
        print("\nBERT evaluation completed successfully\n")
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation.py: {e}", file=sys.stderr)
        sys.exit(1)


def get_tfidf_vectorizer():
    """
    Create and train a TF-IDF vectorizer on boiled.csv.
    
    Returns:
        TfidfVectorizer: Trained vectorizer
    """
    df = pd.read_csv("data/boiled.csv", dtype=str).fillna('') 
    df = df[df['Generic Name'] != '']
    df['combined_text'] = df['Drug Name'] + ' ' + df['Description']
    
    vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r'\b\w+\b')
    vectorizer.fit(df['combined_text'].values)
    
    print(f"TF-IDF vocab size: {len(vectorizer.get_feature_names_out())}")
    
    return vectorizer


def get_vector_analysis(vectors, name=""):
    """
    Compute and display statistical analysis of cosine similarities.
    
    Args:
        vectors: Array of vectors
        name (str): Name for display purposes
    """
    similarity_matrix = cosine_similarity(vectors)
    similarity_matrix = np.clip(similarity_matrix, 0, 1)

    upper_triangle_indices = np.triu_indices_from(similarity_matrix, k=1)
    non_self_similarities = similarity_matrix[upper_triangle_indices]

    print(f"\n  CS matrix shape: {similarity_matrix.shape}")
    print(f"  CS range: [{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]")
    print(f"  Mean: {np.nanmean(non_self_similarities):.4f}")
    print(f"  Std Dev: {np.nanstd(non_self_similarities):.4f}")
    print(f"  Min: {np.nanmin(non_self_similarities):.4f}")
    print(f"  Max: {np.nanmax(non_self_similarities):.4f}")
    print()


def levels(vectors, n, drug_names):
    """
    Analyze and display similarity statistics for top, middle, and bottom drug pairs.
    
    Args:
        vectors: Drug embedding vectors
        n (int): Number of drugs
        drug_names (list): List of drug names
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
    mid_n = sorted_sims[middle_start:middle_start + n]

    print(f"Top, Middle, and Bottom Pairs of the Top {n} Pairs:")
    print(f"Top pair: {top_n[0][0]} <-> {top_n[0][1]} = {top_n[0][2]}")
    print(f"Middle pair: {top_n[n//2][0]} <-> {top_n[n//2][1]} = {top_n[n//2][2]}")
    print(f"Bottom pair: {top_n[n-1][0]} <-> {top_n[n-1][1]} = {top_n[n-1][2]}")
    print('\n')
    
    print(f"Top, Middle, and Bottom Pairs of the Middle {n} Pairs:")
    print(f"Top pair: {mid_n[0][0]} <-> {mid_n[0][1]} = {mid_n[0][2]}")
    print(f"Middle pair: {mid_n[n//2][0]} <-> {mid_n[n//2][1]} = {mid_n[n//2][2]}")
    print(f"Bottom pair: {mid_n[n-1][0]} <-> {mid_n[n-1][1]} = {mid_n[n-1][2]}")
    print('\n')
    
    print(f"Top, Middle, and Bottom Pairs of the Bottom {n} Pairs:")
    print(f"Top pair: {bott_n[0][0]} <-> {bott_n[0][1]} = {bott_n[0][2]}")
    print(f"Middle pair: {bott_n[n//2][0]} <-> {bott_n[n//2][1]} = {bott_n[n//2][2]}")
    print(f"Bottom pair: {bott_n[n-1][0]} <-> {bott_n[n-1][1]} = {bott_n[n-1][2]}")
    print("\n")

    top_n_cs = np.mean([pair[2] for pair in top_n])
    mid_n_cs = np.mean([pair[2] for pair in mid_n])
    bott_n_cs = np.mean([pair[2] for pair in bott_n])

    print(f"Mean of the Cosine Similarities of the TOP {n} pairs: {top_n_cs:.4f}")
    print(f"Mean of the Cosine Similarities of the MIDDLE {n} pairs: {mid_n_cs:.4f}")
    print(f"Mean of the Cosine Similarities of the BOTTOM {n} pairs: {bott_n_cs:.4f}")
    print("\n")


def sim_per_drug(vectors, drug_names, n, name):
    """
    Calculate average similarity for each drug and save results to CSV.
    
    Args:
        vectors: Drug embedding vectors
        drug_names (list): List of drug names
        n (int): Number of drugs
        name (str): Base filename for output CSV
    """
    similarity_matrix = cosine_similarity(vectors)
    similarity_matrix = np.clip(similarity_matrix, 0, 1)
    
    drug_sims = []
    for i in range(n):
        similarities = []
        for j in range(n):
            if i != j:
                similarities.append(similarity_matrix[i, j])
        
        if similarities:
            avg = np.mean(similarities)
        else:
            avg = 0.0
            
        drug_sims.append(avg)
    
    drug_sim_df = pd.DataFrame({'Drug': drug_names, 'Avg_Similarity': drug_sims})
    drug_sim_df = drug_sim_df.sort_values('Avg_Similarity', ascending=False)

    file_path = Path("tfidf_results")
    file_path.mkdir(parents=True, exist_ok=True)
    drug_sim_df.to_csv(f"tfidf_results/{name}_sims.csv", index=False)
    print(f"Saved TF-IDF results to tfidf_results/{name}_sims.csv")


def tfidf_analysis(vectorizer, csv_name, sample_size=None):
    """
    Perform TF-IDF analysis on a drug CSV file.
    
    Args:
        vectorizer: Trained TfidfVectorizer
        csv_name (str): Name of CSV file in data/ directory
        sample_size (int or None): Number of random samples to use, None for all
        
    Returns:
        dict: Analysis results
    """
    test_drug_name = csv_name.split("_")[0].replace(".csv", "")
    drug_df = pd.read_csv(f"data/{csv_name}", dtype=str, header=0).fillna('')

    # Rename columns to standard format
    drug_df.columns = ['Generic Name', 'Drug Name', 'Description']
    
    # Filter and fix brand names
    processed_rows = []
    for idx, row in drug_df.iterrows():
        generic_name = str(row['Generic Name'])
        drug_name = str(row['Drug Name'])
        
        # Default to generic name if brand name is empty, "unbranded", or "none"
        if drug_name.strip().lower() in ["", "unbranded", "none", "none reformulated"]:
            if not generic_name.strip():
                raise ValueError(
                    f"Error at row {idx}: Brand name is invalid ('{drug_name}') "
                    f"and generic name is empty. Cannot process this drug entry."
                )
            drug_name = generic_name
            row['Drug Name'] = drug_name
        
        processed_rows.append(row)
    
    drug_df = pd.DataFrame(processed_rows)
    
    # Apply random sampling if specified
    if sample_size is not None and sample_size < len(drug_df):
        drug_df = drug_df.sample(n=sample_size, random_state=42)

    drug_df['combined_drug'] = drug_df['Generic Name'] + '/' + drug_df['Drug Name']
    drug_df['combined_text'] = drug_df['Drug Name'] + ' ' + drug_df['Description']

    drug_vec = vectorizer.transform(drug_df['combined_text'])
    drug_names = drug_df['combined_drug'].tolist()

    print(f"\n  TF-IDF TEST: {test_drug_name.upper()}")
    print(f"  Number of drugs: {len(drug_df)}")
    if sample_size is not None:
        print(f"  Sample size: {sample_size}")
    
    get_vector_analysis(drug_vec, test_drug_name)
    levels(drug_vec, len(drug_df), drug_names)
    sim_per_drug(drug_vec, drug_names, len(drug_df), f"{test_drug_name}_tfidf")


def compare_results(bert_results_dir="sbert_results", tfidf_results_dir="tfidf_results"):
    """
    Compare BERT and TF-IDF results.
    
    Args:
        bert_results_dir (str): Directory containing BERT results
        tfidf_results_dir (str): Directory containing TF-IDF results
    """
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│                  BERT vs TF-IDF COMPARISON                      │")
    print("└─────────────────────────────────────────────────────────────────┘\n")
    
    bert_dir = Path(bert_results_dir)
    tfidf_dir = Path(tfidf_results_dir)
    
    if not bert_dir.exists():
        print(f"  Warning: BERT results directory not found: {bert_results_dir}")
        return
    
    if not tfidf_dir.exists():
        print(f"  Warning: TF-IDF results directory not found: {tfidf_results_dir}")
        return
    
    for bert_file in bert_dir.glob("*_bert_sims.csv"):
        drug_name = bert_file.stem.replace("_bert_sims", "")
        tfidf_file = tfidf_dir / f"{drug_name}_tfidf_sims.csv"
        
        if not tfidf_file.exists():
            print(f"  No matching TF-IDF file for {drug_name}")
            continue
        
        bert_df = pd.read_csv(bert_file)
        tfidf_df = pd.read_csv(tfidf_file)
        
        print(f"  {drug_name.upper()}:")
        print(f"    BERT   - Mean: {bert_df['Avg_Similarity'].mean():.4f}  |  Std Dev: {bert_df['Avg_Similarity'].std():.4f}")
        print(f"    TF-IDF - Mean: {tfidf_df['Avg_Similarity'].mean():.4f}  |  Std Dev: {tfidf_df['Avg_Similarity'].std():.4f}")
        print(f"    Difference in means: {abs(bert_df['Avg_Similarity'].mean() - tfidf_df['Avg_Similarity'].mean()):.4f}")
        print()



def run_single_csv_analysis(csv_file, sample_size, vectorizer, hdf5_path="../artifact/drugsberted.h5"):
    """
    Run complete analysis (BERT + TF-IDF + comparison) for a single CSV file.
    
    Args:
        csv_file (str): CSV filename to process
        sample_size (int or None): Number of samples, None for all
        vectorizer: Trained TF-IDF vectorizer
        hdf5_path (str): Path to HDF5 embeddings file
    """
    dataset_name = csv_file.split("_")[0].replace(".csv", "")
    
    print(f"\n┌─────────────────────────────────────────────────────────────────┐")
    print(f"│  DATASET: {dataset_name.upper():^54}│")
    print(f"└─────────────────────────────────────────────────────────────────┘\n")
    
    # Step 1: Run BERT evaluation for this CSV
    print(f"  # [1/3] Running BERT Embeddings Evaluation...")
    cmd = [sys.executable, "evaluation.py", "--hdf5", hdf5_path, "--input", csv_file]
    try:
        subprocess.run(cmd, check=True, capture_output=False, text=True)
    except subprocess.CalledProcessError as e:
        print(f"    Error running evaluation.py: {e}", file=sys.stderr)
        return
    
    # Step 2: Run TF-IDF analysis for this CSV
    print(f"\n  # [2/3] Running TF-IDF Analysis...")
    tfidf_analysis(vectorizer, csv_file, sample_size)
    
    # Step 3: Compare results for this CSV
    print(f"\n  # [3/3] Comparing BERT and TF-IDF Results...")
    compare_single_result(dataset_name)
    print()


def compare_single_result(dataset_name):
    """
    Compare BERT and TF-IDF results for a single dataset.
    
    Args:
        dataset_name (str): Name of the dataset to compare
    """
    bert_file = Path(f"sbert_results/{dataset_name}_bert_sims.csv")
    tfidf_file = Path(f"tfidf_results/{dataset_name}_tfidf_sims.csv")
    
    if not bert_file.exists():
        print(f"    ⚠ BERT results not found: {bert_file}")
        return
    
    if not tfidf_file.exists():
        print(f"    ⚠ TF-IDF results not found: {tfidf_file}")
        return
    
    bert_df = pd.read_csv(bert_file)
    tfidf_df = pd.read_csv(tfidf_file)
    
    print(f"    BERT   - Mean: {bert_df['Avg_Similarity'].mean():.4f}  |  Std Dev: {bert_df['Avg_Similarity'].std():.4f}")
    print(f"    TF-IDF - Mean: {tfidf_df['Avg_Similarity'].mean():.4f}  |  Std Dev: {tfidf_df['Avg_Similarity'].std():.4f}")
    print(f"    Difference in means: {abs(bert_df['Avg_Similarity'].mean() - tfidf_df['Avg_Similarity'].mean()):.4f}")


def main():
    """
    Main function to run evaluation and TF-IDF analysis on specified CSV files.
    """
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│              DRUG SIMILARITY ANALYSIS                           │")
    print("└─────────────────────────────────────────────────────────────────┘")
    
    # Train TF-IDF vectorizer once
    print("\n  Initializing TF-IDF vectorizer...")
    vectorizer = get_tfidf_vectorizer()
    print()
    
    # Process each CSV file completely before moving to the next
    for idx, (csv_file, sample_size) in enumerate(zip(csv_files, samples), 1):
        print(f"Processing {idx}/{len(csv_files)}")
        run_single_csv_analysis(csv_file, sample_size, vectorizer)
    
    print("┌─────────────────────────────────────────────────────────────────┐")
    print("│                   ANALYSIS COMPLETE                             │")
    print("└─────────────────────────────────────────────────────────────────┘\n")
    print("  Results saved to:")
    print("    • BERT results: sbert_results/")
    print("    • TF-IDF results: tfidf_results/\n")


if __name__ == "__main__":
    main()
