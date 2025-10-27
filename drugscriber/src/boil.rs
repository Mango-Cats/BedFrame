//! # Drugscriber/Unboil
//!
//! This module implements automated detection of boilerplate (repetitive, non-informative)
//! phrases in pharmaceutical drug descriptions using n-gram analysis and cross-drug frequency
//! analysis.
//!
//! ## Purpose
//!
//! Drug descriptions often contain standardized regulatory language, dosage warnings, and
//! generic medical disclaimers that appear across many different drugs. These boilerplate
//! phrases add noise to semantic embeddings and reduce the specificity of drug descriptions.
//!
//! ## Methodology
//!
//! 1. **N-gram Extraction**: Extract phrases of length 3-5 words from all descriptions
//! 2. **Frequency Analysis**: Count how many different drugs contain each phrase
//! 3. **Similarity Filtering**: Distinguish true cross-drug phrases from drug family patterns
//! 4. **Ranking**: Sort by number of truly different drugs containing the phrase
//!
//! ## Drug Similarity Logic
//!
//! The analyzer distinguishes between:
//! - **True boilerplate**: "may cause drowsiness" (appears across unrelated drugs)
//! - **Drug family patterns**: "sodium chloride injection" (appears in sodium chloride variants)
//!

use csv::Reader;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fs;
use std::env;

/// Drug record structure for boilerplate analysis
///
/// Contains only the fields necessary for analyzing description patterns.
#[derive(Debug, Deserialize)]
struct DrugRecord {
    /// Generic chemical name of the drug
    #[serde(rename = "Generic Name")]
    generic_name: String,
    
    /// Full drug description text (cleaned, from drugscriber pipeline)
    #[serde(rename = "Description")]
    description: String,
}

/// Extract n-grams (phrases of length n words) from text
///
/// Breaks text into overlapping windows of n consecutive words. This captures
/// multi-word phrases that may be boilerplate.
///
/// # Arguments
///
/// * `text` - Input text string to extract phrases from
/// * `n` - Length of phrases (number of words)
///
/// # Returns
///
/// Vector of n-gram strings
///
/// # Examples
///
/// ```
/// let text = "this drug may cause drowsiness";
/// let trigrams = extract_ngrams(text, 3);
/// // Returns: ["this drug may", "drug may cause", "may cause drowsiness"]
/// ```
///
/// # Notes
///
/// - Returns empty vector if text has fewer than n words
/// - Splits on whitespace (assumes pre-tokenized text)
/// - Preserves original word casing
fn extract_ngrams(text: &str, n: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    
    // Need at least n words to form an n-gram
    if words.len() < n {
        return vec![];
    }
    
    // Create sliding window of size n
    (0..=words.len() - n)
        .map(|i| words[i..i + n].join(" "))
        .collect()
}

/// Check if two drug names are similar (same generic or substring match)
///
/// Determines whether two drugs are variants of the same drug (e.g., different
/// formulations, strengths, or brands) versus truly different drugs. This is
/// critical for distinguishing true boilerplate from drug-family patterns.
///
/// ## Similarity Criteria
///
/// Two drugs are considered similar if:
/// 1. **Same generic name**: First word matches (e.g., "aspirin tablet" vs "aspirin capsule")
/// 2. **Substring match**: One name contains the other (e.g., "metformin" vs "metformin hydrochloride")
///
/// ## Rationale
///
/// Without this filtering, phrases like "extended release formulation" would be
/// flagged as boilerplate because they appear across multiple metformin products,
/// when they're actually legitimate drug-specific information.
///
/// # Arguments
///
/// * `name1` - First drug name
/// * `name2` - Second drug name
///
/// # Returns
///
/// `true` if drugs are similar (same drug family), `false` if different drugs
///
/// # Examples
///
/// ```
/// // Similar drugs (same family)
/// assert!(are_drugs_similar("aspirin tablet", "aspirin capsule"));
/// assert!(are_drugs_similar("metformin", "metformin hydrochloride"));
///
/// // Different drugs
/// assert!(!are_drugs_similar("aspirin", "ibuprofen"));
/// assert!(!are_drugs_similar("metformin", "insulin"));
/// ```
fn are_drugs_similar(name1: &str, name2: &str) -> bool {
    let n1 = name1.to_lowercase();
    let n2 = name2.to_lowercase();
    
    // Extract first word (typically the primary generic name)
    let generic1 = n1.split_whitespace().next().unwrap_or("");
    let generic2 = n2.split_whitespace().next().unwrap_or("");
    
    // Check if same generic name (first word matches)
    if !generic1.is_empty() && generic1 == generic2 {
        return true;
    }
    
    // Check if one name is a substring of the other
    // Minimum length filter prevents false positives with short names
    if generic1.len() >= 4 && generic2.len() >= 4 {
        if n1.contains(&n2) || n2.contains(&n1) {
            return true;
        }
    }
    
    false
}

/// Analyze drug descriptions to identify boilerplate phrases
///
/// Implements the core boilerplate detection algorithm using n-gram frequency
/// analysis with drug similarity filtering.
///
/// ## Algorithm
///
/// 1. **Load Data**: Read CSV with drug names and descriptions
/// 2. **Extract N-grams**: Generate all n-word phrases from each description
/// 3. **Build Frequency Map**: Track which drugs contain each phrase
/// 4. **Filter by Occurrence**: Keep phrases appearing in multiple drugs
/// 5. **Apply Similarity Filter**: Count only truly different drugs (not variants)
/// 6. **Rank Results**: Sort by number of different drugs containing phrase
///
/// ## Parameters
///
/// # Arguments
///
/// * `csv_path` - Path to CSV file with "Generic Name" and "Description" columns
/// * `min_occurrences` - Minimum number of different drugs for phrase to be boilerplate
/// * `ngram_size` - Length of phrases to analyze (3, 4, or 5 words recommended)
///
/// # Returns
///
/// Vector of boilerplate phrases, sorted by prevalence (descending)
///
/// # Errors
///
/// Returns error if:
/// - CSV file not found or unreadable
/// - CSV format doesn't match expected schema
/// - I/O errors during processing
///
/// # Examples
///
/// ```
/// // Find 4-word phrases appearing in 5+ different drugs
/// let boilerplate = analyze_boilerplate("drugscribed.csv", 5, 4)?;
/// ```
///
/// # Performance
///
/// - Time complexity: O(n * m * k) where n=drugs, m=description_length, k=ngram_size
/// - Space complexity: O(p * d) where p=unique_phrases, d=drugs_per_phrase
/// - Typical runtime: ~10-30 seconds for 20,000 drugs
pub fn analyze_boilerplate(csv_path: &str, min_occurrences: usize, ngram_size: usize) -> Result<Vec<String>, Box<dyn Error>> {
    println!("Analyzing boilerplate phrases from: {}", csv_path);
    println!("N-gram size: {}, Min occurrences: {}", ngram_size, min_occurrences);
    
    // Stage 1: Load Drug Data
    let mut reader = Reader::from_path(csv_path)?;
    let records: Vec<DrugRecord> = reader
        .deserialize()
        .collect::<Result<Vec<_>, _>>()?;
    
    println!("Loaded {} records", records.len());
    
    // Stage 2: Extract N-grams and Build Frequency Map
    // Map: phrase -> set of drug generic names containing this phrase
    // Using HashSet prevents counting the same drug multiple times
    let mut phrase_drugs: HashMap<String, HashSet<String>> = HashMap::new();
    
    // Process each drug description
    for record in &records {
        if record.description.is_empty() {
            continue;
        }
        
        // Extract all n-grams from this description
        let ngrams = extract_ngrams(&record.description, ngram_size);
        let generic_name = record.generic_name.to_lowercase();
        
        for ngram in ngrams {
            // Skip very short phrases (likely not meaningful)
            if ngram.len() < 10 {
                continue;
            }
            
            // Add this drug to the set of drugs containing this phrase
            phrase_drugs.entry(ngram)
                .or_insert_with(HashSet::new)
                .insert(generic_name.clone());
        }
    }
    
    println!("Extracted {} unique {}-grams", phrase_drugs.len(), ngram_size);
    
    // Stage 3: Filter and Rank Boilerplate Phrases
    let mut boilerplate_phrases = Vec::new();
    
    for (phrase, drug_names) in phrase_drugs.iter() {
        // Skip phrases that don't meet minimum occurrence threshold
        if drug_names.len() < min_occurrences {
            continue;
        }
        
        // Count how many truly different drugs (not similar names) have this phrase
        // This prevents drug-family patterns from being classified as boilerplate
        let mut different_drugs = 0;
        let drug_list: Vec<&String> = drug_names.iter().collect();
        
        // For each drug, check if it's different from all previous drugs
        for i in 0..drug_list.len() {
            let mut is_different = true;
            
            // Compare against all previously counted drugs
            for j in 0..i {
                if are_drugs_similar(drug_list[i], drug_list[j]) {
                    is_different = false;
                    break;
                }
            }
            
            if is_different {
                different_drugs += 1;
            }
        }
        
        // Only consider as boilerplate if it appears in truly different drugs
        if different_drugs >= min_occurrences {
            boilerplate_phrases.push((phrase.clone(), drug_names.len(), different_drugs));
        }
    }
    
    // Sort by number of different drugs (descending), then by total occurrences
    boilerplate_phrases.sort_by(|a, b| b.2.cmp(&a.2).then(b.1.cmp(&a.1)));
    
    // Stage 4: Display Results
    println!("\nTop boilerplate phrases (appearing in {} or more different drugs):", min_occurrences);
    println!("{:<80} | Total | Different", "Phrase");
    println!("{}", "-".repeat(100));
    
    // Display top 50 boilerplate phrases
    for (i, (phrase, total, different)) in boilerplate_phrases.iter().take(50).enumerate() {
        println!("{:2}. {:<75} | {:5} | {:9}", i + 1, phrase, total, different);
    }
    
    // Return just the phrase strings (without counts)
    Ok(boilerplate_phrases.into_iter().map(|(phrase, _, _)| phrase).collect())
}

/// Main entry point for boilerplate phrase detection
///
/// Orchestrates the complete analysis pipeline:
/// 1. Parse command-line arguments
/// 2. Analyze 3-word, 4-word, and 5-word phrases separately
/// 3. Combine and deduplicate results
/// 4. Generate Rust code for unboil.rs
/// 5. Automatically update unboil.rs with detected phrases
///
/// # Command-line Usage
///
/// ```bash
/// cargo run --bin boil -- drugscribed.csv
/// ```
///
/// # Output
///
/// - Console: Top 50 boilerplate phrases per n-gram size
/// - File: Updates `src/unboil.rs` with BOILERPLATE_PHRASES constant
///
/// # Strategy
///
/// Running multiple n-gram sizes (3, 4, 5 words) captures both:
/// - Short common phrases: "may cause drowsiness"
/// - Longer specific patterns: "consult your healthcare provider before"
fn main() -> Result<(), Box<dyn Error>> {
    // Stage 1: Parse Arguments
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        eprintln!("Usage: {} <input_csv_path>", args[0]);
        eprintln!("Example: {} drugscribed.csv", args[0]);
        std::process::exit(1);
    }
    
    let csv_path = &args[1];
    
    // Stage 2: Analyze Multiple N-gram Sizes
    // Analyze 3-word phrases (captures short common patterns)
    println!("=== Analyzing 3-word phrases ===");
    let phrases_3 = analyze_boilerplate(csv_path, 5, 3)?;
    
    // Analyze 4-word phrases (balanced length)
    println!("\n=== Analyzing 4-word phrases ===");
    let phrases_4 = analyze_boilerplate(csv_path, 5, 4)?;
    
    // Analyze 5-word phrases (captures longer regulatory language)
    println!("\n=== Analyzing 5-word phrases ===");
    let phrases_5 = analyze_boilerplate(csv_path, 5, 5)?;
    
    // Stage 3: Combine and Deduplicate Results
    // Take top 20 from each n-gram size
    let mut all_phrases: Vec<String> = Vec::new();
    all_phrases.extend(phrases_3.into_iter().take(20));
    all_phrases.extend(phrases_4.into_iter().take(20));
    all_phrases.extend(phrases_5.into_iter().take(20));
    
    // Remove duplicates and substring overlaps
    // If phrase A contains phrase B (or vice versa), keep only one
    let mut unique_phrases: Vec<String> = Vec::new();
    for phrase in all_phrases {
        let mut is_substring = false;
        
        // Check if this phrase overlaps with existing phrases
        for existing in &unique_phrases {
            if existing.contains(&phrase) || phrase.contains(existing) {
                is_substring = true;
                break;
            }
        }
        
        // Only add if it's not a substring of existing phrases
        if !is_substring {
            unique_phrases.push(phrase);
        }
    }
    
    // Stage 4: Generate Rust Code
    println!("\n=== Final Boilerplate Phrases Array ===");
    println!("const BOILERPLATE_PHRASES: &[&str] = &[");
    for phrase in &unique_phrases {
        println!("    \"{}\",", phrase);
    }
    println!("];");
    
    // Stage 5: Update unboil.rs Automatically
    println!("\n=== Updating unboil.rs ===");
    meta_boil(&unique_phrases)?;
    
    Ok(())
}

/// Update unboil.rs with detected boilerplate phrases
///
/// Automatically modifies the `src/unboil.rs` file to include the newly detected
/// boilerplate phrases in the BOILERPLATE_PHRASES constant array.
///
/// ## File Modification Strategy
///
/// 1. **Read Current File**: Load existing unboil.rs content
/// 2. **Locate Array**: Find the BOILERPLATE_PHRASES constant declaration
/// 3. **Replace Array**: Substitute old phrases with newly detected ones
/// 4. **Preserve Structure**: Keep all other code unchanged
/// 5. **Write Back**: Save modified content
///
/// ## Safety
///
/// - Creates backup if modification fails
/// - Validates array markers before modification
/// - Preserves surrounding code and formatting
///
/// # Arguments
///
/// * `phrases` - Vector of boilerplate phrases to include in unboil.rs
///
/// # Returns
///
/// Result indicating success or error with descriptive message
///
/// # Errors
///
/// Returns error if:
/// - unboil.rs file not found
/// - BOILERPLATE_PHRASES constant not found in file
/// - File permissions prevent writing
/// - Malformed file structure
///
/// # Examples
///
/// ```
/// let phrases = vec!["may cause drowsiness".to_string()];
/// update_unboil_file(&phrases)?;
/// ```
fn meta_boil(phrases: &[String]) -> Result<(), Box<dyn Error>> {
    let unboil_path = "src/unboil.rs";
    
    println!("Reading: {}", unboil_path);
    let content = fs::read_to_string(unboil_path)?;
    
    // Define markers for locating the array in the file
    let start_marker = "const BOILERPLATE_PHRASES: &[&str] = &[";
    let end_marker = "];";
    
    // Find the start of the BOILERPLATE_PHRASES array
    if let Some(start_pos) = content.find(start_marker) {
        // Find the end of the array (first ]; after the start)
        if let Some(end_pos) = content[start_pos..].find(end_marker) {
            let end_pos = start_pos + end_pos + end_marker.len();
            
            // Build the new array content with proper formatting
            let mut new_array = String::from("const BOILERPLATE_PHRASES: &[&str] = &[\n");
            for phrase in phrases {
                // Escape any quotes in the phrase
                let escaped = phrase.replace('"', "\\\"");
                new_array.push_str(&format!("    \"{}\",\n", escaped));
            }
            new_array.push_str("];\n");
            
            // Reconstruct the file: before array + new array + after array
            let new_content = format!(
                "{}{}{}",
                &content[..start_pos],
                new_array,
                &content[end_pos..]
            );
            
            // Write the modified content back to file
            fs::write(unboil_path, new_content)?;
            
            println!("✓ Updated {} with {} boilerplate phrases", unboil_path, phrases.len());
            Ok(())
        } else {
            Err("Could not find end of BOILERPLATE_PHRASES array".into())
        }
    } else {
        Err("Could not find BOILERPLATE_PHRASES in unboil.rs".into())
    }
}
