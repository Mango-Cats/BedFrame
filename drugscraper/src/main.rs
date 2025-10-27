//! # Drugscraper
//!
//! This module implements a high-performance parallel processing pipeline for cleaning and
//! normalizing drug product data. The pipeline performs ASCII normalization, special
//! character removal, and deduplication using Rayon for multi-threaded execution.
//!
//! ## Pipeline Overview
//!
//! 1. **Data Loading**: Read FDA drug product CSV files into memory
//! 2. **Parallel Processing**: Clean and normalize drug names using all available CPU cores
//! 3. **Deduplication**: Remove duplicate (generic_name, drug_name) pairs
//! 4. **Output Generation**: Write cleaned data to CSV
//!
//! ## Key Features
//!
//! - **Unicode Normalization**: Converts Latin-1 and extended Latin characters to ASCII
//! - **Special Character Removal**: Strips trademark symbols (®, ™, ©) and non-standard characters
//! - **Parenthetical Content Removal**: Eliminates dosage forms and other parenthetical data
//! - **Parallel Execution**: Processes records concurrently using Rayon
//! - **Progress Reporting**: Real-time updates during processing
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release -- <input_csv>
//! ```

use csv::{Reader, Writer};
use rayon::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::env;
use std::error::Error;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Input record structure matching drug product CSV format
///
/// Represents a single drug product entry from the drug database with
/// generic name, brand name, and pharmacologic category.
#[derive(Debug, Deserialize)]
struct DrugProduct {
    /// Generic (chemical) name of the drug
    #[serde(rename = "Generic Name")]
    generic_name: String,

    /// Brand (commercial) name of the drug
    #[serde(rename = "Brand Name")]
    brand_name: String,

    /// Pharmacologic/therapeutic category classification
    #[serde(rename = "Pharmacologic Category")]
    pharmacologic_category: String,
}

/// Output record structure for cleaned and normalized drug data
///
/// Contains processed drug information with ASCII-normalized names
/// and cleaned pharmacologic categories.
#[derive(Debug, Serialize)]
struct OutputRecord {
    /// Cleaned and normalized generic name (lowercase)
    #[serde(rename = "Generic Name")]
    generic_name: String,

    /// Cleaned and normalized brand/drug name (lowercase)
    #[serde(rename = "Drug Name")]
    drug_name: String,

    /// Normalized pharmacologic category
    #[serde(rename = "Pharmacologic Category")]
    pharmacologic_category: String,
}

/// Normalize pharmacologic category field
///
/// Cleans and normalizes the pharmacologic category field by:
/// - Trimming whitespace
/// - Removing placeholder values ("-", "'-'")
/// - Removing special characters
///
/// # Arguments
///
/// * `category` - Raw pharmacologic category string from input data
///
/// # Returns
///
/// Normalized category string, or empty string if invalid
///
/// # Examples
///
/// ```
/// let category = normalize_pharmacologic_category("Analgesic®".to_string());
/// assert_eq!(category, "Analgesic");
/// ```
fn normalize_pharmacologic_category(category: String) -> String {
    let trimmed = category.trim();
    if trimmed.is_empty() || trimmed == "-" || trimmed == "'-'" {
        String::new()
    } else {
        // Also remove special characters from category
        remove_special_chars(trimmed.to_string())
    }
}

/// Clean and normalize generic drug names
///
/// Applies comprehensive cleaning to generic drug names including:
/// 1. **Parenthetical Removal**: Eliminates dosage forms, routes (e.g., "(oral)", "(injection)")
/// 2. **Bracket Removal**: Removes bracketed content (e.g., "[obsolete]")
/// 3. **Combination Drug Formatting**: Normalizes forward slashes for multi-component drugs
/// 4. **ASCII Normalization**: Converts Unicode characters to ASCII equivalents
/// 5. **Whitespace Normalization**: Collapses multiple spaces and trims
///
/// This is critical for entity matching as generic names in FDA data often include
/// extraneous dosage form information that interferes with cross-database matching.
///
/// # Arguments
///
/// * `name` - Raw generic name string from FDA data
///
/// # Returns
///
/// Cleaned generic name suitable for entity matching
///
/// # Examples
///
/// ```
/// let cleaned = clean_generic_name("Acetaminophen (oral)".to_string());
/// assert_eq!(cleaned, "Acetaminophen");
///
/// let combo = clean_generic_name("Amoxicillin/Clavulanate".to_string());
/// assert_eq!(combo, "Amoxicillin / Clavulanate");
/// ```
fn clean_generic_name(name: String) -> String {
    let name = name.trim();

    // Remove all content in parentheses (generalized approach)
    let re_parens = Regex::new(r"\s*\([^)]*\)").unwrap();
    let name = re_parens.replace_all(name, "").to_string();

    // Remove all content in brackets
    let re_brackets = Regex::new(r"\s*\[[^\]]*\]").unwrap();
    let name = re_brackets.replace_all(&name, "").to_string();

    // Replace forward slashes with commas for combination drugs
    let name = name.replace("/", " / ");

    // Remove special characters (®, ™, ©, etc.)
    let name = remove_special_chars(name);

    // Clean up multiple spaces and trim
    let name = name.split_whitespace().collect::<Vec<_>>().join(" ");

    // Return the cleaned name (CSV writer will handle quoting)
    name.trim().to_string()
}

/// Remove special characters and normalize Unicode to ASCII
///
/// Implements comprehensive character normalization essential for pharmaceutical text processing:
///
/// ## Character Handling Strategy
///
/// 1. **Preservation**: ASCII alphanumeric, whitespace, hyphens, slashes, commas, periods
/// 2. **Conversion**: Latin-1 and Extended Latin characters to ASCII equivalents
/// 3. **Removal**: Trademark symbols (®, ™, ©), special Unicode characters
///
/// ## Rationale
///
/// Pharmaceutical data sources use inconsistent Unicode encoding, particularly for:
/// - **Diacritical marks**: Latin drug names (e.g., "Montélukast" → "Montelukast")
/// - **Trademark symbols**: Brand names (e.g., "Advil®" → "Advil")
/// - **Typography**: Em/en dashes in compound names (e.g., "Drug—Generic" → "Drug-Generic")
///
/// This normalization is critical for entity matching across databases with different
/// character encoding standards.
///
/// ## Character Mappings
///
/// ### Latin-1 Supplement (U+00C0 - U+00FF)
/// - à, á, â, ã, ä, å, ā, ą, ạ → a
/// - è, é, ê, ë, ē, ė, ę → e
/// - ì, í, î, ï, ī, į → i
/// - ò, ó, ô, õ, ö, ø, ō → o
/// - ù, ú, û, ü, ū → u
/// - ñ → n, ç → c, ß → s
///
/// ### Punctuation
/// - – (en dash, U+2013) → -
/// - — (em dash, U+2014) → -
/// - \u{00A0} (non-breaking space) → (space)
///
/// # Arguments
///
/// * `text` - Input string potentially containing special characters
///
/// # Returns
///
/// ASCII-normalized string with special characters removed or converted
///
/// # Examples
///
/// ```
/// let text = remove_special_chars("Montélukast®".to_string());
/// assert_eq!(text, "Montelukast");
///
/// let text = remove_special_chars("Naproxen—NSAID".to_string());
/// assert_eq!(text, "Naproxen-NSAID");
/// ```
fn remove_special_chars(text: String) -> String {
    // Remove trademark, registered, copyright symbols and other non-standard characters
    // Also convert non-ASCII characters to ASCII equivalents or remove them
    text.chars()
        .filter_map(|c| {
            // Keep only ASCII alphanumeric, spaces, hyphens, forward slashes, commas, and periods
            if c.is_ascii_alphanumeric()
                || c.is_ascii_whitespace()
                || c == '-'
                || c == '/'
                || c == ','
                || c == '.'
            {
                Some(c)
            } else {
                // Try to convert some common non-ASCII to ASCII equivalents
                match c {
                    'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' | 'ā' | 'ą' | 'ạ' => Some('a'),
                    'è' | 'é' | 'ê' | 'ë' | 'ē' | 'ė' | 'ę' => Some('e'),
                    'ì' | 'í' | 'î' | 'ï' | 'ī' | 'į' => Some('i'),
                    'ò' | 'ó' | 'ô' | 'õ' | 'ö' | 'ø' | 'ō' => Some('o'),
                    'ù' | 'ú' | 'û' | 'ü' | 'ū' => Some('u'),
                    'ñ' => Some('n'),
                    'ç' => Some('c'),
                    'ß' => Some('s'),
                    'æ' => Some('a'),
                    'À' | 'Á' | 'Â' | 'Ã' | 'Ä' | 'Å' => Some('A'),
                    'È' | 'É' | 'Ê' | 'Ë' => Some('E'),
                    'Ì' | 'Í' | 'Î' | 'Ï' => Some('I'),
                    'Ò' | 'Ó' | 'Ô' | 'Õ' | 'Ö' | 'Ø' => Some('O'),
                    'Ù' | 'Ú' | 'Û' | 'Ü' => Some('U'),
                    'Ñ' => Some('N'),
                    'Ç' => Some('C'),
                    '–' | '—' => Some('-'),  // Em dash and en dash to hyphen
                    '\u{00A0}' => Some(' '), // Non-breaking space to regular space
                    _ => None,               // Remove any other non-ASCII character
                }
            }
        })
        .collect()
}

/// Clean and normalize brand drug names
///
/// Applies simpler cleaning to brand names compared to generic names:
/// 1. **Special Character Removal**: Strips trademark symbols and Unicode
/// 2. **Whitespace Normalization**: Collapses multiple spaces
///
/// Brand names typically do not require parenthetical removal as they rarely
/// include dosage form information.
///
/// # Arguments
///
/// * `name` - Raw brand name string from FDA data
///
/// # Returns
///
/// Cleaned brand name
///
/// # Examples
///
/// ```
/// let cleaned = clean_brand_name("Advil®".to_string());
/// assert_eq!(cleaned, "Advil");
/// ```
fn clean_brand_name(name: String) -> String {
    let name = name.trim();

    // Remove special characters (®, ™, ©, etc.)
    let name = remove_special_chars(name.to_string());

    // Clean up multiple spaces and trim
    let name = name.split_whitespace().collect::<Vec<_>>().join(" ");

    name.trim().to_string()
}

/// Main entry point for the FDA drug product processing pipeline
///
/// ## Pipeline Stages
///
/// 1. **Argument Parsing**: Validates command-line arguments for input CSV path
/// 2. **Data Loading**: Reads FDA drug product CSV into memory
/// 3. **Parallel Processing**: Applies cleaning functions to all records using Rayon
/// 4. **Deduplication**: Removes duplicate (generic_name, drug_name) pairs
/// 5. **Output Generation**: Writes cleaned data to `drugscraped.csv`
///
/// ## Parallelization Strategy
///
/// The pipeline uses Rayon's `par_iter()` to process records concurrently across
/// all available CPU cores. This is particularly effective for I/O-bound operations
/// like regex matching and string manipulation.
///
/// ## Progress Reporting
///
/// Progress is reported every 1,000 records using atomic counters to avoid
/// synchronization overhead.
///
/// ## Error Handling
///
/// Returns `Box<dyn Error>` for any I/O or CSV parsing errors, with descriptive
/// error messages printed to stderr.
///
/// # Returns
///
/// `Ok(())` on successful completion, `Err` on any processing error
fn main() -> Result<(), Box<dyn Error>> {
    // Stage 1: Argument Parsing and Validation
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <input_csv_path>", args[0]);
        eprintln!("Example: {} data/ALL_DrugProducts.csv", args[0]);
        std::process::exit(1);
    }

    let input_path = &args[1];

    println!("=== FDA Drug Product Scraper ===\n");
    println!("Reading CSV file: {}", input_path);

    // Stage 2: Data Loading
    let mut reader = Reader::from_path(input_path)?;

    // Collect all records into a vector for parallel processing
    let records: Vec<DrugProduct> = reader.deserialize().collect::<Result<Vec<_>, _>>()?;

    let total_records = records.len();
    println!("Loaded {} records\n", total_records);

    // Stage 3: Parallel Processing
    println!(
        "Processing records with {} threads...",
        rayon::current_num_threads()
    );

    // Atomic counter for thread-safe progress tracking
    let counter = Arc::new(AtomicUsize::new(0));

    // Process records in parallel using Rayon
    let output_records: Vec<OutputRecord> = records
        .into_par_iter()
        .map(|record| {
            let count = counter.fetch_add(1, Ordering::Relaxed) + 1;

            // Print progress every 1000 records
            if count % 1000 == 0 {
                println!("  Processed {} / {} records...", count, total_records);
            }

            OutputRecord {
                generic_name: clean_generic_name(record.generic_name).to_lowercase(),
                drug_name: clean_brand_name(record.brand_name).to_lowercase(),
                pharmacologic_category: normalize_pharmacologic_category(
                    record.pharmacologic_category,
                ),
            }
        })
        .collect();

    println!("  Completed processing {} records\n", total_records);

    // Stage 4: Deduplication
    println!("Removing duplicates...");

    // Deduplicate based on (generic_name, drug_name) pair
    // This is necessary as FDA data contains multiple entries for different
    // dosage forms and formulations of the same drug
    let mut seen = HashSet::new();
    let mut unique_records = Vec::new();
    let mut duplicate_count = 0;

    for record in output_records {
        let key = (
            record.generic_name.to_lowercase(),
            record.drug_name.to_lowercase(),
        );
        if seen.insert(key) {
            unique_records.push(record);
        } else {
            duplicate_count += 1;
        }
    }

    println!("  Removed {} duplicate entries", duplicate_count);
    println!("  Retained {} unique records\n", unique_records.len());

    // Stage 5: Output Generation
    let output_path = "drugscraped.csv";
    println!("Writing output to {}...", output_path);

    let mut writer = Writer::from_path(output_path)?;

    for record in unique_records {
        writer.serialize(record)?;
    }

    writer.flush()?;

    println!("\n=== Processing Complete ===");
    println!("Total records processed: {}", total_records);
    println!("Output saved to: {}", output_path);

    Ok(())
}
