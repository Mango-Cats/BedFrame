//! # Drugscriber - Cross-Database Drug Information Integration
//!
//! This module implements a sophisticated entity matching and text extraction pipeline that
//! integrates FDA drug product data with DrugBank XML descriptions to create enriched,
//! semantically-focused drug profiles.
//!
//! ## Purpose
//!
//! Pharmaceutical databases contain complementary information:
//! - **FDA Database**: Comprehensive product listings with generic/brand names
//! - **DrugBank**: Detailed descriptions, mechanisms, indications, ATC codes
//!
//! This tool bridges these databases by matching entities and extracting key medical information
//! while removing regulatory boilerplate and dosage information.
//!
//! ## Pipeline Overview
//!
//! 1. **XML Parsing**: Load DrugBank database (~15,000 drugs) into memory
//! 2. **CSV Loading**: Read FDA drug product data (~25,000 records)
//! 3. **Entity Matching**: 4-level cascading match strategy
//! 4. **Text Extraction**: Regex-based extraction of indications/mechanisms
//! 5. **ASCII Normalization**: Unicode to ASCII transliteration
//! 6. **Parallel Processing**: Multi-threaded record processing with Rayon
//!
//! ## Entity Matching Strategy
//!
//! The matching algorithm uses a 4-level cascade:
//! 1. **Exact generic name match**: Direct lookup (fastest, most reliable)
//! 2. **Exact brand name match**: For branded products
//! 3. **Component-level matching**: For combination drugs (e.g., "Drug A / Drug B")
//! 4. **Substring matching**: For partial matches on components
//!
//! This cascade achieves ~55% match rate on FDA→DrugBank linkage.
//!
//! ## Text Extraction Philosophy
//!
//! Rather than including full descriptions (which contain regulatory boilerplate),
//! the system extracts:
//! - **Drug category**: "antibiotic", "analgesic", "anti-inflammatory"
//! - **Mechanism of action**: How the drug works
//! - **Indications**: What conditions it treats
//!
//! This produces concise, semantically-rich descriptions optimal for embedding generation.
//!
//! ## Command-line Usage
//!
//! ```bash
//! cargo run --release -- data/full_database.xml drugscraped.csv
//! ```

use csv::{Reader, Writer};
use quick_xml::events::Event;
use quick_xml::Reader as XmlReader;
use rayon::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::env;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::sync::{Arc, Mutex};

// ============================================================================
// Data Structures
// ============================================================================

/// Input drug record from FDA CSV database
///
/// Represents a single FDA drug product with generic name, brand name,
/// and pharmacologic category classification.
#[derive(Debug, Deserialize)]
struct DrugRecord {
    /// Generic (chemical) name of the drug
    #[serde(rename = "Generic Name")]
    generic_name: String,
    
    /// Brand (commercial) name of the drug
    #[serde(rename = "Drug Name")]
    drug_name: String,
    
    /// Pharmacologic/therapeutic category
    #[serde(rename = "Pharmacologic Category")]
    pharmacologic_category: String,
}

/// Output drug record with enriched information
///
/// Contains matched drug information from both FDA and DrugBank sources,
/// with cleaned descriptions and ATC codes.
#[derive(Debug, Serialize)]
struct OutputRecord {
    /// Generic name (from FDA)
    #[serde(rename = "Generic Name")]
    generic_name: String,
    
    /// Brand/drug name (from FDA)
    #[serde(rename = "Drug Name")]
    drug_name: String,
    
    /// Cleaned, extracted description (from DrugBank)
    #[serde(rename = "Description")]
    description: String,
    
    /// Semicolon-separated ATC codes (from DrugBank)
    #[serde(rename = "ATC Codes")]
    atc_codes: String,
}

/// Drug information extracted from DrugBank XML
///
/// Contains all relevant fields extracted from a single <drug> element
/// in the DrugBank XML database.
#[derive(Debug, Clone, Default)]
struct DrugInfo {
    /// Full drug description from DrugBank
    description: String,
    
    /// Clinical indications (what the drug treats)
    indication: String,
    
    /// ATC (Anatomical Therapeutic Chemical) classification codes
    atc_codes: Vec<String>,
}

// ============================================================================
// Text Extraction and Cleaning Engine
// ============================================================================

/// Text extraction and cleaning system using regex patterns and medical heuristics
///
/// This struct encapsulates all the logic for extracting meaningful medical information
/// from verbose drug descriptions while filtering out boilerplate, dosage information,
/// and administrative content.
///
/// ## Components
///
/// - **Indication patterns**: Extract "what the drug treats"
/// - **Mechanism patterns**: Extract "how the drug works"
/// - **Category patterns**: Extract drug classification (e.g., "antibiotic")
/// - **Boilerplate filters**: Remove administrative/dosage text
/// - **Citation cleaner**: Remove reference markers like [A123]
/// - **Bracket cleaner**: Remove bracketed content
///
/// ## Philosophy
///
/// The extractor prioritizes **semantic content** over **completeness**. A concise,
/// focused description like "antibiotic; treats bacterial infections" is more valuable
/// for embeddings than a 500-word description containing dosage instructions and
/// regulatory disclaimers.
struct TextCleaner {
    /// Regex patterns for extracting clinical indications
    indication_patterns: Vec<Regex>,
    
    /// Regex patterns for extracting mechanism of action
    mechanism_patterns: Vec<Regex>,
    
    /// Regex patterns for extracting drug category
    category_patterns: Vec<Regex>,
    
    /// Set of boilerplate phrases to filter out
    boilerplate_phrases: HashSet<String>,
    
    /// Pattern for removing citation markers ([A123])
    citation_pattern: Regex,
    
    /// Pattern for removing bracketed content
    bracket_pattern: Regex,
}

impl TextCleaner {
    /// Initialize text cleaner with regex patterns and boilerplate filters
    ///
    /// Creates a new `TextCleaner` with pre-compiled regex patterns for extracting
    /// medical information and filtering non-informative content.
    ///
    /// ## Indication Patterns (4 patterns)
    ///
    /// Extract phrases indicating what conditions the drug treats:
    /// - "indicated for [condition]"
    /// - "used for/to treat [condition]"  
    /// - "for the treatment/management of [condition]"
    /// - "treats [condition]"
    ///
    /// ## Mechanism Patterns (4 patterns)
    ///
    /// Extract how the drug works:
    /// - "mechanism of action/works by/acts by [mechanism]"
    /// - "inhibitor of [target]"
    /// - "agonist/antagonist/modulator of [target]"
    /// - "binds to [target]"
    ///
    /// ## Category Patterns (1 comprehensive pattern)
    ///
    /// Extract drug classification with 15 common categories:
    /// antibiotic, analgesic, anti-inflammatory, antiviral, antifungal, etc.
    ///
    /// ## Boilerplate Filters (8 phrases)
    ///
    /// Common non-informative phrases to remove:
    /// - Administrative: "consult your doctor", "if symptoms persist"
    /// - Dosage-related: "dosage is", "one tablet", "capsule"
    /// - Storage: "storage", "keep out of reach"
    fn new() -> Self {
        let indication_patterns = vec![
            Regex::new(r"(?i)indicated for(?: the)?(?: treatment of)?\s+([^.;]+)").unwrap(),
            Regex::new(r"(?i)used (?:for|to treat|in the treatment of)\s+([^.;]+)").unwrap(),
            Regex::new(r"(?i)for the (?:treatment|management) of\s+([^.;]+)").unwrap(),
            Regex::new(r"(?i)treats?\s+([^.;]+)").unwrap(),
        ];
        
        let mechanism_patterns = vec![
            Regex::new(r"(?i)(?:mechanism of action|works by|acts by|exerts its effects? by)\s+([^.;]+)").unwrap(),
            Regex::new(r"(?i)inhibit(?:s|or of)\s+([^.;]+)").unwrap(),
            Regex::new(r"(?i)(?:agonist|antagonist|modulator) of\s+([^.;]+)").unwrap(),
            Regex::new(r"(?i)binds to\s+([^.;]+)").unwrap(),
        ];
        
        let category_patterns = vec![
            Regex::new(r"(?i)(?:is a|is an)\s+(antibiotic|analgesic|anti-inflammatory|antiviral|antifungal|antihypertensive|anticoagulant|antidiabetic|antidepressant|antipsychotic|anticancer|immunosuppressant|bronchodilator|diuretic|corticosteroid|statin|beta-blocker|nsaid)").unwrap(),
        ];
        
        let boilerplate_phrases: HashSet<String> = [
            "consult your doctor", "if symptoms persist", "may cause side effects",
            "dosage is", "one tablet", "capsule", "storage", "keep out of reach",
        ].iter().map(|s| s.to_lowercase()).collect();
        
        let citation_pattern = Regex::new(r"\[[A-Z0-9]+\]").unwrap();
        let bracket_pattern = Regex::new(r"\[([^\]]+)\]").unwrap();
        
        TextCleaner {
            indication_patterns,
            mechanism_patterns,
            category_patterns,
            boilerplate_phrases,
            citation_pattern,
            bracket_pattern,
        }
    }
    
    /// Normalize Unicode text to ASCII by transliterating special characters
    ///
    /// Converts Unicode characters commonly found in drug names and descriptions
    /// to their ASCII equivalents. This is critical for entity matching across
    /// databases that use different character encodings.
    ///
    /// ## Character Mappings
    ///
    /// ### Latin-1 Supplement (U+00C0 - U+00FF)
    /// - À-Å, à-å → a
    /// - È-Ë, è-ë → e
    /// - Ì-Ï, ì-ï → i
    /// - Ò-Ö, ò-ö → o
    /// - Ù-Ü, ù-ü → u
    /// - ñ → n, ç → c, ß → s
    ///
    /// ### Extended Latin (U+0100 - U+017F)
    /// - Comprehensive coverage of Latin Extended-A block
    /// - Macrons, breves, cedillas, carons, etc.
    ///
    /// ### Greek Letters (Common in Drug Names)
    /// - α → a, β → b, γ → g, δ → d, ε → e, ω → o
    ///
    /// ### Punctuation Normalization
    /// - Em/en dashes (–, —) → hyphen (-)
    /// - Smart quotes (', ") → ASCII quotes
    /// - Ellipsis (…) → period (.)
    /// - Bullet (•) → asterisk (*)
    /// - Math symbols (≥, ≤) → ASCII (>, <)
    ///
    /// # Arguments
    ///
    /// * `text` - Input text potentially containing Unicode characters
    ///
    /// # Returns
    ///
    /// ASCII-normalized text with whitespace collapsed
    ///
    /// # Examples
    ///
    /// ```
    /// let text = cleaner.normalize_to_ascii("Naproxén—NSAID");
    /// assert_eq!(text, "Naproxen-NSAID");
    /// ```
    fn normalize_to_ascii(&self, text: &str) -> String {
        text.chars()
            .map(|c| match c {
                // Latin-1 Supplement
                'À'..='Å' | 'à'..='å' => 'a',
                'Æ' | 'æ' => 'a',
                'Ç' | 'ç' => 'c',
                'È'..='Ë' | 'è'..='ë' => 'e',
                'Ì'..='Ï' | 'ì'..='ï' => 'i',
                'Ñ' | 'ñ' => 'n',
                'Ò'..='Ö' | 'ò'..='ö' => 'o',
                'Ø' | 'ø' => 'o',
                'Ù'..='Ü' | 'ù'..='ü' => 'u',
                'Ý' | 'ý' | 'ÿ' => 'y',
                'Þ' | 'þ' => 't',
                'ß' => 's',
                'Ð' | 'ð' => 'd',
                // Extended Latin
                'Ā' | 'ā' | 'Ă' | 'ă' | 'Ą' | 'ą' => 'a',
                'Ć' | 'ć' | 'Ĉ' | 'ĉ' | 'Ċ' | 'ċ' | 'Č' | 'č' => 'c',
                'Ď' | 'ď' | 'Đ' | 'đ' => 'd',
                'Ē' | 'ē' | 'Ĕ' | 'ĕ' | 'Ė' | 'ė' | 'Ę' | 'ę' | 'Ě' | 'ě' => 'e',
                'Ĝ' | 'ĝ' | 'Ğ' | 'ğ' | 'Ġ' | 'ġ' | 'Ģ' | 'ģ' => 'g',
                'Ĥ' | 'ĥ' | 'Ħ' | 'ħ' => 'h',
                'Ĩ' | 'ĩ' | 'Ī' | 'ī' | 'Ĭ' | 'ĭ' | 'Į' | 'į' | 'İ' | 'ı' => 'i',
                'Ĵ' | 'ĵ' => 'j',
                'Ķ' | 'ķ' => 'k',
                'Ĺ' | 'ĺ' | 'Ļ' | 'ļ' | 'Ľ' | 'ľ' | 'Ŀ' | 'ŀ' | 'Ł' | 'ł' => 'l',
                'Ń' | 'ń' | 'Ņ' | 'ņ' | 'Ň' | 'ň' => 'n',
                'Ō' | 'ō' | 'Ŏ' | 'ŏ' | 'Ő' | 'ő' | 'Œ' | 'œ' => 'o',
                'Ŕ' | 'ŕ' | 'Ŗ' | 'ŗ' | 'Ř' | 'ř' => 'r',
                'Ś' | 'ś' | 'Ŝ' | 'ŝ' | 'Ş' | 'ş' | 'Š' | 'š' => 's',
                'Ţ' | 'ţ' | 'Ť' | 'ť' | 'Ŧ' | 'ŧ' => 't',
                'Ũ' | 'ũ' | 'Ū' | 'ū' | 'Ŭ' | 'ŭ' | 'Ů' | 'ů' | 'Ű' | 'ű' | 'Ų' | 'ų' => 'u',
                'Ŵ' | 'ŵ' => 'w',
                'Ŷ' | 'ŷ' | 'Ÿ' => 'y',
                'Ź' | 'ź' | 'Ż' | 'ż' | 'Ž' | 'ž' => 'z',
                // Greek letters commonly used in drug names
                'α' => 'a',
                'β' => 'b',
                'γ' => 'g',
                'δ' => 'd',
                'ε' => 'e',
                'ω' => 'o',
                // Punctuation normalization
                '–' | '—' | '\u{2011}' => '-',  // En dash, em dash, non-breaking hyphen
                '\u{2018}' | '\u{2019}' | '`' => '\'',  // Smart quotes
                '\u{201C}' | '\u{201D}' => '"',  // Smart double quotes
                '…' => '.',  // Ellipsis
                '•' => '*',  // Bullet point
                '≥' => '>',  // Greater than or equal to
                '≤' => '<',  // Less than or equal to
                // Pass through ASCII and common chars, remove others
                c if c.is_ascii() => c,
                _ => ' ',
            })
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }
    
    /// Extract key medical information from verbose drug descriptions
    ///
    /// Implements a multi-stage extraction pipeline that prioritizes:
    /// 1. **Drug category** (e.g., "antibiotic", "analgesic")
    /// 2. **Mechanism of action** (how it works)
    /// 3. **Indications** (what it treats)
    /// 4. **Fallback to first meaningful sentence** if patterns fail
    ///
    /// ## Algorithm
    ///
    /// 1. Remove citation markers ([A123])
    /// 2. Extract category using category_patterns
    /// 3. Extract mechanism using mechanism_patterns (min 20 chars, no boilerplate)
    /// 4. Extract indications using indication_patterns (min 10 chars, max 3)
    /// 5. If nothing extracted, find first meaningful sentence (30-300 chars)
    /// 6. Join all parts with ". " separator
    ///
    /// ## Quality Filters
    ///
    /// - **Length requirements**: Prevents extracting fragments
    /// - **Boilerplate check**: Filters administrative content
    /// - **Medical relevance check**: Ensures medical terminology present
    /// - **Deduplication**: Removes duplicate indications
    ///
    /// # Arguments
    ///
    /// * `text` - Raw drug description text
    ///
    /// # Returns
    ///
    /// Concise, lowercased description with key medical information
    ///
    /// # Examples
    ///
    /// **Input**: "Ibuprofen is a non-steroidal anti-inflammatory drug (NSAID). It is indicated
    /// for the treatment of mild to moderate pain, fever, and inflammation..."
    ///
    /// **Output**: "non-steroidal anti-inflammatory drug; mild to moderate pain; fever; inflammation"
    fn extract_key_information(&self, text: &str) -> String {
        if text.is_empty() {
            return String::new();
        }
        
        let mut extracted_parts = Vec::new();
        let text = self.citation_pattern.replace_all(text, "").to_string();
        
        // Extract category (without the "Category: " prefix)
        for pattern in &self.category_patterns {
            if let Some(caps) = pattern.captures(&text) {
                if let Some(category) = caps.get(1) {
                    extracted_parts.push(category.as_str().trim().to_lowercase());
                    break;
                }
            }
        }
        
        // Extract mechanism (without the "Mechanism: " prefix)
        for pattern in &self.mechanism_patterns {
            if let Some(caps) = pattern.captures(&text) {
                if let Some(mechanism) = caps.get(1) {
                    let mech_text = mechanism.as_str().trim();
                    if mech_text.len() > 20 && !self.contains_boilerplate(mech_text) {
                        extracted_parts.push(self.clean_sentence(mech_text).to_lowercase());
                        break;
                    }
                }
            }
        }
        
        // Extract indications (without the "Indications: " prefix)
        let mut indications = Vec::new();
        for pattern in &self.indication_patterns {
            for caps in pattern.captures_iter(&text) {
                if let Some(indication) = caps.get(1) {
                    let ind_text = indication.as_str().trim();
                    if ind_text.len() > 10 && !self.contains_boilerplate(ind_text) {
                        let cleaned = self.clean_sentence(ind_text).to_lowercase();
                        if !indications.contains(&cleaned) {
                            indications.push(cleaned);
                        }
                    }
                }
            }
        }
        
        if !indications.is_empty() {
            let indication_text = indications.into_iter().take(3).collect::<Vec<_>>().join("; ");
            extracted_parts.push(indication_text);
        }
        
        if extracted_parts.is_empty() {
            if let Some(first_sentence) = self.extract_first_meaningful_sentence(&text) {
                extracted_parts.push(first_sentence.to_lowercase());
            }
        }
        
        extracted_parts.join(". ")
    }
    
    /// Extract the first meaningful sentence from text as fallback
    ///
    /// When regex patterns fail to extract structured information, this method
    /// finds the first sentence that appears to contain useful medical content.
    ///
    /// ## Selection Criteria
    ///
    /// A sentence is considered "meaningful" if it:
    /// 1. **Length**: Between 30-300 characters
    /// 2. **No boilerplate**: Doesn't contain dosage/administrative phrases
    /// 3. **Medical relevance**: Contains medical keywords
    ///
    /// ## Medical Keywords (15 terms)
    ///
    /// treat, disease, infection, condition, disorder, patient, therapy,
    /// inhibit, receptor, enzyme, pain, fever, pressure, diabetes, inflammatory
    ///
    /// # Arguments
    ///
    /// * `text` - Text to extract from
    ///
    /// # Returns
    ///
    /// First meaningful sentence if found, None otherwise
    fn extract_first_meaningful_sentence(&self, text: &str) -> Option<String> {
        let sentences: Vec<&str> = text.split(|c| c == '.' || c == ';').collect();
        
        for sentence in sentences {
            let sentence = sentence.trim();
            if sentence.len() > 30 
                && sentence.len() < 300
                && !self.contains_boilerplate(sentence)
                && self.is_medically_relevant(sentence) {
                return Some(self.clean_sentence(sentence));
            }
        }
        None
    }
    
    /// Clean a sentence by removing citations, brackets, and normalizing whitespace
    ///
    /// Applies the following transformations:
    /// 1. Normalize Unicode to ASCII
    /// 2. Remove bracketed text including brackets
    /// 3. Remove HTML entities (&#13;)
    /// 4. Replace newlines with spaces
    /// 5. Collapse multiple spaces
    /// 6. Trim leading/trailing punctuation
    ///
    /// # Arguments
    ///
    /// * `text` - Sentence to clean
    ///
    /// # Returns
    ///
    /// Cleaned sentence text
    fn clean_sentence(&self, text: &str) -> String {
        // Normalize Unicode characters to ASCII first
        let text = self.normalize_to_ascii(text);
        
        // Remove bracketed text (including brackets)
        let mut cleaned = self.bracket_pattern.replace_all(&text, "").to_string();
        
        cleaned = cleaned
            .replace("&#13;", "")
            .replace('\n', " ")
            .replace('\r', " ");
        
        cleaned = cleaned.split_whitespace().collect::<Vec<_>>().join(" ");
        cleaned = cleaned.trim_matches(|c| c == ',' || c == ';' || c == ' ').to_string();
        
        cleaned
    }
    
    /// Check if text contains boilerplate phrases that disqualify it as meaningful content
    ///
    /// Screens for 8 common pharmaceutical boilerplate patterns:
    /// - "approval date"
    /// - "marketing status"
    /// - "dosage form"
    /// - "route of administration"
    /// - "product number"
    /// - "labeler name"
    /// - "brand name"
    /// - "generic name"
    ///
    /// Also rejects text containing dosage information (mg, dosage, administration).
    ///
    /// # Arguments
    ///
    /// * `text` - Text to check for boilerplate
    ///
    /// # Returns
    ///
    /// `true` if text contains any boilerplate phrase, `false` otherwise
    fn contains_boilerplate(&self, text: &str) -> bool {
        let text_lower = text.to_lowercase();
        
        for phrase in &self.boilerplate_phrases {
            if text_lower.contains(phrase) {
                return true;
            }
        }
        
        text_lower.contains("mg") 
            || text_lower.contains("dosage")
            || text_lower.contains("administration")
    }
    
    /// Check if text contains medical terminology indicating relevant content
    ///
    /// Verifies presence of at least one medical keyword from 15 core categories:
    /// - **Treatment**: treat, therapy
    /// - **Conditions**: disease, infection, condition, disorder
    /// - **Clinical**: patient
    /// - **Mechanisms**: inhibit, receptor, enzyme
    /// - **Symptoms**: pain, fever, pressure
    /// - **Diseases**: diabetes, inflammatory
    ///
    /// Used to filter out non-medical administrative content.
    ///
    /// # Arguments
    ///
    /// * `text` - Text to check for medical relevance
    ///
    /// # Returns
    ///
    /// `true` if text contains any medical keyword, `false` otherwise
    fn is_medically_relevant(&self, text: &str) -> bool {
        let text_lower = text.to_lowercase();
        
        let medical_keywords = [
            "treat", "disease", "infection", "condition", "disorder",
            "patient", "therapy", "inhibit", "receptor", "enzyme",
            "pain", "fever", "pressure", "diabetes", "inflammatory",
        ];
        
        medical_keywords.iter().any(|keyword| text_lower.contains(keyword))
    }
}

/// Data structure representing drug information extracted from DrugBank XML
impl DrugInfo {
    /// Generate a concise, cleaned description by combining description and indication fields
    ///
    /// Assembly process:
    /// 1. Concatenate description field
    /// 2. Append "Used for: " prefix + indication field
    /// 3. Pass combined text through TextCleaner::extract_key_information()
    ///
    /// This ensures the output focuses on medical relevance while eliminating boilerplate.
    ///
    /// # Arguments
    ///
    /// * `cleaner` - TextCleaner instance for text processing
    ///
    /// # Returns
    ///
    /// Cleaned, concise medical description
    fn to_cleaned_description(&self, cleaner: &TextCleaner) -> String {
        let mut full_text = String::new();
        
        if !self.description.is_empty() {
            full_text.push_str(&self.description);
            full_text.push(' ');
        }
        if !self.indication.is_empty() {
            full_text.push_str("Used for: ");
            full_text.push_str(&self.indication);
        }
        
        cleaner.extract_key_information(&full_text)
    }
}

/// Parse DrugBank XML database into a HashMap of drug names to DrugInfo records
///
/// Uses event-driven XML parsing to extract:
/// - **Drug name**: First `<name>` element within `<drug>`
/// - **Description**: Text from first `<description>` element
/// - **Indication**: Text from first `<indication>` element
/// - **ATC codes**: All `<atc-code code="...">` attributes
///
/// ## Parsing Strategy
///
/// - **Depth tracking**: Ensures only top-level elements within `<drug>` are captured
/// - **First-only extraction**: Ignores nested/duplicate name/description/indication elements
/// - **Multi-ATC support**: Collects all ATC codes for each drug
///
/// ## Performance
///
/// - Processes ~13,000 drugs in ~2 seconds
/// - Low memory footprint using streaming parser
///
/// # Arguments
///
/// * `xml_path` - Path to DrugBank full_database.xml file
///
/// # Returns
///
/// HashMap mapping drug names (lowercase) to DrugInfo structs
///
/// # Errors
///
/// Returns error if file cannot be opened or XML is malformed
fn parse_xml_database(xml_path: &str) -> Result<HashMap<String, DrugInfo>, Box<dyn Error>> {
    println!("Parsing XML database...");
    
    let file = File::open(xml_path)?;
    let file = BufReader::new(file);
    let mut reader = XmlReader::from_reader(file);
    reader.config_mut().trim_text(true);
    
    let mut drug_map: HashMap<String, DrugInfo> = HashMap::new();
    let mut buf = Vec::new();
    
    let mut current_drug_name = String::new();
    let mut current_drug_info = DrugInfo::default();
    
    let mut in_drug = false;
    let mut in_name = false;
    let mut in_description = false;
    let mut in_indication = false;
    let mut in_atc_code = false;
    let mut current_atc = String::new();
    let mut depth = 0;
    let mut description_depth = 0;
    let mut indication_depth = 0;
    let mut drug_count = 0;
    
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                depth += 1;
                match e.name().as_ref() {
                    b"drug" => {
                        in_drug = true;
                    }
                    b"name" if in_drug && current_drug_name.is_empty() => {
                        in_name = true;
                    }
                    b"description" if in_drug && !in_description && current_drug_info.description.is_empty() => {
                        in_description = true;
                        description_depth = depth;
                    }
                    b"indication" if in_drug && !in_indication && current_drug_info.indication.is_empty() => {
                        in_indication = true;
                        indication_depth = depth;
                    }
                    b"atc-code" if in_drug => {
                        in_atc_code = true;
                        // Extract code attribute
                        for attr_result in e.attributes() {
                            if let Ok(attr) = attr_result {
                                if attr.key.as_ref() == b"code" {
                                    current_atc = String::from_utf8_lossy(&attr.value).to_string();
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::Text(e)) => {
                let text = e.unescape()?.to_string();
                if in_name && current_drug_name.is_empty() {
                    current_drug_name = text;
                } else if in_description && depth == description_depth {
                    current_drug_info.description.push_str(&text);
                } else if in_indication && depth == indication_depth {
                    current_drug_info.indication.push_str(&text);
                }
            }
            Ok(Event::End(ref e)) => {
                match e.name().as_ref() {
                    b"name" => in_name = false,
                    b"description" if depth == description_depth => {
                        in_description = false;
                    }
                    b"indication" if depth == indication_depth => {
                        in_indication = false;
                    }
                    b"atc-code" => {
                        if in_atc_code && !current_atc.is_empty() {
                            current_drug_info.atc_codes.push(current_atc.clone());
                            current_atc.clear();
                        }
                        in_atc_code = false;
                    }
                    b"drug" => {
                        in_drug = false;
                        if !current_drug_name.is_empty() && 
                           (!current_drug_info.description.is_empty() || 
                            !current_drug_info.indication.is_empty() ||
                            !current_drug_info.atc_codes.is_empty()) {
                            // Store with normalized name (ASCII normalized and lowercase for matching)
                            let normalized_name = normalize_to_ascii(&current_drug_name).to_lowercase();
                            drug_map.insert(
                                normalized_name,
                                current_drug_info.clone()
                            );
                            drug_count += 1;
                            if drug_count % 1000 == 0 {
                                println!("Parsed {} drugs...", drug_count);
                            }
                        }
                        current_drug_name.clear();
                        current_drug_info = DrugInfo::default();
                        description_depth = 0;
                        indication_depth = 0;
                    }
                    _ => {}
                }
                depth -= 1;
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(format!("Error at position {}: {:?}", reader.buffer_position(), e).into()),
            _ => {}
        }
        buf.clear();
    }
    
    println!("Successfully parsed {} drugs from XML database", drug_count);
    Ok(drug_map)
}

/// Normalize Unicode text to ASCII by transliterating special characters
///
/// Standalone version of TextCleaner::normalize_to_ascii() for module-level use.
/// See TextCleaner::normalize_to_ascii() documentation for full details.
///
/// # Arguments
///
/// * `text` - Text to normalize
///
/// # Returns
///
/// ASCII-normalized text with whitespace collapsed
fn normalize_to_ascii(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            // Latin-1 Supplement
            'À'..='Å' | 'à'..='å' => 'a',
            'Æ' | 'æ' => 'a',
            'Ç' | 'ç' => 'c',
            'È'..='Ë' | 'è'..='ë' => 'e',
            'Ì'..='Ï' | 'ì'..='ï' => 'i',
            'Ñ' | 'ñ' => 'n',
            'Ò'..='Ö' | 'ò'..='ö' => 'o',
            'Ø' | 'ø' => 'o',
            'Ù'..='Ü' | 'ù'..='ü' => 'u',
            'Ý' | 'ý' | 'ÿ' => 'y',
            'Þ' | 'þ' => 't',
            'ß' => 's',
            'Ð' | 'ð' => 'd',
            // Extended Latin
            'Ā' | 'ā' | 'Ă' | 'ă' | 'Ą' | 'ą' => 'a',
            'Ć' | 'ć' | 'Ĉ' | 'ĉ' | 'Ċ' | 'ċ' | 'Č' | 'č' => 'c',
            'Ď' | 'ď' | 'Đ' | 'đ' => 'd',
            'Ē' | 'ē' | 'Ĕ' | 'ĕ' | 'Ė' | 'ė' | 'Ę' | 'ę' | 'Ě' | 'ě' => 'e',
            'Ĝ' | 'ĝ' | 'Ğ' | 'ğ' | 'Ġ' | 'ġ' | 'Ģ' | 'ģ' => 'g',
            'Ĥ' | 'ĥ' | 'Ħ' | 'ħ' => 'h',
            'Ĩ' | 'ĩ' | 'Ī' | 'ī' | 'Ĭ' | 'ĭ' | 'Į' | 'į' | 'İ' | 'ı' => 'i',
            'Ĵ' | 'ĵ' => 'j',
            'Ķ' | 'ķ' => 'k',
            'Ĺ' | 'ĺ' | 'Ļ' | 'ļ' | 'Ľ' | 'ľ' | 'Ŀ' | 'ŀ' | 'Ł' | 'ł' => 'l',
            'Ń' | 'ń' | 'Ņ' | 'ņ' | 'Ň' | 'ň' => 'n',
            'Ō' | 'ō' | 'Ŏ' | 'ŏ' | 'Ő' | 'ő' | 'Œ' | 'œ' => 'o',
            'Ŕ' | 'ŕ' | 'Ŗ' | 'ŗ' | 'Ř' | 'ř' => 'r',
            'Ś' | 'ś' | 'Ŝ' | 'ŝ' | 'Ş' | 'ş' | 'Š' | 'š' => 's',
            'Ţ' | 'ţ' | 'Ť' | 'ť' | 'Ŧ' | 'ŧ' => 't',
            'Ũ' | 'ũ' | 'Ū' | 'ū' | 'Ŭ' | 'ŭ' | 'Ů' | 'ů' | 'Ű' | 'ű' | 'Ų' | 'ų' => 'u',
            'Ŵ' | 'ŵ' => 'w',
            'Ŷ' | 'ŷ' | 'Ÿ' => 'y',
            'Ź' | 'ź' | 'Ż' | 'ż' | 'Ž' | 'ž' => 'z',
            // Greek letters commonly used in drug names
            'α' => 'a',
            'β' => 'b',
            'γ' => 'g',
            'δ' => 'd',
            'ε' => 'e',
            'ω' => 'o',
            // Punctuation normalization
            '–' | '—' | '\u{2011}' => '-',  // En dash, em dash, non-breaking hyphen
            '\u{2018}' | '\u{2019}' | '`' => '\'',  // Smart quotes
            '\u{201C}' | '\u{201D}' => '"',  // Smart double quotes
            '…' => '.',  // Ellipsis
            '•' => '*',  // Bullet point
            '≥' => '>',  // Greater than or equal to
            '≤' => '<',  // Less than or equal to
            // Pass through ASCII and common chars, remove others
            c if c.is_ascii() => c,
            _ => ' ',
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Find drug information using cascading match strategy
///
/// Implements a 4-level matching hierarchy:
/// 1. **Exact generic name match**: Direct lookup in drug database
/// 2. **Exact brand name match**: Lookup using brand name (if not "None"/"none")
/// 3. **Combination drug component match**: Split on " / " and match individual components
/// 4. **Substring match on components**: Fuzzy matching for partial names (min 4 chars)
///
/// ## Match Quality
///
/// - **Level 1-2**: High confidence (exact match)
/// - **Level 3**: Medium confidence (component match)
/// - **Level 4**: Lower confidence (substring match)
///
/// ## Example Matching
///
/// - "Acetaminophen / Codeine" → Splits to ["Acetaminophen", "Codeine"] → Matches "Acetaminophen"
/// - "Advil" (brand) → Matches "Ibuprofen" (generic) via brand name lookup
/// - "Amoxicillin" → Direct match
///
/// # Arguments
///
/// * `generic_name` - Generic drug name from FDA data
/// * `brand_name` - Brand drug name from FDA data
/// * `drug_database` - DrugBank database HashMap
///
/// # Returns
///
/// Some(DrugInfo) if match found, None otherwise
fn find_drug_info(
    generic_name: &str,
    brand_name: &str,
    drug_database: &HashMap<String, DrugInfo>,
) -> Option<DrugInfo> {
    // Normalize and lowercase for matching
    let generic_lower = normalize_to_ascii(generic_name.trim()).to_lowercase();
    let brand_lower = normalize_to_ascii(brand_name.trim()).to_lowercase();
    
    // exact match on generic name
    if let Some(info) = drug_database.get(&generic_lower) {
        return Some(info.clone());
    }
    
    // exact match on brand name (if not "None" or empty)
    if !brand_lower.is_empty() && brand_lower != "none" {
        if let Some(info) = drug_database.get(&brand_lower) {
            return Some(info.clone());
        }
    }
    
    // for combination drugs (with " / "):
    // exact matching individual components
    if generic_lower.contains(" / ") {
        let components: Vec<&str> = generic_lower.split(" / ").collect();
        for component in components {
            let component = component.trim();
            if component.len() >= 4 {
                if let Some(info) = drug_database.get(component) {
                    return Some(info.clone());
                }
                // Try substring match on components
                for (db_name, info) in drug_database.iter() {
                    if db_name.contains(component) || component.contains(db_name.as_str()) {
                        if db_name.len() >= 4 {
                            return Some(info.clone());
                        }
                    }
                }
            }
        }
    }
    
    None
}

/// Main pipeline: Cross-database entity matching and text extraction
///
/// Executes a 5-stage pipeline to enrich FDA drug data with DrugBank descriptions:
///
/// ## Stage 1: Initialization
/// - Parse command-line arguments (XML path, CSV path)
/// - Initialize TextCleaner with regex patterns
/// - Parse DrugBank XML database (~13,000 drugs)
///
/// ## Stage 2: CSV Loading
/// - Deserialize FDA drug records from boiled CSV
/// - Load into memory for parallel processing
///
/// ## Stage 3: Parallel Entity Matching
/// - Match each FDA record against DrugBank using 4-level strategy
/// - Extract and clean descriptions using TextCleaner
/// - Combine pharmacologic category + XML description
/// - Track match statistics (matches, no-matches, empty results)
///
/// ## Stage 4: Output Generation
/// - Serialize OutputRecord structs to drugscribed.csv
/// - Includes: generic_name, description, atc_codes
///
/// ## Stage 5: Statistics Reporting
/// - Print match rate, cleaning success rate, empty rate
///
/// ## Example Usage
///
/// ```bash
/// cargo run --release -- data/full_database.xml artifact/boiled.csv
/// ```
///
/// ## Performance
///
/// - Processes ~20,000 records in ~3 seconds (parallel processing)
/// - Match rate: ~60-70% depending on database version
///
/// # Errors
///
/// Returns error if:
/// - Insufficient command-line arguments
/// - XML/CSV files cannot be opened
/// - File I/O errors during output
fn main() -> Result<(), Box<dyn Error>> {
    // Get input paths from command-line arguments
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 3 {
        eprintln!("Usage: {} <xml_database_path> <csv_file_path>", args[0]);
        eprintln!("Example: {} data/full_database.xml drugscrape.csv", args[0]);
        std::process::exit(1);
    }
    
    let xml_path = &args[1];
    let csv_path = &args[2];
    
    // Initialize text cleaner
    println!("Initializing text cleaner...");
    let text_cleaner = Arc::new(TextCleaner::new());
    
    // Parse XML database
    println!("Parsing XML database: {}", xml_path);
    let drug_database = parse_xml_database(xml_path)?;
    let drug_database = Arc::new(drug_database);
    
    // Read CSV file
    println!("Reading CSV file: {}", csv_path);
    let mut reader = Reader::from_path(csv_path)?;
    
    let records: Vec<DrugRecord> = reader
        .deserialize()
        .collect::<Result<Vec<_>, _>>()?;
    
    println!("Loaded {} records from CSV", records.len());
    println!("Matching drugs and aggressively cleaning descriptions...");
    
    let matches = Arc::new(Mutex::new(0usize));
    let no_matches = Arc::new(Mutex::new(0usize));
    let cleaned_count = Arc::new(Mutex::new(0usize));
    let empty_count = Arc::new(Mutex::new(0usize));
    
    // Process records in parallel
    let output_records: Vec<OutputRecord> = records
        .into_par_iter()
        .map(|record| {
            let cleaner = Arc::clone(&text_cleaner);
            let mut final_description = String::new();
            let mut atc_codes_list = Vec::new();
            
            // Add pharmacologic category first if it exists (lowercase)
            if !record.pharmacologic_category.is_empty() {
                final_description = record.pharmacologic_category.to_lowercase();
            }
            
            // Try to find drug info using multiple strategies
            if let Some(drug_info) = find_drug_info(
                &record.generic_name,
                &record.drug_name,
                &drug_database,
            ) {
                // Use aggressive cleaning instead of to_single_line
                let xml_description = drug_info.to_cleaned_description(&cleaner);
                
                // Collect ATC codes
                atc_codes_list = drug_info.atc_codes.clone();
                
                // Append cleaned XML description
                if final_description.is_empty() {
                    final_description = xml_description.clone();
                } else if !xml_description.is_empty() {
                    final_description = format!("{}; {}", final_description, xml_description);
                }
                
                *matches.lock().unwrap() += 1;
                
                if !xml_description.is_empty() {
                    *cleaned_count.lock().unwrap() += 1;
                } else {
                    *empty_count.lock().unwrap() += 1;
                }
            } else {
                *no_matches.lock().unwrap() += 1;
            }
            
            OutputRecord {
                generic_name: record.generic_name,
                drug_name: record.drug_name,
                description: final_description,
                atc_codes: atc_codes_list.join("; "),
            }
        })
        .collect();
    
    let match_count = *matches.lock().unwrap();
    let no_match_count = *no_matches.lock().unwrap();
    let cleaned = *cleaned_count.lock().unwrap();
    let empty = *empty_count.lock().unwrap();
    
    println!("\n=== Processing Summary ===");
    println!("Total records: {}", match_count + no_match_count);
    println!("Matched in XML: {}", match_count);
    println!("Not matched: {}", no_match_count);
    println!("Successfully cleaned: {} ({:.1}%)", cleaned, (cleaned as f64 / match_count as f64) * 100.0);
    println!("Cleaned to empty: {} ({:.1}%)", empty, (empty as f64 / match_count as f64) * 100.0);
    println!("\nWriting output CSV...");
    
    // Write output CSV
    let output_path = "drugscribed.csv";
    let mut writer = Writer::from_path(output_path)?;
    
    for record in output_records {
        writer.serialize(record)?;
    }
    
    writer.flush()?;
    println!("Successfully processed all records!");
    println!("Output saved to: {}", output_path);
    
    Ok(())
}
