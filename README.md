# BedFrame

**By**: Zhean Robby Ganituen <br>
- GitHub: [`@zrygan`](https://github.com/zrygan)
- ([DLSU Mail](zhean_robby_ganituen@dlsu.edu.ph))
- [MangoCats]([url](https://github.com/Mango-Cats))

A sentence em**bed**dings **frame**work for pharmaceutical data from drug databases and DrugBank using sentence transformers.

This framework enables semantic similarity comparison of pharmacologic drug descriptions.

As a test **bed**, the Philippine Food Drug Administration (FDA) database of registered human drugs was used for demonstration and validation; [it is available here](https://verification.fda.gov.ph/ALL_DrugProductslist.php).

## Overview

BedFrame implements a complete pharmaceutical data processing pipeline that:
1. **Cleans** raw FDA drug product data
2. **Enriches** it with DrugBank descriptions and ATC codes
3. **Removes** boilerplate phrases using automated detection
4. **Generates** semantic embeddings using BioBERT
5. **Enables** similarity-based drug retrieval and comparison

## Architecture

```
┌─────────────────┐
│  Raw FDA Data   │  drug_products.csv
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Drugscraper    │  Rust - Parallel text cleaning & normalization
└────────┬────────┘
         │ drugscraped.csv
         ▼
┌─────────────────┐
│  Drugscriber    │  Rust - Entity matching with DrugBank XML
└────────┬────────┘
         │ drugscribed.csv
         ▼
┌─────────────────┐
│     Boil        │  Rust - Automated boilerplate detection
└────────┬────────┘
         │ (updates unboil.rs)
         ▼
┌─────────────────┐
│    Unboil       │  Rust - Remove detected boilerplate
└────────┬────────┘
         │ boiled.csv
         ▼
┌─────────────────┐
│  Drugsberter    │  Python - Generate BioBERT embeddings
└────────┬────────┘
         │ drugsberted.h5
         ▼
┌─────────────────┐
│ Embedding Store │  HDF5 - Compressed semantic vectors
└─────────────────┘
```

## Components

### 1. Drugscraper (Rust)
*Drug scraper*
**Purpose**: Clean and normalize raw drug product data

**Features**:
- Unicode tp ASCII normalization
- Parallel processing with Rayon
- Deduplication
- Special character removal

**Output**: `drugscraped.csv`

### 2. Drugscriber (Rust)
*Drug describer*
**Purpose**: Cross-database entity matching and text extraction

**Features**:
- 4-level cascading match strategy
- DrugBank XML parsing
- Regex-based information extraction
- ATC code collection

**Modules**:
- `main.rs` - Entity matching pipeline
- `boil.rs` - Boilerplate detection using n-gram analysis
- `unboil.rs` - Boilerplate removal

**Requires**: DrugBank complete data (an XML file).
**Input**: `drugscraped.csv`

**Output**: `drugscribed.csv` → `boiled.csv`

### 3. Drugsberter (Python)
**Purpose**: Generate semantic embeddings using a sentence transformer model

**Features**:
- Separate embeddings for descriptions and ATC codes
- HDF5 storage with gzip compression
- GPU/NPU acceleration support

**Input**: `boiled.csv`  
**Output**: `drugsberted.h5`

## Quick Start

### Prerequisites
```bash
# Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Python 3.11+
pip install sentence-transformers torch h5py pandas tqdm
```

### Running the Full Pipeline
```bash
# Run the Rust pipeline (drugscraper → drugscriber → boil → unboil)
python scripts/runscript.py

# Generate embeddings
cd drugsberter && uv run main.py
```

## Configuration

Edit `config.toml` in the root directory:
```toml
[paths]
ph_fda_drugs_csv = "data/drug_products.csv"
drugbank_data = "data/full_database.xml"
output_file = "artifact/drugscribed.csv"
unboiled_output_file = "artifact/boiled.csv"
```

Edit `drugsberter/config.toml` for embedding configuration:
```toml
[model]
name = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
device = "cuda"  # or "cpu", "npu"

[data]
csv_file = "../artifact/boiled.csv"
```

## Data Requirements

### Input Files
1. **Drug Products CSV**
   - Required Columns: Generic Name, Brand Name, Pharmacologic Category

2. **DrugBank XML** (`data/full_database.xml`)
   - Download from: https://go.drugbank.com/releases/latest

### Output Files
- `artifact/drugscraped.csv` - Cleaned FDA data
- `artifact/drugscribed.csv` - Enriched with DrugBank
- `artifact/boiled.csv` - Boilerplate removed
- `artifact/drugsberted.h5` - Semantic embeddings

## Project Structure

```
eacl/
├── config.toml              # Root configuration
├── README.md                # This file
├── drugscraper/             # Rust: drug data extraction and cleaning
│   ├── Cargo.toml
│   └── src/main.rs
├── drugscriber/             # Rust: Entity matching + boilerplate
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs          # Entity matching
│       ├── boil.rs          # Boilerplate detection
│       └── unboil.rs        # Boilerplate removal
├── drugsberter/             # Python: Embedding generation
│   ├── config.toml
│   ├── pyproject.toml
│   └── main.py
└── scripts/
    ├── runscript.py         # Automated pipeline
    └── csv_to_sql.py        # Utility: CSV→SQL conversion
```

## Citation

> Forthcoming

## License

Apache License 2.0 - See LICENSE file for details

## Acknowledgments

- **Philippine FDA** - Regulatory drug list
- **DrugBank** - Pharmaceutical database
- **BioBERT** - Biomedical language model
- **Rust Community** - Rayon, CSV, Quick-XML
- **Python Community** - Sentence-Transformers, HuggingFace 

