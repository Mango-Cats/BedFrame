#!/usr/bin/env python3
"""
Drug Processing Pipeline Automation Script

This script automates the drug data processing pipeline:
1. Runs drugscraper to clean and extract drug data from FDA CSV
2. Runs drugscriber to enrich the data with DrugBank information
3. Outputs the final enriched drug database
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path

try:
    import tomli
except ImportError:
    print("Error: tomli package not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tomli"])
    import tomli


def load_config(config_path="config.toml"):
    """Load configuration from TOML file."""
    try:
        with open(config_path, "rb") as f:
            config = tomli.load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading configuration: {e}")
        sys.exit(1)


def check_file_exists(file_path, description):
    """Check if a required file exists."""
    if not os.path.exists(file_path):
        print(f"Error: {description} not found at: {file_path}")
        return False
    return True


def run_command(command, description, cwd=None):
    """Run a shell command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print("Warnings/Info:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {description} failed!")
        print(f"Exit code: {e.returncode}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False


def main():
    """Main execution function."""
    print("="*60)
    print("Drug Processing Pipeline")
    print("="*60)
    
    # Get root directory (parent of scripts directory)
    root_dir = Path(__file__).parent.parent.absolute()
    data_dir = root_dir / "data"
    
    # Load configuration
    config = load_config(root_dir / "config.toml")
    
    # Extract paths from config (all relative to root)
    fda_csv = root_dir / config["paths"]["ph_fda_drugs_csv"]
    drugbank_xml = root_dir / config["paths"]["drugbank_data"]
    final_output = root_dir / config["paths"]["output_file"]
    unboiled_output_file = root_dir / config["paths"].get("unboiled_output_file", "artifact/boiled.csv")
    
    # Get intermediate paths (with defaults in data/)
    settings = config.get("settings", {})
    scraped_csv_name = settings.get("scraped_csv", "data/drugscraped.csv")
    scraped_csv = root_dir / scraped_csv_name
    
    # Validate input files
    print("\nValidating input files...")
    if not check_file_exists(str(fda_csv), "FDA drug products CSV"):
        sys.exit(1)
    if not check_file_exists(str(drugbank_xml), "DrugBank XML database"):
        sys.exit(1)
    
    print("✓ All input files found")
    
    # Project directories
    drugscraper_dir = root_dir / "drugscraper"
    drugscriber_dir = root_dir / "drugscriber"
    
    # Check if Rust projects exist
    if not (drugscraper_dir / "Cargo.toml").exists():
        print(f"Error: drugscraper project not found at {drugscraper_dir}")
        sys.exit(1)
    if not (drugscriber_dir / "Cargo.toml").exists():
        print(f"Error: drugscriber project not found at {drugscriber_dir}")
        sys.exit(1)
    
    # Ensure data directory exists
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Build drugscraper (if needed)
    print("\n" + "="*60)
    print("Building drugscraper...")
    print("="*60)
    if not run_command(
        ["cargo", "build", "--release"],
        "Building drugscraper",
        cwd=str(drugscraper_dir)
    ):
        print("Failed to build drugscraper")
        sys.exit(1)
    
    # Step 2: Run drugscraper with input from data/ and output to data/
    drugscraper_exe = drugscraper_dir / "target" / "release" / "drugscraper.exe"
    if not drugscraper_exe.exists():
        # Try Unix executable name
        drugscraper_exe = drugscraper_dir / "target" / "release" / "drugscraper"
    
    if not run_command(
        [str(drugscraper_exe), str(fda_csv.absolute())],
        "Running drugscraper to clean and extract drug data",
        cwd=str(drugscraper_dir)
    ):
        print("Failed to run drugscraper")
        sys.exit(1)
    
    # Move output from drugscraper/ to data/
    drugscraper_output = drugscraper_dir / "drugscraped.csv"
    if not drugscraper_output.exists():
        print(f"Error: Scraped CSV not found at {drugscraper_output}")
        sys.exit(1)
    
    scraped_csv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(drugscraper_output, scraped_csv)
    print(f"✓ Scraped data saved to: {scraped_csv}")
    
    # Clean up temporary file in drugscraper directory
    drugscraper_output.unlink()
    print(f"✓ Removed temporary file: {drugscraper_output}")
    
    # Step 3: Build drugscriber (if needed)
    print("\n" + "="*60)
    print("Building drugscriber...")
    print("="*60)
    if not run_command(
        ["cargo", "build", "--release"],
        "Building drugscriber",
        cwd=str(drugscriber_dir)
    ):
        print("Failed to build drugscriber")
        sys.exit(1)
    
    # Step 4: Run drugscriber with files from data/
    drugscriber_exe = drugscriber_dir / "target" / "release" / "drugscriber.exe"
    if not drugscriber_exe.exists():
        # Try Unix executable name
        drugscriber_exe = drugscriber_dir / "target" / "release" / "drugscriber"
    
    if not run_command(
        [
            str(drugscriber_exe),
            str(drugbank_xml.absolute()),
            str(scraped_csv.absolute())
        ],
        "Running drugscriber to enrich data with DrugBank information",
        cwd=str(drugscriber_dir)
    ):
        print("Failed to run drugscriber")
        sys.exit(1)
    
    # Move output from drugscriber/ to data/
    drugscriber_output = drugscriber_dir / "drugscribed.csv"
    if not drugscriber_output.exists():
        print(f"Error: Enriched CSV not found at {drugscriber_output}")
        sys.exit(1)
    
    enriched_csv = data_dir / "drugscribed.csv"
    shutil.copy2(drugscriber_output, enriched_csv)
    print(f"✓ Enriched data saved to: {enriched_csv}")
    
    # Clean up temporary file in drugscriber directory
    drugscriber_output.unlink()
    print(f"✓ Removed temporary file: {drugscriber_output}")
    
    # Step 5: Run boil to analyze boilerplate and update unboil
    print("\n" + "="*60)
    print("Running boil to analyze boilerplate phrases...")
    print("="*60)
    
    boil_exe = drugscriber_dir / "target" / "release" / "boil.exe"
    if not boil_exe.exists():
        boil_exe = drugscriber_dir / "target" / "release" / "boil"
    
    if not run_command(
        [str(boil_exe), str(enriched_csv.absolute())],
        "Analyzing boilerplate phrases",
        cwd=str(drugscriber_dir)
    ):
        print("Failed to run boil")
        sys.exit(1)
    
    # Step 6: Run unboil to remove boilerplate
    print("\n" + "="*60)
    print("Running unboil to remove boilerplate phrases...")
    print("="*60)
    
    unboil_exe = drugscriber_dir / "target" / "release" / "unboil.exe"
    if not unboil_exe.exists():
        unboil_exe = drugscriber_dir / "target" / "release" / "unboil"
    
    if not run_command(
        [str(unboil_exe), str(enriched_csv.absolute())],
        "Removing boilerplate phrases",
        cwd=str(drugscriber_dir)
    ):
        print("Failed to run unboil")
        sys.exit(1)
    
    # Move unboiled output to artifact directory
    unboiled_output = drugscriber_dir / "unboiled.csv"
    if not unboiled_output.exists():
        print(f"Error: Unboiled CSV not found at {unboiled_output}")
        sys.exit(1)
    
    # Save unboiled file separately, don't overwrite the enriched file
    unboiled_final = unboiled_output_file
    unboiled_final.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(unboiled_output, unboiled_final)
    print(f"✓ Unboiled (clean) data saved to: {unboiled_final}")
    
    # Clean up temporary file in drugscriber directory
    unboiled_output.unlink()
    print(f"✓ Removed temporary file: {unboiled_output}")
    
    print("\n" + "="*60)
    print("SUCCESS!")
    print("="*60)
    print(f"Enriched drug database saved to: {final_output.absolute()}")
    print(f"Clean (unboiled) database saved to: {unboiled_final.absolute()}")
    print(f"Clean file size: {unboiled_final.stat().st_size:,} bytes")
    
    # Show some statistics
    try:
        with open(unboiled_final, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f) - 1  # Subtract header
        print(f"Total drugs processed: {line_count:,}")
    except Exception as e:
        print(f"Could not read final file for statistics: {e}")
    
    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
