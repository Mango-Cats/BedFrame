import csv
import sys
import os

def csv_to_sql(csv_file, table_name=None):
    """Convert CSV file to SQL INSERT statements"""
    
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found")
        sys.exit(1)
    
    # Use filename (without extension) as table name if not provided
    if table_name is None:
        table_name = os.path.splitext(os.path.basename(csv_file))[0]
    
    # Clean table name (replace spaces and special chars with underscores)
    table_name = ''.join(c if c.isalnum() else '_' for c in table_name).lower()
    
    output_file = csv_file.rsplit('.', 1)[0] + '.sql'
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        
        if not headers:
            print("Error: CSV file has no headers")
            sys.exit(1)
        
        # Clean column names
        clean_headers = [''.join(c if c.isalnum() else '_' for c in h).lower() for h in headers]
        
        # Read all rows
        rows = list(reader)
        
        if not rows:
            print("Warning: CSV file has no data rows")
    
    # Write SQL file
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write CREATE TABLE statement
        f.write(f"-- Generated from {os.path.basename(csv_file)}\n")
        f.write(f"-- Total rows: {len(rows)}\n\n")
        
        f.write(f"DROP TABLE IF EXISTS {table_name};\n\n")
        f.write(f"CREATE TABLE {table_name} (\n")
        for i, col in enumerate(clean_headers):
            f.write(f"    {col} TEXT")
            if i < len(clean_headers) - 1:
                f.write(",")
            f.write("\n")
        f.write(");\n\n")
        
        # Write INSERT statements
        for row in rows:
            values = []
            for header in headers:
                value = row[header]
                # Escape single quotes
                value = value.replace("'", "''")
                values.append(f"'{value}'")
            
            f.write(f"INSERT INTO {table_name} ({', '.join(clean_headers)}) VALUES ({', '.join(values)});\n")
    
    print(f"Successfully converted {len(rows)} rows from '{csv_file}'")
    print(f"SQL output saved to: {output_file}")
    print(f"Table name: {table_name}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python csv_to_sql.py <csv_file> [table_name]")
        print("\nExample:")
        print("  python csv_to_sql.py data.csv")
        print("  python csv_to_sql.py data.csv my_table")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    table_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    csv_to_sql(csv_file, table_name)

if __name__ == "__main__":
    main()
