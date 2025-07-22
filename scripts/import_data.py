#!/usr/bin/env python3
"""
Import CSV data into Supabase database
"""
import os
import sys
import pandas as pd
from typing import Dict, List
import json

# Add parent directory to path to import from project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_csv_data(file_path: str) -> pd.DataFrame:
    """Load and preprocess CSV data"""
    print(f"Loading CSV data from {file_path}...")
    
    # Read CSV file
    df = pd.read_csv(file_path)
    
    print(f"Loaded {len(df)} records")
    print(f"Columns: {list(df.columns)}")
    
    return df

def prepare_data_for_supabase(df: pd.DataFrame) -> List[Dict]:
    """Convert DataFrame to format suitable for Supabase insertion"""
    print("Preparing data for Supabase...")
    
    # Rename columns to match database schema (snake_case)
    column_mapping = {
        'customerID': 'customer_id',
        'SeniorCitizen': 'senior_citizen',
        'Partner': 'partner',
        'Dependents': 'dependents',
        'PhoneService': 'phone_service',
        'MultipleLines': 'multiple_lines',
        'InternetService': 'internet_service',
        'OnlineSecurity': 'online_security',
        'OnlineBackup': 'online_backup',
        'DeviceProtection': 'device_protection',
        'TechSupport': 'tech_support',
        'StreamingTV': 'streaming_tv',
        'StreamingMovies': 'streaming_movies',
        'Contract': 'contract',
        'PaperlessBilling': 'paperless_billing',
        'PaymentMethod': 'payment_method',
        'MonthlyCharges': 'monthly_charges',
        'TotalCharges': 'total_charges',
        'Churn': 'churn',
        'tenure_group': 'tenure_group',
        'monthly_charges_group': 'monthly_charges_group'
    }
    
    # Rename columns
    df_renamed = df.rename(columns=column_mapping)
    
    # Handle data type conversions
    # Convert numeric columns
    df_renamed['monthly_charges'] = pd.to_numeric(df_renamed['monthly_charges'], errors='coerce')
    df_renamed['total_charges'] = pd.to_numeric(df_renamed['total_charges'], errors='coerce')
    df_renamed['tenure'] = pd.to_numeric(df_renamed['tenure'], errors='coerce')
    df_renamed['senior_citizen'] = pd.to_numeric(df_renamed['senior_citizen'], errors='coerce')
    df_renamed['churn'] = pd.to_numeric(df_renamed['churn'], errors='coerce')
    
    # Handle missing values
    df_renamed = df_renamed.fillna('')
    
    # Convert to list of dictionaries
    records = df_renamed.to_dict('records')
    
    print(f"Prepared {len(records)} records for insertion")
    return records

def batch_insert_data(records: List[Dict], batch_size: int = 1000) -> bool:
    """Insert data in batches using Supabase"""
    print(f"Inserting {len(records)} records in batches of {batch_size}...")
    
    total_batches = (len(records) + batch_size - 1) // batch_size
    
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} records)...")
        
        # Convert batch to SQL INSERT statement
        values_list = []
        for record in batch:
            # Escape single quotes and format values
            values = []
            for key, value in record.items():
                if pd.isna(value) or value == '':
                    values.append('NULL')
                elif isinstance(value, str):
                    # Escape single quotes
                    escaped_value = str(value).replace("'", "''")
                    values.append(f"'{escaped_value}'")
                else:
                    values.append(str(value))
            
            values_list.append(f"({', '.join(values)})")
        
        # Create INSERT statement
        columns = ', '.join(records[0].keys())
        values_str = ', '.join(values_list)
        
        sql_query = f"""
        INSERT INTO customers ({columns})
        VALUES {values_str}
        ON CONFLICT (customer_id) DO NOTHING;
        """
        
        # Save SQL to file for manual execution (since we're using MCP)
        sql_file = f"batch_{batch_num}.sql"
        with open(sql_file, 'w') as f:
            f.write(sql_query)
        
        print(f"Generated SQL file: {sql_file}")
    
    print("Data preparation complete. SQL files generated for manual execution.")
    return True

def main():
    """Main function to import data"""
    csv_file = "cleaned_telco_customer_churn.csv"
    
    if not os.path.exists(csv_file):
        print(f"Error: CSV file '{csv_file}' not found")
        return False
    
    try:
        # Load CSV data
        df = load_csv_data(csv_file)
        
        # Prepare data for Supabase
        records = prepare_data_for_supabase(df)
        
        # Take a sample for testing first
        sample_records = records[:100]  # First 100 records for testing
        
        print(f"Sample record structure:")
        print(json.dumps(sample_records[0], indent=2, default=str))
        
        # Generate SQL for sample
        batch_insert_data(sample_records, batch_size=50)
        
        print("\nData import preparation completed successfully!")
        print("Next steps:")
        print("1. Review the generated SQL files")
        print("2. Execute them using Supabase MCP tools")
        print("3. Validate the data")
        
        return True
        
    except Exception as e:
        print(f"Error during data import: {e}")
        return False

if __name__ == "__main__":
    main()