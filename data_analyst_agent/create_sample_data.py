import os
import pandas as pd
import numpy as np

def generate_dirty_data():
    """Generates a dirty dataset and saves it as an Excel file to test the agent."""
    # Ensure sample_data directory exists
    os.makedirs("sample_data", exist_ok=True)
    
    # Raw data lists with duplicates, missing values, formatting issues, and noise
    data = {
        "Transaction_ID": [1001, 1002, 1003, 1004, 1005, 1002, 1006, 1007, 1008, 1009, 1010, 1011],
        "Date": [
            "2026/06/01", 
            "2026-06-02", 
            "03-06-2026", 
            "2026/06/04", 
            None,          # Missing date
            "2026-06-02",  # Duplicate row values
            "2026/06/06", 
            "07-06-2026", 
            "2026-06-08", 
            "2026/06/09", 
            "2026-06-10", 
            "11-06-2026"
        ],
        "Customer_Name": [
            "  Alice Smith  ", 
            "Bob Jones", 
            "charlie brown", # lowercase
            "  david miller", 
            "Eva Green  ", 
            "Bob Jones",      # Duplicate
            "Frank Castle", 
            "Grace Hopper ", 
            "  Heidi Klum", 
            "Ivan the Terrible", 
            "  Jack Sparrow", 
            "Karen Page"
        ],
        "Product_Category": [
            "Electronics", 
            "Clothing", 
            "Home", 
            "Electronics", 
            "Clothing", 
            "Clothing", 
            "Home", 
            None,          # Missing category
            "Electronics", 
            "Home", 
            "Clothing", 
            None           # Missing category
        ],
        "Amount": [
            "$1,200.50",   # Formatting issues (currency symbol, comma)
            "450.00", 
            " 150.75 ", 
            "$89.99", 
            "620.00", 
            "450.00",      # Duplicate
            "-999.00",     # Outlier/anomaly
            "310.20", 
            None,          # Missing value
            "120.00", 
            "$2,500.00", 
            " 75.00 "
        ],
        "Returned": [
            "No", 
            "no",          # Casing inconsistency
            "Yes", 
            "NO", 
            "No", 
            "no", 
            "YES", 
            "No", 
            "No", 
            "Yes", 
            None,          # Missing
            "No"
        ]
    }
    
    df = pd.DataFrame(data)
    
    file_path = "sample_data/dirty_data.xlsx"
    df.to_excel(file_path, index=False)
    print(f"Sample dirty data generated successfully at: {file_path}")
    print("\nDataset Preview:")
    print(df.head())

if __name__ == "__main__":
    generate_dirty_data()
