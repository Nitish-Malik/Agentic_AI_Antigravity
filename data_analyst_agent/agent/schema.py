import os
import pandas as pd
from typing import Dict, Any, List

def get_dataset_metadata(file_path: str, sheet_name: str = None) -> Dict[str, Any]:
    """
    Reads an Excel or CSV file and extracts schema metadata to share with the LLM.
    Avoids loading full rows into the prompt, just provides structure.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    _, ext = os.path.splitext(file_path.lower())
    
    sheets = []
    df = None
    
    # 1. Handle Excel files
    if ext in ['.xlsx', '.xls', '.xlsm']:
        try:
            excel_file = pd.ExcelFile(file_path)
            sheets = excel_file.sheet_names
            
            if sheet_name is None:
                # Default to the first sheet
                sheet_name = sheets[0]
            elif sheet_name not in sheets:
                raise ValueError(f"Sheet '{sheet_name}' not found in Excel file. Available sheets: {sheets}")
                
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        except Exception as e:
            raise ValueError(f"Error reading Excel file: {str(e)}")
            
    # 2. Handle CSV files
    elif ext == '.csv':
        try:
            df = pd.read_csv(file_path)
            sheets = ["Default"]
            sheet_name = "Default"
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {str(e)}")
            
    else:
        raise ValueError(f"Unsupported file format: {ext}. Only Excel and CSV files are supported.")
        
    # 3. Extract schema statistics
    if df is None:
        raise ValueError("Failed to load data dataframe.")
        
    # Columns list
    columns = [str(col) for col in df.columns.tolist()]
    
    # Data types
    dtypes = {str(col): str(dtype) for col, dtype in df.dtypes.items()}
    
    # Shape [rows, cols]
    shape = list(df.shape)
    
    # Missing values count per column
    missing_values = {str(col): int(val) for col, val in df.isnull().sum().items()}
    
    # Take a 3-row sample, convert to dict list, handling NaN values for JSON compatibility
    sample_df = df.head(3).copy()
    # Replace NaN/NaT with None so it translates to JSON null
    sample_df = sample_df.astype(object).where(pd.notnull(sample_df), None)
    
    # For datetimes, convert to string
    for col in sample_df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            sample_df[col] = sample_df[col].apply(lambda x: x.isoformat() if x is not None else None)
            
    sample_data = sample_df.to_dict(orient='records')
    
    return {
        "columns": columns,
        "dtypes": dtypes,
        "shape": shape,
        "missing_values": missing_values,
        "sample_data": sample_data,
        "sheets": sheets,
        "active_sheet": sheet_name
    }
