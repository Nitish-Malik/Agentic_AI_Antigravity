import os
import sys
import shutil

# Make sure agent packages are importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_tests():
    print("=========================================")
    print("      Testing Core Agent Components      ")
    print("=========================================")
    
    # 1. Test Metadata Extraction
    print("\n--- Test 1: Metadata Extraction ---")
    try:
        from agent.schema import get_dataset_metadata
        metadata = get_dataset_metadata("sample_data/dirty_data.xlsx")
        print("\033[92m[SUCCESS] Metadata loaded successfully.\033[0m")
        print(f"  - Sheets found: {metadata['sheets']}")
        print(f"  - Active sheet: {metadata['active_sheet']}")
        print(f"  - Dimensions: {metadata['shape']}")
        print(f"  - Columns found: {metadata['columns']}")
        print(f"  - Sample row keys: {list(metadata['sample_data'][0].keys())}")
    except Exception as e:
        print(f"\033[91m[FAILURE] Metadata extraction failed: {str(e)}\033[0m")
        sys.exit(1)
        
    # 2. Test Code Execution Sandbox
    print("\n--- Test 2: Code Execution Sandbox ---")
    try:
        from agent.executor import execute_python_code
        
        # Test code that reads the sample data, does a simple check, and saves a dummy text file
        test_code = """
import pandas as pd
import os

print("Hello from code sandbox!")
df = pd.read_excel("sample_data/dirty_data.xlsx")
print(f"Loaded DataFrame with shape: {df.shape}")

# Create output folder if not exist
os.makedirs("output", exist_ok=True)
with open("output/test_execution_file.txt", "w") as f:
    f.write("Successfully executed in sandbox.")
print("Created txt file.")
"""
        cwd = os.getcwd()
        # Clean previous test files
        test_txt_path = os.path.join(cwd, "output", "test_execution_file.txt")
        if os.path.exists(test_txt_path):
            os.remove(test_txt_path)
            
        result = execute_python_code(test_code, cwd)
        
        if result["success"] and os.path.exists(test_txt_path):
            print("\033[92m[SUCCESS] Code execution sandbox succeeded.\033[0m")
            print(f"  - Stdout: {result['stdout'].strip()}")
            print(f"  - New files detected: {result['new_files']}")
        else:
            print(f"\033[91m[FAILURE] Code execution failed or file not created.\033[0m")
            print(f"  - Success: {result['success']}")
            print(f"  - Stdout: {result['stdout']}")
            print(f"  - Stderr: {result['stderr']}")
            sys.exit(1)
    except Exception as e:
        print(f"\033[91m[FAILURE] Code execution module error: {str(e)}\033[0m")
        sys.exit(1)
        
    # 3. Test Graph compilation and imports
    print("\n--- Test 3: LangGraph Compilation ---")
    try:
        from agent import app
        print("\033[92m[SUCCESS] Graph imported and compiled successfully.\033[0m")
    except Exception as e:
        print(f"\033[91m[FAILURE] LangGraph compilation failed: {str(e)}\033[0m")
        sys.exit(1)

    print("\n=========================================")
    print("      All local core components OK!      ")
    print("=========================================")

if __name__ == "__main__":
    run_tests()
