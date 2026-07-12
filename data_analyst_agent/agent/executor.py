import sys
import os
import subprocess
import tempfile
from typing import Dict, Any, List

def execute_python_code(code_string: str, workspace_dir: str) -> Dict[str, Any]:
    """
    Executes Python code in a subprocess using the project's virtual environment.
    Captures stdout, stderr, and tracks any files created during execution.
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.join(workspace_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # We will write the code to a temporary file
    temp_file_fd, temp_file_path = tempfile.mkstemp(suffix=".py", dir=workspace_dir)
    
    # Record files in the output directory before execution
    before_files = set()
    if os.path.exists(output_dir):
        before_files = set(os.listdir(output_dir))
        
    try:
        with os.fdopen(temp_file_fd, 'w') as f:
            f.write(code_string)
            
        # Determine the virtual environment python interpreter
        venv_python = os.path.join(workspace_dir, "venv", "bin", "python")
        if not os.path.exists(venv_python):
            # Fallback to system python if venv python doesn't exist (e.g. testing)
            venv_python = sys.executable
            
        # Execute the code in a subprocess
        result = subprocess.run(
            [venv_python, temp_file_path],
            cwd=workspace_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60  # Prevent infinite loops
        )
        
        stdout = result.stdout
        stderr = result.stderr
        success = result.returncode == 0
        
    except subprocess.TimeoutExpired:
        stdout = ""
        stderr = "Timeout expired: The code execution took longer than 60 seconds."
        success = False
    except Exception as e:
        stdout = ""
        stderr = f"Execution failed to launch: {str(e)}"
        success = False
    finally:
        # Clean up temporary script file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
    # Record files in output directory after execution
    after_files = set()
    if os.path.exists(output_dir):
        after_files = set(os.listdir(output_dir))
        
    new_files = list(after_files - before_files)
    new_file_paths = [os.path.join("output", f) for f in new_files]
    
    return {
        "success": success,
        "stdout": stdout,
        "stderr": stderr,
        "new_files": new_file_paths
    }
