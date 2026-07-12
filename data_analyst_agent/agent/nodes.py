import re
import os
import json
from typing import Dict, Any, List, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

from agent.state import AnalystState
from agent.schema import get_dataset_metadata
from agent.executor import execute_python_code

# Lazy loader for the LLM to avoid validation errors at import time.
_llm = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    return _llm

def extract_files_from_text(text: str) -> List[str]:
    """Helper to find CSV/Excel file paths mentioned in text."""
    # Matches patterns like: path/to/file.csv, "file.xlsx", 'data/dirty.xlsx'
    matches = re.findall(r'[\'"]?([^\s\'"]+\.(?:xlsx|xls|csv|xlsm))[\'"]?', text)
    return [m.strip('\'"') for m in matches]

def input_processor_node(state: AnalystState) -> Dict[str, Any]:
    """
    Scans the latest user message for file paths, extracts schemas for new files,
    and updates the state.
    """
    messages = state.get("messages", [])
    if not messages:
        return {}
        
    latest_message = messages[-1].content
    file_paths = extract_files_from_text(latest_message)
    
    current_files = list(state.get("files", []))
    data_schemas = dict(state.get("data_schemas", {}))
    
    updated = False
    
    # Root workspace directory is current directory
    cwd = os.getcwd()
    
    for path in file_paths:
        # Resolve path relative to current working directory
        abs_path = os.path.abspath(path)
        
        # Check if the file exists and is not already processed
        if os.path.exists(abs_path) and abs_path not in current_files:
            try:
                # Extract metadata
                metadata = get_dataset_metadata(abs_path)
                current_files.append(abs_path)
                data_schemas[abs_path] = metadata
                updated = True
            except Exception as e:
                # Add a message to indicate import failure, but don't crash
                messages.append(AIMessage(content=f"Error importing file '{path}': {str(e)}"))
                updated = True
                
    if updated:
        return {
            "files": current_files,
            "data_schemas": data_schemas,
            "messages": messages
        }
    return {}

def analyst_router_node(state: AnalystState) -> Dict[str, Any]:
    """
    Decides whether the user request requires running Python code to clean/analyze data
    or if it's a general query that can be answered directly.
    """
    messages = state.get("messages", [])
    data_schemas = state.get("data_schemas", {})
    
    schema_context = ""
    if data_schemas:
        schema_context = "Available datasets:\n"
        for filepath, schema in data_schemas.items():
            filename = os.path.basename(filepath)
            schema_context += f"- File: {filename} (Path: {filepath})\n"
            schema_context += f"  Columns: {schema['columns']}\n"
            schema_context += f"  Shape: {schema['shape']} (rows, cols)\n"
            schema_context += f"  Sheet name: {schema.get('active_sheet', 'Default')}\n"
            if schema.get("sheets") and len(schema["sheets"]) > 1:
                schema_context += f"  All sheets: {schema['sheets']}\n"
            schema_context += f"  Null counts: {schema['missing_values']}\n\n"
            
    system_prompt = f"""You are an associate-level Data Analyst Agent Router.
Your job is to read the conversation and decide if the user's latest request requires writing and executing Python/Pandas code to manipulate, clean, analyze, or plot the data, OR if it's a general question that can be answered directly using the schemas/metadata or conversation context.

{schema_context}

Respond in JSON format with two keys:
1. "requires_code": true (if we need to run python code to load files, clean rows, filter, compute statistics, or generate charts) or false (if it's a general hello, explanation of columns, or basic query we can answer without running scripts).
2. "reason": A brief explanation of your choice.

Respond ONLY with valid JSON. Do not include markdown code block formatting in your raw response, just the JSON object.
"""
    
    response = get_llm().invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=messages[-1].content)
    ])
    
    # Clean output block if LLM returned markdown code blocks
    content = response.content.strip()
    if content.startswith("```json"):
        content = content[7:-3].strip()
    elif content.startswith("```"):
        content = content[3:-3].strip()
        
    try:
        decision = json.loads(content)
        requires_code = decision.get("requires_code", False)
    except Exception:
        # Default fallback
        requires_code = True
        
    # We store the routing decision in state or use it in the conditional edge
    return {
        "next_step": "code_generator" if requires_code else "general_responder"
    }

def code_generator_node(state: AnalystState) -> Dict[str, Any]:
    """Generates the Python code to perform the data cleaning, manipulation, or analysis."""
    messages = state.get("messages", [])
    data_schemas = state.get("data_schemas", {})
    
    schema_context = "Available datasets and schemas:\n"
    for filepath, schema in data_schemas.items():
        filename = os.path.basename(filepath)
        schema_context += f"### Dataset: {filename}\n"
        schema_context += f"- File path: {filepath}\n"
        schema_context += f"- Sheet name: {schema.get('active_sheet', 'Default')}\n"
        schema_context += f"- Shape: {schema['shape']} (rows, cols)\n"
        schema_context += f"- Columns and types:\n"
        for col in schema["columns"]:
            dtype = schema["dtypes"].get(col, "unknown")
            null_cnt = schema["missing_values"].get(col, 0)
            schema_context += f"  - {col} ({dtype}): {null_cnt} missing values\n"
        schema_context += f"- First 3 rows sample:\n"
        schema_context += f"  {json.dumps(schema['sample_data'], indent=2)}\n\n"
        
    system_prompt = f"""You are an expert Data Analyst Python Programmer.
Your task is to write a single, clean, self-contained Python script to clean, manipulate, or analyze the datasets listed below.

{schema_context}

RULES FOR WRITING THE PYTHON CODE:
1. ALWAYS read the file(s) using Pandas:
   - For Excel: `pd.read_excel(path, sheet_name=...)`
   - For CSV: `pd.read_csv(path)`
2. For CLEANING requests, implement logic for:
   - Handling missing values (imputing or dropping).
   - Removing duplicate rows.
   - Trimming whitespace from string columns: `df[col] = df[col].astype(str).str.strip()`
   - Correcting data types (e.g. string to datetime or float).
   - Saving the CLEANED dataframe back to a new excel file in the 'output' directory. Save it as: `output/<original_filename_no_ext>_cleaned.xlsx` or `.csv`.
3. For PLOTTING/VISUALIZATION requests:
   - Save the plot as a PNG image in the 'output' directory, e.g., `output/plot_name.png`. Do NOT attempt to show the plot using `plt.show()`.
   - Use `matplotlib` or `seaborn` and ensure proper styling (title, labels, grid, colors).
   - Call `plt.close()` at the end to release memory.
4. For REPORTING/OUTPUT:
   - Use `print()` to output summaries, descriptive statistics, or results. This standard output will be read by the agent to report back to the user.
   - Print markdown tables or clean text summaries.
5. DO NOT import any libraries other than pandas, numpy, matplotlib, seaborn, openpyxl, os.
6. The code must be self-contained and run without user interaction.

Respond ONLY with the raw python code inside a markdown code block starting with ```python and ending with ```.
"""
    
    response = get_llm().invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=messages[-1].content)
    ])
    
    code = response.content.strip()
    # Extract code from code block
    match = re.search(r'```python\s*(.*?)\s*```', code, re.DOTALL)
    if match:
        code = match.group(1)
        
    return {
        "generated_code": code,
        "retry_count": 0,
        "error_feedback": ""
    }

def execute_code_node(state: AnalystState) -> Dict[str, Any]:
    """Executes the generated python code and captures results."""
    code = state.get("generated_code", "")
    cwd = os.getcwd()
    
    if not code:
        return {"execution_output": "Error: No code generated to execute."}
        
    result = execute_python_code(code, cwd)
    
    if result["success"]:
        output = f"Execution Succeeded.\n\nStdout:\n{result['stdout']}"
        if result["new_files"]:
            output += f"\n\nNew Files Created:\n" + "\n".join([f"- {f}" for f in result["new_files"]])
        return {
            "execution_output": output,
            "error_feedback": "",
            "new_files_created": result["new_files"] # pass to reporter
        }
    else:
        return {
            "execution_output": f"Execution Failed.\n\nStdout:\n{result['stdout']}\n\nStderr:\n{result['stderr']}",
            "error_feedback": result["stderr"]
        }

def code_refiner_node(state: AnalystState) -> Dict[str, Any]:
    """Regenerates the Python code to fix errors based on traceback feedback."""
    messages = state.get("messages", [])
    data_schemas = state.get("data_schemas", {})
    code = state.get("generated_code", "")
    error = state.get("error_feedback", "")
    retry_count = state.get("retry_count", 0)
    
    schema_context = "Available datasets:\n"
    for filepath, schema in data_schemas.items():
        filename = os.path.basename(filepath)
        schema_context += f"- {filename} (Path: {filepath})\n"
        
    system_prompt = f"""You are a senior debugger. The Python script you wrote failed during execution.
Here is the script you wrote:
```python
{code}
```

Here is the error traceback:
```
{error}
```

{schema_context}

Please rewrite the script to resolve the error. Ensure you follow all code generation guidelines:
1. Read files properly.
2. Clean data or plot as requested.
3. Save any output plots or files to the 'output' directory.
4. Output results using print().
5. Avoid syntax errors, index errors, or missing imports.

Respond ONLY with the updated raw python code inside a markdown code block starting with ```python and ending with ```.
"""
    
    response = get_llm().invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=messages[-1].content)
    ])
    
    new_code = response.content.strip()
    match = re.search(r'```python\s*(.*?)\s*```', new_code, re.DOTALL)
    if match:
        new_code = match.group(1)
        
    return {
        "generated_code": new_code,
        "retry_count": retry_count + 1,
        "error_feedback": ""
    }

def summarizer_reporter_node(state: AnalystState) -> Dict[str, Any]:
    """Summarizes code execution outputs, tables, and charts to present to the user."""
    messages = state.get("messages", [])
    code = state.get("generated_code", "")
    output = state.get("execution_output", "")
    error = state.get("error_feedback", "")
    
    system_prompt = f"""You are an associate-level Data Analyst.
Review the python code that was executed and its execution output (stdout/stderr/files generated).
Summarize the results of the operation and present the final answer clearly to the user.

Executed Code:
```python
{code}
```

Execution Output:
{output}

{f"Error trace if failed: {error}" if error else ""}

Guidelines:
- Explain what actions were performed (e.g. cleaned duplicates, imputed nulls).
- Present key analytical insights or statistical summaries clearly. Use Markdown tables if appropriate.
- If plots/images were generated (listed in "New Files Created"), notify the user of their filenames and locations.
- Be concise but professional, like an associate data analyst presenting to their manager.
- If the execution failed and could not be corrected, explain the failure and suggest how the user might help (e.g., clarify sheet name, correct column names).
"""
    
    response = get_llm().invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=messages[-1].content)
    ])
    
    messages.append(AIMessage(content=response.content))
    
    return {
        "messages": messages
    }

def general_responder_node(state: AnalystState) -> Dict[str, Any]:
    """Answers general questions directly from schema or metadata without executing code."""
    messages = state.get("messages", [])
    data_schemas = state.get("data_schemas", {})
    
    schema_context = ""
    if data_schemas:
        schema_context = "Available datasets structure:\n"
        for filepath, schema in data_schemas.items():
            filename = os.path.basename(filepath)
            schema_context += f"### {filename}\n"
            schema_context += f"- Path: {filepath}\n"
            schema_context += f"- Columns: {schema['columns']}\n"
            schema_context += f"- Data Types: {schema['dtypes']}\n"
            schema_context += f"- Dimensions: {schema['shape']} (rows, columns)\n"
            schema_context += f"- Missing Values count: {schema['missing_values']}\n"
            schema_context += f"- Sheet names: {schema['sheets']}\n"
            schema_context += f"- Active sheet: {schema.get('active_sheet')}\n\n"
            
    system_prompt = f"""You are an associate-level Data Analyst.
Answer the user's question directly based on the conversation history and the schema/metadata of the imported files.

{schema_context}

Be helpful, concise, and professional. If the user asks you to clean or analyze the data, you should remind them that you can do so if they describe what they want.
"""
    
    response = get_llm().invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=messages[-1].content)
    ])
    
    messages.append(AIMessage(content=response.content))
    
    return {
        "messages": messages
    }
