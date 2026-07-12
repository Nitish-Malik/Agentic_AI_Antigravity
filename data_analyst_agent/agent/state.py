from typing import TypedDict, Annotated, List, Dict, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class DatasetSchema(TypedDict):
    columns: List[str]
    dtypes: Dict[str, str]
    shape: List[int]  # [rows, cols]
    missing_values: Dict[str, int]
    sample_data: List[Dict[str, Any]]
    sheets: List[str]  # Excel sheets if applicable

class AnalystState(TypedDict):
    # Chat history between user and agent
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Track files currently imported
    files: List[str]
    
    # Store schema/metadata of tracked files
    data_schemas: Dict[str, DatasetSchema]
    
    # Current python code generated for the task
    generated_code: str
    
    # Output of code execution (stdout, stderr, generated files)
    execution_output: str
    
    # Error message if code execution failed
    error_feedback: str
    
    # Counter to limit code correction loops
    retry_count: int
