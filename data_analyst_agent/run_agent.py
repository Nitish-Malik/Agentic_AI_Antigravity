import os
import sys
from dotenv import load_dotenv

# Load env variables from .env file
load_dotenv()

def check_env_credentials():
    """Verify GEMINI_API_KEY is configured before running the agent."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        print("\033[93m[Warning] GEMINI_API_KEY is not configured in .env file.\033[0m")
        print("Please obtain an API key from Google AI Studio: https://aistudio.google.com/app/api-keys")
        user_key = input("Enter your GEMINI_API_KEY here to run the agent (or press Enter to exit): ").strip()
        if user_key:
            os.environ["GEMINI_API_KEY"] = user_key
            # Write to .env for persistence
            try:
                with open(".env", "w") as f:
                    f.write(f"GEMINI_API_KEY={user_key}\n")
                print("\033[92mSaved API key to .env file.\033[0m\n")
            except Exception as e:
                print(f"Could not save key to .env: {str(e)}")
        else:
            print("Exit. Please set the GEMINI_API_KEY in .env and restart.")
            sys.exit(1)

# Run credential check
check_env_credentials()

# Import LangGraph app and LangChain components after credentials are set
try:
    from agent import app
    from langchain_core.messages import HumanMessage
except ImportError as e:
    print(f"Error importing dependencies: {str(e)}")
    print("Ensure you have activated the virtual environment and installed requirements.txt.")
    sys.exit(1)

def print_banner():
    banner = """
============================================================
       Associate Data Analyst Agent (LangGraph CLI)
============================================================
Capabilities:
  * Import Excel/CSV files (e.g. mention path in your prompt)
  * Auto-extract dataset structures (schemas, shapes, columns)
  * Clean raw data (drop duplicates, handle nulls, fix formats)
  * Analyze data (descriptive statistics, groupings, filters)
  * Generate visualization plots (saved in output/ directory)
  
Commands:
  * /list  - List all imported datasets and their shapes
  * /exit  - Quit the chat
============================================================
"""
    print(banner)

def main():
    print_banner()
    
    # Initialize the local state
    current_state = {
        "messages": [],
        "files": [],
        "data_schemas": {},
        "generated_code": "",
        "execution_output": "",
        "error_feedback": "",
        "retry_count": 0
    }
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() == "/exit":
                print("Goodbye!")
                break
                
            if user_input.lower() == "/list":
                if not current_state["files"]:
                    print("\nAgent: No datasets have been imported yet. Mention an Excel or CSV path to import one (e.g., 'Load sample_data/dirty_data.xlsx').")
                else:
                    print("\nImported Datasets:")
                    for filepath in current_state["files"]:
                        filename = os.path.basename(filepath)
                        schema = current_state["data_schemas"].get(filepath, {})
                        shape = schema.get("shape", "unknown")
                        sheet = schema.get("active_sheet", "Default")
                        print(f"  - {filename} | Path: {filepath} | Sheet: {sheet} | Shape: {shape}")
                continue
            
            # Append human message to list of messages in state
            current_state["messages"].append(HumanMessage(content=user_input))
            
            print("\nAgent is thinking and processing (running nodes)...")
            
            # Invoke the LangGraph compiled workflow
            # Note: We pass the entire current_state dictionary
            response_state = app.invoke(current_state)
            
            # Update the local state with the returned state from graph
            current_state.update({
                "messages": response_state["messages"],
                "files": response_state["files"],
                "data_schemas": response_state["data_schemas"],
                "generated_code": response_state.get("generated_code", ""),
                "execution_output": response_state.get("execution_output", ""),
                "error_feedback": response_state.get("error_feedback", ""),
                "retry_count": response_state.get("retry_count", 0)
            })
            
            # Print the latest agent response
            latest_message = current_state["messages"][-1]
            print(f"\nAgent: {latest_message.content}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError running agent turn: {str(e)}")

if __name__ == "__main__":
    main()
