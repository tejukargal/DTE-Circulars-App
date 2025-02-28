import streamlit as st
import pandas as pd
import numpy as np
import json
from anthropic import Anthropic
from dotenv import load_dotenv
import os
import chardet
import io
import sys
import traceback

# Loading environment variables from .env file
load_dotenv()

def detect_encoding(file_content):
    """Detect the encoding of the file content"""
    result = chardet.detect(file_content)
    return result['encoding']

def read_csv_with_encoding(file):
    """Read CSV file with automatic encoding detection"""
    try:
        content = file.read()
        file.seek(0)
        encoding = detect_encoding(content)
        
        try:
            df = pd.read_csv(file, encoding=encoding)
            return df, None
        except Exception as e:
            common_encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            for enc in common_encodings:
                try:
                    file.seek(0)
                    df = pd.read_csv(file, encoding=enc)
                    return df, None
                except:
                    continue
            
            return None, f"Failed to read file with any encoding. Error: {str(e)}"
            
    except Exception as e:
        return None, f"Error reading file: {str(e)}"

def get_dataframe_info(df):
    """Get information about the dataframe"""
    # Create a safe JSON-serializable version of the describe data
    describe_df = df.describe(include='all').fillna("NA")
    
    # Convert NumPy types to Python native types for JSON serialization
    describe_dict = {}
    for col in describe_df.columns:
        describe_dict[col] = {}
        for idx in describe_df.index:
            val = describe_df.loc[idx, col]
            if isinstance(val, (np.int64, np.int32, np.int16, np.int8)):
                val = int(val)
            elif isinstance(val, (np.float64, np.float32, np.float16)):
                val = float(val)
            elif val == "NA":
                val = None
            describe_dict[col][idx] = val
    
    # Get basic info
    info = {
        "columns": list(df.columns),
        "shape": df.shape,
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "sample_rows": df.head(5).to_dict(orient="records"),
        "describe": describe_dict,
        "null_counts": df.isnull().sum().to_dict(),
        "column_unique_counts": {col: int(df[col].nunique()) for col in df.columns}
    }
    return info

def infer_financial_data_structure(df):
    """Try to infer if this is financial data and how it's structured"""
    financial_indicators = {
        'credits': ['credit', 'income', 'revenue', 'received', 'deposit', 'receipt', 'inflow', 'cr'],
        'debits': ['debit', 'expense', 'payment', 'paid', 'withdrawal', 'outflow', 'dr'],
        'amount': ['amount', 'value', 'sum', 'total', 'price', 'cost', 'fee'],
        'date': ['date', 'day', 'transaction_date', 'txn_date', 'time']
    }
    
    # Check column names for financial indicators
    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        for category, indicators in financial_indicators.items():
            if any(indicator in col_lower for indicator in indicators):
                column_mapping[category] = col
                break
                
    # If we don't have explicit credit/debit columns, see if there's a single amount column
    # and possibly a transaction type column
    if 'credits' not in column_mapping and 'debits' not in column_mapping and 'amount' in column_mapping:
        # Look for a type column that might indicate credit/debit
        for col in df.columns:
            col_lower = col.lower()
            if 'type' in col_lower or 'category' in col_lower or 'transaction' in col_lower:
                column_mapping['type'] = col
                # See if values contain credit/debit indicators
                if df[col].dtype == 'object':
                    values = df[col].dropna().astype(str).str.lower()
                    has_credit = any(credit in ' '.join(values) for credit in financial_indicators['credits'])
                    has_debit = any(debit in ' '.join(values) for debit in financial_indicators['debits'])
                    if has_credit and has_debit:
                        column_mapping['has_transaction_types'] = True
                        
    return column_mapping

def extract_and_execute_code(text, df):
    """Extract Python code from Claude's response and execute it"""
    results = []
    
    # Try to infer if this is financial data
    financial_structure = infer_financial_data_structure(df)
    
    # Check if there's any code block
    if "```python" in text and "```" in text:
        code_blocks = []
        
        # Extract all code blocks
        parts = text.split("```")
        for i, part in enumerate(parts):
            if i % 2 == 1 and (part.startswith("python") or part.startswith("py")):
                code = part.replace("python", "").replace("py", "").strip()
                code_blocks.append(code)
        
        # Execute each code block
        for code in code_blocks:
            try:
                # Create execution environment with DataFrame already loaded
                local_vars = {"df": df, "pd": pd, "np": np}
                
                # Capture stdout
                output_buffer = io.StringIO()
                original_stdout = sys.stdout
                sys.stdout = output_buffer
                
                # Execute the code
                exec(code, {"pd": pd, "np": np}, local_vars)
                
                # Reset stdout
                sys.stdout = original_stdout
                
                # Get print outputs
                output = output_buffer.getvalue()
                if output:
                    results.append({"type": "text", "content": output})
                
                # Look for new variables that might be DataFrames
                for var_name, var_value in local_vars.items():
                    if var_name not in ["df", "pd", "np"] and isinstance(var_value, pd.DataFrame):
                        results.append({"type": "dataframe", "content": var_value, "name": var_name})
                    elif var_name not in ["df", "pd", "np"] and isinstance(var_value, list):
                        results.append({"type": "list", "content": var_value, "name": var_name})
                
                # If no output was captured and no new variables, show the original df with filters
                if not results:
                    # Try to find filtered dataframes in the local vars
                    for var_name, var_value in local_vars.items():
                        if var_name not in ["df", "pd", "np"]:
                            if isinstance(var_value, pd.DataFrame):
                                results.append({"type": "dataframe", "content": var_value, "name": var_name})
                            elif isinstance(var_value, (list, tuple)):
                                results.append({"type": "list", "content": var_value, "name": var_name})
                            else:
                                results.append({"type": "variable", "content": str(var_value), "name": var_name})
                
                # If still no results, evaluate the last line as an expression
                if not results and len(code.strip().split('\n')) > 0:
                    last_line = code.strip().split('\n')[-1]
                    # Check if the last line is an expression (not an assignment)
                    if "=" not in last_line or "==" in last_line or "<=" in last_line or ">=" in last_line:
                        try:
                            result_value = eval(last_line, {"pd": pd, "np": np}, local_vars)
                            if isinstance(result_value, pd.DataFrame):
                                results.append({"type": "dataframe", "content": result_value, "name": "Result"})
                            elif isinstance(result_value, (list, tuple)):
                                results.append({"type": "list", "content": result_value, "name": "Result"})
                            else:
                                results.append({"type": "variable", "content": str(result_value), "name": "Result"})
                        except:
                            pass
                
            except Exception as e:
                error_msg = f"Error executing code: {str(e)}\n{traceback.format_exc()}"
                results.append({"type": "error", "content": error_msg})
                
                # Try fallback strategies based on the query context
                
                # Fallback for credits and debits merging
                if "merge" in text.lower() and ("credit" in text.lower() or "debit" in text.lower()):
                    try:
                        # Check if we identified financial columns
                        if financial_structure:
                            merged_data = None
                            
                            # Case 1: Separate credit and debit dataframes
                            if "credits" in financial_structure and "debits" in financial_structure:
                                # Create a merged view with transaction type indicator
                                credits = df.copy()
                                credits['Transaction_Type'] = 'Credit'
                                
                                debits = df.copy()
                                debits['Transaction_Type'] = 'Debit'
                                
                                merged_data = pd.concat([credits, debits], ignore_index=True)
                                if 'date' in financial_structure:
                                    merged_data = merged_data.sort_values(by=financial_structure['date'])
                                    
                            # Case 2: Single dataframe with amount and transaction type
                            elif 'amount' in financial_structure and 'type' in financial_structure:
                                merged_data = df.copy()
                                if 'date' in financial_structure:
                                    merged_data = merged_data.sort_values(by=financial_structure['date'])
                            
                            # Case 3: Just show the data as is
                            else:
                                merged_data = df.copy()
                            
                            if merged_data is not None:
                                results.append({"type": "dataframe", "content": merged_data, "name": "Merged_Transactions"})
                    except Exception as fallback_error:
                        # If the fallback also fails, add the original dataframe
                        results.append({"type": "dataframe", "content": df, "name": "Original_Data"})
                        
                # Student filter fallbacks from previous implementation
                elif "CE" in code and "1st Yr" in code:
                    try:
                        # Fallback for CE students in 1st Yr
                        ce_first_year = df[(df['Course'] == 'CE') & (df['Year'] == '1st Yr')]
                        results.append({"type": "dataframe", "content": ce_first_year, "name": "CE_First_Year"})
                    except:
                        pass
                elif "EE" in code and "1st Yr" in code:
                    try:
                        # Fallback for EE students in 1st Yr
                        ee_first_year = df[(df['Course'] == 'EE') & (df['Year'] == '1st Yr')]
                        results.append({"type": "dataframe", "content": ee_first_year, "name": "EE_First_Year"})
                    except:
                        pass
    
    return results, text

def chat_with_csv_claude(df, query):
    """Analyze CSV data with Claude API directly"""
    load_dotenv()
    anthropic_api_key = os.environ.get('ANTHROPIC_API_KEY')
    
    if not anthropic_api_key:
        return "Error: ANTHROPIC_API_KEY not found in environment variables"
    
    client = Anthropic(api_key=anthropic_api_key)
    
    # Get dataframe information
    df_info = get_dataframe_info(df)
    
    # Sample data (first 5 rows)
    sample_data = df.head(5).to_string()
    
    # Create a prompt for Claude
    prompt = f"""
You are a data analysis assistant. I have a CSV file with the following information:

Shape: {df_info['shape'][0]} rows x {df_info['shape'][1]} columns
Columns: {', '.join(df_info['columns'])}

Here's a sample of the data (first 5 rows):

{sample_data}

Column data types:
{json.dumps({col: df_info['dtypes'][col] for col in df_info['columns']}, indent=2)}

Null values per column:
{json.dumps(df_info['null_counts'], indent=2)}

Unique values count per column:
{json.dumps(df_info['column_unique_counts'], indent=2)}

My question about this data is:
{query}

CRITICALLY IMPORTANT: Before writing any code, carefully review the columns available in the dataframe. The DataFrame is already loaded in a variable called 'df'. DO NOT try to read any CSV files. Work directly with the 'df' variable and ONLY use column names that exist in the data.

First, provide a clear concise explanation of how you'll approach this question.

Then, provide the solution using pandas code to get precise results. Include only essential Python code in a code block that will directly answer the question when executed. The code should be complete and work with the existing 'df' variable. Do not include code to read CSV files.

Example of good code:
```python
# First check what columns we have
print(df.columns)

# Then use exact column names from the data
result_df = df[(df['Column1'] == 'Value1') & (df['Column2'] > 50)]
print(result_df)
```

Finally, summarize the results and insights briefly. Remember that your code will be automatically executed, so make sure it outputs the final result.
"""
    
    try:
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=2048,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.content[0].text
        
        # Extract and execute code from the response
        execution_results, text = extract_and_execute_code(response_text, df)
        
        return {"text": text, "results": execution_results}
    except Exception as e:
        return {"text": f"Error calling Anthropic API: {str(e)}", "results": []}

def display_result(result):
    """Helper function to display results"""
    if isinstance(result, dict) and "text" in result and "results" in result:
        # Extract explanation parts from text without the code blocks
        explanation = []
        code_parts = False
        for line in result["text"].split("\n"):
            if line.strip() == "```python" or line.strip() == "```py":
                code_parts = True
                continue
            if line.strip() == "```" and code_parts:
                code_parts = False
                continue
            if not code_parts:
                explanation.append(line)
        
        # Show the explanation parts
        explanation_text = "\n".join([line for line in explanation if line.strip()])
        if explanation_text:
            st.write(explanation_text)
        
        # Show the execution results
        if result["results"]:
            st.markdown("### Analysis Results")
            for item in result["results"]:
                if item["type"] == "dataframe":
                    st.markdown(f"**{item['name']}**")
                    st.dataframe(item["content"], use_container_width=True)
                elif item["type"] == "text":
                    st.text(item["content"])
                elif item["type"] == "list":
                    st.markdown(f"**{item['name']}**")
                    for value in item["content"]:
                        st.markdown(f"- {value}")
                elif item["type"] == "variable":
                    st.markdown(f"**{item['name']}:** {item['content']}")
                elif item["type"] == "error":
                    # Only show errors if there are no other results
                    if len(result["results"]) == 1:
                        st.error(item["content"])
        else:
            st.warning("No executable results were found. Try a different question or check the data structure.")
    elif isinstance(result, str):
        st.write(result)
    else:
        st.write(result)

# Initialize session states
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'current_response' not in st.session_state:
    st.session_state.current_response = None
if 'current_question' not in st.session_state:
    st.session_state.current_question = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

def process_query():
    """Handle query submission"""
    if st.session_state.text_input.strip():
        st.session_state.query_history.append(st.session_state.text_input)
        return True
    return False

# Set custom CSS
st.set_page_config(layout='wide')
st.markdown("""
    <style>
    .main > div {
        padding-top: 1rem;
    }
    .stTextInput > div > div > input {
        height: 35px;
    }
    .error-message {
        color: #ff4b4b;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #ffe5e5;
    }
    .stButton button {
        margin-top: 0 !important;
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
    }
    .stCodeBlock {
        max-height: 300px;
        overflow-y: auto;
    }
    </style>
""", unsafe_allow_html=True)

st.title("CSV Analysis with Claude")
st.caption("Powered by Anthropic's Claude")

# Upload multiple CSV files
input_csvs = st.sidebar.file_uploader("Upload your CSV files", type=['csv'], accept_multiple_files=True)

if input_csvs:
    # Select a CSV file from the uploaded files
    file_options = [file.name for file in input_csvs]
    selected_file = st.selectbox("Select a CSV file", file_options)
    selected_index = file_options.index(selected_file)
    
    # Load the selected CSV file only once or when selection changes
    if not st.session_state.data_loaded or st.session_state.get('selected_file') != selected_file:
        with st.spinner('Loading CSV file...'):
            data, error = read_csv_with_encoding(input_csvs[selected_index])
            if not error:
                st.session_state.data_loaded = True
                st.session_state.current_data = data
                st.session_state.selected_file = selected_file
            else:
                st.session_state.data_loaded = False
    else:
        data = st.session_state.current_data
        error = None
        
    if error:
        st.error(f"Error loading file: {error}")
    elif st.session_state.data_loaded:
        # Data preview with expander
        with st.expander("Preview Data", expanded=False):
            st.dataframe(data.head(5), use_container_width=True)
            st.caption(f"Total rows: {len(data)}, Total columns: {len(data.columns)}")
        
        # Create two columns for input and button with better ratio
        col1, col2 = st.columns([6, 1])
        
        # Query input in first column
        with col1:
            st.text_input(
                "Enter your query about the data", 
                key="text_input",
                placeholder="e.g., 'What is the average of column X?' or 'Show me trends in this data'"
            )
        
        # Submit button in second column, aligned horizontally
        with col2:
            submitted = st.button("Submit", key="submit_btn", use_container_width=True)
            
        # Handle query submission and display
        if submitted and process_query():
            current_query = st.session_state.text_input
            try:
                # Display the question as regular text
                st.write("Question:", current_query)
                st.markdown("---")
                
                with st.spinner('Analyzing your data...'):
                    # Process the query and display results
                    result = chat_with_csv_claude(data, current_query)
                    
                    # Store current question and response
                    st.session_state.current_question = current_query
                    st.session_state.current_response = result
                    
                    # Display the results
                    display_result(result)
                        
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                st.info("Try rephrasing your question or check if your ANTHROPIC_API_KEY is properly set in the .env file")
                
        # Display previous response if it exists
        elif st.session_state.current_question is not None and st.session_state.current_response is not None:
            st.write("Question:", st.session_state.current_question)
            st.markdown("---")
            display_result(st.session_state.current_response)
            
else:
    st.info("👈 Please upload your CSV files using the sidebar")
