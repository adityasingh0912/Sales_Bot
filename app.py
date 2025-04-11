import os
import sys
import requests
import re
import json
import io
from contextlib import redirect_stdout
from flask import Flask, request, jsonify, render_template, url_for
from dotenv import load_dotenv
from astrapy import DataAPIClient
from flask_cors import CORS
import datetime 
from collections import Counter, defaultdict 
import math 
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import time 


load_dotenv()

ASTRA_DB_TOKEN = os.getenv("ASTRA_DB_TOKEN")
ASTRA_DB_ENDPOINT = os.getenv("ASTRA_DB_ENDPOINT")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not (ASTRA_DB_TOKEN and ASTRA_DB_ENDPOINT and GROQ_API_KEY):
    print("Error: One or more required credentials are missing in the environment.")
    print("Ensure ASTRA_DB_TOKEN, ASTRA_DB_ENDPOINT, and GROQ_API_KEY are set in your .env file.")
    sys.exit(1)

try:
    client = DataAPIClient(ASTRA_DB_TOKEN)
    db = client.get_database_by_api_endpoint(ASTRA_DB_ENDPOINT)
    sales_collection = db.get_collection("sales")
    print("✅ Connected to Astra DB and accessed 'sales' collection.")
except Exception as e:
    print(f"❌ Failed to connect or access the sales collection in Astra DB: {e}")
    print(f"Endpoint: {ASTRA_DB_ENDPOINT}")
    sys.exit(1)

def fetch_sales_documents(limit=500):
    """Fetches sales documents from the Astra DB collection and cleans numerical types."""
    print("⏳ Fetching raw data from Astra DB...")
    try:
        raw_docs = list(sales_collection.find({}, limit=limit, projection={"_id": 0, "$vector": 0}))
        print(f"📄 Retrieved {len(raw_docs)} raw document(s). Starting cleaning...")

        cleaned_docs = []
        skipped_count = 0
        numerical_fields = ['unit_price', 'quantity', 'discount', 'total_price']

        for doc in raw_docs:
            valid_doc = True
            for field in numerical_fields:
                if field in doc:
                    try:
                        if doc[field] is None or str(doc[field]).strip() == "":
                             doc[field] = 0.0 # Assign default or handle as needed
                        else:
                             doc[field] = float(doc[field])
                    except (ValueError, TypeError) as e:
                        print(f"⚠️ Warning: Could not convert field '{field}' ('{doc[field]}') to float in doc: {doc.get('product', 'N/A')}. Error: {e}. Skipping field conversion, may cause issues.")

            if 'order_date' in doc:
                 try:
                    datetime.datetime.strptime(str(doc['order_date']), '%Y-%m-%d')
                 except (ValueError, TypeError):
                    print(f"⚠️ Warning: Invalid date format for 'order_date' ('{doc['order_date']}') in doc: {doc.get('product', 'N/A')}. Skipping doc.")
                    valid_doc = False


            if valid_doc:
                cleaned_docs.append(doc)
            else:
                skipped_count += 1

        if skipped_count > 0:
            print(f"⚠️ Skipped {skipped_count} documents due to conversion/validation errors.")
        print(f"✨ Cleaned {len(cleaned_docs)} documents.")
        return cleaned_docs

    except Exception as e:
        print(f"❌ Error fetching or cleaning data from Astra DB: {e}")
        return [] # Return empty list on error

print("⏳ Loading and cleaning sales data...")
sales_docs = fetch_sales_documents(limit=500)
if not sales_docs:
     print("🚨 Critical: Could not load or clean sales data. Visualizations may be empty.")
else:
    print(f"✅ Sales data loaded and cleaned successfully ({len(sales_docs)} documents).")

    try:
        sales_df = pd.DataFrame(sales_docs)
        if 'order_date' in sales_df.columns:
            sales_df['order_date'] = pd.to_datetime(sales_df['order_date'], errors='coerce') # errors='coerce' turns invalid dates into NaT
            sales_df.dropna(subset=['order_date'], inplace=True) # Remove rows with invalid dates if necessary
        print("✅ Data converted to Pandas DataFrame.")
    except Exception as e:
        print(f"❌ Error converting data to DataFrame: {e}. Plotting might fail.")
        sales_df = pd.DataFrame() # Create empty DataFrame if conversion fails

def query_to_code(nl_query, current_sales_data):
    groq_url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    # Get keys from the first cleaned document if available
    field_keys = list(current_sales_data[0].keys()) if current_sales_data else 'unknown'
    data_info = (
        f"Each dictionary in 'sales_docs' has keys like: {field_keys}. "
        "IMPORTANT: Numerical fields ('unit_price', 'quantity', 'discount', 'total_price') are already Python floats. "
        "'order_date' is a string in 'YYYY-MM-DD' format."
    )

    prompt = (
        "You are a Python data analysis assistant. A Python list of dictionaries named 'sales_docs' "
        "is already loaded and cleaned in the execution environment.\n"
        f"{data_info}\n\n"
        "Write a Python code snippet using the existing 'sales_docs' list to answer the user's query. "
        "The code MUST calculate the answer and print the final result clearly to standard output. "
        "Assume standard libraries like 'datetime', 'collections', 'math' are available. "
        "Return ONLY the valid Python code snippet inside a markdown code block (```python ... ```). No explanations outside the block.\n\n"
        f"User query: {nl_query}"
    )

    payload = {
        "model": "llama-3.3-70b-versatile", # Consistent model
        "messages": [
            {"role": "system", "content": "You are a Python data analysis assistant generating executable code snippets using pre-loaded, cleaned data."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1, # Lower temperature for more deterministic code
        "max_tokens": 600, # Slightly increase if code gets complex
    }

    try:
        response = requests.post(groq_url, headers=headers, json=payload, timeout=45) # Increased timeout
        response.raise_for_status()
        data = response.json()

        if not data.get("choices") or not data["choices"][0].get("message") or not data["choices"][0]["message"].get("content"):
            print("❌ Groq API Response Error (Code Gen): Unexpected structure", data)
            return None

        raw_content = data["choices"][0]["message"]["content"]
        code_block_match = re.search(r"```python\s*([\s\S]+?)\s*```", raw_content)

        if code_block_match:
            extracted_code = code_block_match.group(1).strip()
            # Basic safety check - enhance if needed
            unsafe_keywords = ['os.', 'sys.', 'requests.', 'subprocess.', 'eval(', 'exec(', 'open(', '__import__']
            if any(keyword in extracted_code for keyword in unsafe_keywords):
                 print(f"⚠️ Warning: Generated code contains potentially unsafe keyword: {extracted_code}")
                 return None # Reject potentially unsafe code
            try:
                compile(extracted_code, '<string>', 'exec')
            except SyntaxError as e:
                print(f"❌ Generated code has Syntax Error: {e}")
                print(f"--- Code with Syntax Error ---\n{extracted_code}\n--- End Code ---")
                return None
            print(f"🐍 Generated Code (for server logs):\n{extracted_code}") # Log code server-side
            return extracted_code
        else:
            print("❌ Fallback failed: No Python code block found in Groq response.")
            print(f"Raw response content: {raw_content}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"❌ Groq API Request Error (Code Gen): {e}")
        return None
    except Exception as e:
        print(f"❌ Error processing Groq response (Code Gen): {e}")
        return None


def get_ai_advice(query, result):
    if not result or result.strip() == "" or "error" in result.lower():
        return "No actionable result provided to generate advice."

    groq_url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    prompt = (
         f"You are a concise business advisor. Based *only* on the following information:\n"
        f"1. The user asked about sales data: '{query}'\n"
        f"2. The calculated result was:\n---\n{result}\n---\n\n" # Delimit result clearly
        "Provide 1-2 short sentences of specific, practical, and actionable business advice or insight directly related *only* to this query and result. "
        "Focus on potential actions or interpretations. Be direct. Do not repeat the query or result. Do not add greetings or closings."
    )

    payload = {
        "model": "llama-3.3-70b-specdec", # Using a model confirmed to work for code gen
        "messages": [
            {"role": "system", "content": "You provide short, actionable business advice based *only* on the provided query and result."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.6, 
        "max_tokens": 150, 
    }

    try:
        response = requests.post(groq_url, headers=headers, json=payload, timeout=30) 
        response.raise_for_status() # Will raise exception for 404, 500 etc.
        data = response.json()

        if data.get("choices") and data["choices"][0].get("message") and data["choices"][0]["message"].get("content"):
             advice = data["choices"][0]["message"]["content"].strip()
             # Basic filter for non-advice responses
             if "sorry" in advice.lower() or "cannot provide" in advice.lower() or "don't have enough information" in advice.lower():
                 print(f"⚠️ AI could not generate specific advice: {advice}")
                 return "AI could not generate specific advice based on this result."
             print(f"💡 Generated Advice: {advice}")
             return advice
        else:
            print("❌ Groq Advice API Response Error: Unexpected structure", data)
            return "Unable to generate advice due to API response issue."

    except requests.exceptions.HTTPError as e:
         # Specifically catch HTTP errors like 404
         print(f"❌ Groq Advice API HTTP Error: {e.response.status_code} - {e.response.text}")
         return f"Unable to generate advice. (API Error: {e.response.status_code})"
    except requests.exceptions.RequestException as e:
        print(f"❌ Groq Advice API Request Error: {e}")
        return "Unable to generate advice due to network error."
    except Exception as e:
        print(f"❌ Error getting AI advice: {e}")
        return "Unable to generate advice at this time."


# --- Flask Application Setup ---
app = Flask(__name__)
CORS(app)

# --- Create Static Directory for Plots ---
# Define the path relative to the app.py file
plots_dir = os.path.join(app.static_folder, 'plots')
os.makedirs(plots_dir, exist_ok=True)
print(f"✅ Static plots directory ensured at: {plots_dir}")


# --- NEW Plotting Functions ---

def generate_sales_by_category(df, filename="sales_by_category.png"):
    if df.empty or 'category' not in df.columns or 'total_price' not in df.columns:
        print("⚠️ Cannot generate 'sales_by_category': DataFrame empty or missing columns.")
        return None
    filepath = os.path.join(plots_dir, filename)
    try:
        plt.figure(figsize=(10, 6))
        category_sales = df.groupby('category')['total_price'].sum().sort_values(ascending=False)
        sns.barplot(x=category_sales.values, y=category_sales.index, palette="viridis", hue=category_sales.index, legend=False)
        plt.title('Total Sales by Product Category')
        plt.xlabel('Total Sales ($)')
        plt.ylabel('Category')
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close() # Close the plot to free memory
        print(f"📊 Plot saved: {filepath}")
        return f'plots/{filename}'# Return relative path for url_for
    except Exception as e:
        print(f"❌ Error generating 'sales_by_category' plot: {e}")
        return None

def generate_sales_by_region(df, filename="sales_by_region.png"):
    if df.empty or 'region' not in df.columns or 'total_price' not in df.columns:
        print("⚠️ Cannot generate 'sales_by_region': DataFrame empty or missing columns.")
        return None
    filepath = os.path.join(plots_dir, filename)
    try:
        plt.figure(figsize=(10, 6))
        region_sales = df.groupby('region')['total_price'].sum().sort_values(ascending=False)
        sns.barplot(x=region_sales.values, y=region_sales.index, palette="magma", hue=region_sales.index, legend=False)
        plt.title('Total Sales by Region')
        plt.xlabel('Total Sales ($)')
        plt.ylabel('Region')
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        print(f"📊 Plot saved: {filepath}")
        return f'plots/{filename}'
    except Exception as e:
        print(f"❌ Error generating 'sales_by_region' plot: {e}")
        return None

def generate_sales_over_time(df, filename="sales_over_time.png"):
    if df.empty or 'order_date' not in df.columns or 'total_price' not in df.columns:
        print("⚠️ Cannot generate 'sales_over_time': DataFrame empty or missing columns.")
        return None
    # Ensure 'order_date' is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['order_date']):
        print("⚠️ 'order_date' column is not datetime type for time series plot.")
        return None

    filepath = os.path.join(plots_dir, filename)
    try:
        plt.figure(figsize=(12, 6))
        # Resample data by month ('ME' for Month End frequency) and sum total_price
        monthly_sales = df.set_index('order_date')['total_price'].resample('ME').sum()
        if monthly_sales.empty:
             print("⚠️ No data after resampling monthly for 'sales_over_time'.")
             return None
        monthly_sales.plot(kind='line', marker='o')
        plt.title('Total Sales Over Time (Monthly)')
        plt.xlabel('Month')
        plt.ylabel('Total Sales ($)')
        plt.grid(True, axis='y')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        print(f"📊 Plot saved: {filepath}")
        return f'plots/{filename}'
    except Exception as e:
        print(f"❌ Error generating 'sales_over_time' plot: {e}")
        return None

def generate_customer_type_distribution(df, filename="customer_type_dist.png"):

    if df.empty or 'customer_type' not in df.columns:
         print("⚠️ Cannot generate 'customer_type_distribution': DataFrame empty or missing 'customer_type' column.")
         return None
    filepath = os.path.join(plots_dir, filename)
    try:
        plt.figure(figsize=(8, 8))
        customer_counts = df['customer_type'].value_counts()
        plt.pie(customer_counts, labels=customer_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("Set2"))
        plt.title('Distribution of Customer Types')
        # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        print(f"📊 Plot saved: {filepath}")
        return f'plots/{filename}'
    except Exception as e:
        print(f"❌ Error generating 'customer_type_distribution' plot: {e}")
        return None
@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def process_query():
    """Handles natural language queries from the frontend."""
    global sales_docs # Access the globally loaded and cleaned data

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    query = data.get('query')

    if not query:
        return jsonify({"error": "Query parameter is missing"}), 400

    if not sales_docs: # Check if data failed to load initially
         print("ERROR: sales_docs is empty or None in process_query")
         return jsonify({"error": "Sales data is not available. Cannot process query."}), 500

    print(f"❓ Received Query: {query}")

    # 1. Generate Python code
    generated_code = query_to_code(query, sales_docs)

    if not generated_code:
        print("❌ Failed to generate code for query.")
        return jsonify({"error": "I couldn't translate that query into an action. Could you try rephrasing it?"}), 400

    # 2. Execute the generated code safely
    output_capture = io.StringIO()
    execution_result = None

    # Prepare the execution context with necessary imports and the cleaned data
    local_execution_scope = {
        "sales_docs": sales_docs, # The pre-loaded and cleaned data
        "datetime": datetime,
        "Counter": Counter,
        "defaultdict": defaultdict,
        "math": math,

    }
    global_execution_scope = {} # Keep globals minimal

    try:
        print("🚀 Executing generated code...")
        with redirect_stdout(output_capture):
            exec(generated_code, global_execution_scope, local_execution_scope)
        execution_result = output_capture.getvalue().strip()
        if not execution_result:
             print("⚠️ Code executed but produced no output.")
             # Return a specific message instead of just empty
             execution_result = "(Query processed, but no specific output was generated by the code)"
        else:
             print(f"✅ Code Execution Result:\n{execution_result}")

    except Exception as e:
        # Catch errors during execution (like the previous TypeError, or others)
        error_details = f"Error executing generated code: {type(e).__name__}: {str(e)}"
        print(f"❌ {error_details}")
        # Log the code that failed for debugging on the server
        print(f"--- Failing Code Start ---\n{generated_code}\n--- Failing Code End ---")
 
        return jsonify({"error": "An internal error occurred while processing the query results. Please try again or rephrase."}), 500


    advice = get_ai_advice(query, execution_result)

    return jsonify({
        "query": query,
        "result": execution_result,
        "advice": advice
    })

@app.route('/plots')
def plots_page():
    """Generates plots and serves the plots HTML page."""
    global sales_df # Use the pre-loaded DataFrame

    if sales_df.empty:
        print("⚠️ Plot page requested, but DataFrame is empty.")
        return render_template('plots.html', plot_files=[], error_message="Sales data is not available or failed to load.")

    print("📈 Generating plots for /plots page...")
    plot_paths = [] # Store relative paths of generated plots

    plot_paths.append(generate_sales_by_category(sales_df))
    plot_paths.append(generate_sales_by_region(sales_df))
    plot_paths.append(generate_sales_over_time(sales_df))
    plot_paths.append(generate_customer_type_distribution(sales_df))

    # Filter out None values (if a plot failed to generate)
    valid_plot_paths = [path for path in plot_paths if path is not None]

    timestamp = int(time.time())
    plot_urls = [url_for('static', filename=path) + f'?v={timestamp}' for path in valid_plot_paths]


    print(f"🖼️ Rendering plots page with {len(plot_urls)} plot(s).")
    return render_template('plots.html', plot_urls=plot_urls) # Pass URLs to template
# --- Main Execution ---
if __name__ == "__main__":
    templates_dir = 'templates'
    if not os.path.exists(templates_dir):
        print(f"⚠️ Warning: '{templates_dir}' directory not found. Creating it.")
        os.makedirs(templates_dir)
    index_path = os.path.join(templates_dir, 'index.html')
    if not os.path.exists(index_path):
         print(f"⚠️ Warning: '{index_path}' not found. Creating a placeholder.")
         with open(index_path, 'w') as f:
              f.write('<html><head><title>App Running</title></head><body><h1>Sales Chatbot Backend Running</h1><p>Your real index.html should be here.</p></body></html>')

    print(f"🚀 Starting Flask server on http://0.0.0.0:5000 (Debug: {app.debug})")
    # Use debug=False for production
    app.run(debug=True, port=5000, host='0.0.0.0')