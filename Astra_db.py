import os
import sys
from dotenv import load_dotenv
from astrapy import DataAPIClient

# --- Load environment variables from .env ---
load_dotenv()

ASTRA_DB_TOKEN = os.getenv("ASTRA_DB_TOKEN")
ASTRA_DB_ENDPOINT = os.getenv("ASTRA_DB_ENDPOINT")

if not ASTRA_DB_TOKEN or not ASTRA_DB_ENDPOINT:
    print("Error: ASTRA_DB_TOKEN and/or ASTRA_DB_ENDPOINT not set in the environment.")
    sys.exit(1)

# --- Connect to Astra DB using DataAPIClient ---
try:
    client = DataAPIClient(ASTRA_DB_TOKEN)
    db = client.get_database_by_api_endpoint(ASTRA_DB_ENDPOINT)
    sales_collection = db.get_collection("sales")
    print("✅ Connected to Astra DB and accessed 'sales' collection.")
except Exception as e:
    print("❌ Failed to connect or access the sales collection:", e)
    sys.exit(1)

# --- Function to fetch sales documents from Astra DB ---
def fetch_sales_documents(collection, limit=None):
    try:
        sales_docs = list(collection.find({}, limit=limit))
        print(f"📄 Retrieved {len(sales_docs)} sales document(s) from the database.")
        return sales_docs
    except Exception as e:
        print("❌ Error fetching data:", e)
        sys.exit(1)

# --- Function to find the sale with the highest total_price ---
def find_highest_total_price_sale(sales_docs):
    if not sales_docs:
        print("⚠️ No sales data found.")
        return None
    highest_sale = max(sales_docs, key=lambda x: x.get('total_price', 0))
    print("💰 Highest total_price sale:")
    print(highest_sale)
    return highest_sale

# --- Main function to run the query ---
def main():
    sales_docs = fetch_sales_documents(sales_collection, limit=5000)
    find_highest_total_price_sale(sales_docs)

if __name__ == "__main__":
    main()
