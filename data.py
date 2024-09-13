import requests
import pandas as pd

# Etherscan API key
api_key = '1RY32CCK7M9Z9AQPFZIMXB6HZE8PW2XMDN'

# Function to get Ethereum transactions
def get_ethereum_transactions(address, start_block, end_block, total_transactions=500):
    all_transactions = []
    page = 1
    offset = 100  # Increase the number of records per request to 100

    while len(all_transactions) < total_transactions:
        url = f"https://api.etherscan.io/api?module=account&action=txlist&address={address}&startblock={start_block}&endblock={end_block}&page={page}&offset={offset}&apikey={api_key}"
        
        response = requests.get(url)
        data = response.json()
        
        if data['status'] == '1' and data['message'] == 'OK':
            transactions = data['result']
            all_transactions.extend(transactions)
            page += 1
        else:
            print(f"Error fetching data: {data['message']}")
            break
    
    return pd.DataFrame(all_transactions[:total_transactions])

# Function to calculate unique from addresses for each transaction
def calculate_unique_from_addresses(transactions_df):
    unique_from_counts = []

    # Calculate the unique count of from addresses for the entire dataframe
    unique_from_addresses = transactions_df['from'].nunique()

    # Assign the unique from count to each transaction
    transactions_df['Unique_Received_From_Addresses'] = unique_from_addresses-1
    return transactions_df

# Example: Get transactions for a specific address
address = '0x00009277775ac7d0d59eaad8fee3d10ac6c805e8'
start_block = 0
end_block = 99999999
transactions = get_ethereum_transactions(address, start_block, end_block, total_transactions=2500)

# Calculate unique from addresses
if not transactions.empty:
    transactions = calculate_unique_from_addresses(transactions)
    csv_file = 'ethereum_transactions.csv'
    transactions.to_csv(csv_file, index=False)
    print(f"Transaction data with unique from addresses has been saved to '{csv_file}'.")
else:
    print("No transaction data retrieved.")

# Load the data back into a DataFrame
loaded_transactions = pd.read_csv(csv_file)
print("Loaded Transactions DataFrame with Unique_Received_From_Addresses:")
print(loaded_transactions)
