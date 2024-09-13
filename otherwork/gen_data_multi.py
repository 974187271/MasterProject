import requests
import pandas as pd
from datetime import datetime

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
            print(f"Error fetching data for address {address}: {data['message']}")
            break
    
    return pd.DataFrame(all_transactions[:total_transactions])

# Function to get ERC20 transactions
def get_erc20_transactions(address, start_block, end_block, total_transactions=500):
    all_transactions = []
    page = 1
    offset = 100  # Increase the number of records per request to 100

    while len(all_transactions) < total_transactions:
        url = f"https://api.etherscan.io/api?module=account&action=tokentx&address={address}&startblock={start_block}&endblock={end_block}&page={page}&offset={offset}&apikey={api_key}"
        
        response = requests.get(url)
        data = response.json()
        
        if data['status'] == '1' and data['message'] == 'OK':
            transactions = data['result']
            all_transactions.extend(transactions)
            page += 1
        else:
            print(f"Error fetching ERC20 data for address {address}: {data['message']}")
            break
    
    return pd.DataFrame(all_transactions[:total_transactions])

# Function to get account balance
def get_account_balance(address):
    url = f"https://api.etherscan.io/api?module=account&action=balance&address={address}&tag=latest&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    
    if data['status'] == '1' and data['message'] == 'OK':
        balance = float(data['result']) / 10**18  # Convert balance to Ether
    else:
        print(f"Error fetching balance for address {address}: {data['message']}")
        balance = 0.0
    
    return balance

# Function to calculate unique from addresses for each transaction
def calculate_unique_from_addresses(transactions_df):
    unique_from_counts = transactions_df['from'].nunique()
    transactions_df['Unique Received From Addresses'] = unique_from_counts - 1
    return transactions_df

# Function to calculate time difference between first and last transaction in minutes
def calculate_time_difference(transactions_df):
    if 'timeStamp' in transactions_df.columns:
        transactions_df['timeStamp'] = pd.to_datetime(transactions_df['timeStamp'], unit='s')
        first_transaction_time = transactions_df['timeStamp'].min()
        last_transaction_time = transactions_df['timeStamp'].max()
        time_diff_minutes = (last_transaction_time - first_transaction_time).total_seconds() / 60
        transactions_df['Time Diff between first and last (Mins)'] = time_diff_minutes
    return transactions_df

def calculate_additional_features(transactions_df, erc20_transactions_df, address):
    # Initialize all necessary features
    features = {
        "ERC20 max val rec": 0,
        "Total ERC20 tnxs": 0,
        "total transactions (including tnx to create contract)": len(transactions_df),
        "ERC20 total Ether received": 0,
        "total ether balance": get_account_balance(address),
        "avg val received": 0,
        "total ether received": 0,
        "Avg min between received tnx": 0,
        "ERC20 total ether sent": 0,
        "ERC20 avg val rec": 0,
        "Received Tnx": 0,
        "min val sent": 0,  
        "ERC20 min val rec": 0,
        "min value received": 0,
        "ERC20 uniq rec contract addr": 0,
        "max value received": 0,
        "total Ether sent": 0,
        "ERC20 uniq sent token name": 0,
        "ERC20 uniq sent addr": 0,
        "max val sent": 0,
        "ERC20 avg val sent": 0,
        "Unique Sent To Addresses": 0,
        "ERC20 max val sent": 0,
        "ERC20 uniq rec token name": 0,
        "Number of Created Contracts": 0,
        "Avg min between sent tnx": 0,
        "Sent tnx": 0,
        "avg val sent": 0,
        "min value sent to contract": 0,
        "max val sent to contract": 0,
        "avg value sent to contract": 0,
        "total ether sent contracts": 0,
        "ERC20 total Ether sent contract": 0,
        "ERC20 uniq rec addr": 0
    }

    # Process received ERC20 transactions
    if 'to' in erc20_transactions_df.columns:
        erc20_received = erc20_transactions_df[erc20_transactions_df['to'].str.lower() == address.lower()]
        if not erc20_received.empty:
            erc20_received['value'] = erc20_received['value'].astype(float) / 10**18  # Convert to Ether
            features["ERC20 max val rec"] = erc20_received['value'].max()
            features["ERC20 min val rec"] = erc20_received['value'].min()
            features["ERC20 avg val rec"] = erc20_received['value'].mean()
            features["ERC20 total Ether received"] = erc20_received['value'].sum()
            features["ERC20 uniq rec token name"] = erc20_received['tokenName'].nunique()
            features["ERC20 uniq rec contract addr"] = erc20_received['contractAddress'].nunique()
            features["Total ERC20 tnxs"] = len(erc20_received)
            features["ERC20 uniq rec addr"] = erc20_received['from'].nunique()

    # Process sent ERC20 transactions
    if 'from' in erc20_transactions_df.columns:
        erc20_sent = erc20_transactions_df[erc20_transactions_df['from'].str.lower() == address.lower()]
        if not erc20_sent.empty:
            erc20_sent['value'] = erc20_sent['value'].astype(float) / 10**18  # Convert to Ether
            features["ERC20 max val sent"] = erc20_sent['value'].max()
            features["ERC20 avg val sent"] = erc20_sent['value'].mean()
            features["ERC20 total ether sent"] = erc20_sent['value'].sum()
            features["ERC20 uniq sent token name"] = erc20_sent['tokenName'].nunique()
            features["ERC20 uniq sent addr"] = erc20_sent['to'].nunique()
            features["ERC20 total Ether sent contract"] = erc20_sent['value'].sum()

    # Process normal transactions
    if 'from' in transactions_df.columns:
        sent_transactions = transactions_df[transactions_df['from'].str.lower() == address.lower()]
        if not sent_transactions.empty:
            sent_transactions['value'] = sent_transactions['value'].astype(float) / 10**18  # Convert to Ether
            features["Avg min between sent tnx"] = sent_transactions['timeStamp'].diff().mean().total_seconds() / 60
            features["Sent tnx"] = len(sent_transactions)
            features["avg val sent"] = sent_transactions['value'].mean()
            features["min val sent"] = sent_transactions['value'].min()
            features["max val sent"] = sent_transactions['value'].max()
            features["total Ether sent"] = sent_transactions['value'].sum()
            features["Unique Sent To Addresses"] = sent_transactions['to'].nunique()

            # Assuming there are transactions sent to contracts, need to confirm with actual data structure
            sent_to_contracts = sent_transactions[sent_transactions['to'].apply(lambda x: x.startswith('0x'))]  # Assuming contract addresses start with '0x'
            if not sent_to_contracts.empty:
                features["min value sent to contract"] = sent_to_contracts['value'].min()
                features["max val sent to contract"] = sent_to_contracts['value'].max()
                features["avg value sent to contract"] = sent_to_contracts['value'].mean()
                features["total ether sent contracts"] = sent_to_contracts['value'].sum()

    # Number of Created Contracts
    features["Number of Created Contracts"] = len(transactions_df[transactions_df['isError'] == '0'])

    # Create a DataFrame from features
    features_df = pd.DataFrame([features])

    return features_df

# Read the addresses and flags from a CSV file
input_csv_path = './address_phishing_check_results2.csv'
address_df = pd.read_csv(input_csv_path)

# Load the modified dataset
uploaded_file_path = '../modified_mean_values_dataset.csv'
df_modified = pd.read_csv(uploaded_file_path)

# Loop through each address
for index, row in address_df.iterrows():
    address = row['Address']
    flag = row['Tag']

    # Get transactions for the current address
    transactions = get_ethereum_transactions(address, start_block=0, end_block=99999999, total_transactions=2500)
    erc20_transactions = get_erc20_transactions(address, start_block=0, end_block=99999999, total_transactions=2500)

    if not transactions.empty:
        # Calculate features for the current address
        transactions = calculate_unique_from_addresses(transactions)
        transactions = calculate_time_difference(transactions)
        features_df = calculate_additional_features(transactions, erc20_transactions, address)

        # Add the flag to the features DataFrame
        features_df['flag'] = flag

        # Append the features DataFrame to the modified dataset DataFrame
        df_modified = pd.concat([df_modified, features_df], ignore_index=True)

        # Save the updated DataFrame to a new CSV file
        updated_df_path = './updated_mean_values_dataset.csv'
        df_modified.to_csv(updated_df_path, index=True)

print("All addresses processed and results saved.")
