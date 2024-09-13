import requests
import pandas as pd
from datetime import datetime

# Etherscan API key
api_key = '1RY32CCK7M9Z9AQPFZIMXB6HZE8PW2XMDN'

# Function to get block number by timestamp
def get_block_number_by_timestamp(timestamp):
    url = f"https://api.etherscan.io/api?module=block&action=getblocknobytime&timestamp={timestamp}&closest=before&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    if data['status'] == '1':
        return int(data['result'])
    else:
        print(f"Error fetching block number by timestamp: {data['message']}")
        return None

# Function to get transactions in a block
def get_block_transactions(block_number):
    url = f"https://api.etherscan.io/api?module=proxy&action=eth_getBlockByNumber&tag={hex(block_number)}&boolean=true&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    if 'result' in data:
        transactions = data['result']['transactions']
        return transactions
    else:
        print(f"Error fetching block transactions: {data['message']}")
        return []

# Function to extract unique addresses from transactions
def extract_unique_addresses(transactions):
    addresses = set()
    for tx in transactions:
        addresses.add(tx['from'])
        if tx['to']:
            addresses.add(tx['to'])
    return addresses

# 获取2023年4月的起始和结束区块号
start_timestamp = int(datetime(2023, 4, 1, 0, 0).timestamp())
end_timestamp = int(datetime(2023, 4, 1, 0, 59).timestamp())

# end_timestamp = int(datetime(2023, 4, 30, 23, 59).timestamp())

start_block = get_block_number_by_timestamp(start_timestamp)
end_block = get_block_number_by_timestamp(end_timestamp)

print(f"Start block: {start_block}, End block: {end_block}")

import time
s_t = time.time()


# 提取所有交易地址
all_addresses = set()
for block_number in range(start_block, end_block + 1):
    transactions = get_block_transactions(block_number)
    block_addresses = extract_unique_addresses(transactions)
    all_addresses.update(block_addresses)

e_t = time.time()
print(e_t - s_t)


# 保存交易地址到CSV文件
addresses_df = pd.DataFrame(list(all_addresses), columns=['Address'])
csv_file = 'ethereum_addresses_april_2023.csv'
addresses_df.to_csv(csv_file, index=False)


print(f"All transaction addresses for April 2023 have been saved to '{csv_file}'.")     
