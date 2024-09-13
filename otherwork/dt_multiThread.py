import requests
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

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
        print(f"Error fetching block transactions for block {block_number}: {data}")
        return []

# Function to extract unique addresses from transactions
def extract_unique_addresses(transactions):
    addresses = set()
    for tx in transactions:
        addresses.add(tx['from'])
        if tx['to']:
            addresses.add(tx['to'])
    return addresses

# Function to process a single block and return unique addresses
def process_block(block_number):
    try:
        transactions = get_block_transactions(block_number)
        return extract_unique_addresses(transactions)
    except Exception as e:
        print(f"Error processing block {block_number}: {e}")
        return set()

# 获取系统的CPU核心数量
num_cores = multiprocessing.cpu_count()
print(f"Number of CPU cores: {num_cores}")

# 设置max_workers为CPU核心数量的1到2倍
max_workers = num_cores * 2

# 获取2023年4月的起始和结束区块号
start_timestamp = int(datetime(2023, 4, 1, 0, 0).timestamp())
end_timestamp = int(datetime(2023, 4, 1, 0, 1).timestamp())

start_block = get_block_number_by_timestamp(start_timestamp)
end_block = get_block_number_by_timestamp(end_timestamp)

print(f"Start block: {start_block}, End block: {end_block}")


import time
s_t = time.time()


# 并行处理提取所有交易地址
all_addresses = set()
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_block = {executor.submit(process_block, block_number): block_number for block_number in range(start_block, end_block + 1)}
    for future in as_completed(future_to_block):
        block_number = future_to_block[future]
        try:
            block_addresses = future.result()
            all_addresses.update(block_addresses)
        except Exception as e:
            print(f"Error processing future for block {block_number}: {e}")

e_t = time.time()
print(e_t - s_t)

# 保存交易地址到CSV文件
addresses_df = pd.DataFrame(list(all_addresses), columns=['Address'])
csv_file = 'ethereum_addresses_april_2023.csv'
addresses_df.to_csv(csv_file, index=False)
print(f"All transaction addresses for April 2023 have been saved to '{csv_file}'.")
