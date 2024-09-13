import requests
from bs4 import BeautifulSoup
import csv

def check_phishing_warning(address):
    url = f'https://etherscan.io/address/{address}'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 查找特定的警告消息
        alert_div = soup.find('div', {'class': 'alert alert-danger alert-dismissible fade show mb-3', 'role': 'alert'})
        if alert_div:
            return "1"
        else:
            return "0"
    else:
        return "Error"

def read_addresses_from_csv(input_csv_file):
    addresses = []
    with open(input_csv_file, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # 跳过表头
        for row in csvreader:
            addresses.append(row[0])
    return addresses

def check_addresses_and_save_to_csv(input_csv_file, output_csv_file):
    addresses = read_addresses_from_csv(input_csv_file)
    results = []
    for address in addresses:
        tag = check_phishing_warning(address)
        results.append([address, tag])
        print(f"Address: {address}, Tag: {tag}")
    
    # 保存结果到CSV文件
    with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Address', 'Tag'])
        csvwriter.writerows(results)

# 输入CSV文件路径
input_csv_file = 'ethereum_addresses_april_2023.csv'
# 输出CSV文件路径
output_csv_file = 'address_phishing_check_results2.csv'

check_addresses_and_save_to_csv(input_csv_file, output_csv_file)
print(f"Results saved to {output_csv_file}")
