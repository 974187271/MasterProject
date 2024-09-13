import requests
from bs4 import BeautifulSoup

def check_phishing_warning(address, output_html_file=None):
    url = f'https://etherscan.io/address/{address}'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 保存解析后的HTML内容
        if output_html_file:
            with open(output_html_file, 'w', encoding='utf-8') as file:
                file.write(str(soup))
        
        # 查找特定的警告消息
        alert_div = soup.find('div', {'class': 'alert alert-danger alert-dismissible fade show mb-3', 'role': 'alert'})
        if alert_div:
            warning_text = alert_div.get_text(strip=True)
            return warning_text
        else:
            return "This address is normal. No phishing warning found."
    else:
        return f"Failed to retrieve page. Status code: {response.status_code}"

# address = '0x051005cDCecd916FB8b98643d923646Acc7e07cd'
address = '0xA1eD5Df1278e2326e302cf2f923F4079926bE25C'
output_html_file = 'etherscan_address_page.html'

result = check_phishing_warning(address, output_html_file)
print(result)
