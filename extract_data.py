import numpy as np 
import pandas as pd
df = pd.read_csv(
    './datasets/transaction_dataset.csv',
).drop(columns=['Unnamed: 0'])
df_copy = df.copy()
import pandas as pd

# 假设df是已经加载的原始DataFrame
# 计算数值型列的均值
numeric_means = df.mean(numeric_only=True)

# 确保所有列都在新的数据中
all_columns = df.columns.tolist()
new_data_mean_complete = {col: numeric_means.get(col, None) for col in all_columns}
new_data_mean_complete.update({
    'Index': df['Index'].max() + 1,
    'Address': '0xNewAddressForTestWithMeanValues',
    'FLAG': 0
})

# 将新的完整数据转换为DataFrame
df_mean_complete = pd.DataFrame([new_data_mean_complete])

# 删除最后一列（假设最后一列是多余的特征）
# df_mean_complete = df_mean_complete.iloc[:, :-1]

# 删除 'Unnamed: 0' 列，如果存在的话
if 'Unnamed: 0' in df_mean_complete.columns:
    df_mean_complete = df_mean_complete.drop(columns=['Unnamed: 0'])

# 保存为CSV文件
final_csv_path = 'modified_mean_values_dataset.csv'
df_mean_complete.to_csv(final_csv_path, index=True)

# 显示文件路径
print(final_csv_path)