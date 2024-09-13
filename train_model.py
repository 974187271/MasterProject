import numpy as np 
import pandas as pd
import warnings
import joblib
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier, early_stopping
from typing import Union, Optional  # 确保正确导入这些类型
from sklearn.utils.validation import check_is_fitted  # 添加这一行

warnings.filterwarnings('ignore')

class DataPrep(BaseEstimator, TransformerMixin):
    """Implementation preprocess dataset in several strategies"""
    
    def __init__(self, num_feature_list: list, cat_feature_list: list, drop_feature_list: Optional[list] = None,
                cat_encoder_type: Union[str, list] = 'label', cat_min_count: int = 10,
                fillna: Union[int, str] = 0, q_up_clip_outliers: Optional[float] = None,
                q_down_clip_outliers: Optional[float] = None, build_feature=False):
        self.cat_feature_list = cat_feature_list
        self.num_feature_list = num_feature_list
        self.cat_encoder_type = cat_encoder_type
        self.drop_feature_list = drop_feature_list
        self.cat_min_count = 50
        self.fillna = fillna
        self.q_up_clip_outliers = q_up_clip_outliers
        self.q_down_clip_outliers = q_down_clip_outliers
        self.build_feature = build_feature
        
        
    def fit(self, df):
        self.num_fillna_dict = {}
        self.num_q_up_dict = {}
        self.num_q_down_dict = {}
        self.cat_emb_dict = {}
        
        # numerical fillna fit
        if self.fillna == 'median':
            for feature in self.num_feature_list:
                self.num_fillna_dict[feature] = df[feature].median()
        elif self.fillna == 'mean':
            for feature in self.num_feature_list:
                self.num_fillna_dict[feature] = df[feature].mean()
        elif self.fillna == 0:
            for feature in self.num_feature_list:
                self.num_fillna_dict[feature] = 0                
        else:
            for feature in self.num_feature_list:
                self.num_fillna_dict[feature] = None
            
        # numerical outliers fit
        if self.q_up_clip_outliers:
            for feature in self.num_feature_list:
                self.num_q_up_dict[feature] = df[feature].quantile(self.q_up_clip_outliers)
                
        if self.q_down_clip_outliers:
            for feature in self.num_feature_list:
                self.num_q_down_dict[feature] = df[feature].quantile(self.q_down_clip_outliers)
            
        # cat fit
        for feature in self.cat_feature_list:
            cat_series = df[feature].value_counts()
            cat_series[cat_series.lt(self.cat_min_count)] = 1
            self.cat_emb_dict[feature] = cat_series.to_dict()
            
        if self.drop_feature_list:
            self.num_feature_list = list(set(self.num_feature_list) - set(self.drop_feature_list))
            self.cat_feature_list = list(set(self.cat_feature_list) - set(self.drop_feature_list))
            
        return self
        
    def transform(self, df):
        check_is_fitted(self, attributes=['num_fillna_dict', 'cat_emb_dict'])
        
        # drop features
        if self.drop_feature_list:
            df = df.drop(columns=self.drop_feature_list)
        
        
        # numerical fillna
        for feature in self.num_feature_list:
            df.loc[df[feature].isna(), feature] = self.num_fillna_dict[feature]
        
        
        # numerical outliers
        if self.q_up_clip_outliers:
            for feature in self.num_feature_list:
                df.loc[df[feature] > self.num_q_up_dict[feature], feature] = self.num_q_up_dict[feature]
                
        if self.q_down_clip_outliers:
            for feature in self.num_feature_list:
                df.loc[df[feature] < self.num_q_down_dict[feature], feature] = self.num_q_down_dict[feature]
        
        
        # categorical embed
        df[self.cat_feature_list] = df[self.cat_feature_list].fillna('None') 
        for feature in self.cat_feature_list:
            df[feature] = df[feature].map(self.cat_emb_dict[feature]).fillna(1)
            
        cat_encoder_type_list = self.cat_encoder_type if isinstance(self.cat_encoder_type, list) else [self.cat_encoder_type]
        
        if 'dummy' in cat_encoder_type_list:
            for feature in self.cat_feature_list:
                df_dummy = pd.get_dummies(df[feature], prefix=feature)
                df = df.merge(df_dummy, left_index=True, right_index=True)
                
        if 'label' not in cat_encoder_type_list:
            df = df.drop(columns=self.cat_feature_list)
            
        # feature engineering example
        if self.build_feature:
            df['total_Ether_ratio'] = df['total Ether sent'] / (df['total ether received'] + 1)
            df['total_Ether_ratio_v2'] = (df['total Ether sent'] - df['total ether received']) / (df['total Ether sent'] + df['total ether received'] + 1)
            
            df['ERC20_uniq_addr_ratio'] = df[' ERC20 uniq sent addr'] / (df[' ERC20 uniq rec addr'] + 1)
            df['ERC20_uniq_addr_ratio_v2'] = (df[' ERC20 uniq sent addr'] - df[' ERC20 uniq rec addr']) / (df[' ERC20 uniq sent addr'] + df[' ERC20 uniq rec addr'] + 1)
        
        return df

# Load data and preprocess
df = pd.read_csv('./datasets/transaction_dataset.csv').drop(columns=['Unnamed: 0'])
df = df.drop(columns='Index').drop_duplicates()
zero_feature_list = df.columns[(df.nunique() == 1)].tolist()
feature_list = list(set(df.columns) - set(['Address', 'FLAG']) - set(zero_feature_list))
num_feature_list = list(set(feature_list) - set(df.dtypes[df.dtypes == 'object'].index) - set(zero_feature_list))
cat_feature_list = list(set(feature_list) - set(num_feature_list))

# Split data
df_train, df_test = train_test_split(df, test_size=0.1, random_state=0, stratify=df['FLAG'])
df_train, df_valid = train_test_split(df_train, test_size=0.1, random_state=0, stratify=df_train['FLAG'])

dp = DataPrep(
    num_feature_list=num_feature_list,
    cat_feature_list=cat_feature_list,
    cat_encoder_type=None,
    cat_min_count=10,
    fillna='median',
)

df_train_prep = dp.fit_transform(df_train)
df_valid_prep = dp.transform(df_valid)
df_test_prep = dp.transform(df_test)

scaler = StandardScaler()
df_train_prep[num_feature_list] = scaler.fit_transform(df_train_prep[num_feature_list])
df_valid_prep[num_feature_list] = scaler.transform(df_valid_prep[num_feature_list])
df_test_prep[num_feature_list] = scaler.transform(df_test_prep[num_feature_list])

# Train model
model = LGBMClassifier(
    n_estimators=1000,
    random_state=0
)
model.fit(
    X=df_train_prep[num_feature_list],
    y=df_train['FLAG'],
    eval_set=[(df_valid_prep[num_feature_list], df_valid['FLAG'])],
    eval_metric='roc_auc',
    callbacks=[early_stopping(5)]
)

# Save the model and preprocessors
joblib.dump(model, './lgbm_model.pkl')
joblib.dump(dp, './data_prep.pkl')
joblib.dump(scaler, './scaler.pkl')
