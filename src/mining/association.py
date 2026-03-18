import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

def run_fp_growth(df, min_support=0.15, max_len=3):
    print(f"Đang chạy FP-Growth (min_support={min_support})...")
    df_bool = pd.get_dummies(df).astype(bool)
    frequent_itemsets = fpgrowth(df_bool, min_support=min_support, use_colnames=True, max_len=max_len)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    return rules