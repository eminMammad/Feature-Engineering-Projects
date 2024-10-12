import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def check_df(df, head=5):
    print("##################### Shape #####################")
    print(df.shape)
    print("##################### Types #####################")
    print(df.dtypes)
    print("##################### Head #####################")
    print(df.head(head))
    print("##################### Tail #####################")
    print(df.tail(head))
    print("##################### NA #####################")
    print(df.isnull().sum())
    print("##################### Quantiles #####################")
    numeric_df = df.select_dtypes(include=[np.float64, np.int64])
    print(numeric_df.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

def grab_columns(df,cat_th=10, car_th=20):
    cat_cols = [col for col in df.columns if df[col].dtype not in [int, float]]
    num_cols = [col for col in df.columns if df[col].dtype in [int, float]]
    num_but_cat = [col for col in df.columns if df[col].dtype in [int,float] and df[col].nunique() < cat_th]
    cat_but_car = [col for col in df.columns if df[col].dtype in ["object", "category", "bool"] and df[col].nunique() > car_th]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {df.shape[0]}")
    print(f"Variables: {df.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


def cat_summary(df,col_name,plot=False):
    print(pd.DataFrame({
        col_name:df[col_name].value_counts(),
        'Ratio': 100*(df[col_name].value_counts())/len(df)
    }))
    print("##########################################")

    if plot:
        sns.countplot(x=df[col_name], data=df)
        plt.show(block=True)



def num_summary(df, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(df[numerical_col].describe(quantiles))

    if plot:
        # dataframe[numerical_col].hist(bins=20)
        # plt.xlabel(numerical_col)
        # plt.title(numerical_col)
        # OR
        sns.boxplot(x=df[numerical_col], data=df)
        plt.show()

def target_summary_with_num(df, target, numerical_col):
    print(df.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


def show_corr(df):
    corr = df.corr()
    f, ax = plt.subplots(figsize=[20,15])
    sns.heatmap(corr,cmap="magma", ax=ax, fmt=".2f", annot=True)
    ax.set_title("Correlation Matrix", fontsize=20)
    plt.show(block=True)


def binary_cols(df):
    return [col for col in df.columns if df[col].dtype in ["object", "category", "bool", np.int64] and df[col].nunique() == 2]

def one_hot_encoder(df, ohe_cols, drop_first=False):
    return  pd.get_dummies(df, columns=ohe_cols, drop_first=drop_first,dtype=int)

def outlier_thresholds(df, col_name, q1_percentage=0.05, q3_percentage=0.95):
    q1 = df[col_name].quantile(q1_percentage)
    q3 = df[col_name].quantile(q3_percentage)
    inter_quartile_range = q3 - q1

    lower_limit = q1 - 1.5 * inter_quartile_range
    upper_limit = q3 + 1.5 * inter_quartile_range

    return lower_limit, upper_limit


def check_outlier(df, col_name, q1_percentage=0.05, q3_percentage=0.95):
    lower, upper = outlier_thresholds(df, col_name, q1_percentage, q3_percentage)

    return df.loc[(df[col_name] < lower) | (df[col_name]) > upper].any(axis=None)


def grab_outliers(df, col_name, index=False, q1_percentage=0.05, q3_percentage=0.95):
    lower_limit, upper_limit = outlier_thresholds(df,col_name, q1_percentage, q3_percentage)
    outliers = df.loc[(df[col_name] < lower_limit) | (df[col_name] > upper_limit)].index
    print(df.iloc[outliers, :].head(10))
    if index:
        return outliers

    