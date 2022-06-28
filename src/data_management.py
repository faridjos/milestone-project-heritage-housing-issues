import streamlit as st
import pandas as pd
import numpy as np
import joblib

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_house_prices_data():
    df = pd.read_csv("outputs/datasets/collection/HousePricesRecords.csv")
    return df

def load_corr():
    df = pd.read_csv("outputs/house_prices_study/v1/corr_df_rev.csv")
    return df

def load_pkl_file(file_path):
    return joblib.load(filename=file_path)