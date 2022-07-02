import streamlit as st
import numpy as np
import pandas as pd
from src.data_management import load_house_price_data, load_pkl_file
#from src.machine_learning.predictive_analysis_ui import (predict_sale_price)

def page_predict_house_price_body():
	
	# load predict house price files
	version = 'v1'
	pipeline = load_pkl_file(f"outputs/predict_price/{version}/best_regressor_pipeline.pkl")
	best_features = (pd.read_csv(f"outputs/predict_price/{version}/X_train.csv")
							.columns
							.to_list()
							)
	df = pd.read_csv("inputs/datasets/raw/house-price-20211124T154130Z-001/house-price/inherited_houses.csv")
	
		
	st.write("### House sale prices from client's inherited houses")
	st.write(
        f"* The table shows the four inherited houses profile"
	)
	st.write(df.head())

	df = df.filter(best_features)
	house_price_prediction = pipeline.predict(df).round(0)
	
	df['Predicted House Sale Price'] = house_price_prediction

	st.write(
        f"* The table shows the predicted sale price for the four houses, together with the house features used in the prediction"
	)
	st.write(df.head())

	sum = df['Predicted House Sale Price'].sum()

	st.write(
        f"* The sum of the predicted sale price for the four houses is: &nbsp; &nbsp; &nbsp;{sum}  \n"
	)

	st.write("---")

	st.write("### Predict house sale prices in Ames, Iowa")

	# Generate Live Data
	X_live = DrawInputsWidgets()

	# predict on live data
	if st.button("Run Predictive Analysis"):
		house_price_prediction = pipeline.predict(X_live.filter(best_features)).round(0)
		st.write(
			f"* The predicted sale price for the house is: &nbsp; &nbsp; &nbsp;{house_price_prediction[0]}  \n"
		)
			

def DrawInputsWidgets():

	# load dataset
	df = load_house_price_data()
	percentageMin, percentageMax = 0.5, 2.0

    # we create input widgets only for 6 features	
	col1, col2 = st.columns(2)

	# We are using these features to feed the ML pipeline - values copied from check_variables_for_UI() result
		

 	# create an empty DataFrame, which will be the live data
	X_live = pd.DataFrame([], index=[0]) 
	
	# from here on we draw the widget based on the variable type (numerical or categorical)
	# and set initial values
	with col1:
		feature = 'GrLivArea'
		st_widget = st.number_input(
	 		label= feature,
			min_value= df[feature].min()*percentageMin,
			max_value= df[feature].max()*percentageMax,
			value= df[feature].median()
			)
	X_live[feature] = st_widget


	with col2:
		feature = "OverallQual"
		st_widget = st.selectbox(
			label= feature,
			options= df[feature].sort_values(ascending=True).unique()
			)
	X_live[feature] = st_widget

	# st.write(X_live)

	return X_live