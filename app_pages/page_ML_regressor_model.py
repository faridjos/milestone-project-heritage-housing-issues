import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data_management import load_pkl_file
from src.evaluate import regression_performance, regression_evaluation, regression_evaluation_plots


def page_ML_regressor_model_body():

    # load house price pipeline files
    version = 'v1'
    pipeline = load_pkl_file(f"outputs/predict_price/{version}/best_regressor_pipeline.pkl")
    house_price_feat_importance = plt.imread(f"outputs/predict_price/{version}/features_importance.png")
    X_train = pd.read_csv(f"outputs/predict_price/{version}/X_train.csv")
    X_test = pd.read_csv(f"outputs/predict_price/{version}/X_test.csv")
    y_train =  pd.read_csv(f"outputs/predict_price/{version}/y_train.csv").squeeze()
    y_test =  pd.read_csv(f"outputs/predict_price/{version}/y_test.csv").squeeze()

 

    st.write("### ML house price pipeline")

    # summary of model performance
    st.info(
        f"* We agreed with the client on an R2 score of at least 0.75 on both train and test "
        f"set.  \n"
        f"* Our pipeline achieves 0.83 and 0.76 on train set and test set respectively  \n"    
        f"* We notice that our model does not predict prices above $457199. This is possibly"
        f" connected to the fact that the distribution of actual sale prices has a very long tail.")
    st.write("---")

    # show pipeline steps
    st.write("* ML pipeline to predict house sale price")
    st.write(pipeline)
    st.write("---")

    # show best features
    st.write("* The features the model was trained on and their importance")
    st.write(X_train.columns.to_list())
    st.image(house_price_feat_importance)
    st.write("---")

    # evaluate performance on both sets
    regression_performance(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, pipeline=pipeline)
    
    regression_evaluation_plots(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, pipeline=pipeline, alpha_scatter=0.5)

