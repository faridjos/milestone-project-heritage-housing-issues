import streamlit as st
from src.snsplot import plot_histogram_and_boxplot
from src.data_management import load_house_price_data

def page_project_hypothesis_body():

    df = load_house_price_data()

    st.write("### Project Hypothesis and Validation")

    # conclusions taken House Sale Prices notebook 
    st.success(
        f"* We suspect that the distribution of the sale prices has a long right tail."
        f" The combined boxplot/histogram below confirms that.  \n"
        f"* Outliers are defined to be outside the upper quantile by a distance equal to"
        f" three times the interquantile range (IQR). This corresponds to sale prices"
        f" above $466075"
         
    )

    df2=df.filter(['SalePrice'])
    plot_histogram_and_boxplot(df2)         

        
    st.info(
        f"* There is a problem when it comes to predicting high sale prices."
        f" This is confirmed by the fact that our model does not predict prices above $457199.  \n"
        f"* This could be connected to the outliers with sale prices above $466075 but it cannot"
        f" be proven. Further investigation is needed.")
        