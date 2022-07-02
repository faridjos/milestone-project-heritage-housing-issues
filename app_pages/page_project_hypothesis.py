import streamlit as st
from src.snsplot import plot_histogram_and_boxplot
from src.data_management import load_house_price_data

def page_project_hypothesis_body():

    df = load_house_price_data()

    st.write("### Project Hypothesis and Validation")

    # conclusions taken House Sale Prices notebook 
    st.success(
        f"* We suspect that there is a few very high sale prices."
        f" The combined boxplot/histogram below confirms that: The histogram extends far to the right."
        f" It has a long tail.  \n"
        f"* The price values well beyond the average range are called outliers and are shown as dots "
        f"to the right of the box in the boxplot. They correspond to sale prices"
        f" above $466075"
         
    )

    df2=df.filter(['SalePrice'])
    plot_histogram_and_boxplot(df2)         

        
    st.info(
        f"* The models we have created do not accurately predict sale prices above $400000 "
        f"(see scatterplots on the ML Regressor Model page). \n"
        f"* This could be connected to the outliers mentioned above (with sale prices above $466075)"
        f" but it cannot be proven. Further investigation is needed.")
        