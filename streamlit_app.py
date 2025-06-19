import streamlit as st
import numpy as np
import joblib

st.title('🪙 Loan Risk Prediction Machine Learning App')
tab1, tab2 = st.tabs(["Predict Loan Risk", "Bulk Predict", "Model Information"])

with tab1: 
    st.subheader('Predict Loan Risk for a Single Applicant')

    # Input fields for the user to enter applicant details
    zipcode_locations = {
        "22690": "22690 - Springfield, VA",
        "05113": "05113 - Montpelier, VT",
        "00813": "00813 - Charlotte Amalie, VI",
        "11650": "11650 - Far Rockaway, NY",
        "30723": "30723 - Dalton, GA",
        "70466": "70466 - Hammond, LA",
        "29597": "29597 - Myrtle Beach, SC",
        "48052": "48052 - Mount Clemens, MI",
        "86630": "86630 - Casa Grande, AZ",
        "93700": "93700 - Fresno, CA"
    }

    selected_zipcode = st.selectbox("Select Zipcode", 
                            list(zipcode_locations.keys()),
                            format_func=lambda x: zipcode_locations[x])
    st.write(f"You selected zipcode: {selected_zipcode}")
    
    subgrades = [f"{grade}{i}" for grade in "ABCDEFG" for i in range(1, 6)]

    selected_subgrade = st.selectbox("Select Subgrade", subgrades)

    selected_term = st.radio("Select Term (months)", [36, 60])
    st.write(f"You selected term: {selected_term} months")