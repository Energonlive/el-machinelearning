import streamlit as st
import numpy as np
import joblib

st.title('🪙 Loan Risk Prediction Machine Learning App')
tab1, tab2, tab3 = st.tabs(["Predict Loan Risk", "Bulk Predict", "Model Information"])

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
    st.write(f"You selected subgrade: {selected_subgrade}")

    annual_inc_input = st.text_input("Enter Annual Income ($)")
    try:
        annual_inc = float(annual_inc_input.replace(',', ''))
        if annual_inc >= 0:
            st.success(f"Annual Income: ${annual_inc:,.2f}")
        else:
            st.warning("Please enter a non-negative value.")
    except ValueError:
        if annual_inc_input:
            st.error("Please enter a valid number.")

    selected_term = st.radio("Select Term (months)", [36, 60])
    st.write(f"You selected term: {selected_term} months")

    dti_input = st.text_input("Enter DTI (Debt-to-Income ratio) (%)")

    try:
        dti = float(dti_input)
        if 0.0 <= dti <= 100.0:
            st.success(f"DTI: {dti}%")
        else:
            st.warning("Please enter a value between 0 and 100.")
    except ValueError:
        if dti_input:
            st.error("Please enter a valid number.")

    revol_util_input = st.text_input("Enter Revolving Line Utilization (%)")

    try:
        revol_util = float(revol_util_input)
        if 0.0 <= revol_util <= 100.0:
            st.success(f"Revolving Utilization: {revol_util}%")
        else:
            st.warning("Please enter a value between 0 and 100.")
    except ValueError:
        if revol_util_input:
            st.error("Please enter a valid number.")

