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

    annual_inc_input = st.text_input("Enter Annual Income ($) (e.g 10000 or 10,000)")
    annual_inc = None
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
    revol_util = None
    try:
        revol_util = float(revol_util_input)
        if 0.0 <= revol_util <= 100.0:
            st.success(f"Revolving Utilization: {revol_util}%")
        else:
            st.warning("Please enter a value between 0 and 100.")
    except ValueError:
        if revol_util_input:
            st.error("Please enter a valid number.")


    revol_bal_input = st.text_input("Enter Revolving Balance ($) (e.g 10000 or 10,000)")
    revol_bal = None
    try:
        revol_bal = float(revol_bal_input.replace(',', ''))
        if revol_bal >= 0:
            st.success(f"Revolving Balance: ${revol_bal:,.2f}")
        else:
            st.warning("Revolving balance cannot be negative.")
    except ValueError:
        if revol_bal_input:
            st.error("Please enter a valid number.")

    open_acc_input = st.text_input("Enter Number of Open Credit Lines")
    open_acc = None
    try:
        open_acc = int(open_acc_input)
        if open_acc >= 0:
            st.success(f"Open Credit Lines: {open_acc}")
        else:
            st.warning("Number of open accounts cannot be negative.")
    except ValueError:
        if open_acc_input:
            st.error("Please enter a valid whole number.")

    loan_amount_input = st.text_input("Enter Loan Amount ($) (e.g 10000 or 10,000)")
    loan_amount = None
    try:
        loan_amount = float(loan_amount_input.replace(',', ''))
        if loan_amount > 0:
            st.success(f"Loan Amount: ${loan_amount:,.2f}")
        else:
            st.warning("Loan amount must be greater than 0.")
    except ValueError:
        if loan_amount_input:
            st.error("Please enter a valid number.")

    int_rate_input = st.text_input("Enter Interest Rate (%) (per annum, e.g 10.5)")
    int_rate = None
    try:
        int_rate = float(int_rate_input)
        if 0.0 <= int_rate <= 100.0:
            st.success(f"Interest Rate: {int_rate:.2f}%")
        else:
            st.warning("Interest rate must be between 0 and 100.")
    except ValueError:
        if int_rate_input:
            st.error("Please enter a valid number.")

    installment_display = ""
    if loan_amount is not None and int_rate is not None and selected_term is not None:
        monthly_rate = int_rate / 100 / 12
        try:
            if monthly_rate == 0:
                installment = loan_amount / selected_term
            else:
                installment = (loan_amount * monthly_rate) / (1 - (1 + monthly_rate) ** -selected_term)
            installment_display = f"{installment:,.2f}"
        except ZeroDivisionError:
            st.error("Installment calculation failed Please check inputs.")
    
    st.text_input("Calculated Monthly Installment ($)", value=f"{installment_display}" if 'installment' in locals() else "", disabled=True)

    if 'installment' in locals():
        st.write(f"Installment: ${installment_display}")
    else:
        st.info("Installment will be calculated after entering all required fields.")
        
    st.markdown(
        """
        **Note:** Installment is calculated using the formula:  
        `Installment = (Loan Amount x Monthly Interest Rate) / (1 - (1 + Monthly Interest Rate)^-Term)`

        Where:  
        - Monthly Interest Rate = Annual Interest Rate / 100 / 12
        """
    )
    
    # home_ownership = st.selectbox("Select Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])


