import streamlit as st
import numpy as np
import pandas as pd
import os
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
    dti = None
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
    
    selected_home_ownership_status = st.selectbox("Select Home Ownership Status", ["Mortgage", "Own", "Rent", "Other"])
    st.write(f"You selected: {selected_home_ownership_status}")

    cred_hist_input = st.text_input("Enter Credit History Length (in years)")
    cred_hist_years = None
    try:
        cred_hist_years = float(cred_hist_input)
        if cred_hist_years >= 0:
            st.success(f"Credit History Length: {cred_hist_years} years")
        else:
            st.warning("Credit history length cannot be negative.")
    except ValueError:
        if cred_hist_input:
            st.error("Please enter a valid whole number.")
    
    selected_verification_status = st.selectbox("Select Verification Status", ["Source Verified", "Verified", "Not Verified"])
    st.write(f"You selected: {selected_verification_status}")

    total_acc_input = st.text_input("Enter Total Number of Credit Accounts")
    total_acc = None
    try:
        total_acc = int(total_acc_input)
        if total_acc >= 0:
            st.success(f"Total Credit Accounts: {total_acc}")
        else:
            st.warning("Total accounts cannot be negative.")
    except ValueError:
        if total_acc_input:
            st.error("Please enter a valid whole number.")

    mort_acc_input = st.text_input("Enter Number of Mortgage Accounts")
    mort_acc = None
    try:
        mort_acc = int(mort_acc_input)
        if mort_acc >= 0:
            st.success(f"Mortgage Accounts: {mort_acc}")
        else:
            st.warning("Mortgage accounts cannot be negative.")
    except ValueError:
        if mort_acc_input:
            st.error("Please enter a valid whole number.")




    subgrade_encoded = subgrades.index(selected_subgrade) if selected_subgrade in subgrades else -1

    try:
        is_valid = all(v is not None for v in [
            annual_inc, dti, revol_util, revol_bal, open_acc, selected_term,
            installment, loan_amount, cred_hist_years, total_acc, mort_acc
        ]) and subgrade_encoded != -1
    except NameError:
        is_valid = False

    if 'predict_clicked' not in st.session_state:
        st.session_state.predict_clicked = False

    if not st.session_state.predict_clicked:
        predict_clicked = st.button("🧮 Predict Loan Risk", disabled=not is_valid)
        
        if predict_clicked:
            st.session_state.predict_clicked = True

            with st.spinner("⏳ Processing prediction..."):
                input_data = np.array(
                    [
                        1 if selected_zipcode == z.split("_")[-1] else 0 for z in [
                            'zip_code_05113', 'zip_code_11650', 'zip_code_22690', 'zip_code_29597',
                            'zip_code_30723', 'zip_code_48052', 'zip_code_70466', 'zip_code_86630', 'zip_code_93700'
                        ]
                    ]
                    + [subgrade_encoded, np.log1p(annual_inc), selected_term,
                        dti, revol_util, np.log1p(revol_bal), open_acc, installment,
                        loan_amount, int_rate
                    ]
                    + [1 if selected_home_ownership_status.upper() == h.split("_")[-1] else 0 for h in [
                        'home_ownership_OTHER', 'home_ownership_OWN', 'home_ownership_RENT'
                    ]]
                    + [cred_hist_years]
                    + [1 if selected_verification_status.replace(" ", "_") == v.split("_", 1)[-1] else 0 for v in [
                        'verification_status_Source_Verified', 'verification_status_Verified'
                    ]]
                    + [total_acc, mort_acc]
                ).reshape(1, -1)

                loaded_model = joblib.load('xgb_final_pipeline.joblib')
                prediction = loaded_model.predict(input_data)
                prediction_proba = loaded_model.predict_proba(input_data)[0][1]

                if prediction[0] == 1:
                    st.subheader("Prediction Result")
                    st.success("✅ Prediction completed successfully!")
                    st.toast("Prediction complete!")
                    result_text = "🟢 **The loan is likely to be fully paid.**"
                else:
                    st.subheader("Prediction Result")
                    st.success("✅ Prediction completed successfully!")
                    st.toast("Prediction complete!")
                    result_text = "🔴 **The loan is likely to default.**"

                st.markdown(f"<h3 style='color:#333;'>{result_text}</h3>", unsafe_allow_html=True)
                st.markdown(f"<b>🧮 Probability of Full Payment:</b> {prediction_proba:.2%}")

                save_data = pd.DataFrame({
                    'zip_code': [selected_zipcode],
                    'subgrade': [selected_subgrade],
                    'annual_inc': [annual_inc],
                    'term': [selected_term],
                    'dti': [dti],
                    'revol_util': [revol_util],
                    'revol_bal': [revol_bal],
                    'open_acc': [open_acc],
                    'loan_amnt': [loan_amount],
                    'int_rate': [int_rate],
                    'installment': [installment],
                    'home_ownership': [selected_home_ownership_status],
                    'cred_hist_years': [cred_hist_years],
                    'verification_status': [selected_verification_status],
                    'total_acc': [total_acc],
                    'mort_acc': [mort_acc],
                    'prediction': [int(prediction[0])],
                    'proba_full_pay': [prediction_proba]
                })

                if not os.path.exists("predictions.csv"):
                    save_data.to_csv("predictions.csv", index=False)
                else:
                    save_data.to_csv("predictions.csv", mode='a', header=False, index=False)

                st.success("📁 Prediction saved to predictions.csv ✅")

                if os.path.exists("predictions.csv"):
                    with open("predictions.csv", "rb") as file:
                        st.download_button(
                            label="📥 Download predictions.csv",
                            data=file,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )

            st.session_state.predict_clicked = False
