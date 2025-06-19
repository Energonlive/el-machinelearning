import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib

st.title('🪙 Loan Risk Prediction Machine Learning App')
tab1, tab2, tab3 = st.tabs(["Predict Loan Risk", "Bulk Predict", "Model Information"])

with tab1: 
    st.header('Predict Loan Risk for a Single Applicant')

    st.subheader("Loan Information")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        loan_amount_input = st.text_input("Enter **Loan Amount** ($) (e.g., 10000 or 10,000)", key="loan_amount_input")
        loan_amount = None
        try:
            loan_amount = float(loan_amount_input.strip().replace(',', ''))
            if loan_amount > 0:
                st.success(f"Loan Amount: ${loan_amount:,.2f}")
            else:
                st.warning("Loan amount must be greater than 0.")
        except ValueError:
            if loan_amount_input:
                st.error("Please enter a valid number.")

        selected_term = st.radio("Select **Term** (months)", [36, 60], horizontal=True, key="selected_term")
        st.write(f"You selected term: {selected_term} months")

    with col2:
        int_rate_input = st.text_input("Enter **Interest Rate** (%) (per annum, e.g., 10.5)", key="int_rate_input")
        int_rate = None
        try:
            int_rate = float(int_rate_input.strip())
            if 0.0 <= int_rate <= 100.0:
                st.success(f"Interest Rate: {int_rate:.2f}%")
            else:
                st.warning("Interest rate must be between 0 and 100.")
        except ValueError:
            if int_rate_input:
                st.error("Please enter a valid number.")

        subgrades = [f"{grade}{i}" for grade in "ABCDEFG" for i in range(1, 6)]
        selected_subgrade = st.selectbox("Select **Subgrade**", subgrades, key="selected_subgrade")
        st.write(f"You selected subgrade: {selected_subgrade}")

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
            st.error("Installment calculation failed. Please check inputs.")

    st.text_input("Calculated **Monthly Installment** ($)", value=f"{installment_display}" if 'installment' in locals() else "", disabled=True, key="calculated_installment")

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

    st.subheader("Personal Information")
    st.markdown("---")

    col3, col4 = st.columns(2)
    with col3:
        annual_inc_input = st.text_input("Enter **Annual Income** ($) (e.g., 10000 or 10,000)", key="annual_inc_input")
        annual_inc = None
        try:
            annual_inc = float(annual_inc_input.strip().replace(',', ''))
            if annual_inc >= 0:
                st.success(f"Annual Income: ${annual_inc:,.2f}")
            else:
                st.warning("Please enter a non-negative value.")
        except ValueError:
            if annual_inc_input:
                st.error("Please enter a valid number.")

        selected_home_ownership_status = st.selectbox("Select **Home Ownership Status**", ["Mortgage", "Own", "Rent", "Other"], key="home_ownership_status")
        st.write(f"You selected: {selected_home_ownership_status}")

    with col4:
        dti_input = st.text_input("Enter **DTI** (Debt-to-Income ratio) (%)", key="dti_input")
        dti = None
        try:
            dti = float(dti_input.strip())
            if 0.0 <= dti <= 100.0:
                st.success(f"DTI: {dti}%")
            else:
                st.warning("Please enter a value between 0 and 100.")
        except ValueError:
            if dti_input:
                st.error("Please enter a valid number.")

        selected_verification_status = st.selectbox("Select **Verification Status**", ["Source Verified", "Verified", "Not Verified"], key="verification_status")
        st.write(f"You selected: {selected_verification_status}")

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
    selected_zipcode = st.selectbox("Select **Zipcode**",
                                    list(zipcode_locations.keys()),
                                    format_func=lambda x: zipcode_locations[x], key="selected_zipcode")
    st.write(f"You selected zipcode: {selected_zipcode}")


    st.subheader("Credit Information")
    st.markdown("---")

    col5, col6 = st.columns(2)
    with col5:
        revol_util_input = st.text_input("Enter **Revolving Line Utilization** (%)", key="revol_util_input")
        revol_util = None
        try:
            revol_util = float(revol_util_input.strip())
            if 0.0 <= revol_util <= 100.0:
                st.success(f"Revolving Utilization: {revol_util}%")
            else:
                st.warning("Please enter a value between 0 and 100.")
        except ValueError:
            if revol_util_input:
                st.error("Please enter a valid number.")

        open_acc_input = st.text_input("Enter **Number of Open Credit Lines**", key="open_acc_input")
        open_acc = None
        try:
            open_acc = int(open_acc_input.strip())
            if open_acc >= 0:
                st.success(f"Open Credit Lines: {open_acc}")
            else:
                st.warning("Number of open accounts cannot be negative.")
        except ValueError:
            if open_acc_input:
                st.error("Please enter a valid whole number.")

        mort_acc_input = st.text_input("Enter **Number of Mortgage Accounts**", key="mort_acc_input")
        mort_acc = None
        try:
            mort_acc = int(mort_acc_input.strip())
            if mort_acc >= 0:
                st.success(f"Mortgage Accounts: {mort_acc}")
            else:
                st.warning("Mortgage accounts cannot be negative.")
        except ValueError:
            if mort_acc_input:
                st.error("Please enter a valid whole number.")

    with col6:
        revol_bal_input = st.text_input("Enter **Revolving Balance** ($) (e.g., 10000 or 10,000)", key="revol_bal_input")
        revol_bal = None
        try:
            revol_bal = float(revol_bal_input.strip().replace(',', ''))
            if revol_bal >= 0:
                st.success(f"Revolving Balance: ${revol_bal:,.2f}")
            else:
                st.warning("Revolving balance cannot be negative.")
        except ValueError:
            if revol_bal_input:
                st.error("Please enter a valid number.")

        total_acc_input = st.text_input("Enter **Total Number of Credit Accounts**", key="total_acc_input")
        total_acc = None
        try:
            total_acc = int(total_acc_input.strip())
            if total_acc >= 0:
                st.success(f"Total Credit Accounts: {total_acc}")
            else:
                st.warning("Total accounts cannot be negative.")
        except ValueError:
            if total_acc_input:
                st.error("Please enter a valid whole number.")

        cred_hist_input = st.text_input("Enter **Credit History Length** (in years)", key="cred_hist_input")
        cred_hist_years = None
        try:
            cred_hist_years = float(cred_hist_input.strip())
            if cred_hist_years >= 0:
                st.success(f"Credit History Length: {cred_hist_years} years")
            else:
                st.warning("Credit history length cannot be negative.")
        except ValueError:
            if cred_hist_input:
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

                loaded_model = joblib.load('xgb_final_pipeline_no_undersample.joblib')
                prediction = loaded_model.predict(input_data)
                prediction_proba = loaded_model.predict_proba(input_data)[0][1]

                if prediction[0] == 1:
                    st.subheader("Prediction Result")
                    st.success("✅ Prediction completed successfully!")
                    st.toast("Prediction complete!")
                    st.markdown("🟢 **The loan is likely to be fully paid.**")
                else:
                    st.subheader("Prediction Result")
                    st.success("✅ Prediction completed successfully!")
                    st.toast("Prediction complete!")
                    st.markdown("🔴 **The loan is likely to default.**")

                st.markdown(f"**🧮 Probability of Full Payment:** {prediction_proba:.2%}")

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

with tab2:

    try:
        st.header('📊 Bulk Predict Loan Risk from CSV File')
        st.code("""zip_code, subgrade, annual_inc, term, dti, revol_util, revol_bal,
                    open_acc, loan_amnt, int_rate, installment, home_ownership, 
                    cred_hist_years, verification_status, total_acc, mort_acc""",
                    language='python'
                )
        
        uploaded_file = st.file_uploader("Upload CSV file with the above columns", type=["csv"])

        if uploaded_file:
            df = pd.read_csv(uploaded_file)

            subgrades = [f"{grade}{i}" for grade in "ABCDEFG" for i in range(1, 6)]
            df['subgrade'] = df['subgrade'].map({sg: i for i, sg in enumerate(subgrades)})

            df['zip_code'] = df['zip_code'].astype(str)
            zip_cols = [
                'zip_code_05113', 'zip_code_11650', 'zip_code_22690',
                'zip_code_29597', 'zip_code_30723', 'zip_code_48052',
                'zip_code_70466', 'zip_code_86630', 'zip_code_93700'
            ]
            for z in zip_cols:
                suffix = z.split('_')[-1]
                df[z] = (df['zip_code'] == suffix).astype(int)

            df['annual_inc'] = np.log1p(df['annual_inc'])
            df['revol_bal'] = np.log1p(df['revol_bal'])

            df['home_ownership'] = df['home_ownership'].str.upper()
            for h in ['OTHER', 'OWN', 'RENT']:
                df[f'home_ownership_{h}'] = (df['home_ownership'] == h).astype(int)

            df['verification_status'] = df['verification_status'].str.replace(" ", "_")
            for v in ['Source_Verified', 'Verified']:
                df[f'verification_status_{v}'] = (df['verification_status'] == v).astype(int)

            final_features = [
                'zip_code_05113', 'zip_code_11650', 'zip_code_22690',
                'zip_code_29597', 'zip_code_30723', 'zip_code_48052',
                'zip_code_70466', 'zip_code_86630', 'zip_code_93700',
                'subgrade', 'annual_inc', 'term', 'dti', 'revol_util',
                'revol_bal', 'open_acc', 'installment', 'loan_amnt', 'int_rate',
                'home_ownership_OTHER', 'home_ownership_OWN', 'home_ownership_RENT',
                'cred_hist_years',
                'verification_status_Source_Verified', 'verification_status_Verified',
                'total_acc', 'mort_acc'
            ]

            input_data = df[final_features]

            model = joblib.load('xgb_final_pipeline_no_undersample.joblib')
            predictions = model.predict(input_data)
            prediction_probas = model.predict_proba(input_data)[:, 1]

            st.success("✅ Bulk predictions completed.")
            st.dataframe(df.head())

            csv_out = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Prediction Results",
                data=csv_out,
                file_name="bulk_predictions.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Error processing file: {e}")


with tab2:
    st.header('📊 Bulk Predict Loan Risk from CSV File')
    st.code("""zip_code, subgrade, annual_inc, term, dti, revol_util, revol_bal,
                open_acc, loan_amnt, int_rate, installment, home_ownership, 
                cred_hist_years, verification_status, total_acc, mort_acc""",
                language='python'
            )

    uploaded_file = st.file_uploader("Upload CSV file with the above columns", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            subgrades = [f"{grade}{i}" for grade in "ABCDEFG" for i in range(1, 6)]
            df['subgrade'] = df['subgrade'].map({sg: i for i, sg in enumerate(subgrades)})

            df['zip_code'] = df['zip_code'].astype(str)
            zip_cols = [
                'zip_code_05113', 'zip_code_11650', 'zip_code_22690',
                'zip_code_29597', 'zip_code_30723', 'zip_code_48052',
                'zip_code_70466', 'zip_code_86630', 'zip_code_93700'
            ]
            for z in zip_cols:
                suffix = z.split('_')[-1]
                df[z] = (df['zip_code'] == suffix).astype(int)

            df['annual_inc'] = np.log1p(df['annual_inc'])
            df['revol_bal'] = np.log1p(df['revol_bal'])

            df['home_ownership'] = df['home_ownership'].str.upper()
            for h in ['OTHER', 'OWN', 'RENT']:
                df[f'home_ownership_{h}'] = (df['home_ownership'] == h).astype(int)

            df['verification_status'] = df['verification_status'].str.replace(" ", "_", regex=False)
            for v in ['Source_Verified', 'Verified']:
                df[f'verification_status_{v}'] = (df['verification_status'] == v).astype(int)

            final_features = [
                'zip_code_05113', 'zip_code_11650', 'zip_code_22690',
                'zip_code_29597', 'zip_code_30723', 'zip_code_48052',
                'zip_code_70466', 'zip_code_86630', 'zip_code_93700',
                'subgrade', 'annual_inc', 'term', 'dti', 'revol_util',
                'revol_bal', 'open_acc', 'installment', 'loan_amnt', 'int_rate',
                'home_ownership_OTHER', 'home_ownership_OWN', 'home_ownership_RENT',
                'cred_hist_years',
                'verification_status_Source_Verified', 'verification_status_Verified',
                'total_acc', 'mort_acc'
            ]

            input_data = df[final_features]

            model = joblib.load('xgb_final_pipeline_no_undersample.joblib')
            df["prediction"] = model.predict(input_data)
            df["proba_full_pay"] = model.predict_proba(input_data)[:, 1]

            st.success("✅ Bulk predictions completed.")
            st.dataframe(df.head())

            csv_out = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Prediction Results",
                data=csv_out,
                file_name="bulk_predictions.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error processing file: {e}")

