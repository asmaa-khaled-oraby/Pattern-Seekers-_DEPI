import streamlit as st
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# --- Load Model and Scaler ---
@st.cache_resource
def load_model():
    try:
        model = XGBClassifier()
        model.load_model('xgb_hr_model.json')
        return model
    except Exception as e:
        st.sidebar.error(f"Error loading model (xgb_hr_model.json): {e}")
        st.stop()

@st.cache_resource
def load_scaler():
    scaler_file_pkl = 'scaler.pkl'
    scaler_file_joblib = 'scaler.joblib'
    scaler = None
    try:
        scaler = joblib.load(scaler_file_joblib)
    except:
        try:
            with open(scaler_file_pkl, 'rb') as f:
                scaler = pickle.load(f)
        except:
            st.sidebar.error("Scaler file not found.")
            st.stop()
    return scaler

model = load_model()
scaler = load_scaler()

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Predict Attrition", "About", "How to Use", "Additional Info"])
st.sidebar.markdown("---")
st.sidebar.markdown("<p style='text-align: center; color: gray;'>HR Attrition Prediction App | Powered by Machine Learning</p>", unsafe_allow_html=True)

# --- Page Content ---
if page == "Predict Attrition":
    st.title("🔮 HR Attrition Prediction App")
    st.write("Upload employee data (CSV) to predict if they will leave or stay.")
    st.markdown("---")

    uploaded_file = st.file_uploader("📂 Upload Employee CSV File", type=["csv"])

    if uploaded_file is not None and model is not None and scaler is not None:
        try:
            data = pd.read_csv(uploaded_file)
            data_orig = data.copy()
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

        expected_cols = [
            'Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome', 'Education',
            'EducationField', 'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
            'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome',
            'MonthlyRate', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike',
            'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
            'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
            'YearsSinceLastPromotion', 'YearsWithCurrManager'
        ]

        # Add missing columns if needed
        for col in expected_cols:
            if col not in data.columns:
                data[col] = 0

        # Mapping for categorical fields
        categorical_mappings = {
            'BusinessTravel': {'Non-Travel': 2, 'Travel_Rarely': 0, 'Travel_Frequently': 1},
            'Department': {'Sales': 0, 'Research & Development': 1, 'Human Resources': 2},
            'EducationField': {'Life Sciences': 0, 'Medical': 1, 'Marketing': 2, 'Technical Degree': 3, 'Human Resources': 4, 'Other': 5},
            'OverTime': {'Yes': 1, 'No': 0},
            'Gender': {'Male': 1, 'Female': 0},
            'JobRole': {'Sales Executive': 0, 'Research Scientist': 1, 'Laboratory Technician': 2, 'Manufacturing Director': 3,
                        'Healthcare Representative': 4, 'Manager': 5, 'Sales Representative': 6, 'Research Director': 7, 'Human Resources': 8},
            'MaritalStatus': {'Single': 0, 'Married': 1, 'Divorced': 2}
        }

        for col, mapping in categorical_mappings.items():
            if col in data.columns:
                data[col] = data[col].map(mapping).fillna(-1)

        data = data[expected_cols].fillna(0)
        data_scaled = scaler.transform(data)

        predictions = model.predict(data_scaled)
        probabilities = model.predict_proba(data_scaled)[:, 1]

        result_df = pd.DataFrame({
            "Employee #": range(1, len(predictions)+1),
            "Prediction": ["🔴 Will Leave" if p == 1 else "🟢 Will Stay" for p in predictions],
            "Probability (Leave)": [f"{prob:.0%}" for prob in probabilities], # Display as percentage
            "Risk Level": ["🔴 High" if prob > 0.7 else "🟢 Low" for prob in probabilities]
        })

        st.success("✅ Predictions Generated:")
        st.dataframe(result_df)

        csv_output = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Prediction Table", data=csv_output, file_name="employee_attrition_summary.csv", mime="text/csv")
    else:
        st.info("⬆️ Please upload a CSV file containing employee data to get predictions.")

elif page == "About":
    st.title("ℹ️ About the HR Attrition Prediction App")
    st.markdown("""
    This application leverages a machine learning model to forecast employee attrition.
    By analyzing relevant employee data, the model identifies patterns and predicts the likelihood
    of an employee leaving the organization.

    **Key Features:**
    - Predicts the probability of employee attrition.
    - Categorizes attrition risk levels (High/Low).
    - Provides a downloadable summary of predictions.

    This tool aims to empower HR professionals and organizational leaders to proactively
    address potential turnover and implement effective retention strategies.
    """)
    st.markdown("---")
    st.subheader("Our Commitment")
    st.markdown("We are committed to providing accurate and insightful predictions to help you build a stronger, more stable workforce.")
    # You can add more visually appealing elements here, like images or styled text

elif page == "How to Use":
    st.title("📝 How to Use the HR Attrition Prediction App")
    st.markdown("""
    Follow these simple steps to predict employee attrition:

    1.  **Upload Your Data:** Click on the "Browse files" button under "Upload Employee CSV File"
        in the main section of the app. Select the CSV file containing your employee data.
        Ensure your file includes the necessary columns for accurate prediction.

    2.  **Wait for Processing:** Once the file is uploaded, the application will automatically
        process the data and generate the attrition predictions. This may take a few moments
        depending on the size of your dataset.

    3.  **View Predictions:** The prediction results will be displayed in a table below the upload button.
        The table includes:
        - **Employee #:** A sequential identifier for each employee.
        - **Prediction:** Indicates whether the employee is predicted to "Will Leave" (🔴) or "Will Stay" (🟢).
        - **Probability (Leave):** The likelihood of the employee leaving, expressed as a percentage (e.g., 85%).
        - **Risk Level:** Categorizes the attrition risk as "High" (🔴, probability > 70%) or "Low" (🟢).

    4.  **Download Results:** To save the prediction summary for further analysis, click the
        "📥 Download Prediction Table" button. The results will be downloaded as a CSV file.

    **Important Notes:**
    - Ensure your CSV file adheres to the expected format and includes the required columns.
    - The accuracy of the predictions depends on the quality and relevance of the data provided.
    """)
    # Add more visual elements or formatting here

elif page == "Additional Info":
    st.title("📌 Additional Information")
    st.markdown("""
    **Understanding the Predictions:**

    - **Probability (Leave):** This percentage represents the model's confidence that an employee will leave the organization. A higher percentage indicates a greater likelihood of attrition.

    - **Risk Level:** This is a simplified categorization of the probability:
        - **🔴 High:** Employees with a probability of leaving greater than 70% are considered high-risk.
        - **🟢 Low:** Employees with a probability of leaving 70% or lower are considered low-risk.

    **Using the Insights:**

    The predictions generated by this app can be valuable for:
    - **Proactive Retention Efforts:** Identifying high-risk employees allows HR to implement targeted retention strategies.
    - **Resource Planning:** Understanding potential turnover can help in workforce planning and recruitment.
    - **Analyzing Contributing Factors:** While this app provides predictions, further analysis of the data can help identify factors driving attrition within your organization.

    **Disclaimer:**

    The predictions are based on patterns learned from historical data. While the model strives for accuracy, predictions are not guarantees. Use these insights as a tool to inform your HR decisions, in conjunction with your professional judgment and other relevant information.
    """)
    # Add more visual enhancements to this page