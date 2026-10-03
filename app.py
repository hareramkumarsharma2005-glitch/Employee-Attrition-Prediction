import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👨‍💼",
    layout="wide"
)

st.title("👨‍💼 Employee Attrition Prediction")
st.write(
    "Enter employee information to predict the probability "
    "of employee attrition."
)

# -----------------------------
# Load dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv(
        "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    )

df = load_data()

# -----------------------------
# Prepare model
# -----------------------------
@st.cache_resource
def train_model(df):

    data = df.copy()

    # Remove columns that are not useful for prediction
    data = data.drop(
        columns=[
            "EmployeeNumber",
            "EmployeeCount",
            "Over18",
            "StandardHours"
        ],
        errors="ignore"
    )

    # Convert target
    data["Attrition"] = data["Attrition"].map({
        "Yes": 1,
        "No": 0
    })

    X = data.drop("Attrition", axis=1)
    y = data["Attrition"]

    # Convert categorical variables
    X = pd.get_dummies(X, drop_first=True)

    # Numerical columns
    numerical_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns

    scaler = StandardScaler()

    X[numerical_columns] = scaler.fit_transform(
        X[numerical_columns]
    )

    # Gradient Boosting model
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )

    model.fit(X, y)

    return model, X.columns, scaler, numerical_columns


model, model_columns, scaler, numerical_columns = train_model(df)

# -----------------------------
# Employee inputs
# -----------------------------

st.header("Employee Information")

col1, col2, col3 = st.columns(3)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=70,
        value=30
    )

    business_travel = st.selectbox(
        "Business Travel",
        sorted(df["BusinessTravel"].unique())
    )

    daily_rate = st.number_input(
        "Daily Rate",
        min_value=0,
        value=800
    )

    department = st.selectbox(
        "Department",
        sorted(df["Department"].unique())
    )

    distance_from_home = st.number_input(
        "Distance From Home",
        min_value=0,
        value=5
    )

    education = st.slider(
        "Education",
        1, 5, 3
    )

    education_field = st.selectbox(
        "Education Field",
        sorted(df["EducationField"].unique())
    )

    environment_satisfaction = st.slider(
        "Environment Satisfaction",
        1, 4, 3
    )

    gender = st.selectbox(
        "Gender",
        sorted(df["Gender"].unique())
    )

with col2:

    hourly_rate = st.number_input(
        "Hourly Rate",
        min_value=0,
        value=65
    )

    job_involvement = st.slider(
        "Job Involvement",
        1, 4, 3
    )

    job_level = st.slider(
        "Job Level",
        1, 5, 2
    )

    job_role = st.selectbox(
        "Job Role",
        sorted(df["JobRole"].unique())
    )

    job_satisfaction = st.slider(
        "Job Satisfaction",
        1, 4, 3
    )

    marital_status = st.selectbox(
        "Marital Status",
        sorted(df["MaritalStatus"].unique())
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=0,
        value=5000
    )

    monthly_rate = st.number_input(
        "Monthly Rate",
        min_value=0,
        value=15000
    )

    num_companies_worked = st.number_input(
        "Number of Companies Worked",
        min_value=0,
        value=2
    )

with col3:

    overtime = st.selectbox(
        "Overtime",
        sorted(df["OverTime"].unique())
    )

    percent_salary_hike = st.slider(
        "Percent Salary Hike",
        0, 30, 15
    )

    performance_rating = st.slider(
        "Performance Rating",
        1, 5, 3
    )

    relationship_satisfaction = st.slider(
        "Relationship Satisfaction",
        1, 4, 3
    )

    stock_option_level = st.slider(
        "Stock Option Level",
        0, 3, 1
    )

    total_working_years = st.number_input(
        "Total Working Years",
        min_value=0,
        value=8
    )

    training_times = st.number_input(
        "Training Times Last Year",
        min_value=0,
        value=3
    )

    work_life_balance = st.slider(
        "Work Life Balance",
        1, 4, 3
    )

    years_at_company = st.number_input(
        "Years At Company",
        min_value=0,
        value=5
    )

    years_current_role = st.number_input(
        "Years In Current Role",
        min_value=0,
        value=3
    )

    years_promotion = st.number_input(
        "Years Since Last Promotion",
        min_value=0,
        value=1
    )

    years_manager = st.number_input(
        "Years With Current Manager",
        min_value=0,
        value=3
    )

# -----------------------------
# Prediction
# -----------------------------

st.divider()

if st.button(
    "🔮 Predict Employee Attrition",
    use_container_width=True
):

    input_data = pd.DataFrame({

        "Age": [age],
        "BusinessTravel": [business_travel],
        "DailyRate": [daily_rate],
        "Department": [department],
        "DistanceFromHome": [distance_from_home],
        "Education": [education],
        "EducationField": [education_field],
        "EnvironmentSatisfaction": [environment_satisfaction],
        "Gender": [gender],
        "HourlyRate": [hourly_rate],
        "JobInvolvement": [job_involvement],
        "JobLevel": [job_level],
        "JobRole": [job_role],
        "JobSatisfaction": [job_satisfaction],
        "MaritalStatus": [marital_status],
        "MonthlyIncome": [monthly_income],
        "MonthlyRate": [monthly_rate],
        "NumCompaniesWorked": [num_companies_worked],
        "OverTime": [overtime],
        "PercentSalaryHike": [percent_salary_hike],
        "PerformanceRating": [performance_rating],
        "RelationshipSatisfaction": [relationship_satisfaction],
        "StockOptionLevel": [stock_option_level],
        "TotalWorkingYears": [total_working_years],
        "TrainingTimesLastYear": [training_times],
        "WorkLifeBalance": [work_life_balance],
        "YearsAtCompany": [years_at_company],
        "YearsInCurrentRole": [years_current_role],
        "YearsSinceLastPromotion": [years_promotion],
        "YearsWithCurrManager": [years_manager]
    })

    # Encode categorical values
    input_data = pd.get_dummies(
        input_data,
        drop_first=True
    )

    # Make input columns identical to training columns
    input_data = input_data.reindex(
        columns=model_columns,
        fill_value=0
    )

    # Scale numerical values
    input_data[numerical_columns] = scaler.transform(
        input_data[numerical_columns]
    )

    # Prediction
    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(
        input_data
    )[0][1]

    # -----------------------------
    # Display result
    # -----------------------------

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error(
            "⚠️ High Risk of Employee Attrition"
        )

    else:

        st.success(
            "✅ Low Risk of Employee Attrition"
        )

    st.metric(
        "Probability of Attrition",
        f"{probability * 100:.2f}%"
    )

    st.progress(float(probability))