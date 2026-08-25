import os
from io import BytesIO
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Dropout Early Warning System",
    page_icon="🎓",
    layout="wide",
)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_PATH = "models/final_dropout_model.pkl"
FEATURE_NAMES_PATH = "models/feature_names.pkl"
DEFAULT_VALUES_PATH = "models/default_values.pkl"

DATA_PATH = "data/student_data.csv"
HISTORY_PATH = "data/prediction_history.csv"


# ============================================================
# PROFESSIONAL UI STYLING
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #F8FAFC;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    [data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #E2E8F0;
        padding: 15px;
        border-radius: 10px;
    }

    .stButton > button {
        border-radius: 8px;
        height: 45px;
        font-weight: 600;
    }

    .stDownloadButton > button {
        border-radius: 8px;
        height: 45px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

def check_required_files():

    required_files = [
        MODEL_PATH,
        FEATURE_NAMES_PATH,
        DEFAULT_VALUES_PATH,
        DATA_PATH,
    ]

    missing_files = [
        file_path
        for file_path in required_files
        if not os.path.exists(file_path)
    ]

    if missing_files:

        st.error("❌ Required project files are missing.")

        st.write("Missing files:")

        for file_path in missing_files:
            st.write(f"- `{file_path}`")

        st.stop()


check_required_files()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)

    feature_names = joblib.load(FEATURE_NAMES_PATH)

    default_values = joblib.load(DEFAULT_VALUES_PATH)

    return model, feature_names, default_values


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv(DATA_PATH)


try:

    model, feature_names, default_values = load_model()

    data = load_data()

except Exception as e:

    st.error(
        "❌ Unable to load the machine learning model or dataset."
    )

    st.exception(e)

    st.stop()


# ============================================================
# VALIDATE MODEL RESOURCES
# ============================================================

if not isinstance(feature_names, list):

    feature_names = list(feature_names)


if not isinstance(default_values, dict):

    try:

        default_values = dict(default_values)

    except Exception:

        st.error(
            "❌ default_values.pkl does not contain a valid dictionary."
        )

        st.stop()


missing_default_features = [
    feature
    for feature in feature_names
    if feature not in default_values
]

if missing_default_features:

    st.error(
        "❌ Some model features are missing from default_values.pkl."
    )

    st.write(missing_default_features)

    st.stop()


# ============================================================
# AI PREDICTION FUNCTION
# ============================================================

def predict_risk(
    age,
    gender,
    marital_status,
    grade_1,
    grade_2,
    approved,
    tuition,
    scholarship,
    debtor,
):

    input_data = default_values.copy()

    if "Age at enrollment" in input_data:

        input_data["Age at enrollment"] = age

    if "Curricular units 1st sem (grade)" in input_data:

        input_data[
            "Curricular units 1st sem (grade)"
        ] = grade_1

    if "Curricular units 2nd sem (grade)" in input_data:

        input_data[
            "Curricular units 2nd sem (grade)"
        ] = grade_2

    if "Curricular units 1st sem (approved)" in input_data:

        input_data[
            "Curricular units 1st sem (approved)"
        ] = approved

    if "Tuition fees up to date" in input_data:

        input_data[
            "Tuition fees up to date"
        ] = 1 if tuition == "Yes" else 0

    if "Scholarship holder" in input_data:

        input_data[
            "Scholarship holder"
        ] = 1 if scholarship == "Yes" else 0

    if "Debtor" in input_data:

        input_data[
            "Debtor"
        ] = 1 if debtor == "Yes" else 0

    if "Gender" in input_data:

        input_data["Gender"] = 1 if gender == "Male" else 0

    if "Marital status" in input_data:

        input_data[
            "Marital status"
        ] = 1 if marital_status == "Married" else 0

    input_df = pd.DataFrame(
        [input_data],
        columns=feature_names,
    )

    try:

        probability = model.predict_proba(
            input_df
        )[0][1]

    except Exception as e:

        raise RuntimeError(
            f"Model prediction failed: {e}"
        )

    return float(probability)


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def classify_risk(probability):

    if probability < 0.30:

        return (
            "LOW",
            "Routine Monitoring",
        )

    elif probability < 0.60:

        return (
            "MEDIUM",
            "Additional Academic Support",
        )

    else:

        return (
            "HIGH",
            "Immediate Attention Required",
        )


# ============================================================
# AI PERSONALIZED STUDENT ANALYSIS
# ============================================================

def generate_ai_analysis(
    student_name,
    risk_level,
    risk_percentage,
    risk_factors,
    grade_1,
    grade_2,
    approved,
    tuition,
    scholarship,
    debtor,
):

    analysis = []

    name = (
        student_name.strip()
        if student_name
        else "The student"
    )

    if risk_level == "LOW":

        analysis.append(
            f"{name} is currently showing a low probability "
            "of academic dropout."
        )

        analysis.append(
            "The student's overall academic and financial "
            "indicators appear stable."
        )

    elif risk_level == "MEDIUM":

        analysis.append(
            f"{name} shows a moderate probability "
            "of academic dropout."
        )

        analysis.append(
            "Early intervention and regular monitoring "
            "are recommended."
        )

    else:

        analysis.append(
            f"{name} shows a high probability "
            "of academic dropout."
        )

        analysis.append(
            "Immediate academic intervention and close "
            "monitoring are strongly recommended."
        )

    if grade_1 < 10:

        analysis.append(
            f"Low 1st semester performance detected "
            f"with a grade of {grade_1}."
        )

    if grade_2 < 10:

        analysis.append(
            f"Low 2nd semester performance detected "
            f"with a grade of {grade_2}."
        )

    if approved < 5:

        analysis.append(
            f"Only {approved} approved subjects were recorded, "
            "which may indicate academic difficulty."
        )

    if tuition == "No":

        analysis.append(
            "Tuition fees are not up to date, which may "
            "create financial pressure."
        )

    if debtor == "Yes":

        analysis.append(
            "Outstanding student debt has been identified."
        )

    if scholarship == "Yes":

        analysis.append(
            "The student has scholarship support, which "
            "may help reduce financial pressure."
        )

    if not risk_factors:

        analysis.append(
            "No major manually identified risk factors "
            "were detected."
        )

    analysis.append(
        f"Overall predicted dropout risk: "
        f"{risk_percentage:.2f}% ({risk_level})."
    )

    return analysis


# ============================================================
# SMART INTERVENTION PLAN
# ============================================================

def generate_intervention_plan(
    risk_level,
    risk_factors,
):

    plan = {}

    if risk_level == "HIGH":

        plan["Week 1"] = [
            "Schedule an immediate meeting with the student",
            "Assign a faculty mentor",
            "Identify weak subjects and academic difficulties",
        ]

        plan["Week 2"] = [
            "Provide academic counseling",
            "Create a subject improvement strategy",
            "Check for financial or personal difficulties",
        ]

        plan["Week 3"] = [
            "Conduct a progress review",
            "Provide additional support for weak subjects",
            "Meet the faculty mentor for feedback",
        ]

        plan["Week 4"] = [
            "Evaluate academic improvement",
            "Review completed intervention actions",
            "Perform a new dropout risk assessment",
        ]

    elif risk_level == "MEDIUM":

        plan["Week 1"] = [
            "Assign an academic mentor",
            "Discuss academic challenges with the student",
        ]

        plan["Week 2"] = [
            "Monitor grades and subject performance",
            "Provide additional learning resources",
        ]

        plan["Week 3"] = [
            "Review academic progress",
            "Identify any new risk factors",
        ]

        plan["Week 4"] = [
            "Conduct a follow-up performance review",
            "Reassess student dropout risk",
        ]

    else:

        plan["Week 1"] = [
            "Continue routine academic monitoring",
        ]

        plan["Week 2"] = [
            "Encourage consistent academic performance",
        ]

        plan["Week 3"] = [
            "Review attendance and subject performance",
        ]

        plan["Week 4"] = [
            "Conduct a routine follow-up",
        ]

    if "Outstanding financial debt" in risk_factors:

        plan["Week 1"].append(
            "Discuss available financial support options"
        )

    if "Tuition fees are not up to date" in risk_factors:

        plan["Week 2"].append(
            "Contact the student regarding tuition support options"
        )

    if "Low 1st semester performance" in risk_factors:

        plan["Week 1"].append(
            "Provide additional support for 1st semester subjects"
        )

    if "Low 2nd semester performance" in risk_factors:

        plan["Week 2"].append(
            "Create an improvement plan for 2nd semester subjects"
        )

    return plan


# ============================================================
# SAVE PREDICTION HISTORY
# ============================================================

def save_prediction_history(
    student_name,
    student_id,
    department,
    semester,
    risk_percentage,
    risk_level,
    priority,
):

    os.makedirs(
        "data",
        exist_ok=True,
    )

    new_prediction = pd.DataFrame(
        [
            {
                "Prediction Date":
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                "Student Name":
                    student_name,

                "Student ID":
                    student_id,

                "Department":
                    department,

                "Semester":
                    semester,

                "Dropout Risk (%)":
                    round(
                        risk_percentage,
                        2,
                    ),

                "Risk Level":
                    risk_level,

                "Priority":
                    priority,
            }
        ]
    )

    try:

        history = pd.read_csv(
            HISTORY_PATH
        )

        if history.empty:

            history = new_prediction

        else:

            history = pd.concat(
                [
                    history,
                    new_prediction,
                ],
                ignore_index=True,
            )

    except (
        FileNotFoundError,
        pd.errors.EmptyDataError,
    ):

        history = new_prediction

    history.to_csv(
        HISTORY_PATH,
        index=False,
    )


# ============================================================
# CREATE PDF REPORT
# ============================================================

def create_pdf_report(
    student_name,
    student_id,
    department,
    semester,
    age,
    gender,
    marital_status,
    grade_1,
    grade_2,
    approved,
    tuition,
    scholarship,
    debtor,
    risk_percentage,
    risk_level,
    priority,
    risk_factors,
    recommendations,
):

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=18,
        spaceAfter=20,
    )

    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=13,
        spaceBefore=12,
        spaceAfter=8,
    )

    normal_style = styles["Normal"]

    story = []

    story.append(
        Paragraph(
            "STUDENT DROPOUT EARLY WARNING REPORT",
            title_style,
        )
    )

    story.append(
        Spacer(1, 15)
    )

    # STUDENT INFORMATION

    story.append(
        Paragraph(
            "Student Information",
            heading_style,
        )
    )

    student_data = [

        ["Student Name", student_name],
        ["Student ID", student_id],
        ["Department", department],
        ["Semester", str(semester)],
        ["Age", str(age)],
        ["Gender", gender],
        ["Marital Status", marital_status],

    ]

    student_table = Table(
        student_data,
        colWidths=[
            2.2 * inch,
            4 * inch,
        ],
    )

    student_table.setStyle(
        TableStyle(
            [

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),

                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.lightgrey,
                ),

                (
                    "FONTNAME",
                    (0, 0),
                    (0, -1),
                    "Helvetica-Bold",
                ),

                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    8,
                ),

            ]
        )
    )

    story.append(student_table)

    story.append(
        Spacer(1, 20)
    )

    # PREDICTION RESULT

    story.append(
        Paragraph(
            "AI Prediction Result",
            heading_style,
        )
    )

    prediction_data = [

        [
            "Dropout Risk",
            f"{risk_percentage:.2f}%",
        ],

        [
            "Risk Level",
            risk_level,
        ],

        [
            "Priority",
            priority,
        ],

    ]

    prediction_table = Table(
        prediction_data,
        colWidths=[
            2.2 * inch,
            4 * inch,
        ],
    )

    prediction_table.setStyle(
        TableStyle(
            [

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),

                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.lightgrey,
                ),

                (
                    "FONTNAME",
                    (0, 0),
                    (0, -1),
                    "Helvetica-Bold",
                ),

                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    8,
                ),

            ]
        )
    )

    story.append(prediction_table)

    story.append(
        Spacer(1, 20)
    )

    # ACADEMIC INFORMATION

    story.append(
        Paragraph(
            "Academic & Financial Information",
            heading_style,
        )
    )

    academic_data = [

        [
            "1st Semester Grade",
            str(grade_1),
        ],

        [
            "2nd Semester Grade",
            str(grade_2),
        ],

        [
            "Approved Subjects",
            str(approved),
        ],

        [
            "Tuition Fees Up To Date",
            tuition,
        ],

        [
            "Scholarship Holder",
            scholarship,
        ],

        [
            "Student Debt",
            debtor,
        ],

    ]

    academic_table = Table(
        academic_data,
        colWidths=[
            2.2 * inch,
            4 * inch,
        ],
    )

    academic_table.setStyle(
        TableStyle(
            [

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),

                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.lightgrey,
                ),

                (
                    "FONTNAME",
                    (0, 0),
                    (0, -1),
                    "Helvetica-Bold",
                ),

                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    8,
                ),

            ]
        )
    )

    story.append(academic_table)

    story.append(
        Spacer(1, 20)
    )

    # RISK FACTORS

    story.append(
        Paragraph(
            "Potential Risk Factors",
            heading_style,
        )
    )

    if risk_factors:

        for factor in risk_factors:

            story.append(
                Paragraph(
                    f"• {factor}",
                    normal_style,
                )
            )

    else:

        story.append(
            Paragraph(
                "No major risk factors detected.",
                normal_style,
            )
        )

    story.append(
        Spacer(1, 15)
    )

    # RECOMMENDATIONS

    story.append(
        Paragraph(
            "Recommended Intervention",
            heading_style,
        )
    )

    for recommendation in recommendations:

        story.append(
            Paragraph(
                f"• {recommendation}",
                normal_style,
            )
        )

    document.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# DASHBOARD HEADER
# ============================================================

st.markdown(
    """
    <div style="
        padding:30px;
        border-radius:15px;
        background:#1E3A8A;
        text-align:center;
        margin-bottom:25px;
    ">

    <h1 style="color:white;">
    🎓 Student Academic Performance & Dropout Risk Predictor
    </h1>

    <p style="color:white;">
    AI-Powered Early Warning System for Identifying
    Students at Risk of Academic Dropout
    </p>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🏠 Dashboard",
        "👤 Single Student",
        "👥 Batch Prediction",
        "📊 Analytics",
        "📜 Prediction History",
    ]
)


# ============================================================
# TAB 1 - DASHBOARD
# ============================================================

with tab1:

    st.header("🏠 Dashboard Overview")

    try:

        if os.path.exists(HISTORY_PATH):

            dashboard_df = pd.read_csv(HISTORY_PATH)

        else:

            dashboard_df = pd.DataFrame()

        if dashboard_df.empty:

            st.info(
                "📭 No student predictions available yet. "
                "Go to the Single Student tab and make your first prediction."
            )

        else:

            total_predictions = len(dashboard_df)

            low_risk = len(
                dashboard_df[
                    dashboard_df["Risk Level"] == "LOW"
                ]
            )

            medium_risk = len(
                dashboard_df[
                    dashboard_df["Risk Level"] == "MEDIUM"
                ]
            )

            high_risk = len(
                dashboard_df[
                    dashboard_df["Risk Level"] == "HIGH"
                ]
            )

            average_risk = dashboard_df[
                "Dropout Risk (%)"
            ].mean()

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:

                st.metric(
                    "👥 Total Students",
                    total_predictions,
                )

            with col2:

                st.metric(
                    "🟢 Low Risk",
                    low_risk,
                )

            with col3:

                st.metric(
                    "🟡 Medium Risk",
                    medium_risk,
                )

            with col4:

                st.metric(
                    "🔴 High Risk",
                    high_risk,
                )

            with col5:

                st.metric(
                    "📈 Average Risk",
                    f"{average_risk:.2f}%",
                )

            st.divider()

            # RISK DISTRIBUTION

            st.subheader(
                "📊 Student Risk Distribution"
            )

            risk_counts = (
                dashboard_df["Risk Level"]
                .value_counts()
                .reindex(
                    ["LOW", "MEDIUM", "HIGH"],
                    fill_value=0,
                )
            )

            st.bar_chart(risk_counts)

            st.divider()

            # VISUAL RISK INSIGHTS

            st.subheader(
                "📈 Visual Risk Insights"
            )

            col1, col2 = st.columns(2)

            # -------------------------------
            # PIE CHART
            # -------------------------------

            with col1:

                st.markdown(
                    "### 🥧 Risk Distribution"
                )

                chart_data = risk_counts[
                    risk_counts > 0
                ]

                if not chart_data.empty and chart_data.sum() > 0:

                    fig, ax = plt.subplots()

                    ax.pie(
                        chart_data.values,
                        labels=chart_data.index,
                        autopct="%1.1f%%",
                        startangle=90,
                    )

                    ax.set_title(
                        "Student Risk Distribution"
                    )

                    st.pyplot(fig)

                    plt.close(fig)

                else:

                    st.info(
                        "📭 No prediction data available yet."
                    )

            # -------------------------------
            # DEPARTMENT CHART
            # -------------------------------

            with col2:

                st.markdown(
                    "### 🏢 Average Risk by Department"
                )

                if "Department" in dashboard_df.columns:

                    department_risk = (
                        dashboard_df
                        .groupby(
                            "Department"
                        )["Dropout Risk (%)"]
                        .mean()
                        .sort_values(
                            ascending=False
                        )
                    )

                    if not department_risk.empty:

                        st.bar_chart(
                            department_risk
                        )

                    else:

                        st.info(
                            "Department data is not available."
                        )

                else:

                    st.info(
                        "Department data is not available."
                    )

            # RISK BY STUDENT

            st.divider()

            st.subheader(
                "📊 Dropout Risk by Student"
            )

            if (
                "Student Name" in dashboard_df.columns
                and "Dropout Risk (%)" in dashboard_df.columns
            ):

                student_risk = (
                    dashboard_df[
                        [
                            "Student Name",
                            "Dropout Risk (%)",
                        ]
                    ]
                    .sort_values(
                        "Dropout Risk (%)",
                        ascending=False,
                    )
                )

                st.bar_chart(
                    student_risk.set_index(
                        "Student Name"
                    )
                )

            # TOP 5

            st.divider()

            st.subheader(
                "🚨 Top 5 Highest-Risk Students"
            )

            top_5_students = (
                dashboard_df
                .sort_values(
                    "Dropout Risk (%)",
                    ascending=False,
                )
                .head(5)
            )

            display_columns = [
                "Student Name",
                "Student ID",
                "Department",
                "Semester",
                "Dropout Risk (%)",
                "Risk Level",
            ]

            available_columns = [
                column
                for column in display_columns
                if column in top_5_students.columns
            ]

            st.dataframe(
                top_5_students[available_columns],
                use_container_width=True,
            )

            # HIGH RISK

            st.divider()

            st.subheader(
                "🚨 High-Risk Students Requiring Attention"
            )

            high_risk_students = dashboard_df[
                dashboard_df["Risk Level"] == "HIGH"
            ]

            if not high_risk_students.empty:

                st.error(
                    f"⚠️ {len(high_risk_students)} "
                    "high-risk student(s) detected."
                )

                st.dataframe(
                    high_risk_students,
                    use_container_width=True,
                )

            else:

                st.success(
                    "🎉 No high-risk students detected."
                )

            # RECENT PREDICTIONS

            st.divider()

            st.subheader(
                "🕒 Recent Predictions"
            )

            recent_predictions = (
                dashboard_df.tail(10)
            )

            st.dataframe(
                recent_predictions,
                use_container_width=True,
            )

    except Exception as e:

        st.error(
            "❌ Dashboard could not be loaded."
        )

        st.exception(e)


# ============================================================
# TAB 2 - SINGLE STUDENT
# ============================================================

with tab2:

    st.header(
        "👤 Student Information"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        student_name = st.text_input(
            "Student Name"
        )

    with col2:

        student_id = st.text_input(
            "Student ID"
        )

    with col3:

        department = st.text_input(
            "Department"
        )

    with col4:

        semester = st.selectbox(
            "Semester",
            [1, 2, 3, 4, 5, 6, 7, 8],
        )

    st.divider()

    st.header(
        "📚 Academic & Personal Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input(
            "Age at Enrollment",
            min_value=15,
            max_value=100,
            value=20,
        )

    with col2:

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"],
        )

    with col3:

        marital_status = st.selectbox(
            "Marital Status",
            ["Single", "Married"],
        )

    st.divider()

    st.header(
        "📖 Academic Performance"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        grade_1 = st.number_input(
            "1st Semester Grade",
            min_value=0.0,
            max_value=20.0,
            value=12.0,
        )

    with col2:

        grade_2 = st.number_input(
            "2nd Semester Grade",
            min_value=0.0,
            max_value=20.0,
            value=12.0,
        )

    with col3:

        approved = st.number_input(
            "Approved Subjects",
            min_value=0,
            max_value=30,
            value=5,
        )

    st.divider()

    st.header(
        "💰 Financial Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        tuition = st.selectbox(
            "Tuition Fees Up To Date?",
            ["Yes", "No"],
        )

    with col2:

        scholarship = st.selectbox(
            "Scholarship Holder?",
            ["No", "Yes"],
        )

    with col3:

        debtor = st.selectbox(
            "Student Has Debt?",
            ["No", "Yes"],
        )

    st.divider()

    if st.button(
        "🔍 Predict Student Dropout Risk",
        use_container_width=True,
    ):

        if not student_name.strip():

            st.warning(
                "⚠️ Please enter the student's name."
            )

            st.stop()

        if not student_id.strip():

            st.warning(
                "⚠️ Please enter the Student ID."
            )

            st.stop()

        if not department.strip():

            st.warning(
                "⚠️ Please enter the department."
            )

            st.stop()

        try:

            probability = predict_risk(
                age,
                gender,
                marital_status,
                grade_1,
                grade_2,
                approved,
                tuition,
                scholarship,
                debtor,
            )

        except Exception as e:

            st.error(
                "❌ Prediction could not be completed."
            )

            st.exception(e)

            st.stop()

        risk_percentage = probability * 100

        risk_level, priority = classify_risk(
            probability
        )

        if risk_level == "LOW":

            recommendations = [
                "Continue regular academic monitoring",
                "Encourage continued academic performance",
            ]

        elif risk_level == "MEDIUM":

            recommendations = [
                "Assign an academic mentor",
                "Monitor academic performance regularly",
                "Provide counseling if required",
            ]

        else:

            recommendations = [
                "Immediate academic intervention",
                "Assign a faculty mentor",
                "Review low-performing subjects",
                "Assess possible financial difficulties",
            ]

        risk_factors = []

        if grade_1 < 10:

            risk_factors.append(
                "Low 1st semester performance"
            )

        if grade_2 < 10:

            risk_factors.append(
                "Low 2nd semester performance"
            )

        if approved < 5:

            risk_factors.append(
                "Low number of approved subjects"
            )

        if tuition == "No":

            risk_factors.append(
                "Tuition fees are not up to date"
            )

        if debtor == "Yes":

            risk_factors.append(
                "Outstanding financial debt"
            )

        ai_analysis = generate_ai_analysis(
            student_name,
            risk_level,
            risk_percentage,
            risk_factors,
            grade_1,
            grade_2,
            approved,
            tuition,
            scholarship,
            debtor,
        )

        intervention_plan = generate_intervention_plan(
            risk_level,
            risk_factors,
        )

        try:

            save_prediction_history(
                student_name,
                student_id,
                department,
                semester,
                risk_percentage,
                risk_level,
                priority,
            )

        except Exception as e:

            st.warning(
                "⚠️ Prediction completed, but prediction history "
                "could not be saved."
            )

            st.exception(e)

        st.subheader(
            "🤖 AI Prediction Result"
        )

        if risk_level == "LOW":

            st.success(
                f"""
                🟢 LOW RISK

                Dropout Probability:
                {risk_percentage:.2f}%

                Priority:
                {priority}
                """
            )

        elif risk_level == "MEDIUM":

            st.warning(
                f"""
                🟡 MEDIUM RISK

                Dropout Probability:
                {risk_percentage:.2f}%

                Priority:
                {priority}
                """
            )

        else:

            st.error(
                f"""
                🔴 HIGH RISK

                Dropout Probability:
                {risk_percentage:.2f}%

                Priority:
                {priority}
                """
            )

        st.progress(
            min(
                max(
                    probability,
                    0.0,
                ),
                1.0,
            )
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "👤 Student",
                student_name,
            )

        with col2:

            st.metric(
                "📊 Dropout Risk",
                f"{risk_percentage:.2f}%",
            )

        with col3:

            st.metric(
                "⚡ Priority",
                priority,
            )

        st.subheader(
            "⚠️ Potential Risk Factors"
        )

        if risk_factors:

            for factor in risk_factors:

                st.warning(
                    f"⚠️ {factor}"
                )

        else:

            st.success(
                "✅ No major risk factors detected."
            )

        st.divider()

        st.subheader(
            "🧠 AI Personalized Student Analysis"
        )

        st.markdown(
            "### 📋 Student Academic Risk Assessment"
        )

        for point in ai_analysis:

            st.info(
                f"🧠 {point}"
            )

        st.divider()

        st.subheader(
            "📅 Smart 4-Week Intervention Plan"
        )

        st.caption(
            "Personalized intervention actions based on "
            "the predicted risk level."
        )

        for week, actions in intervention_plan.items():

            with st.expander(
                f"📅 {week}",
                expanded=True,
            ):

                for action in actions:

                    st.write(
                        f"✅ {action}"
                    )

        st.subheader(
            "💡 Recommended Intervention"
        )

        for recommendation in recommendations:

            st.info(
                f"💡 {recommendation}"
            )

        st.divider()

        st.subheader(
            "📄 Download Student Report"
        )

        report = f"""
STUDENT DROPOUT EARLY WARNING REPORT
====================================

Prediction Date:
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Student Name: {student_name}
Student ID: {student_id}
Department: {department}
Semester: {semester}

Age: {age}
Gender: {gender}
Marital Status: {marital_status}

1st Semester Grade: {grade_1}
2nd Semester Grade: {grade_2}
Approved Subjects: {approved}

Tuition Fees Up To Date: {tuition}
Scholarship Holder: {scholarship}
Student Debt: {debtor}

Dropout Risk: {risk_percentage:.2f}%
Risk Level: {risk_level}
Priority: {priority}

Potential Risk Factors:
"""

        if risk_factors:

            for factor in risk_factors:

                report += f"\n- {factor}"

        else:

            report += "\n- No major risk factors detected."

        report += "\n\nRecommended Intervention:\n"

        for recommendation in recommendations:

            report += f"\n- {recommendation}"

        col1, col2 = st.columns(2)

        with col1:

            st.download_button(
                "⬇️ Download TXT Report",
                data=report,
                file_name=f"{student_id}_risk_report.txt",
                mime="text/plain",
                use_container_width=True,
            )

        try:

            pdf_file = create_pdf_report(
                student_name,
                student_id,
                department,
                semester,
                age,
                gender,
                marital_status,
                grade_1,
                grade_2,
                approved,
                tuition,
                scholarship,
                debtor,
                risk_percentage,
                risk_level,
                priority,
                risk_factors,
                recommendations,
            )

            with col2:

                st.download_button(
                    "📄 Download PDF Report",
                    data=pdf_file,
                    file_name=f"{student_id}_risk_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

        except Exception as e:

            with col2:

                st.error(
                    "❌ PDF report could not be generated."
                )

                st.exception(e)


# ============================================================
# TAB 3 - BATCH PREDICTION
# ============================================================

with tab3:

    st.header(
        "👥 Batch Student Prediction"
    )

    uploaded_file = st.file_uploader(
        "📤 Upload Student CSV File",
        type=["csv"],
    )

    if uploaded_file is not None:

        try:

            batch_data = pd.read_csv(
                uploaded_file
            )

        except Exception as e:

            st.error(
                "❌ Unable to read the uploaded CSV file."
            )

            st.exception(e)

            st.stop()

        st.subheader(
            "📋 Uploaded Student Data"
        )

        st.dataframe(
            batch_data,
            use_container_width=True,
        )

        required_batch_columns = [
            "Age",
            "Gender",
            "Marital_Status",
            "Grade_1",
            "Grade_2",
            "Approved_Subjects",
            "Tuition",
            "Scholarship",
            "Debt",
            "Student_ID",
            "Student_Name",
            "Department",
            "Semester",
        ]

        missing_batch_columns = [
            column
            for column in required_batch_columns
            if column not in batch_data.columns
        ]

        if missing_batch_columns:

            st.error(
                "❌ The uploaded CSV is missing required columns."
            )

            st.write("Missing columns:")

            for column in missing_batch_columns:

                st.write(f"- `{column}`")

        else:

            if st.button(
                "🤖 Predict All Students",
                use_container_width=True,
            ):

                results = []

                progress_bar = st.progress(0)

                total_rows = len(batch_data)

                if total_rows == 0:

                    st.warning(
                        "⚠️ The uploaded CSV contains no students."
                    )

                else:

                    for index, (_, row) in enumerate(
                        batch_data.iterrows()
                    ):

                        try:

                            probability = predict_risk(
                                row["Age"],
                                row["Gender"],
                                row["Marital_Status"],
                                row["Grade_1"],
                                row["Grade_2"],
                                row["Approved_Subjects"],
                                row["Tuition"],
                                row["Scholarship"],
                                row["Debt"],
                            )

                            risk_percentage = probability * 100

                            risk_level, _ = classify_risk(
                                probability
                            )

                            results.append(
                                {
                                    "Student ID":
                                        row["Student_ID"],

                                    "Student Name":
                                        row["Student_Name"],

                                    "Department":
                                        row["Department"],

                                    "Semester":
                                        row["Semester"],

                                    "Dropout Risk (%)":
                                        round(
                                            risk_percentage,
                                            2,
                                        ),

                                    "Risk Level":
                                        risk_level,
                                }
                            )

                        except Exception as e:

                            results.append(
                                {
                                    "Student ID":
                                        row["Student_ID"],

                                    "Student Name":
                                        row["Student_Name"],

                                    "Department":
                                        row["Department"],

                                    "Semester":
                                        row["Semester"],

                                    "Dropout Risk (%)":
                                        None,

                                    "Risk Level":
                                        f"ERROR: {str(e)}",
                                }
                            )

                        progress_bar.progress(
                            (index + 1) / total_rows
                        )

                    results_df = pd.DataFrame(
                        results
                    )

                    st.session_state[
                        "results_df"
                    ] = results_df

                    st.success(
                        f"✅ Processed {len(results_df)} student(s)."
                    )

    if "results_df" in st.session_state:

        results_df = st.session_state[
            "results_df"
        ]

        st.subheader(
            "📋 Batch Prediction Results"
        )

        st.dataframe(
            results_df,
            use_container_width=True,
        )

        st.download_button(
            "⬇️ Download Batch Results",
            data=results_df.to_csv(index=False),
            file_name="batch_prediction_results.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ============================================================
# TAB 4 - ANALYTICS
# ============================================================

with tab4:

    st.header(
        "📊 Dataset & Model Analytics"
    )

    if "Target" in data.columns:

        outcome_counts = data[
            "Target"
        ].value_counts()

        st.subheader(
            "Student Academic Outcomes"
        )

        st.bar_chart(
            outcome_counts
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "🎓 Graduates",
                outcome_counts.get(
                    "Graduate",
                    0,
                ),
            )

        with col2:

            st.metric(
                "🚪 Dropouts",
                outcome_counts.get(
                    "Dropout",
                    0,
                ),
            )

        with col3:

            st.metric(
                "📚 Enrolled",
                outcome_counts.get(
                    "Enrolled",
                    0,
                ),
            )

    else:

        st.warning(
            "⚠️ Target column is not available in the dataset."
        )

    st.divider()

    st.subheader(
        "🧠 AI Insights: Feature Importance"
    )

    if hasattr(
        model,
        "feature_importances_",
    ):

        importance_values = (
            model.feature_importances_
        )

        if len(importance_values) == len(
            feature_names
        ):

            importance_df = pd.DataFrame(
                {
                    "Feature":
                        feature_names,

                    "Importance":
                        importance_values,
                }
            )

            importance_df = (
                importance_df
                .sort_values(
                    by="Importance",
                    ascending=False,
                )
            )

            top_features = (
                importance_df.head(10)
            )

            st.bar_chart(
                top_features.set_index(
                    "Feature"
                )
            )

            st.dataframe(
                top_features,
                use_container_width=True,
            )

        else:

            st.warning(
                "⚠️ Model feature importance count "
                "does not match feature names."
            )

    else:

        st.info(
            "ℹ️ This model does not expose feature importance."
        )

    st.divider()

    st.subheader(
        "📈 Model Performance"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Accuracy",
            "88.36%",
        )

    with col2:

        st.metric(
            "Precision",
            "83.39%",
        )

    with col3:

        st.metric(
            "Recall",
            "79.58%",
        )

    with col4:

        st.metric(
            "F1 Score",
            "81.44%",
        )


# ============================================================
# TAB 5 - PREDICTION HISTORY
# ============================================================

with tab5:

    st.header(
        "📜 Student Prediction History"
    )

    try:

        if os.path.exists(HISTORY_PATH):

            history_df = pd.read_csv(
                HISTORY_PATH
            )

        else:

            history_df = pd.DataFrame()

        if history_df.empty:

            st.info(
                "📭 No prediction history yet. "
                "Predict a student first."
            )

        else:

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "👥 Total Predictions",
                    len(history_df),
                )

            with col2:

                st.metric(
                    "🟢 Low Risk",
                    len(
                        history_df[
                            history_df["Risk Level"]
                            == "LOW"
                        ]
                    ),
                )

            with col3:

                st.metric(
                    "🟡 Medium Risk",
                    len(
                        history_df[
                            history_df["Risk Level"]
                            == "MEDIUM"
                        ]
                    ),
                )

            with col4:

                st.metric(
                    "🔴 High Risk",
                    len(
                        history_df[
                            history_df["Risk Level"]
                            == "HIGH"
                        ]
                    ),
                )

            st.divider()

            col1, col2 = st.columns(2)

            with col1:

                search_history = st.text_input(
                    "🔎 Search Student Name or ID"
                )

            with col2:

                history_risk_filter = st.selectbox(
                    "🚦 Filter Risk Level",
                    [
                        "ALL",
                        "LOW",
                        "MEDIUM",
                        "HIGH",
                    ],
                )

            filtered_history = history_df.copy()

            if search_history:

                search_text = (
                    search_history
                    .strip()
                    .lower()
                )

                name_match = (
                    filtered_history[
                        "Student Name"
                    ]
                    .astype(str)
                    .str.lower()
                    .str.contains(
                        search_text,
                        na=False,
                    )
                )

                id_match = (
                    filtered_history[
                        "Student ID"
                    ]
                    .astype(str)
                    .str.lower()
                    .str.contains(
                        search_text,
                        na=False,
                    )
                )

                filtered_history = filtered_history[
                    name_match | id_match
                ]

            if history_risk_filter != "ALL":

                filtered_history = filtered_history[
                    filtered_history[
                        "Risk Level"
                    ]
                    == history_risk_filter
                ]

            st.subheader(
                "📋 Student Predictions"
            )

            st.dataframe(
                filtered_history,
                use_container_width=True,
            )

            st.download_button(
                "⬇️ Download Prediction History",
                data=filtered_history.to_csv(
                    index=False
                ),
                file_name="student_prediction_history.csv",
                mime="text/csv",
                use_container_width=True,
            )

            st.divider()

            st.subheader(
                "🚨 High-Risk Students"
            )

            high_risk_history = history_df[
                history_df["Risk Level"] == "HIGH"
            ]

            if not high_risk_history.empty:

                st.error(
                    f"⚠️ {len(high_risk_history)} "
                    "high-risk student(s) detected."
                )

                st.dataframe(
                    high_risk_history,
                    use_container_width=True,
                )

            else:

                st.success(
                    "🎉 No high-risk students found."
                )

            st.divider()

            st.subheader(
                "🗑️ History Management"
            )

            if st.button(
                "🗑️ Clear Prediction History",
                use_container_width=True,
            ):

                empty_history = pd.DataFrame(
                    columns=history_df.columns
                )

                empty_history.to_csv(
                    HISTORY_PATH,
                    index=False,
                )

                st.success(
                    "✅ Prediction history cleared!"
                )

                st.rerun()

    except Exception as e:

        st.error(
            "❌ Prediction history could not be loaded."
        )

        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🎓 AI-Based Student Academic Performance "
    "& Dropout Risk Prediction System | "
    "Machine Learning Project"
)