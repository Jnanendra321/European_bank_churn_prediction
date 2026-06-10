import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("European_Bank.csv")

# Set page configuration
st.set_page_config(page_title="Churn Prediction",
                   page_icon="🏦",
                   layout="wide"
                   )

# Load the trained model
model = joblib.load("xgb_pipeline.pkl")

#custom css
st.markdown("""
<style>

/* Main Background */
.stApp{
    background-color: #f4f7fc;
}

/* Main sidebar */
[data-testid="stSidebar"] {
    background-color: #0b1f66;
}

/* All sidebar text */
[data-testid="stSidebar"] * {
    color: white !important;
}

/* Radio button labels */
.stRadio label {
    color: white !important;
    font-size: 20px !important;
}

/* Sidebar title */
[data-testid="stSidebar"] h1 {
    color: white !important;
}

/* Sidebar markdown */
[data-testid="stSidebar"] p {
    color: white !important;
}
.main {
    background-color: #f5f5f5;
}

/* Main Title */
.main-title{
    
    font-size:40px !important;
    font-weight:bold;
    color:#0B1F5E;
}

/* Subtitle */
.sub-title{
    font-size:28px;
    color:#4a4a4a;
    margin-bottom:25px;
}
            
/* Input Labels */
label {
    font-size:18px !important;
    font-weight:600 !important;
    color:#0B1F5E !important;
}


/* Cards */
.card{
    background-color:white;
    padding:25px;
    border-radius:20px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.08);
}

/* Result Box Positive */
.success-box{
    background-color:#d1fae5;
    padding:30px;
    border-radius:20px;
    text-align:center;
    font-size:28px;
    font-weight:bold;
    color:#065f46;
}

/* Result Box Negative */
.danger-box{
    background-color:#fee2e2;
    padding:30px;
    border-radius:20px;
    text-align:center;
    font-size:28px;
    font-weight:bold;
    color:#991b1b;
}

/* Predict Button */
div.stButton > button:first-child{
    background-color:#2563eb;
    color:white;
    border:none;
    border-radius:12px;
    height:55px;
    width:100%;
    font-size:20px;
    font-weight:bold;
}

/* Hover Effect */
div.stButton > button:first-child:hover{
    background-color:#1d4ed8;
    color:white;
}

</style>
""", unsafe_allow_html=True)

#sidebar
with st.sidebar:

    st.markdown("""
        <h2 style='color:white;font-size:24px;'>
        🏦 Bank Dashboard
        </h2>
        """, unsafe_allow_html=True)

    st.sidebar.title("Navigation")

    page = st.sidebar.radio(
            "Go to",
            ["🏠 Home", "📊 Prediction"]
    )
    st.markdown("---")
    st.info(
        "Machine Learning based customer churn prediction system using XGBoost."
    )

if page == "🏠 Home":

    st.title("European Bank Churn Prediction")

    st.write("""
    This app predicts whether a customer
    will leave the bank or not.
    """)

    st.markdown("---")

    # Images
    col1, col2 = st.columns(2)
    st.header("📊 Business Insights Dashboard")

    col1, col2 = st.columns(2)

    with col1:

    # Pie Chart
        fig, ax = plt.subplots(figsize=(5,5))
        df["Exited"].value_counts().plot(
            kind="pie",
            autopct="%1.1f%%",
            labels=["Stayed", "Churned"],
            ax=ax
        )
        st.pyplot(fig)

    with col2:

        geo_churn = (
            df.groupby("Geography")["Exited"]
            .mean()
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(5,5))
        sns.barplot(
            data=geo_churn,
            x="Geography",
            y="Exited",
            ax=ax
        )

        st.pyplot(fig)

    st.markdown("---")

    fig, ax = plt.subplots(figsize=(10,4))

    sns.histplot(
        data=df,
        x="Age",
        hue="Exited",
        bins=30,
        kde=True,
        ax=ax
    )

    st.pyplot(fig)

    st.markdown("---")

    # About Section
    st.header("📖 About This Project")

    st.write("""
    This project predicts customer churn using Machine Learning.

    The model analyzes customer information such as:

    • Credit Score

    • Geography

    • Gender

    • Age

    • Balance

    • Number of Products

    • Active Membership Status

    • Estimated Salary

    The prediction is powered by an XGBoost model trained on historical banking data.
    """)

    st.markdown("---")

    # Feature Cards
    c1, c2, c3 = st.columns(3)

    with c1:
        st.success("""
        ### 📊 Customer Analysis

        Analyze customer behavior patterns.
        """)

    with c2:
        st.info("""
        ### 🤖 Churn Prediction

        Predict customer exit probability.
        """)

    with c3:
        st.warning("""
        ### 💡 Business Insights

        Support customer retention strategies.
        """)
else:
    # MAIN HEADER
    st.markdown(
        '<p class="main-title">Bank Customer Churn Prediction</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="sub-title">Predict whether a customer is likely to leave the bank using Artificial Intelligence.</p>',
        unsafe_allow_html=True
    )
    # TWO COLUMN LAYOUT
    left_col, right_col = st.columns([1.2,1])

    # LEFT COLUMN INPUTS
    with left_col:

        st.markdown("""
        <h2 style='color:#0B1F5E;font-size:24px;'>
        👤 Enter Customer Details
        </h2>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:

            credit_score = st.text_input(
            "Credit Score",
            placeholder="Enter Credit Score(300-900)"
            )

            geography = st.selectbox(
            "Geography",
            ["France", "Germany", "Spain"],
            index=None,
            placeholder="Select customer's country"
            )

            gender = st.selectbox(
            "Gender",
            ["Male", "Female"],
            index=None,
            placeholder="Select customer's gender"
            )

            age = st.text_input(
            "Age",
            placeholder="Enter Age(18-100)"
            )

            tenure = st.slider(
            "Tenure (Years)",
            0, 10, 5,
            help="Number of years the customer has been with the bank"
            )
    
            # st.caption(
            # "Number of years the customer has been with the bank"
            # )

        # RIGHT SIDE
        with c2:

            balance = st.number_input(
                "Balance",
                min_value=0.0,
                help="Current account balance of the customer"
            )

            num_products = st.slider(
                "Number Of Products",
                1, 4, 2
            )

            has_card = st.selectbox(
                "Has Credit Card",
                [1,0]
            )

            active_member = st.selectbox(
                "Is Active Member",
                [1,0]
            )

            salary = st.number_input(
                "Estimated Salary",
                min_value=0.0,
                value=70000.0
            )

        st.markdown("<br>", unsafe_allow_html=True)

        predict_button = st.button("🔍 Predict Churn")

        st.markdown("</div>", unsafe_allow_html=True)

    # RIGHT COLUMN RESULT
    with right_col:

        st.markdown("""
        <h2 style='color:#0B1F5E;font-size:24px;'>
        📈 Prediction Result
        </h2>
        """, unsafe_allow_html=True)

        if predict_button:

            input_df = pd.DataFrame({

                'CreditScore':[credit_score],
                'Geography':[geography],
                'Gender':[gender],
                'Age':[age],
                'Tenure':[tenure],
                'Balance':[balance],
                'NumOfProducts':[num_products],
                'HasCrCard':[has_card],
                'IsActiveMember':[active_member],
                'EstimatedSalary':[salary]

            })

            prediction = model.predict(input_df)[0]

            probability = model.predict_proba(input_df)[0][1]

            # ================= RESULT =================
            if prediction == 1:

                st.markdown(f"""
                <div class='danger-box'>

                <h2>
                Customer is likely to
                </h2>

                <h1 style='font-size:60px;'>
                CHURN
                </h1>

                <h2>
                (Will Leave The Bank)
                </h2>

                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                st.markdown("""
                <h3 style='color:#0B1F5E;'>
                Churn Probability
                </h3>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <h1 style='color:red;'>
                {probability:.2%}
                </h1>
                """, unsafe_allow_html=True)

                st.progress(int(probability*100))

            else:

                st.markdown(f"""
                <div class='success-box'>

                <h2>
                Customer is NOT likely to
                </h2>

                <h1 style='font-size:60px;'>
                CHURN
                </h1>

                <h2>
                (Customer Will Stay)
                </h2>

                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                st.markdown("""
                <h3 style='color:#0B1F5E;'>
                Retention Probability
                </h3>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <h1 style='color:green;'>
                {(1-probability):.2%}
                </h1>
                """, unsafe_allow_html=True)

                st.progress(int((1-probability)*100))

        else:

            st.info(
                "Enter customer details and click Predict Churn"
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # ======================================================
    # FOOTER
    # ======================================================
    st.markdown("<br>", unsafe_allow_html=True)

    st.info(
        "This prediction is based on the Machine Learning model (XGBoost) trained on historical bank customer data."
    )

