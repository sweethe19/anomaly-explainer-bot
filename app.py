import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore
import openai
from openai import OpenAI
import seaborn as sns
import matplotlib.pyplot as plt

# ✅ Set OpenAI key securely
client = OpenAI(api_key=st.secrets["openai"]["api_key"])

# ✅ UI Setup
st.set_page_config(page_title="Anomaly Explainer Bot", layout="wide")
st.title("📉 Anomaly Explainer Bot")
st.markdown("Upload CSV or Excel, detect outliers, and understand them with AI 🧠")

# ✅ File Upload
uploaded_file = st.file_uploader("📂 Upload CSV or Excel file", type=["csv", "xlsx"])
if uploaded_file:
    # Support for CSV & Excel
    file_ext = uploaded_file.name.split(".")[-1]
    if file_ext == "xlsx":
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    st.subheader("📊 Preview of Your Data")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.warning("❌ No numeric columns found.")
    else:
        selected_cols = st.multiselect("🔢 Select columns for anomaly detection", numeric_cols, default=numeric_cols[:1])
        method = st.radio("🧠 Choose detection method", ["Isolation Forest", "Z-Score"])
        sensitivity = st.slider("📏 Anomaly sensitivity (contamination)", 0.01, 0.5, 0.2, step=0.01)

        if selected_cols:
            df_copy = df.copy()

            # Anomaly Detection Logic
            if method == "Isolation Forest":
                model = IsolationForest(contamination=sensitivity)
                model.fit(df_copy[selected_cols])
                df_copy['Anomaly_Score'] = model.decision_function(df_copy[selected_cols])
                df_copy['Anomaly'] = model.predict(df_copy[selected_cols])
                df_copy['Anomaly'] = df_copy['Anomaly'].apply(lambda x: '🔴 Anomaly' if x == -1 else '✅ Normal')
            else:
                zscores = df_copy[selected_cols].apply(zscore)
                df_copy['Anomaly_Score'] = zscores.abs().max(axis=1)
                df_copy['Anomaly'] = df_copy['Anomaly_Score'].apply(lambda x: '🔴 Anomaly' if x > 2 else '✅ Normal')

            # Rule-based explanation
            def explain_reason(row):
                reasons = []
                for col in selected_cols:
                    mean = df_copy[col].mean()
                    std = df_copy[col].std()
                    if abs(row[col] - mean) > 2 * std:
                        reasons.append(f"{col} deviates from mean")
                return ", ".join(reasons) if reasons else "Normal range"

            df_copy["Anomaly_Reason"] = df_copy.apply(explain_reason, axis=1)

            # Optional filter
            show_only = st.checkbox("🔎 Show only anomalies", value=False)
            display_df = df_copy[df_copy['Anomaly'] == '🔴 Anomaly'] if show_only else df_copy

            # Show Results Table
            st.subheader("✅ Anomaly Detection Results")
            st.dataframe(display_df[selected_cols + ['Anomaly_Score', 'Anomaly', 'Anomaly_Reason']])

            # Summary Stats + Boxplot
            st.subheader("📊 Summary Statistics")
            st.dataframe(df_copy[selected_cols].describe())

            for col in selected_cols:
                st.markdown(f"📈 Boxplot: `{col}`")
                fig = px.box(df_copy, y=col, points="all", color='Anomaly')
                st.plotly_chart(fig)

            # 🔥 Anomaly Heatmap
            st.subheader("🔥 Correlation Heatmap")
            fig2, ax = plt.subplots()
            sns.heatmap(df_copy[selected_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig2)

            # Download Button
            st.download_button("⬇️ Download Results", df_copy.to_csv(index=False), "anomalies.csv", "text/csv")

            # GPT Dataset Summary
            st.subheader("🧠 GPT: Dataset Summary")
            if st.button("📋 Summarize Dataset"):
                stats = df_copy[selected_cols].describe().to_string()
                prompt = f"Here are the summary statistics of a dataset:\n{stats}\nGive a summary of what this data reveals."
                with st.spinner("Analyzing..."):
                    try:
                        resp = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "You're a data analyst."},
                                {"role": "user", "content": prompt}
                            ]
                        )
                        summary = resp.choices[0].message.content
                        st.success("Dataset Summary:")
                        st.write(summary)
                    except Exception as e:
                        st.error(f"OpenAI API Error: {e}")

            # GPT Fix Suggestions
            st.subheader("🛠 GPT: Suggest Fixes for Anomalies")
            if st.button("🔧 Suggest Fixes"):
                anomalies_only = df_copy[df_copy['Anomaly'] == '🔴 Anomaly'][selected_cols]
                if not anomalies_only.empty:
                    prompt = f"The following data points were flagged as anomalies:\n{anomalies_only.to_string()}\nSuggest corrected values if these are likely errors."
                    with st.spinner("Generating suggestions..."):
                        try:
                            resp = client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {"role": "system", "content": "You're a helpful data analyst."},
                                    {"role": "user", "content": prompt}
                                ]
                            )
                            suggestion = resp.choices[0].message.content
                            st.success("Suggested Fixes:")
                            st.write(suggestion)
                        except Exception as e:
                            st.error(f"OpenAI API Error: {e}")
                else:
                    st.info("🎉 No anomalies found. Nothing to fix!")
