import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
import re
import plotly.express as px
import os
from dotenv import load_dotenv

# --- 1. SECURE KEY LOADING ---
# Locally: Looks for a .env file. Cloud: Looks for Streamlit Secrets.
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

# --- 2. SETUP & CONFIG ---
st.set_page_config(page_title="Hypothesis-Hatch", layout="wide", page_icon="🧪")
st.title("🧪 Hypothesis-Hatch")
st.subheader("Transforming health data into AI-driven stories")

if not api_key:
    st.error("⚠️ API Key not detected. Please add GEMINI_API_KEY to your .env file or Streamlit Secrets.")
    st.stop()

# --- 3. FILE UPLOADER ---
uploaded_file = st.file_uploader("Upload your Health Dataset (CSV)", type=["csv"])

if uploaded_file:
    # Initialize the 2026 Stable Client
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(api_version="v1beta")
    )

    try:
        # Rewind and Clean Data
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        
        # Global Fix: Remove hidden spaces and drop empty rows
        df.columns = df.columns.str.strip()
        df = df.dropna()

        if df.empty:
            st.warning("The dataset is empty after removing missing values.")
            st.stop()

        st.write("### Data Preview", df.head(5))
        
        # Prepare the statistical DNA for the AI
        stats_summary = f"""
        COLUMNS: {list(df.columns)}
        DESCRIPTIVE STATS: {df.describe().to_string()}
        CORRELATIONS: {df.corr(numeric_only=True).to_string()}
        """

        # --- 4. THE ANALYSIS ENGINE ---
        if st.button("Generate Hypotheses & Charts"):
            with st.spinner("Gemini 3 is analyzing your data..."):
                try:
                    response = client.models.generate_content(
                        model="gemini-3-flash-preview", 
                        contents=f"""
                        ACT AS A SENIOR DATA SCIENTIST. 
                        DATASET CONTEXT: {stats_summary}

                        TASK:
                        1. Propose 3 data-driven hypotheses.
                        2. For each, provide a Plotly Express code block using 'px'.
                        3. Use the variable name 'df'. 
                        4. IMPORTANT: Do not use trendlines or statsmodels.
                        5. Wrap code strictly in ```python [code] ```
                        """
                    )
                    
                    st.markdown(response.text)

                    # --- 5. THE VISUALIZATION ENGINE ---
                    code_blocks = re.findall(r'```python\n(.*?)\n```', response.text, re.DOTALL)
                    
                    for i, code in enumerate(code_blocks):
                        st.divider()
                        st.write(f"### Visualizing Hypothesis {i+1}")
                        try:
                            # Execute the AI-generated code in a safe sandbox
                            local_vars = {"df": df, "px": px, "st": st}
                            exec(code, {}, local_vars)
                            
                            # Find any plotly chart object in memory
                            chart_shown = False
                            for val in local_vars.values():
                                if "plotly.graph_objs._figure.Figure" in str(type(val)):
                                    st.plotly_chart(val, use_container_width=True)
                                    chart_shown = True
                                    break
                            
                            if not chart_shown:
                                st.info("Analysis complete, but no chart object was found.")

                        except Exception as e:
                            st.error(f"Chart execution failed: {e}")
                
                except Exception as e:
                    st.error(f"API Connection Error: {e}")

    except Exception as e:
        st.error(f"Processing Error: {e}")

else:
    st.info("👋 Welcome! Please upload a CSV file to begin the analysis.")
