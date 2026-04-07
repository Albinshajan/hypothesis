import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
import re
import plotly.express as px

# 1. Setup & Config
st.set_page_config(page_title="Hypothesis-Hatch", layout="wide")
st.title("🧪 Hypothesis-Hatch")
st.subheader("Turn raw health data into AI-driven stories")

# Sidebar for API Key
with st.sidebar:
    st.header("Settings")
    # Using a generic label to avoid exposing your key in the UI
    api_key = st.text_input("AIzaSyCqDVRbHldZopM1F7iglBQNK1nz1GpZCks", type="password")
    st.info("Get your key at aistudio.google.com")

# 2. File Uploader
uploaded_file = st.file_uploader("Upload your Health Dataset (CSV)", type=["csv"])

if uploaded_file and api_key:
    # --- 1. SETUP THE CLIENT ---
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(api_version="v1beta")
    )

    try:
        # --- 2. THE DATA CLEANING FIX ---
        uploaded_file.seek(0)  # Rewind the file pointer
        df = pd.read_csv(uploaded_file)
        
        # This fixes the "Adult Mortality" error by removing hidden spaces
        df.columns = df.columns.str.strip()
        
        # This fixes the "nan" error by removing missing data rows
        df = df.dropna()

        if df.empty:
            st.error("The dataset is empty after cleaning. Please check your CSV.")
            st.stop()

        st.write("### Data Preview", df.head(5))
        
        # Prepare the summary for Gemini
        stats_summary = f"""
        Columns available: {list(df.columns)}
        Correlations: {df.corr(numeric_only=True).to_string()}
        Stats: {df.describe().to_string()}
        """

        # --- 3. THE UPDATED BUTTON LOGIC ---
        if st.button("Generate Hypotheses & Charts"):
            with st.spinner("Gemini 3 is analyzing..."):
                try:
                    # The prompt now FORCES the AI to use your cleaned columns
                    response = client.models.generate_content(
                        model="gemini-3-flash-preview", 
                        contents=f"""
                        ACT AS A SENIOR DATA SCIENTIST. 
                        DATA SUMMARY: {stats_summary}

                        TASK:
                        1. Propose 3 hypotheses.
                        2. Provide Plotly Express code using 'px'.
                        3. CRITICAL: Use the exact column names from this list: {list(df.columns)}
                        4. CRITICAL: Do not use trendlines or statsmodels.
                        5. Use variable name 'df' and wrap in ```python [code] ```
                        """
                    )
                    
                    st.markdown(response.text)

                    # --- 4. THE VISUALIZATION ENGINE ---
                    code_blocks = re.findall(r'```python\n(.*?)\n```', response.text, re.DOTALL)
                    
                    for i, code in enumerate(code_blocks):
                        st.write(f"### Visualizing Hypothesis {i+1}")
                        try:
                            local_vars = {"df": df, "px": px, "st": st}
                            exec(code, {}, local_vars)
                            
                            # Search memory for the figure and display it
                            for val in local_vars.values():
                                if "plotly.graph_objs._figure.Figure" in str(type(val)):
                                    st.plotly_chart(val, use_container_width=True)
                                    break
                        except Exception as e:
                            st.error(f"Chart Error: {e}")
                
                except Exception as e:
                    st.error(f"API Error: {e}")

    except Exception as e:
        st.error(f"Processing Error: {e}")