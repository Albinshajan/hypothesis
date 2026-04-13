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
    api_key = st.text_input("AIzaSyDWAQi1RGE-3HJOiGhGx8mc4xfxQ6kn5vk", type="password")
    st.info("Get your key at aistudio.google.com")

# 2. File Uploader
uploaded_file = st.file_uploader("Upload your Health Dataset (CSV)", type=["csv"])

if uploaded_file and api_key:
    # Initialize the Client once per session
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(api_version="v1beta")
    )

    try:
        # Rewind and Clean Data
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        
        # CLEANING: Fix hidden spaces and missing values
        df.columns = df.columns.str.strip()
        df = df.dropna()

        if df.empty:
            st.warning("The dataset is empty after removing missing values.")
            st.stop()

        st.write("### Data Preview", df.head(5))
        
        # Prepare data intelligence for the AI
        stats_summary = f"""
        CLEANED COLUMNS: {list(df.columns)}
        NUMERIC STATS: {df.describe().to_string()}
        CORRELATIONS: {df.corr(numeric_only=True).to_string()}
        """

        # 3. Trigger analysis
        if st.button("Generate Hypotheses & Charts"):
            with st.spinner("Gemini 3 is crunching the numbers..."):
                try:
                    response = client.models.generate_content(
                        model="gemini-3-flash-preview", 
                        contents=f"""
                        ACT AS A SENIOR DATA SCIENTIST. 
                        
                        DATASET INFO:
                        {stats_summary}

                        TASK:
                        1. Propose 3 specific hypotheses.
                        2. For each, provide a brief story and a Plotly Express code block using 'px'.
                        3. CRITICAL: Use the EXACT column names from the list above.
                        4. CRITICAL: Do not use trendlines or external libraries like statsmodels.
                        5. Use variable name 'df' and wrap in ```python [code] ```
                        """
                    )
                    
                    # Display the AI's Analysis text
                    st.markdown(response.text)

                    # 4. Visualization Engine
                    code_blocks = re.findall(r'```python\n(.*?)\n```', response.text, re.DOTALL)
                    
                    for i, code in enumerate(code_blocks):
                        st.write(f"---")
                        st.write(f"### Visualizing Hypothesis {i+1}")
                        try:
                            # Sandbox for AI code
                            local_vars = {"df": df, "px": px, "st": st}
                            exec(code, {}, local_vars)
                            
                            # Find and render the figure
                            chart_found = False
                            for val in local_vars.values():
                                if "plotly.graph_objs._figure.Figure" in str(type(val)):
                                    st.plotly_chart(val, use_container_width=True)
                                    chart_found = True
                                    break
                            
                            if not chart_found:
                                st.info("Analysis complete, but no chart object was generated.")

                        except Exception as e:
                            st.error(f"Chart Error: {e}")
                
                except Exception as e:
                    if "429" in str(e):
                        st.error("Free-tier limit reached. Please wait 60 seconds.")
                    else:
                        st.error(f"API Error: {e}")

    except Exception as e:
        st.error(f"Processing Error: {e}")

elif not api_key:
    st.warning("Please enter your API Key in the sidebar.")
