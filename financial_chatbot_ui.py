import streamlit as st
import openai  # For OpenAI API integration
import os
from pathlib import Path
from utils import load_buffett_letters
from openai_chatbot_api import BuffettRAG
import pandas as pd
import numpy as np
from retrieve_faiss import answer_with_faiss_or_t5
from load_financial_sheet import get_sheet_and_eva, get_continues_plot, analysis_continues, plot_financial_metrics

# Set page layout
st.set_page_config(layout="wide", page_title="Chat & Finance Insight Bot")

# Initialize session state if not already done
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_company' not in st.session_state:
    st.session_state.current_company = None
if 'current_sheet' not in st.session_state:
    st.session_state.current_sheet = None


# Letters data - keyed by ID
buffett_context = load_buffett_letters()
letters_data, company_sector_index, term_index = buffett_context

# Initialize RAG system
rag = BuffettRAG(letters_data, term_index, company_sector_index)

def get_bot_response(question):
    answer = answer_with_faiss_or_t5(question)
    return rag.get_refined_answer(question, answer)


def answer_with_example(question, initial_answer):
    """Use OpenAI API to generate Buffett-style responses with real-world examples"""
    return rag.get_real_world_examples(question, initial_answer)



# 1. Top title
st.title("Chat & Finance Insight Bot")

# 2. Split page into two columns
chat_col, finance_col = st.columns([1.2, 2])

# --- Left Side: ChatBot Area ---
with chat_col:
    st.header("💬 Chat with Warren Buffett")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    user_input = st.chat_input("Ask Warren Buffett about the financial data...")

    # process inout
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Warren is thinking about his investments..."):
                response = get_bot_response(user_input)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        # save response for later
        st.session_state.last_response = response
        st.session_state.last_user_input = user_input

    # 🏛️ Past Investment
    if "last_response" in st.session_state and st.button("🏛️ Buffett’s Past Investments"):
        with st.chat_message("assistant"):
            with st.spinner("Warren is thinking about his past investments..."):
                example_response = answer_with_example(
                    st.session_state.last_user_input, st.session_state.last_response
                )
                st.markdown("### Based on my investment history:")
                st.markdown(example_response)
        st.session_state.messages.append({"role": "assistant", "content": example_response})


# --- Right Side: Financial Analysis Module ---
with finance_col:
    st.header("🏢 Company Financial Chart Analysis")

    # Company selection
    company = st.selectbox("Select Company:", ["Apple", "Tesla", "Meta","Microsoft"])
    st.session_state.current_company = company

    # Report selection
    sheet_type = st.radio("Select Report:", ["Income Sheet", "Balance Sheet", "Cashflow Sheet"])
    st.session_state.current_sheet = sheet_type

    # get required sheet
    all_sheets, eva_sheets = get_sheet_and_eva(company)
    df_sheet = all_sheets[sheet_type]
    # Get the evaluation dataframe for the selected sheet
    df_eva = eva_sheets[sheet_type]

    # Get list of years from the dataframe
    year_list = df_sheet.columns
    st.markdown(
        f"## The {year_list[-1].strftime('%Y') if not isinstance(year_list[-1], str) else year_list[-1]}-{year_list[0].strftime('%Y') if not isinstance(year_list[0], str) else year_list[0]} {sheet_type} Statement:")

    # Display the selected sheet
    df_display = df_sheet.copy()
    df_display = df_display.dropna(how='all', axis=1)
    if isinstance(df_display.columns, pd.DatetimeIndex):
        df_display.columns = df_display.columns.strftime('%Y-%m-%d')
    st.dataframe(df_display)


    # Generate trend data and visualizations
    df_cleaned, trend_fig = get_continues_plot(df_sheet)
    trend_info = analysis_continues(df_cleaned)


    st.markdown("### Metric Trends & Stats")
    col1, col2 = st.columns([7, 3])

    with col1:
        # Add unique key for the trend chart
        st.plotly_chart(trend_fig, use_container_width=True, key="trend_chart")

    with col2:
        st.markdown("#### Trend in Motion")
        st.markdown("##### 📈Climbing Higher")
        if trend_info['increasing']:
            for metric, rate in trend_info['increasing'].items():
                st.markdown(f"- **{metric}**: +{rate:.2%}")
        else:
            st.markdown("_None found._")

        st.markdown("##### 📉Slipping Steadily")
        if trend_info['decreasing']:
            for metric, rate in trend_info['decreasing'].items():
                st.markdown(f"- **{metric}**: -{rate:.2%}")
        else:
            st.markdown("_None found._")

        # Add a "Ask Buffett" button for this specific chart
        if st.button("🤠 Ask Buffett about this trend", key="buffett_trend"):
            if "messages" not in st.session_state:
                st.session_state.messages = []

            increasing_metrics = ", ".join(list(trend_info['increasing'].keys())[:3]) if trend_info[
                'increasing'] else "none"
            decreasing_metrics = ", ".join(list(trend_info['decreasing'].keys())[:3]) if trend_info[
                'decreasing'] else "none"

            st.session_state.messages.append({
                "role": "user",
                "content": f"What does it mean for {company} that metrics like {increasing_metrics} are increasing while {decreasing_metrics} are decreasing over the last 3 years?"
            })


    # Allow user to select year
    year = st.selectbox("Select Year:", [col.strftime('%Y') if not isinstance(col, str) else col for col in year_list])
    st.markdown(f"### {year} {sheet_type}")

    # Plot the metrics
    fig1, fig2, fig3 = plot_financial_metrics(df_sheet, df_eva, year, sheet_type)

    # Create tabs for different threshold categories
    tab1, tab2, tab3 = st.tabs([
        "📈Key Financial Metrics",
        "🔺Metrics Should ABOVE Threshold",
        "🔻Metrics Should BELOW Threshold",
    ])

    with tab1:
        st.markdown(f"#### 📊 Key Financial Metrics")
        st.markdown(f"""- This chart displays key financial metrics from the {sheet_type} for {year}, highlighting the performance relative to target thresholds.  
                     - Use it to identify consistent growth, declines, and any deviations from expected benchmarks.""")
        #Add a "Ask Buffett" button for this specific chart
        if st.button("🤠 Ask Buffett about this trend", key="buffett_og_value"):
            if "messages" not in st.session_state:
                st.session_state.messages = []

            increasing_metrics = ", ".join(list(trend_info['increasing'].keys())[:3]) if trend_info[
                'increasing'] else "none"
            decreasing_metrics = ", ".join(list(trend_info['decreasing'].keys())[:3]) if trend_info[
                'decreasing'] else "none"

            st.session_state.messages.append({
                "role": "user",
                "content": f"What does it mean for {company} that metrics like {increasing_metrics} are increasing while {decreasing_metrics} are decreasing over the last 3 years?"
            })

        st.plotly_chart(fig1, use_container_width=True, key="Original Values")

    with tab2:
        # Show explanation for metrics that should be above threshold
        st.markdown("#### 📊 Metrics Explanation")
        st.markdown("""
                This chart shows metrics that should ideally be **above** their threshold values:
                - **Blue bars**: These metrics are meeting expectations (above threshold)
                - **Red bars**: These metrics are below their expected thresholds (needs attention)
                """)

        #Add a "Ask Buffett" button for this specific chart
        if st.button("🤠 Ask Buffett about above-threshold metrics", key="buffett_above"):
            if "messages" not in st.session_state:
                st.session_state.messages = []

            st.session_state.messages.append({
                "role": "user",
                "content": f"What does it mean that some of {company}'s metrics are below their expected thresholds on the 'Should be Above' chart?"
            })

        # Add unique key for the above threshold chart
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig1, use_container_width=True, key="Original Values for above")
        with col2:
            st.plotly_chart(fig2, use_container_width=True, key="above_threshold_chart")

    with tab3:
        # Show explanation for metrics that should be below threshold
        st.markdown("#### 📊 Metrics Explanation")
        st.markdown("""
                This chart shows metrics that should ideally be **below** their threshold values:
                - **Blue bars**: These metrics are meeting expectations (below threshold)
                - **Red bars**: These metrics are above their expected thresholds (needs attention)
                """)

        #Add a "Ask Buffett" button for this specific chart
        if st.button("🤠 Ask Buffett about below-threshold metrics", key="buffett_below"):
            if "messages" not in st.session_state:
                st.session_state.messages = []

            st.session_state.messages.append({
                "role": "user",
                "content": f"What does it mean that some of {company}'s metrics are above their expected thresholds on the 'Should be Below' chart?"
            })

        # Add unique key for the below threshold chart
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig1, use_container_width=True, key="Original Values for below")

        with col2:
            st.plotly_chart(fig3, use_container_width=True, key="below_threshold_chart")