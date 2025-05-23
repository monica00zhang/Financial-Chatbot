import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

coms2code = {
    'Microsoft': 'MSFT',
    'Apple': 'AAPL',
    'NVIDIA': 'NVDA',
    'Alphabet': 'GOOGL',
    'Amazon': 'AMZN',
    'Meta': 'META',
    'Intel': 'INTC',
    'Johnson & Johnson': 'JNJ',
    'Pfizer': 'PFE',
    'UnitedHealth': 'UNH',
    'Moderna': 'MRNA',
    'Eli Lilly': 'LLY',
    'Abbott': 'ABT',
    'Merck': 'MRK',
    'Tesla': 'TSLA',
    'First Solar': 'FSLR',
    'NextEra Energy': 'NEE',
    'Plug Power': 'PLUG',
    'Bloom Energy': 'BE',
    'SunRun': 'RUN',
    'ChargePoint': 'CHPT',
    'JPMorgan': 'JPM',
    'Bank of America': 'BAC',
    'Goldman Sachs': 'GS',
    'Visa': 'V',
    'Mastercard': 'MA',
    'Morgan Stanley': 'MS',
    'American Airlines': 'AAL',
    'Nike': 'NKE',
    'Coca-Cola': 'KO',
    'Procter & Gamble': 'PG',
    'Starbucks': 'SBUX',
    "McDonald's": 'MCD',
    'Disney': 'DIS',
    'Costco': 'COST',
    'Walmart': 'WMT',
    'Target': 'TGT',
    'Home Depot': 'HD',
    'Best Buy': 'BBY',
    "Lowe's": 'LOW',
    'Kroger': 'KR',
    'United Airlines': 'UAL',
    'Delta Airlines': 'DAL',
    'FedEx': 'FDX',
    'UPS': 'UPS',
    'Boeing': 'BA',
    'Southwest Airlines': 'LUV'
}



class Financial_statement:
    def __init__(self, ticker):
        self.ticker = ticker
        self.stocks = yf.Ticker(ticker)

    def _get_eps_growth(self, df, date_strings_sorted):
        df = df.copy()
        eps_growth = []
        for i in range(len(date_strings_sorted) - 1):
            current_eps = df.loc['Basic EPS', date_strings_sorted[i]]
            next_eps = df.loc['Basic EPS', date_strings_sorted[i + 1]]

            growth = current_eps / next_eps  # growth equation
            eps_growth.append(growth)
        return eps_growth+[np.nan]

    def get_income_sheet(self):

        df_financials = self.stocks.financials
        date_strings = [ts.strftime('%Y-%m-%d') for ts in df_financials.columns.tolist()]
        date_strings_sorted = sorted(date_strings, reverse=True)

        # Compute the ratios
        financial_statement = pd.DataFrame({
            'Gross Margin': df_financials.loc['Gross Profit'] / df_financials.loc['Total Revenue'], # 40% or higher
            'SG & A Expense Margin': df_financials.loc['Selling General And Administration'] / df_financials.loc['Gross Profit'], # 30% or lower
            'R & D Expense Margin': df_financials.loc['Research And Development'] / df_financials.loc['Gross Profit'], # 30% or lower
            'Depreciation Margin': df_financials.loc['Reconciled Depreciation'] / df_financials.loc['Gross Profit'], # 10% or lower
            'Interest Expense Margin': df_financials.loc['Interest Expense'] / df_financials.loc['Operating Income'],# 15% or lower
            'Income Tax Expense': df_financials.loc['Tax Provision'] / df_financials.loc['Pretax Income'], # Current Corporate Tax Rate
            'Profit Margin': df_financials.loc['Net Income'] / df_financials.loc['Total Revenue'], # 20% or higher
            'Earnings Per Share Growth': self._get_eps_growth(df_financials, date_strings_sorted) # > 1 Positive & Growing
        }).T
        return financial_statement


    def get_balance_sheet(self):
        stocks = self.stocks
        df = stocks.balancesheet
        if 'Current Debt' in df.index:
            balance_sheet = pd.DataFrame({
                'Cash > Debt': stocks.balancesheet.loc['Cash And Cash Equivalents'] / stocks.balancesheet.loc[
                    'Current Debt'],  # Rule: cash > debt
                'Adjusted Debt to Equity': stocks.balancesheet.loc['Total Debt'] / (
                            stocks.balancesheet.loc['Total Assets'] - stocks.balancesheet.loc['Total Debt'])
                # Rule : < 0.80
            }).T
        elif "Current Debt And Capital Lease Obligation" in df.index:
            balance_sheet = pd.DataFrame({
                'Cash > Debt': stocks.balancesheet.loc['Cash And Cash Equivalents'] / stocks.balancesheet.loc[
                    'Current Debt And Capital Lease Obligation'],  # Rule: cash > debt
                'Adjusted Debt to Equity': stocks.balancesheet.loc['Total Debt'] / (
                            stocks.balancesheet.loc['Total Assets'] - stocks.balancesheet.loc['Total Debt'])
                # Rule : < 0.80
            }).T

        else:
            balance_sheet = pd.DataFrame({
                'Adjusted Debt to Equity': stocks.balancesheet.loc['Total Debt'] / (
                            stocks.balancesheet.loc['Total Assets'] - stocks.balancesheet.loc['Total Debt'])
                # Rule : < 0.80
            }).T
        return balance_sheet

    def get_cashflow_sheet(self):
        stocks = self.stocks
        cashflow_sheet = pd.DataFrame({
            'CapEx Margin': -stocks.cashflow.loc['Capital Expenditure'] / stocks.cashflow.loc[
                'Net Income From Continuing Operations'],  # Rule: <25%
        }).T
        return cashflow_sheet

    def _converted(self, df):
        df_cleaned = df.dropna(how='all', axis=1)

        try:
            # 尝试将列名转换为整数年份并排序
            sorted_columns = sorted(df_cleaned.columns, key=lambda x: int(x))
            df_cleaned = df_cleaned[sorted_columns]
        except:
            # 如果列名不是纯数字（如包含前缀），直接按字符串排序
            df_cleaned = df_cleaned.sort_index(axis=1)

        return df_cleaned

    def get_indicator(self):
        income_sheet = self.get_income_sheet()
        balance_sheet = self.get_balance_sheet()
        cashflow_sheet = self.get_cashflow_sheet()

        return pd.concat([income_sheet, balance_sheet, cashflow_sheet])


    def get_indicator_by_date(self, all_sheet, date):
        years = all_sheet.columns.strftime('%Y').tolist()
        all_sheet.columns = years
        return dict(all_sheet[str(date)])

class Criteria:
    def __init__(self):
        self.income_criteria = {'Gross Margin': [0.4, 1],
                                'SG & A Expense Margin': [0.3, -1],
                                'R & D Expense Margin': [0.3, -1],
                                'Depreciation Margin': [0.1, -1],
                                'Interest Expense Margin': [0.15, -1],
                                'Income Tax Expense': [0, 0],
                                'Profit Margin': [0.2, 1],
                                'Earnings Per Share Growth': [1, 1]}

        self.balance_criteria = {'Cash > Debt': [1, 1],
                                 'Adjusted Debt to Equity': [0.8, -1]}

        self.cashflow_criteria = {'CapEx Margin': [0.25, -1]}

    def get_standard(self, sheet_type):
        if sheet_type =="Income Sheet":
            return self.income_criteria
        elif sheet_type =="Balance Sheet":
            return self.balance_criteria
        elif sheet_type =="Cashflow Sheet":
            return self.cashflow_criteria
        else:
            print("Dont have the sheet")


indicator_criteria = Criteria()

class Evaluation:
    def __init__(self):
        self.stock = ""

    def _cal_situation_matrix(self, df_sheet, criteria):
        matrix = np.nan_to_num(df_sheet.copy().values.astype(np.float64), nan=float('inf'))
        criteria_matrix = np.array(list(criteria.values()), dtype=np.float64)

        sub_matrix = criteria_matrix[:,0].reshape(criteria_matrix.shape[0],1)
        multiply_matrix = criteria_matrix[:,1].reshape(criteria_matrix.shape[0],1)
        result = (matrix - sub_matrix)* multiply_matrix

      # situation matrix -> tem repot：cal (matrix - subtract_matrix) * multiply_matrix
      # np.where(np.isinf(result), np.nan, (result >= 0).astype(int))
        return result

    def assess_income_sheet(self, income_sheet):
        return self._cal_situation_matrix(income_sheet, indicator_criteria.income_criteria)

    def assess_balance_sheet(self, balance_sheet):
        return self._cal_situation_matrix(balance_sheet, indicator_criteria.balance_criteria)

    def assess_cashflow_sheet(self, cashflow_sheet):
        return self._cal_situation_matrix(cashflow_sheet, indicator_criteria.cashflow_criteria)

def get_sheet_and_eva(company_name):

    # Get ticker for selected company
    ticker = coms2code[company_name]

    # Get financial statements
    financial_statement = Financial_statement(ticker)

    income_sheet = financial_statement.get_income_sheet()
    balance_sheet = financial_statement.get_balance_sheet()
    cashflow_sheet = financial_statement.get_cashflow_sheet()

    # Create a dictionary mapping sheet types to dataframes
    all_sheets = {
        "Income Sheet": income_sheet,
        "Balance Sheet": balance_sheet,
        "Cashflow Sheet": cashflow_sheet
    }

    # Get evaluation sheets

    evaluation = Evaluation()
    income_eva = evaluation.assess_income_sheet(income_sheet)
    balance_eva = evaluation.assess_balance_sheet(balance_sheet)
    cashflow_eva = evaluation.assess_cashflow_sheet(cashflow_sheet)

    # Create a dictionary mapping sheet types to evaluation dataframes
    eva_sheets = {
        "Income Sheet": income_eva,
        "Balance Sheet": balance_eva,
        "Cashflow Sheet": cashflow_eva
    }

    return all_sheets, eva_sheets

def old_get_continues_plot(df_sheet):
    """
    Create a plotly line chart showing trends over time for each metric

    Parameters:
    df_sheet (DataFrame): Financial data sheet

    Returns:
    tuple: Cleaned dataframe and the plotly figure
    """
    # Clean data: remove rows that are all NaN
    df_cleaned = df_sheet.dropna(how='all', axis=1)

    # Create a Plotly figure for interactive visualization
    fig = go.Figure()

    for idx, row in df_cleaned.iterrows():
        if row.dropna().shape[0] > 1:  # At least two points needed to draw a line
            fig.add_trace(go.Scatter(
                x=df_cleaned.columns,
                y=row.values,
                mode='lines+markers',
                name=idx,
                hovertemplate=f"{idx}: %{{y:.2f}}<extra></extra>"
            ))

    fig.update_layout(
        title={
            'text': "High-Level Financial Metrics Trend",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Date",
        yaxis_title="Value",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05
        ),
        hovermode="x unified",
        height=600,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return df_cleaned, fig

def get_continues_plot(df_sheet):
    """
    Create a plotly line chart showing trends over time for each metric

    Parameters:
    df_sheet (DataFrame): Financial data sheet

    Returns:
    tuple: Cleaned dataframe and the plotly figure
    """
    # Clean data: remove rows that are all NaN
    df_cleaned = df_sheet.dropna(how='all', axis=1)

    # ========== adjust col order for 2021-2024 ==========
    try:
        # renmae col
        sorted_columns = sorted(df_cleaned.columns, key=lambda x: int(x))
        df_cleaned = df_cleaned[sorted_columns]
    except:
        #
        df_cleaned = df_cleaned.sort_index(axis=1)
    # ================================================

    # Ensure columns are treated as categorical (for years)
    x_values = df_cleaned.columns.astype(str)  # Convert to strings to prevent numeric interpretation

    # Create a Plotly figure for interactive visualization
    fig = go.Figure()

    for idx, row in df_cleaned.iterrows():
        if row.dropna().shape[0] > 1:  # At least two points needed to draw a line
            fig.add_trace(go.Scatter(
                x=x_values,  # Use the string-converted x-values
                y=row.values,
                mode='lines+markers',
                name=idx,
                hovertemplate=f"{idx}: %{{y:.2f}}<extra></extra>"
            ))

    fig.update_layout(
        title={
            'text': "High-Level Financial Metrics Trend",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Date",
        yaxis_title="Value",
        xaxis=dict(
            type='category',  # Force categorical axis
            tickmode='array',
            tickvals=x_values,  # Explicitly set tick positions
            ticktext=x_values   # And their labels
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05
        ),
        hovermode="x unified",
        height=600,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return df_cleaned, fig

# Analyze continuous trend for the sheet
def analysis_continues(df_cleaned):
    """
    Analyze trends over the recent 3 years

    Parameters:
    df_cleaned (DataFrame): Cleaned financial data

    Returns:
    dict: Information about increasing and decreasing metrics
    """
    recent_cols = df_cleaned.columns[:3]  # recent 3 year
    df_recent = df_cleaned[recent_cols]

    def check_trend(row):
        vals = row.dropna().values
        if len(vals) < 3:  # Need at least 3 points to determine trend
            return None, None

        if np.all(np.diff(vals) > 0):
            growth_rate = (vals[-1] - vals[0]) / abs(vals[0]) if vals[0] != 0 else np.nan
            return "increasing", growth_rate
        elif np.all(np.diff(vals) < 0):
            decline_rate = (vals[0] - vals[-1]) / abs(vals[0]) if vals[0] != 0 else np.nan
            return "decreasing", decline_rate
        else:
            return None, None

    trend_info = {'increasing': {}, 'decreasing': {}}
    for idx, row in df_recent.iterrows():
        if row.isnull().any():
            continue
        trend, rate = check_trend(row)
        if trend == "increasing" and not np.isnan(rate):
            trend_info['increasing'][idx] = rate
        elif trend == "decreasing" and not np.isnan(rate):
            trend_info['decreasing'][idx] = rate
    return trend_info

def plot_financial_metrics(df, threshold_diff, year, sheet_type):
    """
    Plot financial metrics with threshold comparisons using plotly

    Parameters:
    df (DataFrame): Financial data
    threshold_diff (ndarray): Difference from thresholds
    year (str): Selected year for analysis
    sheet_type (str): Type of financial sheet

    Returns:
    tuple: Three plotly figures for different visualizations
    """
    df = df.copy()
    criteria = indicator_criteria.get_standard(sheet_type)

    # Ensure columns are formatted as years
    if not isinstance(df.columns[0], str):
        df.columns = df.columns.strftime('%Y').tolist()

    year_col = f"{year}"
    if year_col not in df.columns:
        st.error(f"Error: There is no data for {year_col}")
        return None, None, None

    metrics = df.index.tolist()
    values = df[year_col].values
    diffs = threshold_diff[:, list(df.columns).index(year_col)]

    # Original values figure
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=metrics,
        y=values,
        marker_color='#003B5C',
        hovertemplate='%{x}: %{y:.2f}<extra></extra>'
    ))
    fig1.update_layout(
        title={
            'text': 'Original Values',
            'font': {'size': 20, 'color': '#003B5C', 'family': 'Arial, sans-serif'}
        },
        yaxis_title="Value",
        xaxis_tickangle=-45,
        height=500,
        margin=dict(l=20, r=20, t=60, b=100)
    )

    # Metrics that should be above threshold
    above_metrics = []
    above_values = []
    above_colors = []
    for i, metric in enumerate(metrics):
        if metric in criteria and criteria[metric][1] == 1 and not np.isnan(diffs[i]):
            above_metrics.append(metric)
            above_values.append(diffs[i])
            above_colors.append('#49DBEB' if diffs[i] >= 0 else 'red')

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=above_metrics,
        y=above_values,
        marker_color=above_colors,
        text=[f"{v:.2f}" for v in above_values],
        textposition='outside',
        hovertemplate='%{x}<br>Difference: %{y:.2f}<extra></extra>'
    ))
    fig2.add_shape(
        type="line",
        x0=-0.5,
        y0=0,
        x1=len(above_metrics) - 0.5,
        y1=0,
        line=dict(color="black", width=2, dash="dash")
    )
    fig2.update_layout(
        title={
            'text': 'Metrics Should ABOVE Threshold',
            'font': {'size': 20, 'color': '#003B5C', 'family': 'Arial, sans-serif'}
        },
        yaxis_title="Difference from Threshold",
        xaxis_tickangle=-45,
        height=500,
        margin=dict(l=20, r=20, t=60, b=100)
    )

    # Metrics that should be below threshold
    below_metrics = []
    below_values = []
    below_colors = []
    for i, metric in enumerate(metrics):
        if metric in criteria and criteria[metric][1] == -1 and not np.isnan(diffs[i]):
            below_metrics.append(metric)
            below_values.append(-diffs[i])
            below_colors.append('#49DBEB' if -diffs[i] <= 0 else 'red')

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=below_metrics,
        y=below_values,
        marker_color=below_colors,
        text=[f"{v:.2f}" for v in below_values],
        textposition='outside',
        hovertemplate='%{x}<br>Difference: %{y:.2f}<extra></extra>'
    ))
    fig3.add_shape(
        type="line",
        x0=-0.5,
        y0=0,
        x1=len(below_metrics) - 0.5,
        y1=0,
        line=dict(color="black", width=2, dash="dash")
    )
    fig3.update_layout(
        title={
            'text': 'Metrics Should BELOW Threshold',
            'font': {'size': 20, 'color': '#003B5C', 'family': 'Arial, sans-serif'}
        },
        yaxis_title="Difference from Threshold",
        xaxis_tickangle=-45,
        height=500,
        margin=dict(l=20, r=20, t=60, b=100)
    )

    return fig1, fig2, fig3