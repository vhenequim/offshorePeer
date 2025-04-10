import polars as pl
import plotly.express as px
import datetime as dt
import pandas as pd


def graph_per_ticker(df, investment_type=None, value_type="value", exclude_south_america=False):
    """
    Generate a graph showing investment values/percentages grouped by ticker over time.

    Parameters:
    -----------
    df : polars.DataFrame
        The dataframe containing the investment data
    investment_type : str or None, default=None
        Filter for specific investment type ("BDR", "Invest. Ext.", or None for both)
    value_type : str, default="value"
        The column to use for values ("value" for absolute or "percentage" for percentage)
    exclude_south_america : bool, default=False
        If True, excludes investments in companies based in South America.
    """
    filtered_df = df.clone()
    title_parts = []

    # Apply investment type filter
    if investment_type == "BDR":
        filtered_df = filtered_df.filter(pl.col("investment_type") == "BDR")
        type_title = "BDR"
    elif investment_type == "Invest. Ext.":
        filtered_df = filtered_df.filter(pl.col("investment_type") == "Invest. Ext.")
        type_title = "Invest. Ext."
    else:
        # Include both if None or any other value
        filtered_df = filtered_df.filter(
            (pl.col("investment_type") == "BDR") | (pl.col("investment_type") == "Invest. Ext.")
        )
        type_title = "BDR & Invest. Ext."
    title_parts.append(type_title)

    # Apply South America Filter
    if exclude_south_america:
        south_american_countries = [
            "Argentina", "Bolivia", "Brazil", "Chile", "Colombia",
            "Ecuador", "Guyana", "Paraguay", "Peru", "Suriname",
            "Uruguay", "Venezuela"
        ]
        filtered_df = filtered_df.filter(
            ~pl.col("company_country").cast(pl.Utf8).is_in(south_american_countries)
        )
        title_parts.append("(Excluding South America)")

    # Set value type label
    value_label = "Value (R$)" if value_type == "value" else "Percentage (%)"
    title_parts.append(f"({value_label})")

    graph_title = f'Investment {value_label} by Ticker and Month - {" ".join(title_parts)}'

    # Group by month and ticker, sum the values
    agg_expr = pl.sum(value_type)
    if value_type == "percentage":
        agg_expr = (agg_expr * 100) # Multiply by 100 for percentage display

    monthly_ticker_sum = filtered_df.group_by(["month", "ticker", "date_for_sort"]).agg(
        agg_expr.alias("total_value")
    )

    # Convert to pandas for easier manipulation with plotly
    plot_df = monthly_ticker_sum.to_pandas()

    # Check if data exists
    if plot_df.empty:
        print(f"No data found for the specified filters: {investment_type}, Exclude SA: {exclude_south_america}")
        return None

    # Sort by date to ensure chronological order
    plot_df = plot_df.sort_values('date_for_sort')

    # Calculate the total value per ticker across all months
    ticker_totals = plot_df.groupby('ticker')['total_value'].sum().sort_values(ascending=False)

    # Get ordered list of tickers for the color assignment
    ordered_tickers = ticker_totals.index.tolist()

    # Create stacked bar chart with plotly using the custom order for color
    fig = px.bar(
        plot_df,
        x="month",
        y="total_value",
        color="ticker",
        title=graph_title, # Use the constructed title
        labels={
            "month": "Month",
            "total_value": value_label, # Use dynamic label
            "ticker": "Ticker"
        },
        category_orders={
            "month": plot_df["month"].tolist(),  # Preserves chronological order
            "ticker": ordered_tickers,  # Use the custom order based on total values
        },
        height=600,
    )

    # Improve layout
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title=value_label, # Use dynamic label
        legend_title="Ticker",
        barmode='relative',  # Use relative instead of stack to properly handle negative values
        hovermode="closest"
    )

    # Format hover info
    hover_format = "%{y:.2f}%" if value_type == "percentage" else "R$ %{y:,.2f}"
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>" +
                      "Ticker: %{fullData.name}<br>" +
                      f"Value: {hover_format}<br>"
    )

    # Return the figure instead of showing it directly
    return fig



def graph_per_fund(df, fund_name, value_type="percentage", investment_type=None, exclude_south_america=False):
    """
    Generate a graph showing the investment composition of a specific fund over time.
    
    Parameters:
    -----------
    df : polars.DataFrame
        The dataframe containing the investment data
    fund_name : str
        The name of the fund to analyze
    value_type : str, default="value"
        The column to use for values ("value" for absolute or "percentage" for percentage)
    investment_type : str or None, default=None
        Filter for specific investment type ("BDR", "Invest. Ext.", or None for both)
    exclude_south_america : bool, default=False
        If True, excludes investments in companies based in South America.
    """
    # Filter for the specified fund
    filtered_df = df.filter(pl.col("funds") == fund_name)
    title_parts = [fund_name]
    
    # Apply investment type filter if specified
    if investment_type == "BDR":
        filtered_df = filtered_df.filter(pl.col("investment_type") == "BDR")
        type_title = "BDR"
    elif investment_type == "Invest. Ext.":
        filtered_df = filtered_df.filter(pl.col("investment_type") == "Invest. Ext.")
        type_title = "Invest. Ext."
    else:
        type_title = "BDR & Invest. Ext."
    title_parts.append(type_title)
    
    # Apply South America Filter
    if exclude_south_america:
        south_american_countries = [
            "Argentina", "Bolivia", "Brazil", "Chile", "Colombia",
            "Ecuador", "Guyana", "Paraguay", "Peru", "Suriname",
            "Uruguay", "Venezuela"
        ]
        # Ensure the country column is string type for comparison
        filtered_df = filtered_df.filter(
            ~pl.col("company_country").cast(pl.Utf8).is_in(south_american_countries)
        )
        title_parts.append("(Excluding South America)")

    # Set value type label for the title
    value_label = "Value (R$)" if value_type == "value" else "Percentage (%)"
    title_parts.append(f"({value_label})")

    graph_title = " - ".join(title_parts[:2]) # Join fund name and type
    if len(title_parts) > 3: # Add exclusion and value label parts
        graph_title += f" {title_parts[2]}"
    graph_title += f" {title_parts[-1]}" # Add value label part

    # Group by month, ticker, and date_for_sort, sum the values
    # Multiply by 100 only if value_type is percentage
    agg_expr = pl.sum(value_type)
    if value_type == "percentage":
        agg_expr = (agg_expr * 100)
        
    monthly_ticker_sum = filtered_df.group_by(["month", "ticker", "date_for_sort"]).agg(
        agg_expr.alias("total_value")
    )
    
    # Convert to pandas for easier manipulation with plotly
    plot_df = monthly_ticker_sum.to_pandas()
    
    # Check if data exists
    if plot_df.empty:
        print(f"No data found for fund: {fund_name} with specified filters.")
        return None
    
    # Sort by date to ensure chronological order
    plot_df = plot_df.sort_values('date_for_sort')
    
    # Calculate the total value per ticker across all months
    ticker_totals = plot_df.groupby('ticker')['total_value'].sum().sort_values(ascending=False)
    
    # Get ordered list of tickers for the color assignment
    ordered_tickers = ticker_totals.index.tolist()
    
    # Create stacked bar chart with plotly
    fig = px.bar(
        plot_df,
        x="month",
        y="total_value",
        color="ticker",
        title=graph_title, # Use the constructed title
        labels={
            "month": "Month",
            "total_value": value_label,
            "ticker": "Ticker"
        },
        category_orders={
            "month": plot_df["month"].tolist(),  # Preserves chronological order
            "ticker": ordered_tickers,  # Use the custom order based on total values
        },
        height=600,
    )
    
    
    # Improve layout
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title=value_label,
        legend_title="Ticker",
        barmode='relative',  # Use relative instead of stack to properly handle negative values
        hovermode="closest"
    )
    
    # Format hover info
    hover_format = "%{y:.2f}%" if value_type == "percentage" else "R$ %{y:,.2f}"
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>" +
                      "Ticker: %{fullData.name}<br>" +
                      f"Value: {hover_format}<br>"
    )
    
    # Return the figure instead of showing it
    return fig


# Example usage
# graph_per_fund(df_peers, "VINLAND")
# graph_per_fund(df_peers, "Fund Name Here", "percentage")
# graph_per_fund(df_peers, "Fund Name Here", investment_type="BDR")

def graph_percentage_and_rentability(df, fund_name=None, investment_type=None):
    """
    Generate a dual-axis graph showing investment percentages as bars (left axis) 
    and fund rentability as a line (right axis).
    
    Parameters:
    -----------
    df : polars.DataFrame
        The dataframe containing the investment data
    fund_name : str or None, default=None
        The name of the fund to analyze (None for all funds)
    investment_type : str or None, default=None
        Filter for specific investment type ("BDR", "Invest. Ext.", or None for both)
    """
    # Apply filters
    filtered_df = df.clone()
    
    if fund_name is not None:
        filtered_df = filtered_df.filter(pl.col("funds") == fund_name)
        title_prefix = fund_name
    else:
        title_prefix = "All Funds"
    
    if investment_type == "BDR":
        filtered_df = filtered_df.filter(pl.col("investment_type") == "BDR")
        type_title = "BDR"
    elif investment_type == "Invest. Ext.":
        filtered_df = filtered_df.filter(pl.col("investment_type") == "Invest. Ext.")
        type_title = "Invest. Ext."
    else:
        type_title = "BDR & Invest. Ext."
    
    # Group by month, ticker for percentages
    monthly_ticker_pct = filtered_df.group_by(["month", "ticker", "date_for_sort"]).agg(
        pl.sum("percentage").alias("total_percentage")
    )
    
    # Group by month for rentability
    monthly_rentability = filtered_df.group_by(["month", "date_for_sort"]).agg(
        (pl.mean("rentability")*100).alias("avg_rentability")
    ).sort("date_for_sort")
    
    # Convert to pandas
    plot_df_pct = monthly_ticker_pct.to_pandas()
    plot_df_rentability = monthly_rentability.to_pandas()
    
    # Check if data exists
    if plot_df_pct.empty:
        print(f"No data found for: {title_prefix} with investment type: {type_title}")
        return None
    
    # Sort by date
    plot_df_pct = plot_df_pct.sort_values('date_for_sort')
    
    # Calculate the total percentage per ticker across all months
    ticker_totals = plot_df_pct.groupby('ticker')['total_percentage'].sum().sort_values(ascending=False)
    ordered_tickers = ticker_totals.index.tolist()
    
    # Create the bar chart for percentages
    fig = px.bar(
        plot_df_pct,
        x="month",
        y="total_percentage",
        color="ticker",
        title=f'{title_prefix} - {type_title}: Investment Percentage and Rentability',
        labels={
            "month": "Month",
            "total_percentage": "Percentage (%)",
            "ticker": "Ticker"
        },
        category_orders={
            "month": plot_df_pct["month"].tolist(),
            "ticker": ordered_tickers,
        },
        height=600,
    )
    
    # Add the line chart for rentability
    fig.add_trace(
        dict(
            type='scatter',
            x=plot_df_rentability['month'],
            y=plot_df_rentability['avg_rentability'],
            name='Rentability',
            mode='lines+markers',
            line=dict(color='black', width=3),
            marker=dict(size=8, color='black'),
            yaxis='y2'  # Use secondary y-axis
        )
    )
    
    # Set up the secondary y-axis for rentability
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Percentage (%)",
        yaxis2=dict(
            title="Rentability (%)",
            titlefont=dict(color="black"),
            tickfont=dict(color="black"),
            anchor="x",
            overlaying="y",
            side="right"
        ),
        legend_title="Ticker",
        barmode='relative',
        hovermode="closest"
    )
    
    # Format hover info
    fig.update_traces(
        selector=dict(type='bar'),
        hovertemplate="<b>%{x}</b><br>" +
                      "Ticker: %{fullData.name}<br>" +
                      "Percentage: %{y:.2f}%<br>"
    )
    
    # Add hover template for the line
    fig.update_traces(
        selector=dict(name='Rentability'),
        hovertemplate="<b>%{x}</b><br>" +
                      "Rentability: %{y:.2f}%<br>"
    )
    
    # Show the plot
    fig.show()
    
    return None


def graph_percentage_and_patrimony(df, fund_name=None, investment_type=None):
    """
    Generate a dual-axis graph showing investment percentages as bars (left axis) 
    and liquid patrimony as a line (right axis).
    
    Parameters:
    -----------
    df : polars.DataFrame
        The dataframe containing the investment data
    fund_name : str or None, default=None
        The name of the fund to analyze (None for all funds)
    investment_type : str or None, default=None
        Filter for specific investment type ("BDR", "Invest. Ext.", or None for both)
    """
    # Apply filters
    filtered_df = df.clone()
    
    if fund_name is not None:
        filtered_df = filtered_df.filter(pl.col("funds") == fund_name)
        title_prefix = fund_name
    else:
        title_prefix = "All Funds"
    
    if investment_type == "BDR":
        filtered_df = filtered_df.filter(pl.col("investment_type") == "BDR")
        type_title = "BDR"
    elif investment_type == "Invest. Ext.":
        filtered_df = filtered_df.filter(pl.col("investment_type") == "Invest. Ext.")
        type_title = "Invest. Ext."
    else:
        type_title = "BDR & Invest. Ext."
    
    # Group by month, ticker for percentages
    monthly_ticker_pct = filtered_df.group_by(["month", "ticker", "date_for_sort"]).agg(
        pl.sum("percentage").alias("total_percentage")
    )
    
    # Group by month for liquid patrimony
    monthly_patrimony = filtered_df.group_by(["month", "date_for_sort"]).agg(
        pl.mean("liquid_patrimony").alias("avg_patrimony")
    ).sort("date_for_sort")
    
    # Convert to pandas
    plot_df_pct = monthly_ticker_pct.to_pandas()
    plot_df_patrimony = monthly_patrimony.to_pandas()
    
    # Check if data exists
    if plot_df_pct.empty:
        print(f"No data found for: {title_prefix} with investment type: {type_title}")
        return None
    
    # Sort by date
    plot_df_pct = plot_df_pct.sort_values('date_for_sort')
    
    # Calculate the total percentage per ticker across all months
    ticker_totals = plot_df_pct.groupby('ticker')['total_percentage'].sum().sort_values(ascending=False)
    ordered_tickers = ticker_totals.index.tolist()
    
    # Create the bar chart for percentages
    fig = px.bar(
        plot_df_pct,
        x="month",
        y="total_percentage",
        color="ticker",
        title=f'{title_prefix} - {type_title}: Investment Percentage and Liquid Patrimony',
        labels={
            "month": "Month",
            "total_percentage": "Percentage (%)",
            "ticker": "Ticker"
        },
        category_orders={
            "month": plot_df_pct["month"].tolist(),
            "ticker": ordered_tickers,
        },
        height=600,
    )
    
    # Add the line chart for liquid patrimony
    # Instead of adding traces one by one, add a single trace with all points
    fig.add_trace(
        dict(
            type='scatter',
            x=plot_df_patrimony['month'],
            y=plot_df_patrimony['avg_patrimony'],
            name='Liquid Patrimony',
            mode='lines+markers',
            line=dict(color='black', width=3),
            marker=dict(size=8, color='black'),
            yaxis='y2'  # Use secondary y-axis
        )
    )
    
    # Set up the secondary y-axis for liquid patrimony
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Percentage (%)",
        yaxis2=dict(
            title="Liquid Patrimony (R$)",
            titlefont=dict(color="black"),
            tickfont=dict(color="black"),
            anchor="x",
            overlaying="y",
            side="right"
        ),
        legend_title="Ticker",
        barmode='relative',
        hovermode="closest"
    )
    
    # Format hover info
    fig.update_traces(
        selector=dict(type='bar'),
        hovertemplate="<b>%{x}</b><br>" +
                      "Ticker: %{fullData.name}<br>" +
                      "Percentage: %{y:.2f}%<br>"
    )
    
    # Add hover template for the line
    fig.update_traces(
        selector=dict(name='Liquid Patrimony'),
        hovertemplate="<b>%{x}</b><br>" +
                      "Liquid Patrimony: R$ %{y:,.2f}<br>"
    )
    
    # Show the plot
    fig.show()
    
    return None

import polars as pl
import pandas as pd
import plotly.express as px
import numpy as np

def analyze_performance_vs_investment_percentage(
    df,
    investment_type=None,
    exclude_south_america=False,
    top_n_funds=None,
    num_extremes=None # Add parameter for top/bottom N funds
):
    """
    Analyzes the correlation between average fund performance (rentability)
    and the average percentage allocated to specified investment types.

    Allows filtering by investment type, excluding South American companies,
    focusing on the top N performing funds, or focusing on the top and bottom N funds.

    Parameters:
    -----------
    df : polars.DataFrame
        The dataframe containing the investment data.
    investment_type : str or None, default=None
        Filter for specific investment type ("BDR", "Invest. Ext.", or None for both).
    exclude_south_america : bool, default=False
        If True, excludes investments in companies based in South America.
    top_n_funds : int or None, default=None
        If an integer is provided, analyze only the top N funds based on average rentability.
        Mutually exclusive with num_extremes.
    num_extremes : int or None, default=None
        If an integer is provided, analyze the top N and bottom N funds based on average rentability.
        Mutually exclusive with top_n_funds. If None, analyze all eligible funds (or top_n_funds if specified).
    """
    # --- Parameter Validation ---
    if top_n_funds is not None and num_extremes is not None:
        raise ValueError("Parameters 'top_n_funds' and 'num_extremes' are mutually exclusive. Please provide only one.")

    # --- 1. Filter Data ---
    filtered_df = df.clone()
    title_parts = []

    # Apply Investment Type Filter
    if investment_type == "BDR":
        filtered_df = filtered_df.filter(pl.col("investment_type") == "BDR")
        type_title = "BDR"
    elif investment_type == "Invest. Ext.":
        filtered_df = filtered_df.filter(pl.col("investment_type") == "Invest. Ext.")
        type_title = "Invest. Ext."
    else:
        filtered_df = filtered_df.filter(
            (pl.col("investment_type") == "BDR") | (pl.col("investment_type") == "Invest. Ext.")
        )
        type_title = "BDR & Invest. Ext."
    title_parts.append(type_title)

    # Apply South America Filter
    if exclude_south_america:
        south_american_countries = [
            "Argentina", "Bolivia", "Brazil", "Chile", "Colombia",
            "Ecuador", "Guyana", "Paraguay", "Peru", "Suriname",
            "Uruguay", "Venezuela"
        ]
        filtered_df = filtered_df.filter(
            ~pl.col("company_country").cast(pl.Utf8).is_in(south_american_countries)
        )
        title_parts.append("(Excluding South America)")

    analysis_title_base = " ".join(title_parts) # Base title before fund selection

    # --- 2. Calculate Monthly Metrics per Fund ---
    monthly_fund_metrics = filtered_df.group_by(["funds", "month"]).agg(
        pl.sum("percentage").alias("total_monthly_percentage"),
        pl.first("rentability").cast(pl.Float64).alias("monthly_rentability")
    )

    # --- 3. Aggregate Data per Fund ---
    fund_summary = monthly_fund_metrics.group_by("funds").agg(
        (pl.mean("total_monthly_percentage")*100).alias("avg_investment_percentage"),
        (pl.mean("monthly_rentability")*100).alias("avg_rentability")
    ).drop_nulls()

    # --- 3.5 Filter Funds based on Performance ---
    analysis_title = analysis_title_base # Start with base title
    if num_extremes is not None and isinstance(num_extremes, int) and num_extremes > 0:
        if fund_summary.height >= 2 * num_extremes:
            fund_summary_sorted = fund_summary.sort("avg_rentability")
            top_funds = fund_summary_sorted.tail(num_extremes) # Highest rentability
            bottom_funds = fund_summary_sorted.head(num_extremes) # Lowest rentability
            fund_summary = pl.concat([top_funds, bottom_funds])
            title_parts.append(f"(Top & Bottom {num_extremes} Funds)")
            analysis_title = " ".join(title_parts)
        else:
            print(f"Warning: Not enough funds ({fund_summary.height}) to select top and bottom {num_extremes}. Analyzing all available funds.")
            analysis_title = analysis_title_base + " (All Funds)" # Revert title addition

    elif top_n_funds is not None and isinstance(top_n_funds, int) and top_n_funds > 0:
        if fund_summary.height > top_n_funds:
            fund_summary = fund_summary.sort("avg_rentability", descending=True).head(top_n_funds)
            title_parts.append(f"(Top {top_n_funds} Funds)")
            analysis_title = " ".join(title_parts)
        else:
             print(f"Warning: Not enough funds ({fund_summary.height}) to select top {top_n_funds}. Analyzing all available funds.")
             analysis_title = analysis_title_base + " (All Funds)" # Revert title addition
    else:
         analysis_title = analysis_title_base + " (All Funds)" # Explicitly mention all funds if no filter

    # Convert to Pandas for correlation and plotting
    fund_summary_pd = fund_summary.to_pandas()

    if fund_summary_pd.shape[0] < 2:
        print(f"Not enough data points ({fund_summary_pd.shape[0]}) to calculate correlation for {analysis_title}.")
        return None

    # --- 4. Correlation Analysis ---
    fund_summary_pd['avg_investment_percentage'] = pd.to_numeric(fund_summary_pd['avg_investment_percentage'], errors='coerce')
    fund_summary_pd['avg_rentability'] = pd.to_numeric(fund_summary_pd['avg_rentability'], errors='coerce')
    fund_summary_pd = fund_summary_pd.dropna(subset=['avg_investment_percentage', 'avg_rentability'])

    if fund_summary_pd.shape[0] < 2:
        print(f"Not enough numeric data points ({fund_summary_pd.shape[0]}) after cleaning for {analysis_title}.")
        return None

    correlation = fund_summary_pd['avg_investment_percentage'].corr(fund_summary_pd['avg_rentability'])

    print(f"Analysis for: {analysis_title}")
    print(f"Correlation between Average Investment Percentage and Average Rentability: {correlation:.4f}")

    # --- 5. Visualization ---
    plot_title = f'Fund Performance vs. Avg. Investment Percentage ({analysis_title})<br>Correlation: {correlation:.4f}'
    fig = px.scatter(
        fund_summary_pd,
        x="avg_investment_percentage",
        y="avg_rentability",
        title=plot_title,
        labels={
            "avg_investment_percentage": f"Average Monthly Investment Percentage ({type_title}) (%)",
            "avg_rentability": "Average Monthly Rentability (%)"
        },
        hover_name="funds",
        trendline="ols",
        trendline_color_override="red"
    )

    fig.update_traces(
        marker=dict(size=10),
        hovertemplate="<b>%{hovertext}</b><br>" +
                      "Avg. Investment Pct: %{x:.2f}%<br>" +
                      "Avg. Rentability: %{y:.2f}%<extra></extra>"
    )

    fig.update_layout(height=600)

    return fig

# --- Example Usage ---

# Analyze for BDR & Invest. Ext. combined, excluding South America, Top 10 & Bottom 10 Funds
# fig_extremes = analyze_performance_vs_investment_percentage(
#     df_peers,
#     exclude_south_america=True,
#     num_extremes=10
# )
# if fig_extremes: fig_extremes.show()

# # Analyze for BDR only, including South America, Top 5 & Bottom 5 Funds
# fig_extremes_bdr = analyze_performance_vs_investment_percentage(
#     df_peers.filter(pl.col("funds") != "OCEANA"),
#     investment_type="BDR",
#     num_extremes=5
# )
# if fig_extremes_bdr: fig_extremes_bdr.show()

# Analyze for BDR & Invest. Ext. combined, excluding South America, Top 10 Funds (using the other parameter)
# fig_top10 = analyze_performance_vs_investment_percentage(
#     df_peers,
#     exclude_south_america=True,
#     top_n_funds=10
# )
# if fig_top10: fig_top10.show()

# Analyze all funds (default)
# fig_all = analyze_performance_vs_investment_percentage(df_peers)
# if fig_all: fig_all.show()