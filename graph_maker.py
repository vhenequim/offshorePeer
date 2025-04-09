import polars as pl
import plotly.express as px
import datetime as dt
import pandas as pd


def graph_per_ticker(df, investment_type, value_type = "value"):
    if investment_type == "BDR":
        df = df.filter(pl.col("investment_type") == "BDR")
    elif investment_type == "Invest. Ext.":
        df = df.filter(pl.col("investment_type") == "Invest. Ext.")
    elif investment_type == None:
        investment_type = "BDR & Invest. Ext."
    
    # Group by month and ticker, sum the values
    monthly_ticker_sum = df.group_by(["month", "ticker", "date_for_sort"]).agg(
        pl.sum(value_type).alias("total_value")
    )

    # Convert to pandas for easier manipulation with plotly
    plot_df = monthly_ticker_sum.to_pandas()

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
        title=f'Investment Values by Ticker and Month - {investment_type}',
        labels={
            "month": "Month",
            "total_value": "Total Value (R$)",
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
        yaxis_title="Total Value (R$)",
        legend_title="Ticker",
        barmode='relative',  # Use relative instead of stack to properly handle negative values
        hovermode="closest"
    )

    # Format hover info
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>" +
                    "Ticker: %{fullData.name}<br>" +
                    "Value: R$ %{y:,.2f}<br>"
    )

    # Return the figure instead of showing it
    return fig

def graph_per_fund(df, fund_name, value_type="value", investment_type=None):
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
    """
    # Filter for the specified fund
    filtered_df = df.filter(pl.col("funds") == fund_name)
    
    # Apply investment type filter if specified
    if investment_type == "BDR":
        filtered_df = filtered_df.filter(pl.col("investment_type") == "BDR")
        type_title = "BDR"
    elif investment_type == "Invest. Ext.":
        filtered_df = filtered_df.filter(pl.col("investment_type") == "Invest. Ext.")
        type_title = "Invest. Ext."
    else:
        type_title = "BDR & Invest. Ext."
    
    # Set value type label for the title
    value_label = "Percentage" if value_type == "percentage" else "Value (R$)"
    
    # Group by month, ticker, and date_for_sort, sum the values
    monthly_ticker_sum = filtered_df.group_by(["month", "ticker", "date_for_sort"]).agg(
        pl.sum(value_type).alias("total_value")
    )
    
    # Convert to pandas for easier manipulation with plotly
    plot_df = monthly_ticker_sum.to_pandas()
    
    # Check if data exists
    if plot_df.empty:
        print(f"No data found for fund: {fund_name} with investment type: {type_title}")
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
        title=f'{fund_name} - {type_title} ({value_label})',
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