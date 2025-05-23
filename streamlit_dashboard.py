import streamlit as st
import streamlit.components.v1 as components
from graph_maker import graph_per_fund, graph_per_ticker, analyze_performance_vs_investment_percentage, plot_average_performance_vs_allocation_corrected
import polars as pl

# Set up the page
st.set_page_config(page_title="Fund Analysis Dashboard", layout="wide")
st.title("Fund Analysis Dashboard")

# Load data
# You'll need to replace this with your actual data loading logic
@st.cache_data
def load_data():
    # Replace this with your actual data loading code
    # For example: df = pl.read_csv("your_data.csv")
    # This is a placeholder
    try:
        df = pl.read_excel("base_peers_mapped.xlsx").filter(pl.col("funds") != "OCEANA")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

if df is None:
    st.error("Failed to load data. Please check your data source.")
    st.stop()

# Create fund selector dropdown
# Load fund names from the dataframe if possible, otherwise use the hardcoded list
if df is not None:
    funds_from_df = df["funds"].unique().sort().to_list()
    funds = funds_from_df if len(funds_from_df) > 0 else sorted([
        '3 ILHAS', 'ABSOLUTE ENDURANCE', 'ABSOLUTO PARTNERS', 'ARX INCOME', 
        'ASTER', 'ATHENA TR', 'ATMOS', 'BOGARI', 'BRASIL CAPITAL 30',
        'CHARLES', 'CLAVE', 'CONSTELLATION', 'DYNAMO', 'ENCORE', 'EQUITAS',
        'INDIE', 'IP', 'JGP', 'KAPITALO', 'KIRON', 'LEBLON AÇÕES',
        'MANTARO', 'NORTE', 'NÚCLEO', 'OPPORTUNITY', 'ORI CAPITAL',
        'RYO', 'SHARP', 'SPX PATRIOT', 'SQUADRA', 'STK', 'STUDIO',
        'TARUÁ', 'TENAX', 'TORK', 'TRUXT', 'VELT', 'VENTOR',
        'VERDE AM', 'VINCI', 'VINLAND'
    ])
else:
    funds = sorted([
        '3 ILHAS', 'ABSOLUTE ENDURANCE', 'ABSOLUTO PARTNERS', 'ARX INCOME', 
        'ASTER', 'ATHENA TR', 'ATMOS', 'BOGARI', 'BRASIL CAPITAL 30',
        'CHARLES', 'CLAVE', 'CONSTELLATION', 'DYNAMO', 'ENCORE', 'EQUITAS',
        'INDIE', 'IP', 'JGP', 'KAPITALO', 'KIRON', 'LEBLON AÇÕES',
        'MANTARO', 'NORTE', 'NÚCLEO', 'OPPORTUNITY', 'ORI CAPITAL',
        'RYO', 'SHARP', 'SPX PATRIOT', 'SQUADRA', 'STK', 'STUDIO',
        'TARUÁ', 'TENAX', 'TORK', 'TRUXT', 'VELT', 'VENTOR',
        'VERDE AM', 'VINCI', 'VINLAND'
    ])

# Add a search box above the dropdown for easier fund selection
fund_search = st.text_input("Search funds")
filtered_funds = [f for f in funds if fund_search.lower() in f.lower()] if fund_search else funds
selected_fund = st.selectbox("Select a Fund", filtered_funds)

# Your existing code remains unchanged, using the same absolute paths as before
st.markdown("---")
st.subheader("Funds Overview w S.A. Companies")
# Create three columns for the graphs
col1, col2, col3 = st.columns(3)

# Display dynamically generated graphs in each column
with col1:
    st.subheader("BDR Distribution")
    fig_bdr = graph_per_fund(df, selected_fund, investment_type="BDR")
    if fig_bdr is not None:
        st.plotly_chart(fig_bdr, use_container_width=True)
    else:
        st.info("No BDR data available for this fund")

with col2:
    st.subheader("Investment Exterior Distribution")
    fig_ext = graph_per_fund(df, selected_fund, investment_type="Invest. Ext.")
    if fig_ext is not None:
        st.plotly_chart(fig_ext, use_container_width=True)
    else:
        st.info("No Investment Exterior data available for this fund")

with col3:
    st.subheader("Combined Distribution")
    fig_combined = graph_per_fund(df, selected_fund)
    if fig_combined is not None:
        st.plotly_chart(fig_combined, use_container_width=True)
    else:
        st.info("No combined data available for this fund")

        st.markdown("---")  # This adds a horizontal line

st.subheader("Funds Overview w/o S.A. Companies")
col4, col5, col6 = st.columns(3)
# Display HTML graphs in each column
with col4:
    st.subheader("BDR Distribution")
    fig_bdr = graph_per_fund(df, selected_fund, investment_type="BDR", exclude_south_america=True)
    if fig_bdr is not None:
        st.plotly_chart(fig_bdr, use_container_width=True)
    else:
        st.info("No BDR data available for this fund")

with col5:
    st.subheader("Investment Exterior Distribution")
    fig_ext = graph_per_fund(df, selected_fund, investment_type="Invest. Ext.", exclude_south_america=True)
    if fig_ext is not None:
        st.plotly_chart(fig_ext, use_container_width=True)
    else:
        st.info("No Investment Exterior data available for this fund")

with col6:
    st.subheader("Combined Distribution")
    fig_combined = graph_per_fund(df, selected_fund, exclude_south_america=True)
    if fig_combined is not None:
        st.plotly_chart(fig_combined, use_container_width=True)
    else:
        st.info("No combined data available for this fund")



st.markdown("---")  # This adds a horizontal line

# For now, we'll skip the South America section as mentioned by the user
# The offshore ratio sections would be replaced similarly with dynamic charts
st.subheader("Offshore Ratio w/ S.A. Companies")
fig_offshore = graph_per_ticker(df, None)  # This will need proper parameters
if fig_offshore is not None:
    st.plotly_chart(fig_offshore, use_container_width=True)
else:
    st.info("No offshore ratio data available")

st.markdown("---")  # This adds a horizontal line

# For now, we'll skip the South America section as mentioned by the user
# The offshore ratio sections would be replaced similarly with dynamic charts
st.subheader("Offshore Ratio w/o S.A. Companies")
fig_offshore = graph_per_ticker(df, None, exclude_south_america=True)  # This will need proper parameters
if fig_offshore is not None:
    st.plotly_chart(fig_offshore, use_container_width=True)
else:
    st.info("No offshore ratio data available")

st.markdown("---")  # This adds a horizontal line

# Performance vs. Investment Percentage Analysis Section
st.subheader("Performance vs. Investment Percentage Analysis")

# Row 1: Combined (Invest. Ext. + BDR)
row1_col1, row1_col2 = st.columns(2)
with row1_col1:
    st.subheader("Combined w/ S.A.")
    fig_combined_sa = plot_average_performance_vs_allocation_corrected(df, investment_type=None, exclude_south_america=False)
    if fig_combined_sa is not None:
        st.plotly_chart(fig_combined_sa, use_container_width=True)
    else:
        st.info("No combined data available (w/ S.A.)")

with row1_col2:
    st.subheader("Combined w/o S.A.")
    fig_combined_no_sa = plot_average_performance_vs_allocation_corrected(df, investment_type=None, exclude_south_america=True)
    if fig_combined_no_sa is not None:
        st.plotly_chart(fig_combined_no_sa, use_container_width=True)
    else:
        st.info("No combined data available (w/o S.A.)")

st.markdown("---") # Separator between rows

# Row 2: Investment Exterior

row2_col1, row2_col2 = st.columns(2)
with row2_col1:
    st.subheader("Invest. Ext. w/ S.A.")
    fig_ext_sa = plot_average_performance_vs_allocation_corrected(df.filter(~pl.col("funds").is_in(["ARX INCOME", "INDIE", "AZ QUEST", "ABSOLUTE ENDURANCE", "ENCORE", "BAHIA", "TARUÁ"])), investment_type="Invest. Ext.", exclude_south_america=False)
    if fig_ext_sa is not None:
        st.plotly_chart(fig_ext_sa, use_container_width=True)
    else:
        st.info("No Invest. Ext. data available (w/ S.A.)")

with row2_col2:
    st.subheader("Invest. Ext. w/o S.A.")
    fig_ext_no_sa = plot_average_performance_vs_allocation_corrected(df, investment_type="Invest. Ext.", exclude_south_america=True)
    if fig_ext_no_sa is not None:
        st.plotly_chart(fig_ext_no_sa, use_container_width=True)
    else:
        st.info("No Invest. Ext. data available (w/o S.A.)")

st.markdown("---") # Separator between rows

# Row 3: BDR
row3_col1, row3_col2 = st.columns(2)
with row3_col1:
    st.subheader("BDR w/ S.A.")
    fig_bdr_sa = plot_average_performance_vs_allocation_corrected(df, investment_type="BDR", exclude_south_america=False)
    if fig_bdr_sa is not None:
        st.plotly_chart(fig_bdr_sa, use_container_width=True)
    else:
        st.info("No BDR data available (w/ S.A.)")

with row3_col2:
    st.subheader("BDR w/o S.A.")
    fig_bdr_no_sa = plot_average_performance_vs_allocation_corrected(df, investment_type="BDR", exclude_south_america=True)
    if fig_bdr_no_sa is not None:
        st.plotly_chart(fig_bdr_no_sa, use_container_width=True)
    else:
        st.info("No BDR data available (w/o S.A.)")


st.markdown("---")  # This adds a horizontal line

# ... existing code ...

# Performance vs. Investment Percentage Analysis Section
st.subheader("Performance vs. Investment Percentage Analysis - Top 10 & Bottom 10 Funds")

# Row 1: Combined (Invest. Ext. + BDR)
row1_col1, row1_col2 = st.columns(2)
with row1_col1:
    st.subheader("Combined w/ S.A.")
    fig_combined_sa = plot_average_performance_vs_allocation_corrected(df, investment_type=None, exclude_south_america=False, num_extremes=10)
    if fig_combined_sa is not None:
        st.plotly_chart(fig_combined_sa, use_container_width=True)
    else:
        st.info("No combined data available (w/ S.A.)")

with row1_col2:
    st.subheader("Combined w/o S.A.")
    fig_combined_no_sa = plot_average_performance_vs_allocation_corrected(df, investment_type=None, exclude_south_america=True, num_extremes=10)
    if fig_combined_no_sa is not None:
        st.plotly_chart(fig_combined_no_sa, use_container_width=True)
    else:
        st.info("No combined data available (w/o S.A.)")


st.markdown("---")  # This adds a horizontal line

# ... existing code ...

# Performance vs. Investment Percentage Analysis Section
st.subheader("Performance vs. Investment Percentage Analysis - Top 5 & Bottom 5 Funds")

# Row 1: Combined (Invest. Ext. + BDR)
row1_col1, row1_col2 = st.columns(2)
with row1_col1:
    st.subheader("Combined w/ S.A.")
    fig_combined_sa = plot_average_performance_vs_allocation_corrected(df, investment_type=None, exclude_south_america=False, num_extremes=5)
    if fig_combined_sa is not None:
        st.plotly_chart(fig_combined_sa, use_container_width=True)
    else:
        st.info("No combined data available (w/ S.A.)")

with row1_col2:
    st.subheader("Combined w/o S.A.")
    fig_combined_no_sa = plot_average_performance_vs_allocation_corrected(df, investment_type=None, exclude_south_america=True, num_extremes=5)
    if fig_combined_no_sa is not None:
        st.plotly_chart(fig_combined_no_sa, use_container_width=True)
    else:
        st.info("No combined data available (w/o S.A.)")



st.markdown("---")  # This adds a horizontal line

# ... existing code ...

# ... existing code ...

# st.subheader("Offshore Ratio w/ S.A. Companies")
# html_content = read_html_file(f"{GRAPHS_DIR}/offshore_ratio_plot_brazil.html")
# components.html(html_content, height=1000)


    
