import polars as pl
import json

df_base_peers_old = pl.read_excel("C:/Users/VHENEQUIM/Documents/Estudo/offshorePeer/basePeers.xlsx")
df_base_peers_new = pl.read_excel("C:/Users/VHENEQUIM/Documents/Estudo/offshorePeer/basePeers_teste.xlsx")
df_base_peers_new.group_by("Ativo").agg(pl.len()).filter(pl.col("Ativo").str.contains("Invest. Ext.")).write_csv("base_peers_new_invest_ext.csv")
df_base_peers_old.group_by("Ativo").agg(pl.len()).filter(pl.col("Ativo").str.contains("Invest. Ext.")).write_csv("base_peers_old_invest_ext.csv")

# Load the mapping file
with open("mapping.json", "r") as f:
    mapping_data = json.load(f)

# Create two lookup dictionaries - one for exact matches and one for partial matching
exact_lookup = {}
all_investments = []  # List of tuples (name, details) for partial matching

for investment in mapping_data["investments"]:
    details = {
        "ticker": investment["ticker"],
        "company_country": investment["company_country"],
        "area_of_work": investment["area_of_work"]
    }
    
    # Store primary name for both exact and partial matching
    primary_name = investment["investment_name"]
    exact_lookup[primary_name] = details
    all_investments.append((primary_name, details))
    
    # Store alternative names if any
    if "alternative_names" in investment:
        for alt_name in investment["alternative_names"]:
            exact_lookup[alt_name] = details
            all_investments.append((alt_name, details))

# Function to map investment details with both exact and partial matching
def map_investment_details(ativo):
    # Remove prefixes
    clean_name = ativo
    if "Invest. Ext. - " in ativo:
        clean_name = ativo.replace("Invest. Ext. - ", "")
    elif "BDR - " in ativo:
        clean_name = ativo.replace("BDR - ", "")
    
    # First try exact match (faster)
    if clean_name in exact_lookup:
        return exact_lookup[clean_name]
    
    # Then try partial matching
    for name, details in all_investments:
        if name in clean_name:  # Partial match - name is contained in clean_name
            return details
    
    # No match found
    return {"ticker": "N/A", "company_country": "N/A", "area_of_work": "N/A"}

# Similar function to check if an asset is in the lookup
def is_in_lookup(ativo):
    clean_name = ativo
    if "Invest. Ext. - " in ativo:
        clean_name = ativo.replace("Invest. Ext. - ", "")
    elif "BDR - " in ativo:
        clean_name = ativo.replace("BDR - ", "")
    
    # Exact match
    if clean_name in exact_lookup:
        return True
    
    # Partial match
    for name, _ in all_investments:
        if name in clean_name:
            return True
    
    return False

def map_investments(df):
    return df.with_columns([
        pl.col("Ativo").map_elements(lambda x: map_investment_details(x).get("ticker", "N/A"), return_dtype=pl.Utf8).alias("ticker"),
        pl.col("Ativo").map_elements(lambda x: map_investment_details(x).get("company_country", "N/A"), return_dtype=pl.Utf8).alias("company_country"),
        pl.col("Ativo").map_elements(lambda x: map_investment_details(x).get("area_of_work", "N/A"), return_dtype=pl.Utf8).alias("area_of_work"),
    ])

def map_investment_type(df):
    return df.with_columns([
        pl.when(pl.col("Ativo").str.contains("Invest. Ext."))
      .then(pl.lit("Invest. Ext."))
      .when(pl.col("Ativo").str.contains("BDR -"))
      .then(pl.lit("BDR"))
      .when(pl.col("Ativo").map_elements(is_in_lookup, return_dtype=pl.Boolean))
      .then(pl.lit("BDR"))
      .otherwise(pl.lit(""))
      .alias("investment_type")
])

# Apply mapping to the DataFrame
df_base_peers_new = (df_base_peers_new
                     .pipe(map_investments)
                     .pipe(map_investment_type))
df_base_peers_new.write_excel("base_peers_new_mapped.xlsx")

df_base_peers_old = (df_base_peers_old
                     .pipe(map_investments)
                     .pipe(map_investment_type))
df_base_peers_old.write_excel("base_peers_old_mapped.xlsx")

print("Old and new mapped")