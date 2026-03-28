import duckdb
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json

# Configuration
DB_PATH = 'ducklake_energy_uk/dev.duckdb'
REPORTS_DIR = 'reports'
os.makedirs(REPORTS_DIR, exist_ok=True)

def run_eda():
    print("🚀 Starting Advanced EDA on 29.2M UK Energy Records...")
    con = duckdb.connect(DB_PATH)

    # 1. Distribution of Current Energy Ratings (A-G)
    print("📊 Generating Question 1: What is the national distribution of energy ratings?")
    df_ratings = con.execute("""
        SELECT 
            energy_rating_current, 
            COUNT(*) as count 
        FROM fct_certificates 
        WHERE energy_rating_current IN ('A', 'B', 'C', 'D', 'E', 'F', 'G')
        GROUP BY 1 
        ORDER BY 1
    """).pl()
    
    fig_ratings = px.bar(
        df_ratings.to_pandas(), 
        x='energy_rating_current', 
        y='count',
        color='energy_rating_current',
        title='National EPC Rating Distribution (A-G)',
        labels={'energy_rating_current': 'EPC Rating', 'count': 'Number of Properties'},
        color_discrete_map={'A':'#008000', 'B':'#00FF00', 'C':'#ADFF2F', 'D':'#FFFF00', 'E':'#FFD700', 'F':'#FF8C00', 'G':'#FF0000'}
    )
    fig_ratings.write_html(os.path.join(REPORTS_DIR, 'rating_distribution.html'))
    print(f"✅ Saved rating_distribution.html")

    # 2. Top 10 Most Efficient Counties
    print("🌍 Generating Question 2: Which counties have the highest average energy efficiency?")
    df_counties = con.execute("""
        SELECT 
            COUNTY, 
            avg_current_efficiency 
        FROM v_regional_energy_performance 
        WHERE COUNTY IS NOT NULL
        ORDER BY avg_current_efficiency DESC 
        LIMIT 10
    """).pl()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_counties.to_pandas(), x='avg_current_efficiency', y='COUNTY', palette='viridis')
    plt.title('Top 10 Most Efficient Counties (Average Score)')
    plt.xlabel('Average Energy Efficiency Score')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'top_10_counties.png'))
    print(f"✅ Saved top_10_counties.png")

    # 3. Efficiency Gap by Construction Age Band
    print("🏗️ Generating Question 3: How does building age impact energy efficiency?")
    df_age = con.execute("""
        SELECT 
            CONSTRUCTION_AGE_BAND, 
            avg_efficiency 
        FROM v_construction_age_analysis 
        GROUP BY 1, 2
        ORDER BY 1
    """).pl()
    
    fig_age = px.line(
        df_age.to_pandas(), 
        x='CONSTRUCTION_AGE_BAND', 
        y='avg_efficiency',
        title='Energy Efficiency Trend by Construction Age Band',
        markers=True
    )
    fig_age.write_html(os.path.join(REPORTS_DIR, 'efficiency_by_age.html'))
    print(f"✅ Saved efficiency_by_age.html")

    # 4. Property Type Impact (CO2 vs Efficiency) - Sampled for performance
    print("🏠 Generating Question 4: Which property types are the biggest carbon emitters?")
    # We sample 100k rows for the scatter plot to keep it interactive
    df_sample = con.execute("""
        SELECT 
            p.PROPERTY_TYPE,
            f.energy_efficiency_current,
            f.co2_emissions_current_tonnes_per_year
        FROM fct_certificates f
        JOIN dim_properties p ON f.property_id = p.property_id
        USING SAMPLE 100000 ROWS
    """).pl()
    
    fig_scatter = px.scatter(
        df_sample.to_pandas(),
        x='energy_efficiency_current',
        y='co2_emissions_current_tonnes_per_year',
        color='PROPERTY_TYPE',
        title='CO2 Emissions vs Efficiency (Sample of 100k properties)',
        opacity=0.5
    )
    fig_scatter.write_html(os.path.join(REPORTS_DIR, 'co2_vs_efficiency.html'))
    print(f"✅ Saved co2_vs_efficiency.html")

    # 5. National Savings Potential (Summary JSON)
    print("💰 Generating Question 5: What is the total potential carbon saving?")
    savings = con.execute("""
        SELECT 
            SUM(co2_emissions_current_tonnes_per_year) as total_current_co2,
            SUM(co2_emissions_potential_tonnes_per_year) as total_potential_co2,
            (total_current_co2 - total_potential_co2) as total_savings_potential
        FROM fct_certificates
    """).fetchone()

    summary = {
        "total_properties_analyzed": int(con.execute("SELECT COUNT(*) FROM fct_certificates").fetchone()[0]),
        "total_current_co2_tonnes": float(savings[0]),
        "total_potential_co2_tonnes": float(savings[1]),
        "total_savings_potential_tonnes": float(savings[2]),
        "percentage_reduction": round(float(savings[2] / savings[0]) * 100, 2)
    }

    with open(os.path.join(REPORTS_DIR, 'national_savings_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"✅ Saved national_savings_summary.json")

    con.close()
    print(f"\n✨ EDA Completed Successfully! Findings saved to /'{REPORTS_DIR}/'")

if __name__ == "__main__":
    run_eda()
