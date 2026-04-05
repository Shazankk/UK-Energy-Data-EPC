"""
UK EPC — Combined Analytics & ML Dashboard Generator
=====================================================
Generates a single self-contained HTML file (reports/dashboard.html) that
combines:
  • Intro & methodology (plain-English explanation of the entire project)
  • 12 EDA sections (national distribution, county efficiency, geographic map, etc.)
  • 4 ML sections (model selection, feature importance, confusion matrix, per-class)
  • Glossary (every technical term defined for a non-technical reader)
  • Key findings summary

This file is deployed directly to GitHub Pages — no server required.
Plotly.js is loaded once from CDN; all charts embed as <div> snippets.

Prerequisites
-------------
  python bulk_load_epc.py                   # raw CSVs → DuckDB
  cd ducklake_energy_uk && dbt run          # build star schema
  python epc_band_tuned.py                  # train + save models/epc_band_lgbm_tuned.pkl

Usage
-----
  source dbt-env/bin/activate
  python generate_dashboard.py
  # → reports/dashboard.html (open in browser or push to GitHub Pages)
"""

import os
import pickle
import warnings
warnings.filterwarnings('ignore')

import duckdb
import polars as pl
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, classification_report, f1_score

# ── Re-use all EDA chart builders from eda_uk_energy.py ─────────────────────
from eda_uk_energy import (
    build_rating_distribution, build_county_efficiency, build_efficiency_by_age,
    build_co2_by_property_type, build_efficiency_density,
    build_retrofit_priority_matrix, build_local_authority_treemap,
    build_postcode_area_heatmap, build_choropleth_map,
    build_fuel_type_analysis, build_annual_trend, build_cost_breakdown,
    get_national_summary, build_summary_section,
    SECTIONS as EDA_SECTIONS, CHART_BUILDERS,
    TEMPLATE, EPC_ORDER, EPC_COLORS, DB_PATH, REPORTS_DIR,
    _embed, _style, _pl,
)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DASHBOARD = os.path.join(REPORTS_DIR, 'dashboard.html')
PKL_PATH  = 'models/epc_band_lgbm_tuned.pkl'
ML_SAMPLE = 20_000   # rows for on-the-fly ML evaluation (fast, representative)

# ─────────────────────────────────────────────────────────────────────────────
# ML data loading (wall type from raw, same pattern as predict_epc_band.py)
# ─────────────────────────────────────────────────────────────────────────────

_WALL_CASE = """
CASE
    WHEN lower(COALESCE(r.walls_description,'')) LIKE '%cavity%'
     AND (lower(COALESCE(r.walls_description,'')) LIKE '%no insulation%'
       OR lower(COALESCE(r.walls_description,'')) LIKE '%uninsulated%')
        THEN 'Cavity - Uninsulated'
    WHEN lower(COALESCE(r.walls_description,'')) LIKE '%cavity%'
     AND lower(COALESCE(r.walls_description,'')) LIKE '%insulation%'
        THEN 'Cavity - Insulated'
    WHEN lower(COALESCE(r.walls_description,'')) LIKE '%solid%'
     AND (lower(COALESCE(r.walls_description,'')) LIKE '%no insulation%'
       OR lower(COALESCE(r.walls_description,'')) LIKE '%uninsulated%')
        THEN 'Solid - Uninsulated'
    WHEN lower(COALESCE(r.walls_description,'')) LIKE '%solid%'
     AND lower(COALESCE(r.walls_description,'')) LIKE '%insulation%'
        THEN 'Solid - Insulated'
    WHEN lower(COALESCE(r.walls_description,'')) LIKE '%timber%'
        THEN 'Timber Frame'
    WHEN lower(COALESCE(r.walls_description,'')) LIKE '%system%'
        THEN 'System Build'
    ELSE 'Other/Unknown'
END
"""


def _load_ml_sample(con: duckdb.DuckDBPyConnection, n: int = ML_SAMPLE):
    """Load a stratified sample for ML evaluation.  Returns (X_pd, y_pd)."""
    query = f"""
    SELECT * FROM (
        SELECT
            s.energy_rating_current                    AS target,
            CAST(s.total_floor_area_sqm AS DOUBLE)     AS total_floor_area_sqm,
            s.construction_age_band,
            s.main_fuel,
            {_WALL_CASE}                               AS wall_type
        FROM stg_epc__domestic s
        JOIN raw.epc_domestic   r ON s.certificate_id = r.lmk_key
        WHERE s.energy_rating_current IS NOT NULL
          AND s.total_floor_area_sqm  IS NOT NULL
          AND CAST(s.total_floor_area_sqm AS DOUBLE) > 0
          AND s.construction_age_band NOT IN ('Unknown')
          AND s.main_fuel             NOT IN ('Other/Unknown')
    ) USING SAMPLE {n} ROWS (reservoir, 99)
    """
    df = _pl(con.query(query))
    return df.drop('target').to_pandas(), df['target'].to_pandas()


# ─────────────────────────────────────────────────────────────────────────────
# ML chart builders
# ─────────────────────────────────────────────────────────────────────────────

def build_ml_model_comparison() -> go.Figure:
    """
    Bar chart comparing 3-fold CV macro F1 across the three candidate models.
    Numbers are from the epc_band_tuned.py run (60k tuning set, seed=42).
    LightGBM is highlighted in green as the selected model.
    """
    models  = ['HistGradientBoosting\n(sklearn)', 'Random Forest\n(baseline)', 'LightGBM\n(selected)']
    f1      = [0.3066, 0.3242, 0.3424]
    acc     = [0.4043, 0.4199, 0.5292]
    f1_std  = [0.0049, 0.0070, 0.0076]
    colors  = ['#4a9eff', '#4a9eff', '#19b459']

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=models, y=f1,
        marker_color=colors,
        error_y=dict(type='data', array=f1_std, visible=True, color='#8b949e'),
        name='Macro F1',
        hovertemplate='<b>%{x}</b><br>Macro F1: %{y:.4f}<extra></extra>',
        text=[f'{v:.4f}' for v in f1],
        textposition='outside',
    ))
    fig.add_hline(y=1/7, line_dash='dot', line_color='#f85149',
                  annotation_text='Random guessing (0.143)', annotation_position='top left')
    fig.update_layout(
        template=TEMPLATE,
        title=dict(text='Model Comparison — 3-Fold Cross-Validation Macro F1<br>'
                        '<sup>Green = selected model | Error bars = std across folds | '
                        'Red line = random-guessing baseline</sup>',
                   font=dict(size=16)),
        yaxis=dict(title='Macro F1 Score (higher = better)', range=[0, 0.45]),
        xaxis_title='Algorithm',
        showlegend=False,
        width=820, height=480,
    )
    return fig


def build_ml_feature_importance() -> go.Figure:
    """
    Dual-panel importance chart using LightGBM's native gain and split metrics.
    Gain = total information gained by using a feature across all splits.
    Split = how many times a feature is used as a decision point.
    Loaded from the saved pkl — no re-training needed.
    """
    if not os.path.exists(PKL_PATH):
        return go.Figure().update_layout(title='Model not found — run epc_band_tuned.py first')

    with open(PKL_PATH, 'rb') as f:
        artefact = pickle.load(f)

    pipe  = artefact['pipeline']
    names = ['Floor Area (m²)', 'Construction Age', 'Heating Fuel', 'Wall Type']
    clf   = pipe.named_steps['clf']

    gain  = clf.booster_.feature_importance(importance_type='gain')
    split = clf.booster_.feature_importance(importance_type='split')

    gain_pct  = gain  / gain.sum()  * 100
    split_pct = split / split.sum() * 100

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=[
                            'Importance by Gain<br><sup>Total information gained from each feature</sup>',
                            'Importance by Split Count<br><sup>How often each feature is used as a decision point</sup>',
                        ])
    for col, vals, pct in [(1, gain, gain_pct), (2, split, split_pct)]:
        order = np.argsort(vals)
        fig.add_trace(go.Bar(
            x=vals[order],
            y=[names[i] for i in order],
            orientation='h',
            marker_color=['#19b459' if i == order[-1] else '#4a9eff' for i in range(len(names))],
            text=[f'{p:.1f}%' for p in pct[order]],
            textposition='inside',
            showlegend=False,
            hovertemplate='%{y}<br>%{x:,.0f} (%{text})<extra></extra>',
        ), row=1, col=col)

    fig.update_layout(
        template=TEMPLATE,
        title=dict(text='What Drives the EPC Band? — LightGBM Feature Importances',
                   font=dict(size=16)),
        height=380, width=900,
    )
    return fig


def build_ml_confusion_matrix(con: duckdb.DuckDBPyConnection) -> go.Figure:
    """
    Row-normalised confusion matrix: for each actual band (rows), what % did
    the model predict as each band (columns)?  Perfect = 100% on the diagonal.
    Uses a fresh 20k sample from DuckDB so the chart is always based on
    real predictions from the saved pipeline.
    """
    if not os.path.exists(PKL_PATH):
        return go.Figure().update_layout(title='Model not found — run epc_band_tuned.py first')

    with open(PKL_PATH, 'rb') as f:
        pipe = pickle.load(f)['pipeline']

    X, y     = _load_ml_sample(con)
    y_pred   = pipe.predict(X)
    bands    = sorted(set(y) | set(y_pred), key=lambda b: EPC_ORDER.index(b))
    cm       = confusion_matrix(y, y_pred, labels=bands)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct   = np.where(row_sums > 0, cm.astype(float) / row_sums * 100, 0.0)

    # Bold the diagonal (correct predictions) in the text
    text = []
    for i, row in enumerate(cm_pct):
        text.append([f'<b>{v:.0f}%</b>' if j == i else f'{v:.0f}%'
                     for j, v in enumerate(row)])

    fig = go.Figure(go.Heatmap(
        z=cm_pct,
        x=[f'Predicted {b}' for b in bands],
        y=[f'Actual {b}'    for b in bands],
        colorscale='RdYlGn', reversescale=True,
        zmin=0, zmax=100,
        text=text, texttemplate='%{text}',
        hovertemplate='Actual %{y} → Predicted %{x}<br>%{text}<extra></extra>',
        colorbar=dict(title='% of<br>actual band'),
    ))
    fig.update_layout(
        template=TEMPLATE,
        title=dict(text='Prediction Accuracy — Confusion Matrix<br>'
                        '<sup>Each row = one actual band. Percentages show where predictions land. '
                        'Bold diagonal = correct predictions.</sup>',
                   font=dict(size=16)),
        xaxis_title='Predicted Band',
        yaxis_title='Actual Band',
        width=720, height=600,
    )
    return fig


def build_ml_per_class_metrics(con: duckdb.DuckDBPyConnection) -> go.Figure:
    """
    Three-line chart: precision, recall, F1 per band.
    Bars show sample count (how many properties of each band were tested).
    Uses the same 20k sample as the confusion matrix.
    """
    if not os.path.exists(PKL_PATH):
        return go.Figure().update_layout(title='Model not found — run epc_band_tuned.py first')

    with open(PKL_PATH, 'rb') as f:
        pipe = pickle.load(f)['pipeline']

    X, y   = _load_ml_sample(con)
    y_pred = pipe.predict(X)
    bands  = sorted(set(y) | set(y_pred), key=lambda b: EPC_ORDER.index(b))
    report = classification_report(y, y_pred, labels=bands, output_dict=True)

    f1_s   = [report.get(b, {}).get('f1-score',  0.0) for b in bands]
    prec   = [report.get(b, {}).get('precision', 0.0) for b in bands]
    rec    = [report.get(b, {}).get('recall',    0.0) for b in bands]
    supp   = [report.get(b, {}).get('support',   0)   for b in bands]
    colors = [EPC_COLORS.get(b, '#888') for b in bands]

    fig = make_subplots(specs=[[{'secondary_y': True}]])
    for vals, dash, name in [
        (f1_s, 'solid', 'F1-Score'),
        (prec, 'dot',   'Precision'),
        (rec,  'dash',  'Recall'),
    ]:
        fig.add_trace(go.Scatter(
            x=bands, y=vals, name=name, mode='lines+markers',
            line=dict(dash=dash, width=2.5),
            hovertemplate=f'Band %{{x}}<br>{name}: %{{y:.3f}}<extra></extra>',
        ), secondary_y=False)
    fig.add_trace(go.Bar(
        x=bands, y=supp, name='Test samples', marker_color=colors, opacity=0.3,
        hovertemplate='Band %{x}<br>Samples: %{y:,}<extra></extra>',
    ), secondary_y=True)

    fig.update_layout(
        template=TEMPLATE,
        title=dict(text='Prediction Quality by EPC Band<br>'
                        '<sup>Lines = scoring metrics (left axis) · Bars = test sample count (right axis)</sup>',
                   font=dict(size=16)),
        xaxis_title='EPC Band',
        legend=dict(x=0.01, y=0.99),
        width=820, height=500,
    )
    fig.update_yaxes(title_text='Score (0 = worst, 1 = best)', range=[0, 1.05], secondary_y=False)
    fig.update_yaxes(title_text='Number of Test Samples', secondary_y=True)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Static HTML sections (intro, methodology, glossary)
# ─────────────────────────────────────────────────────────────────────────────

def _build_intro_html(summary: dict) -> str:
    co2_mt  = round(summary['co2_saving'] / 1e6, 1)
    cars_eq = round(co2_mt * 1e6 / 2.1 / 1e6, 1)
    return f"""
    <section id="about">
      <h2>About This Dashboard</h2>
      <div class="chart-desc">
        This dashboard analyses <strong>{summary['total']:,} Energy Performance Certificates</strong>
        covering virtually every home sold or rented in England &amp; Wales since 2008.
        It was built to answer one question: <em>where is the UK's housing stock falling short on
        energy efficiency, and where should investment go to fix it?</em>
      </div>
      <div class="info-grid">
        <div class="info-card">
          <h3>🏠 What is an EPC?</h3>
          <p>An <strong>Energy Performance Certificate</strong> is a legal document required in the UK
          whenever a property is built, sold, or rented. A trained assessor visits the property and
          measures its physical characteristics — insulation, windows, heating system, wall construction —
          then produces a score from 1 to 100 (called a <strong>SAP score</strong>) and a letter band
          from <strong>A (most efficient)</strong> to <strong>G (least efficient)</strong>.</p>
          <p style="margin-top:10px">Think of it like the energy label on a washing machine, but for an
          entire house. The UK government has set a legal target for all homes to reach at least
          <strong>Band C by 2035</strong> — a prerequisite for the country's Net Zero 2050 commitment.
          Currently, <strong>{summary['pct_below_c']}% of properties fail this target</strong>.</p>
        </div>
        <div class="info-card">
          <h3>⚡ Why Does This Matter?</h3>
          <p>Buildings are responsible for roughly <strong>20% of the UK's total CO₂ emissions</strong>.
          An energy-inefficient home costs more to heat, produces more pollution, and pushes more
          households into <em>fuel poverty</em> — spending over 10% of income on energy.</p>
          <p style="margin-top:10px">The data shows that if every property in this dataset reached
          its EPC-assessed potential through retrofitting, the UK could save
          <strong>{co2_mt} million tonnes of CO₂ per year</strong> — equivalent to taking
          <strong>{cars_eq} million cars off the road</strong>. This dashboard maps exactly where
          those savings are concentrated.</p>
        </div>
        <div class="info-card">
          <h3>🔧 How Was This Built?</h3>
          <p>The raw data (29.2 million CSV records, ~50 GB) was processed using a modern data
          engineering stack designed to run on a single laptop:</p>
          <ul style="margin-top:8px;padding-left:18px;color:var(--muted);font-size:13px">
            <li><strong>DuckDB</strong> — an analytical database that queries 29M rows in under
            a second without needing a server</li>
            <li><strong>dbt</strong> — organises the transformation pipeline into tested, documented
            SQL models with automatic dependency ordering</li>
            <li><strong>Polars</strong> — a high-performance Python library for data processing,
            10–100× faster than the standard Pandas library</li>
            <li><strong>Plotly</strong> — interactive charts embedded directly in this HTML file;
            no server needed to view them</li>
          </ul>
        </div>
        <div class="info-card">
          <h3>📊 What Does This Dashboard Contain?</h3>
          <ul style="margin-top:8px;padding-left:18px;color:var(--muted);font-size:13px">
            <li><strong>Sections 1–12</strong>: Exploratory analysis — distribution, geography,
            construction era, fuel type, cost breakdowns</li>
            <li><strong>Sections 13–16</strong>: Machine learning — predicting EPC bands from
            physical features, understanding which factors matter most</li>
            <li><strong>Key Findings</strong>: Policy-level takeaways across all analyses</li>
            <li><strong>Glossary</strong>: Plain-English definitions for every technical term</li>
          </ul>
          <p style="margin-top:10px;font-size:12px;color:var(--muted)">
          All charts are interactive — hover for details, click legends to toggle series,
          scroll to zoom on maps.</p>
        </div>
      </div>
    </section>"""


def _build_ml_intro_html() -> str:
    return """
    <section id="ml-intro">
      <h2>Machine Learning: Predicting EPC Bands from Physical Features</h2>
      <div class="chart-desc">
        The following four sections use machine learning to answer a specific question:
        <em>given only four physical measurements about a property — its size, age, heating fuel,
        and wall type — how accurately can we predict its EPC band?</em>
      </div>
      <div class="info-grid">
        <div class="info-card">
          <h3>🤖 What Is Machine Learning?</h3>
          <p>Machine learning means teaching a computer to find patterns in historical data and use
          those patterns to make predictions on new data. Instead of programming rules by hand
          ("if a house was built before 1900 AND has solid walls AND uses solid fuel, it is probably
          Band G"), the model learns these rules automatically by studying millions of labelled
          examples where the answer is already known.</p>
          <p style="margin-top:10px">Once trained, it can estimate the EPC band for any property
          instantly — useful for screening large property portfolios or estimating the rating for
          properties that have never had a formal EPC assessment.</p>
        </div>
        <div class="info-card">
          <h3>📐 The Four Features Used</h3>
          <ul style="margin-top:8px;padding-left:18px;color:var(--muted);font-size:13px">
            <li><strong>Floor area (m²)</strong> — larger homes lose more heat through more surface
            area; directly drives heating energy consumption</li>
            <li><strong>Construction age band</strong> — each decade of building regulations
            produced measurably more efficient homes; Pre-1900 = worst, 2012+ = best</li>
            <li><strong>Heating fuel type</strong> — mains gas, electricity, oil, LPG, solid fuel,
            or biomass; fuel type directly determines CO₂ intensity and cost</li>
            <li><strong>Wall construction type</strong> — cavity insulated, cavity uninsulated,
            solid insulated, solid uninsulated, timber frame, or system build; walls are
            the primary heat loss pathway</li>
          </ul>
        </div>
        <div class="info-card">
          <h3>📏 How We Measure Success</h3>
          <p>We use <strong>Macro F1 Score</strong> rather than simple accuracy. Here is why:</p>
          <p style="margin-top:8px">The UK housing stock is heavily skewed — Band D alone is 40%
          of all properties. A model that always predicts "D" scores 40% accuracy but is completely
          useless. Macro F1 averages the prediction quality equally across <em>all seven bands</em>,
          so a model must perform well on rare bands (A, G) to score well.</p>
          <p style="margin-top:8px">Random guessing scores Macro F1 = <strong>0.14</strong> (1/7 classes).
          Our final model scores <strong>0.35</strong> — 2.5× better than guessing.</p>
        </div>
        <div class="info-card">
          <h3>⚠️ The Feature Ceiling</h3>
          <p>The most important finding: after running 30 automated tuning trials, the best
          hyperparameters improved the model by only <strong>+0.0001 F1</strong> over defaults.
          This is not a failure — it proves the model has extracted essentially all available
          signal from these four features.</p>
          <p style="margin-top:8px">The bottleneck is <em>features, not algorithm</em>. EPC bands
          also depend on roof insulation, window glazing, boiler type, floor insulation, and
          hot water system — none of which are in our feature set. Adding those would improve
          performance far more than any amount of algorithm tuning.</p>
        </div>
      </div>
    </section>"""


def _build_glossary_html() -> str:
    terms = [
        ("SAP Score",
         "Standard Assessment Procedure. A number from 1 (very inefficient) to 100 (very efficient) "
         "that measures how energy-efficient a property is — like a fuel-economy rating for a building. "
         "The UK average is around 65. A score of 69+ qualifies as Band C; below 21 is Band G."),
        ("EPC Band (A–G)",
         "The letter grade derived from the SAP score. <strong>A</strong> (92+) is the most efficient; "
         "<strong>G</strong> (1–20) is the least. Most UK homes are rated <strong>D</strong> (55–68). "
         "The government's 2035 target requires all homes to reach at least Band C (69+)."),
        ("CO₂ (Carbon Dioxide)",
         "The primary greenhouse gas released when burning fossil fuels for heat. Each EPC certificate "
         "records a property's estimated annual CO₂ output in <strong>tonnes per year</strong>. "
         "Reducing these figures is the core goal of the UK's Net Zero policy."),
        ("Net Zero",
         "The UK government's legally binding commitment to reduce national greenhouse gas emissions "
         "to zero (or offset any remainder) by <strong>2050</strong>. Buildings account for ~20% of "
         "UK emissions, making EPC improvement central to this target."),
        ("UPRN",
         "Unique Property Reference Number. A permanent, government-assigned identifier for every "
         "addressable location in Great Britain — think of it as a National Insurance number for "
         "buildings. UPRNs let us track the same property across multiple EPC inspections."),
        ("LAD (Local Authority District)",
         "The administrative tier below county level — roughly equivalent to a borough or district "
         "council. England and Wales have 362 LADs. The geographic map colours each LAD by "
         "its average SAP score."),
        ("Fuel Type",
         "The primary energy source used to heat a property. Common UK types: "
         "<strong>Mains Gas</strong> (most common), <strong>Electricity</strong>, "
         "<strong>Heating Oil</strong> (rural, no gas network), "
         "<strong>LPG</strong> (bottled gas), <strong>Solid Fuel</strong> (coal), "
         "<strong>Biomass</strong> (wood pellets/chips)."),
        ("Efficiency Gap",
         "The difference between a property's current SAP score and its potential SAP score "
         "if all recommended upgrades were installed. A gap of 30+ points means major improvement "
         "is achievable with cost-effective retrofitting."),
        ("Retrofit",
         "Upgrading an existing building's energy efficiency after it was originally constructed — "
         "e.g. adding loft insulation, cavity wall insulation, replacing a boiler with a heat pump, "
         "or installing double glazing. As opposed to building a new efficient home from scratch."),
        ("Median",
         "The middle value when all data points are sorted from lowest to highest. Unlike a mean "
         "(average), the median is not dragged by extreme outliers. In box plots, the horizontal "
         "line inside the box is the median."),
        ("Q1 / Q3 (Quartiles)",
         "<strong>Q1</strong> is the 25th percentile — 25% of properties score below this value. "
         "<strong>Q3</strong> is the 75th percentile — 75% score below. The box in a box plot "
         "spans Q1 to Q3, showing where the middle 50% of the data sits."),
        ("IQR (Interquartile Range)",
         "Q3 minus Q1 — the span containing the middle 50% of values. Box plots use IQR to "
         "show where most data clusters, ignoring extreme outliers shown as individual dots."),
        ("Box Plot",
         "A chart that summarises a distribution in five numbers: minimum, Q1, median, Q3, maximum. "
         "The box = middle 50%; the line = median; whiskers = full spread. Ideal for comparing "
         "distributions across many groups without plotting millions of overlapping individual points."),
        ("Macro F1 Score",
         "A machine learning performance metric that averages the F1 score equally across all "
         "classes, regardless of how many samples each class has. Used here instead of accuracy "
         "because the UK housing stock is heavily skewed toward Band D — accuracy alone would "
         "reward models that always predict D."),
        ("Confusion Matrix",
         "A grid showing how a machine learning model's predictions compare to the actual values. "
         "Rows = actual class; columns = predicted class. Values on the diagonal (top-left to "
         "bottom-right) are correct predictions; everything else is a mistake. A perfect model "
         "has 100% on every diagonal cell."),
        ("Feature Importance",
         "In machine learning, a measure of how much each input variable (feature) contributes "
         "to the model's predictions. A high importance score means the model relies heavily on "
         "that variable to make its decisions. Used to understand <em>why</em> a model predicts "
         "what it does, not just <em>what</em> it predicts."),
        ("LightGBM",
         "Light Gradient Boosting Machine — the machine learning algorithm selected for EPC band "
         "prediction. It builds a sequence of decision trees where each tree corrects the errors "
         "of all previous trees. Particularly effective for structured/tabular data like EPC "
         "certificates, and faster than comparable algorithms."),
        ("Gradient Boosting",
         "A family of machine learning algorithms that build models sequentially: each new model "
         "focuses on the examples that previous models got wrong. This 'boosting' process reduces "
         "bias (systematic error) more effectively than training many independent models in parallel."),
        ("ELT (Extract, Load, Transform)",
         "The modern data pipeline pattern used in this project: raw CSV files are loaded "
         "as-is into DuckDB first (Extract + Load), then transformed into clean tables "
         "using SQL inside the database (Transform). The opposite of the older ETL pattern "
         "where transformation happened before loading."),
        ("Medallion Architecture",
         "A data organisation pattern with three layers: "
         "<strong>Bronze</strong> (raw, unchanged), "
         "<strong>Silver</strong> (cleaned and validated), "
         "<strong>Gold</strong> (structured for analysis). Each layer builds on the previous, "
         "providing an audit trail and making debugging easier."),
        ("Star Schema",
         "A database design that organises data into one central fact table (containing metrics "
         "like CO₂ and costs) surrounded by dimension tables (containing descriptions like property "
         "type and location). Enables fast analytical queries because the database only reads the "
         "specific columns needed."),
        ("Surrogate Key",
         "A system-generated ID created by hashing input columns (e.g. postcode → 32-character "
         "hex string). Unlike natural keys (UPRN, postcode), surrogate keys are guaranteed to be "
         "consistent across data reloads — the same input always produces the same ID."),
    ]

    cards = ""
    for term, definition in terms:
        cards += f"""
        <div class="glossary-card">
          <dt>{term}</dt>
          <dd>{definition}</dd>
        </div>"""

    return f"""
    <section id="glossary">
      <h2>Glossary — Key Terms Explained</h2>
      <div class="chart-desc">
        Plain-English definitions for every technical term used in this dashboard.
        No prior knowledge of data science, energy policy, or statistics required.
      </div>
      <dl class="glossary-grid">{cards}</dl>
    </section>"""


# ─────────────────────────────────────────────────────────────────────────────
# ML section metadata (descriptions shown under each chart title)
# ─────────────────────────────────────────────────────────────────────────────

ML_SECTIONS = [
    ("ml-comparison",
     "13. Model Selection: Which Algorithm Works Best?",
     "<strong>What it shows:</strong> Three machine learning algorithms compared on the same data. "
     "The metric is <em>Macro F1 Score</em> — the average prediction quality across all seven EPC "
     "bands equally. A score of 0.14 = random guessing; higher is better. "
     "<br><br><strong>Why LightGBM won:</strong> It scored highest on both Macro F1 (0.34) and "
     "accuracy (53%). It handles categorical features like fuel type and wall type natively — "
     "rather than needing them split into many separate yes/no columns. It also uses a "
     "<em>gradient boosting</em> strategy (each tree learns from the previous tree's mistakes) "
     "which is more effective than Random Forest's strategy of building many independent trees "
     "when you have only four input features. "
     "<br><br><strong>Why not test more models?</strong> After running 30 automated hyperparameter "
     "tuning trials, the improvement was +0.0001 — effectively zero. This proves the bottleneck "
     "is the <em>features themselves</em>, not the algorithm. More models would yield the same ceiling."),

    ("ml-importance",
     "14. What Physical Features Drive EPC Rating?",
     "<strong>What it shows:</strong> How much each of the four input features contributed to the "
     "model's predictions. The left chart shows <em>total information gained</em> from each feature "
     "(more reliable). The right shows <em>how often</em> each feature was used as a decision point. "
     "<br><br><strong>Gain importance</strong> is the preferred metric: it directly measures how "
     "much each feature reduced prediction uncertainty. A feature used rarely but in critical "
     "high-gain splits scores higher than one used frequently on low-quality splits. "
     "<br><br><strong>What to expect:</strong> Construction age and wall type are likely the top "
     "two features — they directly determine the building fabric's heat loss characteristics. "
     "Fuel type has a strong but more uniform effect (all gas homes score similarly, all solid-fuel "
     "homes score similarly). Floor area provides marginal continuous refinement."),

    ("ml-confusion",
     "15. How Accurate Are the Predictions? — Confusion Matrix",
     "<strong>What it shows:</strong> For each actual EPC band (rows), what percentage did the "
     "model predict as each band (columns). Bold numbers on the diagonal = correct predictions. "
     "Numbers off the diagonal = mistakes. <strong>A perfect model would show 100% on every "
     "diagonal cell and 0% everywhere else.</strong> "
     "<br><br><strong>How to read it:</strong> Look at the row for 'Actual D'. The bold number on "
     "the diagonal is the % of D-rated properties the model correctly identified as D. The numbers "
     "in other columns show where the rest ended up — e.g. some D properties were predicted as C "
     "(slightly over-estimated) or E (slightly under-estimated). "
     "<br><br><strong>Why aren't the extremes (A, G) predicted well?</strong> Band A is extremely "
     "rare (0.34% of properties) — the model sees very few examples. Band G shares physical "
     "characteristics with F (both are old, uninsulated, using inefficient fuels), so the "
     "boundary is blurry. The model performs best in the middle (C, D) where most properties sit "
     "and the physical patterns are clearest."),

    ("ml-perclass",
     "16. Prediction Quality by EPC Band",
     "<strong>What it shows:</strong> Three prediction quality metrics for each band: "
     "<em>Precision</em> (of all properties predicted as band X, what % actually are X?), "
     "<em>Recall</em> (of all actual band X properties, what % did the model correctly find?), "
     "and <em>F1-Score</em> (a balance of the two). Bars show the number of test samples per band. "
     "<br><br><strong>Precision vs Recall trade-off:</strong> High precision + low recall means the "
     "model is cautious — it only calls something band G when it is very sure, but misses many G "
     "properties. High recall + low precision means it casts a wide net but includes false positives. "
     "F1 balances both. "
     "<br><br><strong>Key insight:</strong> The model performs well on C and D (the majority bands) "
     "and poorly on A (too rare) and E/F/G (physically similar to each other). This pattern is "
     "expected when using only four structural features — the features distinguish the <em>extremes</em> "
     "well (pre-1900 solid-wall solid-fuel = almost certainly E–G; 2012 insulated cavity gas = "
     "almost certainly B–C) but struggle with the mid-range boundary cases."),
]


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard assembler
# ─────────────────────────────────────────────────────────────────────────────

def _kpi_card(label: str, value: str, sub: str = '', color: str = '#4f8ef7') -> str:
    return f"""
    <div class="kpi-card">
      <div class="kpi-value" style="color:{color}">{value}</div>
      <div class="kpi-label">{label}</div>
      {f'<div class="kpi-sub">{sub}</div>' if sub else ''}
    </div>"""


def build_combined_dashboard(
    eda_figures: list[go.Figure],
    ml_figures:  list[go.Figure],
    summary:     dict,
) -> None:
    """Assemble and write the combined HTML dashboard to reports/dashboard.html."""

    def _fmt_m(n):
        return f"{n/1e6:.1f}M" if n >= 1e6 else f"{n:,.0f}"

    # ── KPI strip ────────────────────────────────────────────────────────────
    kpi_html = "".join([
        _kpi_card("Properties Analysed",   _fmt_m(summary['total']),         "EPC certificates"),
        _kpi_card("Avg Current SAP Score",  str(summary['avg_eff_current']),  f"potential: {summary['avg_eff_potential']}"),
        _kpi_card("CO₂ Saving Potential",   _fmt_m(summary['co2_saving']),    f"tonnes/yr · {summary['pct_saving']}% reduction", color='#19b459'),
        _kpi_card("Below Band C",           f"{summary['pct_below_c']}%",     "of all properties", color='#ef8023'),
    ])

    # ── Nav ──────────────────────────────────────────────────────────────────
    nav_links = '<a href="#about">About</a>'
    for sid, title, _ in EDA_SECTIONS:
        nav_links += f'<a href="#{sid}">{title.split(". ", 1)[-1]}</a>'
    for sid, title, _ in ML_SECTIONS:
        label = title.split(". ", 1)[-1].split(":")[0].strip()
        nav_links += f'<a href="#{sid}">{label}</a>'
    nav_links += '<a href="#summary">Key Findings</a>'
    nav_links += '<a href="#glossary">Glossary</a>'

    # ── EDA sections ─────────────────────────────────────────────────────────
    eda_html = ""
    for (sid, title, desc), fig in zip(EDA_SECTIONS, eda_figures):
        eda_html += f"""
        <section id="{sid}">
          <h2>{title}</h2>
          <div class="chart-desc">{desc}</div>
          <div class="chart-wrap">{_embed(fig)}</div>
        </section>"""

    # ── ML sections ──────────────────────────────────────────────────────────
    ml_html = _build_ml_intro_html()
    for (sid, title, desc), fig in zip(ML_SECTIONS, ml_figures):
        ml_html += f"""
        <section id="{sid}">
          <h2>{title}</h2>
          <div class="chart-desc">{desc}</div>
          <div class="chart-wrap">{_embed(fig)}</div>
        </section>"""

    # ── Full page ─────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>UK Energy EPC — Analytics &amp; ML Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    :root {{
      --bg:      #0d1117; --surface: #161b22; --border:  #30363d;
      --text:    #c9d1d9; --muted:   #8b949e; --accent:  #58a6ff;
      --green:   #3fb950; --orange:  #d29922; --red:     #f85149;
    }}
    html {{ scroll-behavior: smooth; }}
    body {{ background:var(--bg); color:var(--text);
            font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
            font-size:14px; line-height:1.6; }}

    header {{ background:var(--surface); border-bottom:1px solid var(--border);
              padding:24px 32px 20px; }}
    header h1 {{ font-size:22px; color:#e6edf3; margin-bottom:4px; }}
    header p  {{ color:var(--muted); font-size:13px; }}

    .kpi-strip {{ display:flex; flex-wrap:wrap; gap:16px; padding:20px 32px;
                  background:var(--surface); border-bottom:1px solid var(--border); }}
    .kpi-card  {{ flex:1 1 160px; background:var(--bg); border:1px solid var(--border);
                  border-radius:8px; padding:16px 20px; }}
    .kpi-value {{ font-size:28px; font-weight:700; line-height:1.1; }}
    .kpi-label {{ font-size:12px; color:var(--muted); margin-top:4px;
                  text-transform:uppercase; letter-spacing:.05em; }}
    .kpi-sub   {{ font-size:11px; color:var(--muted); margin-top:2px; }}

    nav {{ position:sticky; top:0; z-index:100; background:var(--surface);
           border-bottom:1px solid var(--border); padding:0 32px;
           display:flex; flex-wrap:wrap; gap:0; overflow-x:auto; }}
    nav a {{ color:var(--muted); text-decoration:none; font-size:12px;
             padding:10px 14px; border-bottom:2px solid transparent;
             white-space:nowrap; transition:color .15s,border-color .15s; }}
    nav a:hover {{ color:var(--accent); border-bottom-color:var(--accent); }}

    main {{ max-width:1400px; margin:0 auto; padding:32px; }}
    section {{ margin-bottom:56px; scroll-margin-top:44px; }}
    section h2 {{ font-size:17px; font-weight:600; color:#e6edf3;
                  padding-left:12px; border-left:3px solid var(--accent);
                  margin-bottom:8px; }}
    .chart-desc {{ color:var(--muted); font-size:13px; line-height:1.7;
                   margin-bottom:16px; max-width:920px; }}
    .chart-desc strong {{ color:var(--text); }}
    .chart-desc em     {{ color:#a5d6ff; font-style:normal; }}
    .chart-wrap {{ background:var(--surface); border:1px solid var(--border);
                   border-radius:8px; padding:8px; overflow:hidden; }}

    /* ── Info grid (about + ML intro cards) ─── */
    .info-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(340px,1fr));
                  gap:16px; margin-top:16px; }}
    .info-card {{ background:var(--surface); border:1px solid var(--border);
                  border-radius:8px; padding:18px 20px; }}
    .info-card h3 {{ font-size:14px; font-weight:600; color:var(--accent);
                     margin-bottom:8px; }}
    .info-card p, .info-card li {{ color:var(--muted); font-size:13px;
                                    line-height:1.65; }}
    .info-card strong {{ color:var(--text); }}
    .info-card em     {{ color:#a5d6ff; font-style:normal; }}

    /* ── Summary grid ─── */
    .summary-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(380px,1fr));
                     gap:20px; margin-top:8px; }}
    .summary-group {{ background:var(--surface); border:1px solid var(--border);
                      border-radius:8px; padding:18px 20px; }}
    .summary-group h3 {{ font-size:14px; font-weight:600; color:var(--accent);
                         margin-bottom:10px; }}
    .summary-group ul {{ list-style:none; padding:0; margin:0; }}
    .summary-group li {{ color:var(--muted); font-size:13px; line-height:1.65;
                         padding:5px 0 5px 14px; border-bottom:1px solid rgba(48,54,61,.6);
                         position:relative; }}
    .summary-group li:last-child {{ border-bottom:none; }}
    .summary-group li::before {{ content:'›'; position:absolute; left:0;
                                  color:var(--accent); font-weight:700; }}
    .summary-group li strong {{ color:var(--text); }}
    .summary-group li em     {{ color:#a5d6ff; font-style:normal; }}

    /* ── Glossary ─── */
    .glossary-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(340px,1fr));
                      gap:14px; margin-top:14px; }}
    .glossary-card {{ background:var(--surface); border:1px solid var(--border);
                      border-radius:8px; padding:14px 18px; }}
    .glossary-card dt {{ font-size:13px; font-weight:600; color:var(--accent);
                         margin-bottom:6px; }}
    .glossary-card dd {{ font-size:12px; color:var(--muted); line-height:1.6; margin:0; }}
    .glossary-card dd strong {{ color:var(--text); }}
    .glossary-card dd em     {{ color:#a5d6ff; font-style:normal; }}

    /* ── ML section separator ─── */
    .ml-divider {{ border:none; border-top:1px solid var(--border);
                   margin:40px 0 32px; }}
    .ml-label {{ font-size:11px; text-transform:uppercase; letter-spacing:.1em;
                 color:var(--muted); margin-bottom:24px; }}

    footer {{ text-align:center; color:var(--muted); font-size:12px;
              padding:24px; border-top:1px solid var(--border); }}
  </style>
</head>
<body>

<header>
  <h1>UK Energy Performance Certificate — Analytics &amp; Machine Learning Dashboard</h1>
  <p>29.2 million domestic EPC certificates · England &amp; Wales · DuckDB + dbt + Polars + LightGBM</p>
</header>

<div class="kpi-strip">{kpi_html}</div>
<nav>{nav_links}</nav>

<main>
  {_build_intro_html(summary)}

  <hr class="ml-divider">
  <p class="ml-label">📊 Data Exploration — Sections 1–12</p>
  {eda_html}

  <hr class="ml-divider">
  <p class="ml-label">🤖 Machine Learning — Sections 13–16</p>
  {ml_html}

  {build_summary_section(summary)}
  {_build_glossary_html()}
</main>

<footer>
  Generated by generate_dashboard.py · Data: UK Government EPC open dataset (England &amp; Wales) ·
  Stack: DuckDB · dbt · Polars · LightGBM · Plotly · scikit-learn
</footer>

</body>
</html>"""

    with open(DASHBOARD, 'w', encoding='utf-8') as fh:
        fh.write(html)
    print(f'  ✅  {DASHBOARD}  ({os.path.getsize(DASHBOARD)/1e6:.1f} MB)')


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print('\n🚀  Combined EPC Dashboard — starting...\n')

    con = duckdb.connect(DB_PATH, read_only=True)

    print('  ⏳  National summary KPIs')
    summary = get_national_summary(con)

    print('  ⏳  Building EDA charts (12 sections)...')
    eda_figures = []
    for label, builder in CHART_BUILDERS:
        print(f'      {label}')
        try:
            eda_figures.append(builder(con))
        except Exception as exc:
            import traceback
            print(f'      ⚠️  SKIPPED — {exc}')
            traceback.print_exc()
            eda_figures.append(go.Figure())

    print('  ⏳  Building ML charts (4 sections)...')
    ml_builders = [
        ('Model comparison',   lambda: build_ml_model_comparison()),
        ('Feature importance', lambda: build_ml_feature_importance()),
        ('Confusion matrix',   lambda: build_ml_confusion_matrix(con)),
        ('Per-class metrics',  lambda: build_ml_per_class_metrics(con)),
    ]
    ml_figures = []
    for label, builder in ml_builders:
        print(f'      {label}')
        try:
            ml_figures.append(builder())
        except Exception as exc:
            import traceback
            print(f'      ⚠️  SKIPPED — {exc}')
            traceback.print_exc()
            ml_figures.append(go.Figure())

    print('  ⏳  Assembling combined dashboard.html...')
    build_combined_dashboard(eda_figures, ml_figures, summary)

    con.close()
    print('\n✨  Done — open reports/dashboard.html in your browser\n')


if __name__ == '__main__':
    main()
