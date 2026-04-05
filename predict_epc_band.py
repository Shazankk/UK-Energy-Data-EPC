"""
EPC Band Prediction Model
=========================
Trains a Random Forest classifier to predict EPC energy rating (A–G) from
four physical features:

  Feature                  Type       Source
  ─────────────────────    ────────   ──────────────────────────────────────
  total_floor_area_sqm     numeric    stg_epc__domestic
  construction_age_band    ordinal    stg_epc__domestic (8 standardized bands)
  main_fuel                nominal    stg_epc__domestic (6 clean categories)
  wall_type                nominal    raw.epc_domestic → WALLS_DESCRIPTION
                                      (grouped into 6 structural categories)

Wall descriptions in the raw data are free text (~200 variants). They are
collapsed into 6 categories via SQL CASE (same pattern as main_fuel
standardization in staging):
  Cavity - Uninsulated  |  Cavity - Insulated
  Solid - Uninsulated   |  Solid - Insulated
  Timber Frame          |  System Build

Prerequisites
─────────────
  python bulk_load_epc.py                    # raw CSVs → DuckDB
  cd ducklake_energy_uk && dbt run           # build stg_epc__domestic view
  source dbt-env/bin/activate

Usage
─────
  python predict_epc_band.py

Output
──────
  Console: accuracy, per-class classification report
  reports/epc_band_confusion_matrix.html      (normalised heatmap)
  reports/epc_band_feature_importance.html    (top-20 features)
  reports/epc_band_per_class_metrics.html     (F1 + support per band)
  models/epc_band_rf.pkl                      (pipeline + feature names)
"""

import os
import pickle

import duckdb
import polars as pl
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DB_PATH     = 'ducklake_energy_uk/dev.duckdb'
REPORTS_DIR = 'reports'
MODELS_DIR  = 'models'
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# EPC ratings in ascending efficiency order
EPC_ORDER = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
EPC_COLORS = {
    'A': '#008054', 'B': '#19b459', 'C': '#8dce46',
    'D': '#ffd500', 'E': '#fcaa65', 'F': '#ef8023', 'G': '#e9153b',
}

# Ordinal order for construction age band (oldest → newest)
AGE_BAND_ORDER = [
    'Pre-1900', '1900-1929', '1930-1949', '1950-1975',
    '1976-1990', '1991-2002', '2003-2011', '2012-Present',
]

# Reservoir sample size — large enough to be representative, small enough to
# train in under 60 seconds on a laptop.  Full dataset is 29.2M rows.
SAMPLE_ROWS = 500_000


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Load data
# ─────────────────────────────────────────────────────────────────────────────

WALL_CASE = """
CASE
    WHEN lower(r.walls_description) LIKE '%cavity%'
     AND (   lower(r.walls_description) LIKE '%no insulation%'
          OR lower(r.walls_description) LIKE '%uninsulated%')
        THEN 'Cavity - Uninsulated'
    WHEN lower(r.walls_description) LIKE '%cavity%'
     AND lower(r.walls_description) LIKE '%insulation%'
        THEN 'Cavity - Insulated'
    WHEN lower(r.walls_description) LIKE '%solid%'
     AND (   lower(r.walls_description) LIKE '%no insulation%'
          OR lower(r.walls_description) LIKE '%uninsulated%')
        THEN 'Solid - Uninsulated'
    WHEN lower(r.walls_description) LIKE '%solid%'
     AND lower(r.walls_description) LIKE '%insulation%'
        THEN 'Solid - Insulated'
    WHEN lower(r.walls_description) LIKE '%timber%'
        THEN 'Timber Frame'
    WHEN lower(r.walls_description) LIKE '%system%'
        THEN 'System Build'
    ELSE 'Other/Unknown'
END
"""


def load_data(conn: duckdb.DuckDBPyConnection, n: int) -> pl.DataFrame:
    """
    Sample n rows from stg_epc__domestic (view) joined to raw.epc_domestic for
    wall description.  Reservoir sampling is unbiased and O(n) over the table.
    """
    query = f"""
    SELECT *
    FROM (
        SELECT
            s.energy_rating_current                    AS target,
            CAST(s.total_floor_area_sqm AS DOUBLE)     AS total_floor_area_sqm,
            s.construction_age_band,
            s.main_fuel,
            {WALL_CASE}                                AS wall_type
        FROM stg_epc__domestic s
        JOIN raw.epc_domestic   r ON s.certificate_id = r.lmk_key
        WHERE s.energy_rating_current IS NOT NULL
          AND s.total_floor_area_sqm  IS NOT NULL
          AND CAST(s.total_floor_area_sqm AS DOUBLE) > 0
          AND s.construction_age_band NOT IN ('Unknown')
          AND s.main_fuel             NOT IN ('Other/Unknown')
    ) AS full_data
    USING SAMPLE {n} ROWS (reservoir, 42)
    """
    result = conn.query(query)
    df = result.pl()
    return df.rename({c: c.lower() for c in df.columns})


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Build sklearn pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """
    Preprocessing:
      - floor_area       : pass through as-is (already numeric)
      - construction_age : ordinal encoded (Pre-1900=0 → 2012-Present=7)
      - main_fuel        : one-hot encoded
      - wall_type        : one-hot encoded

    Classifier: Random Forest with class_weight='balanced' to handle the
    heavy skew toward D/E bands in the UK housing stock.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough',
             ['total_floor_area_sqm']),

            ('age', OrdinalEncoder(
                        categories=[AGE_BAND_ORDER],
                        handle_unknown='use_encoded_value',
                        unknown_value=len(AGE_BAND_ORDER)),
             ['construction_age_band']),

            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             ['main_fuel', 'wall_type']),
        ],
        remainder='drop',
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=20,
        class_weight='balanced',   # corrects for D/E class imbalance
        n_jobs=-1,
        random_state=42,
    )

    return Pipeline([('prep', preprocessor), ('clf', clf)])


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Evaluate
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(pipe: Pipeline, X_test, y_test) -> dict:
    y_pred = pipe.predict(X_test)
    bands_present = sorted(set(y_test) | set(y_pred), key=lambda b: EPC_ORDER.index(b))
    return {
        'accuracy' : accuracy_score(y_test, y_pred),
        'report'   : classification_report(
                         y_test, y_pred,
                         labels=bands_present,
                         output_dict=True),
        'cm'       : confusion_matrix(y_test, y_pred, labels=bands_present),
        'bands'    : bands_present,
        'y_pred'   : y_pred,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Feature names (maps pipeline output back to human-readable labels)
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_names(pipe: Pipeline) -> list[str]:
    prep = pipe.named_steps['prep']
    ohe_names = prep.named_transformers_['cat'] \
                    .get_feature_names_out(['main_fuel', 'wall_type']).tolist()
    return ['floor_area_sqm', 'construction_age_band'] + ohe_names


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Plotly charts
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, bands: list[str]) -> go.Figure:
    """Row-normalised confusion matrix: each cell = % of that actual band."""
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.where(row_sums > 0, cm.astype(float) / row_sums * 100, 0.0)

    text = [[f'{v:.1f}%' for v in row] for row in cm_pct]

    fig = go.Figure(go.Heatmap(
        z=cm_pct,
        x=[f'Pred {b}' for b in bands],
        y=[f'Actual {b}' for b in bands],
        colorscale='RdYlGn',
        reversescale=True,
        zmin=0, zmax=100,
        text=text,
        texttemplate='%{text}',
        hovertemplate='Actual %{y} → Predicted %{x}<br>%{text}<extra></extra>',
        colorbar=dict(title='% of actual'),
    ))
    fig.update_layout(
        title='Confusion Matrix — EPC Band Prediction<br>'
              '<sup>Each cell = % of actual band predicted as each category</sup>',
        template='plotly_dark',
        xaxis_title='Predicted Band',
        yaxis_title='Actual Band',
        width=700, height=600,
    )
    return fig


def plot_feature_importance(pipe: Pipeline, feature_names: list[str]) -> go.Figure:
    importances = pipe.named_steps['clf'].feature_importances_
    df = (
        pl.DataFrame({'feature': feature_names, 'importance': importances})
        .sort('importance', descending=True)
        .head(20)
    )

    # Clean up one-hot feature names for display
    def _clean(name: str) -> str:
        name = name.replace('main_fuel_', 'Fuel: ')
        name = name.replace('wall_type_', 'Wall: ')
        return name

    df = df.with_columns(pl.col('feature').map_elements(_clean, return_dtype=pl.Utf8))

    fig = px.bar(
        df.to_pandas(),
        x='importance', y='feature',
        orientation='h',
        title='Top 20 Feature Importances — EPC Band Random Forest',
        labels={'importance': 'Mean Decrease in Impurity', 'feature': ''},
        template='plotly_dark',
        color='importance',
        color_continuous_scale='Teal',
    )
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False,
        coloraxis_showscale=False,
        width=820, height=570,
    )
    return fig


def plot_per_class_metrics(report: dict, bands: list[str]) -> go.Figure:
    """F1-score per band (bar) with support count overlay (line)."""
    f1       = [report.get(b, {}).get('f1-score',  0.0) for b in bands]
    support  = [report.get(b, {}).get('support',   0)   for b in bands]
    colors   = [EPC_COLORS.get(b, '#888888') for b in bands]

    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(
        go.Bar(name='F1-Score', x=bands, y=f1, marker_color=colors,
               hovertemplate='Band %{x}<br>F1 = %{y:.3f}<extra></extra>'),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(name='Test samples', x=bands, y=support,
                   mode='lines+markers',
                   line=dict(color='white', width=2, dash='dot'),
                   hovertemplate='Band %{x}<br>n = %{y:,}<extra></extra>'),
        secondary_y=True,
    )
    fig.update_layout(
        title='Per-Class F1-Score and Test Sample Count',
        template='plotly_dark',
        xaxis_title='EPC Band',
        legend=dict(x=0.02, y=0.98),
        width=820, height=500,
    )
    fig.update_yaxes(title_text='F1-Score (0–1)', range=[0, 1], secondary_y=False)
    fig.update_yaxes(title_text='Number of Test Samples', secondary_y=True)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print('=' * 62)
    print('  EPC Band Prediction — Random Forest Classifier')
    print('=' * 62)

    # ── 1. Load ──────────────────────────────────────────────────────────────
    print(f'\n[1/5] Loading {SAMPLE_ROWS:,} rows from DuckDB ...')
    conn = duckdb.connect(DB_PATH, read_only=True)
    df   = load_data(conn, SAMPLE_ROWS)
    conn.close()

    print(f'      Shape: {df.shape}')
    dist = df['target'].value_counts().sort('target')
    print(f'      Band distribution (sample):\n{dist}')

    # ── 2. Split ─────────────────────────────────────────────────────────────
    print('\n[2/5] Train / test split (80 / 20, stratified) ...')
    X = df.drop('target').to_pandas()
    y = df['target'].to_pandas()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42,
    )
    print(f'      Train: {len(X_train):,}   Test: {len(X_test):,}')

    # ── 3. Train ─────────────────────────────────────────────────────────────
    print('\n[3/5] Fitting pipeline (preprocessing + Random Forest 300 trees) ...')
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    print('      Done.')

    # ── 4. Evaluate ──────────────────────────────────────────────────────────
    print('\n[4/5] Evaluating on held-out test set ...')
    results = evaluate(pipe, X_test, y_test)

    print(f'\n  Overall accuracy : {results["accuracy"]:.3f}  '
          f'({results["accuracy"]*100:.1f}%)\n')
    print(classification_report(
        y_test, results['y_pred'],
        labels=results['bands'],
        target_names=results['bands'],
    ))

    # ── 5. Save charts + model ───────────────────────────────────────────────
    print('[5/5] Saving charts and model ...')
    feature_names = get_feature_names(pipe)

    cm_path  = os.path.join(REPORTS_DIR, 'epc_band_confusion_matrix.html')
    fi_path  = os.path.join(REPORTS_DIR, 'epc_band_feature_importance.html')
    cls_path = os.path.join(REPORTS_DIR, 'epc_band_per_class_metrics.html')
    pkl_path = os.path.join(MODELS_DIR,  'epc_band_rf.pkl')

    plot_confusion_matrix(results['cm'], results['bands']).write_html(cm_path)
    plot_feature_importance(pipe, feature_names).write_html(fi_path)
    plot_per_class_metrics(results['report'], results['bands']).write_html(cls_path)

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(pkl_path, 'wb') as f:
        pickle.dump({'pipeline': pipe, 'feature_names': feature_names}, f)

    print(f'\n  {cm_path}')
    print(f'  {fi_path}')
    print(f'  {cls_path}')
    print(f'  {pkl_path}')
    print('\nDone.\n')


if __name__ == '__main__':
    main()
