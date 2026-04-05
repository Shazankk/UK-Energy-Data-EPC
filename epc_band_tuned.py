"""
EPC Band — Model Selection & Hyperparameter Tuning
====================================================
Systematically identifies the best classifier for predicting EPC energy rating
(A–G) from four physical building features, then tunes it with Optuna Bayesian
optimisation.

Pipeline
--------
  1. Load 500k representative rows from DuckDB (reservoir sample)
  2. Compare four candidate classifiers on the same 3-fold CV split
     (macro F1 — treats all seven EPC bands equally)
  3. Tune the winning model (LightGBM) via Optuna TPE over 50 trials
  4. Retrain best hyperparameters on the full training set
  5. Evaluate on a held-out 20% test set
  6. Emit three interactive HTML reports and one pickled pipeline

Why macro F1 (not accuracy)?
-----------------------------
The UK housing stock skews heavily toward D/E bands (≈62% of all properties).
A naive model that predicts "D" for everything scores ~40% accuracy but is
useless in practice. Macro F1 averages the F1 score equally across all seven
bands — a model *must* perform on A, B, F, and G (the minority classes) to
score well.

Prerequisites
-------------
  python bulk_load_epc.py                   # raw CSVs → DuckDB
  cd ducklake_energy_uk && dbt run          # build stg_epc__domestic view
  source dbt-env/bin/activate
  pip install lightgbm xgboost optuna       # (already in requirements.txt)

Usage
-----
  python epc_band_tuned.py

Outputs
-------
  Console:    comparison table, Optuna progress, final classification report
  reports/    epc_tuned_model_comparison.html
              epc_tuned_confusion_matrix.html
              epc_tuned_feature_importance.html
              epc_tuned_per_class_metrics.html
              epc_tuned_optuna_history.html
  models/     epc_band_lgbm_tuned.pkl
"""

import os
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

import duckdb
import polars as pl
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import lightgbm as lgb

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import (
    StratifiedKFold, cross_validate, train_test_split,
)
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix,
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DB_PATH     = 'ducklake_energy_uk/dev.duckdb'
REPORTS_DIR = 'reports'
MODELS_DIR  = 'models'
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

EPC_ORDER  = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
EPC_COLORS = {
    'A': '#008054', 'B': '#19b459', 'C': '#8dce46',
    'D': '#ffd500', 'E': '#fcaa65', 'F': '#ef8023', 'G': '#e9153b',
}

# Ordinal order for construction age band (oldest → newest)
AGE_BAND_ORDER = [
    'Pre-1900', '1900-1929', '1930-1949', '1950-1975',
    '1976-1990', '1991-2002', '2003-2011', '2012-Present',
]

SAMPLE_ROWS = 500_000    # total reservoir sample from 29.2 M rows
TUNE_ROWS   =  60_000    # subset for Optuna CV — enough signal, fast per trial
N_TRIALS    =      30    # Optuna trials; TPE converges well in 25-35 on this space
CV_FOLDS    =       3    # folds for both comparison CV and Optuna objective


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Data loading
# ─────────────────────────────────────────────────────────────────────────────

_WALL_CASE = """
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
    Reservoir-sample n rows from the joined staging + raw tables.
    Returns a Polars DataFrame with lowercase column names.

    Reservoir sampling (RESERVOIR keyword in DuckDB) is:
      - Unbiased: every row has equal probability of being selected
      - O(n) memory: only keeps n rows in memory regardless of table size
      - Seeded (42): reproducible across runs
    """
    query = f"""
    SELECT *
    FROM (
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
    ) AS full_data
    USING SAMPLE {n} ROWS (reservoir, 42)
    """
    result = conn.query(query)
    df = result.pl()
    return df.rename({c: c.lower() for c in df.columns})


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Preprocessing factories
# ─────────────────────────────────────────────────────────────────────────────

def _rf_preprocessor() -> ColumnTransformer:
    """
    For tree ensembles that cannot use native categoricals (RF, ExtraTrees):
      - floor area  : passthrough
      - age band    : OrdinalEncoder (preserves ordinal relationship)
      - fuel, wall  : OneHotEncoder  (no ordinal relationship)
    """
    return ColumnTransformer([
        ('num', 'passthrough',
         ['total_floor_area_sqm']),
        ('age', OrdinalEncoder(
                    categories=[AGE_BAND_ORDER],
                    handle_unknown='use_encoded_value',
                    unknown_value=len(AGE_BAND_ORDER)),
         ['construction_age_band']),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
         ['main_fuel', 'wall_type']),
    ], remainder='drop')


def _lgbm_preprocessor() -> ColumnTransformer:
    """
    For LightGBM native categorical support:
      - All categoricals are OrdinalEncoded to non-negative integers
      - LightGBM's optimal-split categorical algorithm then handles them
        better than one-hot encoding (no curse of dimensionality,
        finds optimal splits across all category combinations)

    Column order after transform:
      [0] total_floor_area_sqm   — numeric
      [1] construction_age_band  — ordinal (0-7)
      [2] main_fuel              — categorical int
      [3] wall_type              — categorical int
    Indices 1, 2, 3 are passed as categorical_feature to LGBMClassifier.
    """
    return ColumnTransformer([
        ('num', 'passthrough',
         ['total_floor_area_sqm']),
        ('ord', OrdinalEncoder(
                    categories=[AGE_BAND_ORDER],
                    handle_unknown='use_encoded_value',
                    unknown_value=len(AGE_BAND_ORDER)),
         ['construction_age_band']),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
         ['main_fuel', 'wall_type']),
    ], remainder='drop')


def _hgbm_preprocessor() -> ColumnTransformer:
    """
    HistGradientBoostingClassifier handles missing values natively and can
    treat integer-encoded columns as categoricals via categorical_features param.
    Uses same OrdinalEncoder-only strategy as LightGBM.
    """
    return ColumnTransformer([
        ('num', 'passthrough',
         ['total_floor_area_sqm']),
        ('all_cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
         ['construction_age_band', 'main_fuel', 'wall_type']),
    ], remainder='drop')


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Model comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare_models(
    X_tune: pd.DataFrame,
    y_tune: pd.Series,
    cv: int = CV_FOLDS,
) -> pd.DataFrame:
    """
    Run stratified k-fold cross-validation on three candidate classifiers.
    Reports macro F1, accuracy, and wall-clock training time for each.

    Note on XGBoost: architecturally equivalent to LightGBM (both histogram-
    based gradient boosting) but slower on tabular data and without native
    categorical support. Excluded to keep the comparison focused on the
    bagging vs boosting distinction; see MODEL_DEVELOPMENT.md §3 for rationale.

    Candidate rationale:
      RandomForest          — strong baseline; bagging of deep trees; good with
                              mixed feature types but higher memory & slower
      HistGradientBoosting  — sklearn's native GBM; fast; native categoricals;
                              but limited tuning surface vs LightGBM
      LightGBM              — leaf-wise gradient boosting; fastest on tabular
                              data; best native categorical support; rich
                              regularisation surface for hyperparameter tuning

    All models use balanced class weights (or is_unbalance=True for LightGBM)
    to correct for D/E class dominance in the UK housing stock.

    Uses cross_validate (single pass) instead of two cross_val_score calls so
    both scoring metrics are computed from the same fitted models.
    """
    candidates = {
        'Random Forest (baseline)': Pipeline([
            ('prep', _rf_preprocessor()),
            ('clf',  RandomForestClassifier(
                         n_estimators=100, max_depth=10,
                         class_weight='balanced', n_jobs=-1, random_state=42)),
        ]),
        'HistGradientBoosting (sklearn)': Pipeline([
            ('prep', _hgbm_preprocessor()),
            ('clf',  HistGradientBoostingClassifier(
                         max_iter=200, max_depth=8,
                         categorical_features=[1, 2, 3],  # post-transform indices
                         class_weight='balanced', random_state=42)),
        ]),
        'LightGBM': Pipeline([
            ('prep', _lgbm_preprocessor()),
            ('clf',  lgb.LGBMClassifier(
                         n_estimators=200, num_leaves=63,
                         learning_rate=0.1,
                         categorical_feature=[1, 2, 3],  # post-transform indices
                         is_unbalance=True,
                         n_jobs=-1, random_state=42,
                         verbose=-1)),
        ]),
    }

    skf     = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    records = []

    for name, pipe in candidates.items():
        t0  = time.time()
        cv_results = cross_validate(
            pipe, X_tune, y_tune, cv=skf,
            scoring={'f1_macro': 'f1_macro', 'accuracy': 'accuracy'},
            n_jobs=1,
        )
        elapsed = time.time() - t0

        f1  = cv_results['test_f1_macro']
        acc = cv_results['test_accuracy']
        records.append({
            'Model':       name,
            'Macro F1':    round(f1.mean(),  4),
            'F1 Std':      round(f1.std(),   4),
            'Accuracy':    round(acc.mean(), 4),
            'CV Time (s)': round(elapsed,    1),
        })
        print(f'    {name:<38} '
              f'F1={f1.mean():.4f} ± {f1.std():.4f}  '
              f'Acc={acc.mean():.4f}  t={elapsed:.1f}s')

    return pd.DataFrame(records).sort_values('Macro F1', ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Optuna hyperparameter tuning (LightGBM)
# ─────────────────────────────────────────────────────────────────────────────

def _make_lgbm_objective(X_tune: pd.DataFrame, y_tune: pd.Series, cv: int):
    """
    Returns an Optuna objective function that, for each trial:
      1. Samples hyperparameters from the defined search space
      2. Runs stratified k-fold CV
      3. Returns mean macro F1 (maximised by Optuna)

    Search space rationale:
      num_leaves        : controls model complexity; wider than max_depth alone
      max_depth         : caps tree depth to prevent overfitting
      min_child_samples : minimum data in a leaf; key regulariser for small classes
      learning_rate     : log-scale, coupled with n_estimators
      n_estimators      : more trees = lower variance; diminishing returns past 800
      feature_fraction  : column subsampling per tree (like RF's max_features)
      bagging_fraction  : row subsampling per tree iteration
      bagging_freq      : how often bagging is applied (0 = disabled)
      reg_alpha         : L1 regularisation on leaf weights
      reg_lambda        : L2 regularisation on leaf weights
    """
    label_enc = LabelEncoder().fit(y_tune)
    y_enc     = label_enc.transform(y_tune)
    skf       = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    prep      = _lgbm_preprocessor()
    X_prep    = prep.fit_transform(X_tune)

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            objective         = 'multiclass',
            num_class         = len(label_enc.classes_),
            metric            = 'multi_logloss',
            is_unbalance      = True,
            n_jobs            = -1,
            random_state      = 42,
            verbose           = -1,

            # ── tree structure ────────────────────────────────────────────
            num_leaves        = trial.suggest_int('num_leaves',        15,  300),
            max_depth         = trial.suggest_int('max_depth',          3,   12),
            min_child_samples = trial.suggest_int('min_child_samples', 10,  200),

            # ── learning ─────────────────────────────────────────────────
            n_estimators      = trial.suggest_int('n_estimators',     100,  400),
            learning_rate     = trial.suggest_float('learning_rate', 5e-3, 0.3,
                                                    log=True),

            # ── sampling ─────────────────────────────────────────────────
            feature_fraction  = trial.suggest_float('feature_fraction', 0.4, 1.0),
            bagging_fraction  = trial.suggest_float('bagging_fraction', 0.4, 1.0),
            bagging_freq      = trial.suggest_int('bagging_freq',         1,  10),

            # ── regularisation ────────────────────────────────────────────
            reg_alpha         = trial.suggest_float('reg_alpha',  1e-8, 10.0,
                                                    log=True),
            reg_lambda        = trial.suggest_float('reg_lambda', 1e-8, 10.0,
                                                    log=True),
            min_split_gain    = trial.suggest_float('min_split_gain',  0.0, 0.5),
        )

        f1_scores = []
        for train_idx, val_idx in skf.split(X_prep, y_enc):
            X_tr, X_val = X_prep[train_idx], X_prep[val_idx]
            y_tr, y_val = y_enc[train_idx],  y_enc[val_idx]

            clf = lgb.LGBMClassifier(
                **params,
                categorical_feature=[1, 2, 3],
            )
            clf.fit(X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(20, verbose=False),
                               lgb.log_evaluation(period=-1)])

            y_pred = clf.predict(X_val)
            f1_scores.append(f1_score(y_val, y_pred, average='macro'))

        return float(np.mean(f1_scores))

    return objective, prep, label_enc


def tune_lgbm(
    X_tune: pd.DataFrame,
    y_tune: pd.Series,
    n_trials: int = N_TRIALS,
    cv: int = CV_FOLDS,
) -> tuple[dict, 'ColumnTransformer', 'LabelEncoder', 'optuna.Study']:
    """
    Run Optuna Bayesian optimisation over the LightGBM search space.
    Returns the best hyperparameter dict, fitted preprocessor, and the study.
    """
    objective, prep, label_enc = _make_lgbm_objective(X_tune, y_tune, cv)

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10,
                                           n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best_f1 = study.best_value
    print(f'\n  Best macro F1 from Optuna: {best_f1:.4f}')
    print(f'  Best params: {best}')
    return best, prep, label_enc, study


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Build and evaluate final model
# ─────────────────────────────────────────────────────────────────────────────

def build_final_pipeline(best_params: dict) -> Pipeline:
    """
    Assemble the production-ready sklearn Pipeline with tuned LightGBM params.
    Using Pipeline ensures preprocessing + model are atomic (no data leakage
    when loading the .pkl file for inference).
    """
    lgbm_params = dict(
        **best_params,
        objective     = 'multiclass',
        metric        = 'multi_logloss',
        is_unbalance  = True,
        categorical_feature = [1, 2, 3],
        n_jobs        = -1,
        random_state  = 42,
        verbose       = -1,
    )
    return Pipeline([
        ('prep', _lgbm_preprocessor()),
        ('clf',  lgb.LGBMClassifier(**lgbm_params)),
    ])


def evaluate(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred  = pipe.predict(X_test)
    bands   = sorted(set(y_test) | set(y_pred), key=lambda b: EPC_ORDER.index(b))
    return {
        'accuracy' : accuracy_score(y_test, y_pred),
        'macro_f1' : f1_score(y_test, y_pred, average='macro'),
        'report'   : classification_report(y_test, y_pred, labels=bands,
                                           output_dict=True),
        'cm'       : confusion_matrix(y_test, y_pred, labels=bands),
        'bands'    : bands,
        'y_pred'   : y_pred,
    }


def get_feature_names(pipe: Pipeline) -> list[str]:
    """Map post-transform column indices back to human-readable names."""
    # After _lgbm_preprocessor: [floor_area, construction_age_band, main_fuel, wall_type]
    return [
        'floor_area_sqm',
        'construction_age_band',
        'main_fuel',
        'wall_type',
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Plotly charts
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_comparison(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart: macro F1 per model, sorted best-first."""
    colors = ['#19b459' if 'LightGBM' in m else '#4a9eff' for m in df['Model']]

    fig = go.Figure(go.Bar(
        x=df['Macro F1'],
        y=df['Model'],
        orientation='h',
        marker_color=colors,
        error_x=dict(type='data', array=df['F1 Std'].tolist(), visible=True),
        hovertemplate='%{y}<br>Macro F1: %{x:.4f}<extra></extra>',
    ))
    fig.update_layout(
        title='Model Comparison — 3-Fold CV Macro F1<br>'
              '<sup>Green = selected for tuning | Error bars = std across folds</sup>',
        template='plotly_dark',
        xaxis_title='Macro F1 (higher = better; chance = 0.14 for 7 classes)',
        yaxis=dict(categoryorder='total ascending'),
        width=850, height=420,
        margin=dict(l=260),
    )
    return fig


def plot_confusion_matrix(cm: np.ndarray, bands: list[str]) -> go.Figure:
    """Row-normalised confusion matrix (% of actual class per predicted band)."""
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct   = np.where(row_sums > 0, cm.astype(float) / row_sums * 100, 0.0)

    fig = go.Figure(go.Heatmap(
        z=cm_pct,
        x=[f'Pred {b}' for b in bands],
        y=[f'Actual {b}' for b in bands],
        colorscale='RdYlGn', reversescale=True,
        zmin=0, zmax=100,
        text=[[f'{v:.1f}%' for v in row] for row in cm_pct],
        texttemplate='%{text}',
        hovertemplate='Actual %{y} → Predicted %{x}<br>%{text}<extra></extra>',
        colorbar=dict(title='% of actual'),
    ))
    fig.update_layout(
        title='Confusion Matrix — Tuned LightGBM (held-out test set)<br>'
              '<sup>Diagonal = correct predictions; off-diagonal = misclassifications</sup>',
        template='plotly_dark',
        xaxis_title='Predicted Band',
        yaxis_title='Actual Band',
        width=700, height=620,
    )
    return fig


def plot_feature_importance(pipe: Pipeline) -> go.Figure:
    """
    LightGBM provides two importance types:
      split   — how many times a feature is used to split a node
      gain    — total information gain attributed to a feature (more reliable)
    We plot 'gain' since it is less biased toward high-cardinality features.
    """
    clf   = pipe.named_steps['clf']
    names = ['floor_area_sqm', 'construction_age_band', 'main_fuel', 'wall_type']

    gain  = clf.booster_.feature_importance(importance_type='gain')
    split = clf.booster_.feature_importance(importance_type='split')

    df = pd.DataFrame({'feature': names, 'gain': gain, 'split': split})
    df = df.sort_values('gain', ascending=True)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['Importance by Gain (total info gain)',
                                        'Importance by Split (node usage count)'])
    for col, metric in [(1, 'gain'), (2, 'split')]:
        fig.add_trace(go.Bar(
            x=df[metric], y=df['feature'],
            orientation='h',
            marker_color='#19b459',
            showlegend=False,
            hovertemplate=f'%{{y}}<br>{metric}: %{{x:,.0f}}<extra></extra>',
        ), row=1, col=col)

    fig.update_layout(
        title='LightGBM Feature Importances — Gain vs Split',
        template='plotly_dark',
        height=400, width=900,
    )
    return fig


def plot_per_class_metrics(report: dict, bands: list[str]) -> go.Figure:
    """F1-score per EPC band (bar) overlaid with test sample support (line)."""
    f1      = [report.get(b, {}).get('f1-score',  0.0) for b in bands]
    prec    = [report.get(b, {}).get('precision', 0.0) for b in bands]
    rec     = [report.get(b, {}).get('recall',    0.0) for b in bands]
    support = [report.get(b, {}).get('support',   0)   for b in bands]
    colors  = [EPC_COLORS.get(b, '#888') for b in bands]

    fig = make_subplots(specs=[[{'secondary_y': True}]])
    for metric, dash, name in [
        (f1,   'solid',  'F1-Score'),
        (prec, 'dot',    'Precision'),
        (rec,  'dash',   'Recall'),
    ]:
        fig.add_trace(
            go.Scatter(x=bands, y=metric, name=name, mode='lines+markers',
                       line=dict(dash=dash, width=2)),
            secondary_y=False,
        )
    fig.add_trace(
        go.Bar(x=bands, y=support, name='Test samples',
               marker_color=colors, opacity=0.35,
               hovertemplate='Band %{x}<br>n = %{y:,}<extra></extra>'),
        secondary_y=True,
    )
    fig.update_layout(
        title='Per-Class Precision / Recall / F1 — Tuned LightGBM',
        template='plotly_dark',
        xaxis_title='EPC Band',
        legend=dict(x=0.01, y=0.99),
        width=820, height=520,
    )
    fig.update_yaxes(title_text='Score (0 – 1)',       range=[0, 1.05], secondary_y=False)
    fig.update_yaxes(title_text='Number of Test Rows', secondary_y=True)
    return fig


def plot_optuna_history(study: optuna.Study) -> go.Figure:
    """Scatter of all Optuna trials: trial number vs macro F1, best highlighted."""
    values = [t.value for t in study.trials if t.value is not None]
    best_so_far = [max(values[:i+1]) for i in range(len(values))]
    trial_nums  = list(range(len(values)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trial_nums, y=values,
        mode='markers', name='Trial',
        marker=dict(color='#4a9eff', size=6, opacity=0.7),
        hovertemplate='Trial %{x}<br>Macro F1: %{y:.4f}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=trial_nums, y=best_so_far,
        mode='lines', name='Best so far',
        line=dict(color='#19b459', width=2),
    ))
    fig.update_layout(
        title=f'Optuna Optimisation History ({len(values)} trials)<br>'
              f'<sup>TPE Bayesian optimisation — macro F1 objective</sup>',
        template='plotly_dark',
        xaxis_title='Trial Number',
        yaxis_title='Macro F1',
        width=820, height=460,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print('=' * 66)
    print('  EPC Band — Model Selection & Hyperparameter Tuning')
    print('=' * 66)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    print(f'\n[1/6] Loading {SAMPLE_ROWS:,} rows from DuckDB '
          f'(reservoir sample, seed=42) ...')
    conn = duckdb.connect(DB_PATH, read_only=True)
    df   = load_data(conn, SAMPLE_ROWS)
    conn.close()

    dist = df['target'].value_counts().sort('target')
    print(f'      Shape: {df.shape}')
    print(f'      Band distribution:\n{dist}')

    # ── 2. Train / test split ────────────────────────────────────────────────
    print('\n[2/6] Splitting train / test (80 / 20, stratified) ...')
    X     = df.drop('target').to_pandas()
    y     = df['target'].to_pandas()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42,
    )

    # Subsample from training set for comparison CV + Optuna (speed)
    X_tune, _, y_tune, _ = train_test_split(
        X_train, y_train, train_size=TUNE_ROWS, stratify=y_train, random_state=42,
    )
    print(f'      Full train: {len(X_train):,}  '
          f'| Tune subset: {len(X_tune):,}  '
          f'| Test: {len(X_test):,}')

    # ── 3. Model comparison ──────────────────────────────────────────────────
    print(f'\n[3/6] Comparing candidate models '
          f'({CV_FOLDS}-fold CV, macro F1, n={len(X_tune):,}) ...')
    comparison_df = compare_models(X_tune, y_tune)
    print(f'\n  {comparison_df.to_string(index=False)}')

    # ── 4. Optuna tuning ─────────────────────────────────────────────────────
    print(f'\n[4/6] Tuning LightGBM with Optuna '
          f'({N_TRIALS} trials, {CV_FOLDS}-fold CV, n={len(X_tune):,}) ...')
    best_params, _, _, study = tune_lgbm(X_tune, y_tune, N_TRIALS, CV_FOLDS)

    # Baseline LightGBM CV score (default params on same tune split)
    default_pipe = Pipeline([
        ('prep', _lgbm_preprocessor()),
        ('clf',  lgb.LGBMClassifier(
                     n_estimators=200, num_leaves=63,
                     categorical_feature=[1, 2, 3],
                     is_unbalance=True, n_jobs=-1,
                     random_state=42, verbose=-1)),
    ])
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    baseline_f1 = cross_validate(default_pipe, X_tune, y_tune,
                                 cv=skf, scoring='f1_macro',
                                 n_jobs=1)['test_score'].mean()
    print(f'\n  LightGBM default params CV F1 : {baseline_f1:.4f}')
    print(f'  LightGBM tuned   params CV F1 : {study.best_value:.4f}')
    print(f'  Improvement                   : +{(study.best_value - baseline_f1):.4f}')

    # ── 5. Retrain on full training set ──────────────────────────────────────
    print(f'\n[5/6] Retraining best params on full training set '
          f'(n={len(X_train):,}) ...')
    final_pipe = build_final_pipeline(best_params)
    final_pipe.fit(X_train, y_train)
    print('      Done.')

    # ── 6. Evaluate on held-out test set ─────────────────────────────────────
    print('\n[6/6] Final evaluation on held-out test set ...')
    results = evaluate(final_pipe, X_test, y_test)

    print(f'\n  Overall accuracy : {results["accuracy"]:.4f}')
    print(f'  Macro F1         : {results["macro_f1"]:.4f}\n')
    print(classification_report(
        y_test, results['y_pred'],
        labels=results['bands'],
        target_names=results['bands'],
    ))

    # ── Save artifacts ───────────────────────────────────────────────────────
    print('Saving charts and model ...')
    charts = {
        'epc_tuned_model_comparison.html'   : plot_model_comparison(comparison_df),
        'epc_tuned_confusion_matrix.html'   : plot_confusion_matrix(results['cm'], results['bands']),
        'epc_tuned_feature_importance.html' : plot_feature_importance(final_pipe),
        'epc_tuned_per_class_metrics.html'  : plot_per_class_metrics(results['report'], results['bands']),
        'epc_tuned_optuna_history.html'     : plot_optuna_history(study),
    }
    for fname, fig in charts.items():
        path = os.path.join(REPORTS_DIR, fname)
        fig.write_html(path)
        print(f'  {path}')

    pkl_path = os.path.join(MODELS_DIR, 'epc_band_lgbm_tuned.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'pipeline'     : final_pipe,
            'best_params'  : best_params,
            'feature_names': get_feature_names(final_pipe),
            'epc_order'    : EPC_ORDER,
            'study'        : study,
        }, f)
    print(f'  {pkl_path}')
    print('\nDone.\n')


if __name__ == '__main__':
    main()
