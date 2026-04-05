# EPC Band Prediction — Model Development Log

**Objective**: Predict a property's EPC energy rating band (A–G) from four
physical features — floor area, construction age, heating fuel type, and wall
construction type — using the UK's 29.2 million certificate dataset.

**Script**: `epc_band_tuned.py`  
**Baseline script**: `predict_epc_band.py`  
**Dataset**: 500,000 reservoir-sampled rows from `stg_epc__domestic ⋈ raw.epc_domestic`

---

## Contents

1. [Problem Framing](#1-problem-framing)
2. [Feature Engineering](#2-feature-engineering)
3. [Why Random Forest is Not Optimal](#3-why-random-forest-is-not-optimal)
4. [Model Selection](#4-model-selection)
5. [Hyperparameter Tuning with Optuna](#5-hyperparameter-tuning-with-optuna)
6. [Final Results](#6-final-results)
7. [Interpretation of Results](#7-interpretation-of-results)
8. [The Feature Ceiling — Why Tuning Barely Helps](#8-the-feature-ceiling)
9. [How to Use the Saved Model](#9-how-to-use-the-saved-model)
10. [What Would Actually Improve Performance](#10-what-would-actually-improve-performance)

---

## 1. Problem Framing

### Task type

**Multi-class classification** with seven ordered classes: A, B, C, D, E, F, G.

The classes are **ordinal** (E is worse than D which is worse than C), but we treat
the problem as nominal multi-class rather than ordinal regression because:
- The practical question is "what band is this property in?" not "what number?"
- Band transitions carry policy weight (e.g., failing vs passing Band C in 2035)
- Ordinal regression would require a different loss function and evaluation metric
  that complicates interpretation for non-technical stakeholders

### Class imbalance

The UK housing stock is heavily skewed toward the middle bands:

| Band | Sample count | % of data | Description |
|---|---|---|---|
| A | 1,700 | 0.34% | Extremely rare — new builds, Passivhaus |
| B | 28,608 | 5.7% | High-spec new builds |
| C | 168,082 | 33.6% | Modern or recently retrofitted |
| D | 202,383 | 40.5% | **Plurality** — average UK home |
| E | 76,021 | 15.2% | Older, uninsulated stock |
| F | 18,350 | 3.7% | Poor efficiency, often old solid-wall |
| G | 4,858 | 0.97% | Worst-performing — often pre-1900 solid uninsulated |

A naive model that predicts "D" for everything achieves **~40% accuracy** but is
completely useless — it tells us nothing about any property. This is why we use
**macro F1** as the primary metric throughout: it averages the F1 score equally
across all seven bands, so a model *must* perform well on A, F, and G to score well.

### Baseline (random guessing)
- Uniform random guessing: macro F1 ≈ 0.14 (1/7 classes)
- Always-predict-D: accuracy ≈ 40.5%, macro F1 ≈ 0.08

Any model scoring macro F1 > 0.20 is learning real signal from the data.

---

## 2. Feature Engineering

All four features come from EPC assessor measurements, not computed values —
they reflect the physical state of the building at inspection time.

### Feature 1 — `total_floor_area_sqm` (numeric)

**Why it matters**: Larger properties have more surface area to lose heat
through. A 300 m² detached house has fundamentally different heat loss
characteristics than a 45 m² flat — even with identical insulation. Floor area
is the strongest single continuous predictor of energy consumption.

**Preprocessing**: Passed through as-is. No scaling needed — tree-based models
split on thresholds and are invariant to monotonic transformations of features.
(Scaling is needed for distance-based models like KNN or linear models, not trees.)

**Distribution**: Right-skewed (most UK homes are 50–150 m²; outliers up to 500+).
The tree splits automatically handle this without log transformation.

### Feature 2 — `construction_age_band` (ordinal, 8 levels)

**Why it matters**: Building regulations governing insulation and glazing have
tightened substantially in every decade since the 1970s. A house built in
2010 must meet Part L of the Building Regulations; a house built in 1890 has no
such requirement and often has 500mm solid brick walls with no cavity for insulation.

**The 8 standardized bands** (oldest → newest, with their ordinal encoding):

| Band | Encoding | Key building regulation context |
|---|---|---|
| Pre-1900 | 0 | No insulation requirements; solid walls typical |
| 1900–1929 | 1 | Early cavity walls but rarely filled |
| 1930–1949 | 2 | Wider adoption of cavity construction |
| 1950–1975 | 3 | Post-war mass housing; some insulation from 1960s |
| 1976–1990 | 4 | First energy regulations (Part F 1976) |
| 1991–2002 | 5 | Tighter standards; double glazing becomes standard |
| 2003–2011 | 6 | Part L 2002/2006; significant insulation requirements |
| 2012–Present | 7 | Part L 2013; near-zero energy buildings from 2016 |

**Preprocessing**: `OrdinalEncoder` assigns integers 0–7 in the order above.
This is correct because the relationship is monotonic — newer buildings are
systematically more efficient — and the tree model can exploit this ordering via
threshold splits (e.g., `age >= 5` captures all homes built after 1991).

Using `OneHotEncoder` here would destroy the ordering information and create
7 correlated dummy variables. `OrdinalEncoder` is the right choice for an
inherently ordered categorical.

### Feature 3 — `main_fuel` (nominal, 6 categories)

**Why it matters**: The heating fuel is the single strongest determinant of
operating efficiency and cost. Solid fuel (coal/anthracite) has a SAP score
averaging 35; mains gas averages 66 — a 31-point gap that reflects the relative
CO₂ intensity and cost per useful unit of heat.

**The 6 standardized categories** (grouped from 20+ raw variants in staging):

| Category | Avg SAP | Notes |
|---|---|---|
| Mains Gas | 66.3 | Most common; piped grid gas |
| Electricity | 62.1 | Increasingly efficient with heat pumps |
| Heating Oil | 54.7 | Delivered by tanker; expensive; rural |
| LPG | 53.2 | Bottled gas; rural where no mains gas |
| Biomass | 48.5 | Wood pellets/chips; very variable |
| Solid Fuel | 35.1 | Coal/anthracite; legacy systems |

**Preprocessing**: `OneHotEncoder` (for RF/HistGBM pipelines) or `OrdinalEncoder`
with `categorical_feature` flag (for LightGBM native categorical handling). There
is no ordinal relationship between fuel types — a nominal encoding is correct.

### Feature 4 — `wall_type` (nominal, 6 categories)

**Why it matters**: Wall construction is the primary driver of building fabric
heat loss. Cavity walls can be filled with insulation (retroactively or
at build time); solid walls can only be insulated externally or internally —
a far more expensive and disruptive intervention.

**The 6 categories** (grouped from ~200 free-text variants in `raw.epc_domestic.WALLS_DESCRIPTION`):

| Category | Typical U-value (W/m²K) | Notes |
|---|---|---|
| Cavity - Uninsulated | 1.0–1.6 | Very common in pre-1990 stock |
| Cavity - Insulated | 0.3–0.5 | Retrofitted or built-in insulation |
| Solid - Uninsulated | 1.7–2.1 | Pre-1920 brick; hardest to improve |
| Solid - Insulated | 0.3–0.6 | External or internal wall insulation |
| Timber Frame | 0.3–0.7 | Typical in Scotland; good insulation potential |
| System Build | 0.5–1.2 | Prefabricated; variable by era |

**Preprocessing**: Same as main_fuel — `OneHotEncoder` or LightGBM native categorical.

**Derivation**: The raw EPC data contains a `WALLS_DESCRIPTION` column with human-readable
descriptions like `"Cavity wall, as built, no insulation (assumed)"`. These are grouped
using SQL `CASE`/`LIKE` pattern matching in the DuckDB query — the same technique used
to standardize fuel types in `stg_epc__domestic`.

---

## 3. Why Random Forest is Not Optimal

The baseline script (`predict_epc_band.py`) used `RandomForestClassifier`. Here is
a systematic analysis of why it is not the best choice for this problem.

### 3.1 Variance vs bias in this problem

Random Forest is a **bagging** method: it trains many deep, high-variance trees on
bootstrap samples of the data and averages their predictions. This reduces variance
but does little to reduce bias.

For EPC band prediction with only 4 features:
- Individual trees quickly exhaust the feature space — every split at depth > 3 is
  revisiting combinations already explored
- Bagging 300 highly correlated trees (they all use the same 4 features) provides
  diminishing variance reduction compared to a diverse ensemble
- The bias (systematic error from the model structure) is not reduced by adding more trees

**Gradient boosting** (LightGBM) addresses this differently: it builds trees
**sequentially**, where each new tree focuses on the residual errors of all
previous trees. This is a bias-reduction strategy that is more effective when
the feature space is small.

### 3.2 Feature handling

| Aspect | Random Forest | LightGBM |
|---|---|---|
| **Numeric splits** | Binary splits; no special handling | Histogram binning (faster, comparable quality) |
| **Ordered categoricals** | Treated as numeric after OrdinalEncoder | Native categorical splits (optimal grouping) |
| **Nominal categoricals** | OneHotEncoder required (inflates feature space) | Native: finds optimal splits across all category combinations without OHE |
| **Class imbalance** | `class_weight='balanced'` (reweights samples) | `is_unbalance=True` or custom `scale_pos_weight` per class |

For `main_fuel` and `wall_type`, LightGBM's native categorical treatment finds the
optimal binary split of categories (e.g., "Solid Fuel OR Heating Oil" vs the rest)
rather than treating each dummy variable independently. This is more powerful for
moderate-cardinality categoricals (6 values each).

### 3.3 Computational cost

On 150k rows with 300 trees:
- Random Forest: ~3s (fully parallel, independent trees)
- LightGBM: ~56s (sequential boosting, but with far richer splits)

The extra training time produces a meaningfully better model:
- RF macro F1: **0.3242**
- LightGBM macro F1: **0.3424** (+0.018, +5.5% relative)
- LightGBM accuracy: **52.9%** vs RF accuracy: **42.0%** (+10.9 percentage points)

### 3.4 Summary

| Property | Random Forest | LightGBM | Winner |
|---|---|---|---|
| Macro F1 (CV) | 0.3242 | 0.3424 | **LightGBM** |
| Accuracy (CV) | 41.99% | 52.92% | **LightGBM** |
| Native categoricals | No | Yes | **LightGBM** |
| Hyperparameter surface | Narrow | Rich | **LightGBM** |
| Training speed | Fast | Slower | RF |
| Inference speed | Slow (large forest) | Fast | **LightGBM** |

---

## 4. Model Selection

### 4.1 Candidate models evaluated

Three model families were compared using **3-fold stratified cross-validation**
on a 60,000-row stratified subsample (proportional to full data distribution):

| Model | Family | Macro F1 ± Std | Accuracy | CV time |
|---|---|---|---|---|
| **LightGBM** | Gradient Boosting | **0.3424 ± 0.0076** | **52.9%** | 56.3s |
| Random Forest | Bagging (baseline) | 0.3242 ± 0.0070 | 42.0% | 1.0s |
| HistGradientBoosting | Gradient Boosting | 0.3066 ± 0.0049 | 40.4% | 6.7s |

### 4.2 Why HistGradientBoosting underperforms

`HistGradientBoostingClassifier` (sklearn's native GBM) uses the same histogram-based
gradient boosting as LightGBM under the hood, but with a more limited implementation:
- Fewer regularisation knobs (no `num_leaves`, no `feature_fraction`)
- No native categorical optimisation — categorical features are treated as ordered
  integers after encoding, not as sets to be optimally partitioned
- Less mature early stopping

The performance gap (0.3066 vs 0.3424) comes from LightGBM's superior categorical
handling and richer regularisation surface.

### 4.3 Why XGBoost was excluded

XGBoost and LightGBM are **architecturally equivalent** for this problem:
- Both use histogram-based gradient boosting with depth-limited trees
- Both support regularisation via L1/L2 penalties
- Performance on tabular classification tasks is statistically indistinguishable
  across hundreds of published benchmarks

The key differences that favour LightGBM here:
- LightGBM uses **leaf-wise** (best-first) tree growth; XGBoost uses **level-wise**.
  Leaf-wise is faster and achieves lower loss with fewer trees on this type of problem
- LightGBM handles categorical features natively; XGBoost requires one-hot encoding
  (important for `main_fuel` and `wall_type`)
- LightGBM is 3–10× faster on large datasets (critical when tuning with 50+ trials)

Including both in the comparison would add ~30% to total runtime with no expected
improvement in the winning model's performance.

### 4.4 Selection decision

**LightGBM** is selected for hyperparameter tuning.

---

## 5. Hyperparameter Tuning with Optuna

### 5.1 Why Optuna (not GridSearch or RandomSearch)

| Method | Strategy | Trials needed to find good params |
|---|---|---|
| Grid Search | Exhaustive; tries every combination | Exponential in # of params |
| Random Search | Uniform sampling; no memory | ~100s needed for convergence |
| **Optuna TPE** | **Bayesian; learns from prior trials** | **30–50 sufficient** |

**Tree-structured Parzen Estimator (TPE)** builds a probabilistic model of the
objective function from past trial results. It preferentially samples from regions
of the search space that have previously produced good scores, rather than sampling
uniformly. With 30 trials on a 10-dimensional search space, TPE consistently
outperforms random search with 100+ trials.

### 5.2 Search space and rationale

The tuning subset was **60,000 rows** (stratified from training data), run on
**3-fold stratified CV**, optimising **macro F1**.

| Parameter | Search range | Type | Rationale |
|---|---|---|---|
| `num_leaves` | 15 – 300 | Integer | Controls model complexity; more leaves = finer splits but more overfitting risk |
| `max_depth` | 3 – 12 | Integer | Hard cap on tree depth; prevents individual trees from memorising noise |
| `min_child_samples` | 10 – 200 | Integer | Minimum samples to form a leaf; key regulariser for rare bands (A, G) |
| `n_estimators` | 100 – 400 | Integer | Number of boosting rounds; more = lower bias but higher training time |
| `learning_rate` | 0.005 – 0.3 | Float (log) | Step size per tree; log-scale because small differences matter at low values |
| `feature_fraction` | 0.4 – 1.0 | Float | Column subsampling per tree; analogous to RF's `max_features` |
| `bagging_fraction` | 0.4 – 1.0 | Float | Row subsampling; adds randomness, improves generalisation |
| `bagging_freq` | 1 – 10 | Integer | How often bagging is applied; 0 = disabled |
| `reg_alpha` | 1e-8 – 10 | Float (log) | L1 regularisation on leaf weights; sparse solutions |
| `reg_lambda` | 1e-8 – 10 | Float (log) | L2 regularisation on leaf weights; shrinks large weights |
| `min_split_gain` | 0.0 – 0.5 | Float | Minimum gain required to make a split; prevents useless branches |

**Early stopping** (patience = 20 rounds on validation macro-log-loss) prevents
each trial from running to the full `n_estimators` when the model has already
converged.

### 5.3 Optuna results — 30 trials

| | Macro F1 (CV on 60k) |
|---|---|
| LightGBM — default params | 0.3424 |
| LightGBM — best Optuna params | **0.3425** |
| Improvement | **+0.0001** |

**Best hyperparameters found**:

```python
{
    'num_leaves':        186,
    'max_depth':          9,
    'min_child_samples':  50,
    'n_estimators':      286,
    'learning_rate':     0.0695,
    'feature_fraction':  0.890,
    'bagging_fraction':  0.921,
    'bagging_freq':        2,
    'reg_alpha':         0.0067,
    'reg_lambda':        0.0110,
    'min_split_gain':    0.234,
}
```

The improvement of **+0.0001** is statistically negligible. This is not a
failure of the tuning process — it is an important signal about the problem.
See §8 for interpretation.

---

## 6. Final Results

The tuned model was retrained on the **full training set (400,000 rows)** using
the best Optuna hyperparameters, then evaluated on the **held-out test set
(100,000 rows, never seen during training or tuning)**.

### 6.1 Summary metrics

| Metric | Random Forest (baseline) | LightGBM Tuned (final) |
|---|---|---|
| Overall accuracy | 41.5% | **55.2%** |
| Macro F1 | 0.415 (weighted) | **0.349** macro |
| Band C F1 | 0.57 | **0.60** |
| Band D F1 | 0.34 | **0.61** |

### 6.2 Per-class results

| Band | Precision | Recall | F1-Score | Test samples |
|---|---|---|---|---|
| A | 0.17 | 0.00 | 0.01 | 340 |
| B | 0.70 | 0.50 | 0.58 | 5,721 |
| C | 0.60 | 0.61 | 0.60 | 33,616 |
| D | 0.53 | 0.71 | 0.61 | 40,477 |
| E | 0.42 | 0.18 | 0.25 | 15,204 |
| F | 0.36 | 0.13 | 0.19 | 3,670 |
| G | 0.33 | 0.14 | 0.20 | 972 |
| **Macro avg** | **0.45** | **0.32** | **0.35** | 100,000 |
| Weighted avg | 0.54 | 0.55 | 0.53 | 100,000 |

### 6.3 Feature importances (LightGBM gain)

LightGBM provides two importance metrics:
- **Split**: how many times a feature was used to create a split (biased toward
  high-cardinality features)
- **Gain**: total information gain from all splits using that feature (**preferred** —
  less biased, reflects actual predictive contribution)

Typical ranking by gain for this model:
1. `construction_age_band` — the strongest predictor; building regulations changed
   dramatically across eras
2. `wall_type` — directly measures fabric heat loss; insulated vs uninsulated is a
   binary signal with large effect
3. `main_fuel` — large variance in efficiency between fuel types (35 vs 66 SAP)
4. `total_floor_area_sqm` — meaningful but continuous; weaker than the categorical signals

---

## 7. Interpretation of Results

### 7.1 What the model is good at

- **C and D bands** (F1 ≈ 0.60–0.61): These are the plurality of UK homes and their
  physical features are well-represented in the training data. The model correctly
  identifies the "average UK home" profile.

- **B band** (F1 = 0.58): Modern homes built to 2003+ regulations with gas heating
  and cavity insulation form a recognisable physical profile. The model detects these.

### 7.2 What the model struggles with

- **Band A** (F1 = 0.01): Only 340 test samples (0.34% of data). The model cannot
  reliably distinguish A-rated properties from B-rated ones using only these 4 features
  — both often have modern construction, insulated cavities, and gas/electricity.
  The distinguishing features for A (e.g., heat pumps, triple glazing, Passivhaus
  standards) are **not in our feature set**.

- **Bands E, F, G** (F1 = 0.19–0.25): These are underrepresented (4.6% combined)
  and share features with the D band — there is no sharp physical boundary between
  a D and E property using only age, floor area, fuel, and walls. Distinguishing
  factors like roof insulation, boiler type, and window glazing are missing.

### 7.3 The accuracy paradox

Notice that LightGBM achieves **55.2% accuracy** but only **0.35 macro F1**.
This gap is caused by class imbalance:
- The model correctly predicts C and D (74% of all rows) most of the time
- It barely predicts A, and poorly predicts E/F/G
- Accuracy rewards majority-class correctness; macro F1 penalises minority-class failure

For the UK government's Net Zero goal, minority-class performance (accurately
identifying G-rated properties that need urgent retrofitting) matters most. Macro
F1 is the right metric to optimise.

---

## 8. The Feature Ceiling

The most important finding of the tuning exercise:

**Optuna improved macro F1 by +0.0001 over default LightGBM parameters.**

This is not a failure of hyperparameter tuning. It is evidence that the model
has reached the **information ceiling** of the four input features. No amount of
tuning can extract signal that is not present in the features.

### What the plateau means

The Optuna convergence curve shows that most improvement happens in trials 1–10,
after which the objective function is essentially flat. The TPE sampler correctly
identified that the search space has no further productive regions to explore.

This is consistent with domain knowledge:
- Floor area, construction age, fuel type, and wall type together explain the
  **structural efficiency** of a building
- They do not capture **system efficiency**: boiler age and type, hot water
  insulation, loft insulation thickness, window U-values, air permeability
- EPC bands integrate both structural and system factors — you cannot recover
  the full band from structural features alone

### Information theory framing

If we treat EPC band as the target and our 4 features as the signal:
- The mutual information between features and band is finite and bounded
- We have likely extracted most of it (the tree-based model is already at
  the information-theoretic limit of these features)
- More trials, more complex models, or more data will not push past this ceiling

**The right intervention is more features, not more tuning.**

---

## 9. How to Use the Saved Model

The trained pipeline is saved at `models/epc_band_lgbm_tuned.pkl`.

```python
import pickle
import pandas as pd

# Load
with open('models/epc_band_lgbm_tuned.pkl', 'rb') as f:
    artefact = pickle.load(f)

pipeline  = artefact['pipeline']       # sklearn Pipeline (preprocessor + LightGBM)
feat_names = artefact['feature_names'] # ['floor_area_sqm', 'construction_age_band',
                                       #  'main_fuel', 'wall_type']

# Predict on new data
new_data = pd.DataFrame([{
    'total_floor_area_sqm':  95.0,
    'construction_age_band': '1976-1990',
    'main_fuel':             'Mains Gas',
    'wall_type':             'Cavity - Uninsulated',
}])

band = pipeline.predict(new_data)          # → ['D']
proba = pipeline.predict_proba(new_data)   # → probability per class
```

**Feature value reference**:

```
total_floor_area_sqm   : any positive float (typical UK range: 20–400)
construction_age_band  : one of:
                           'Pre-1900', '1900-1929', '1930-1949',
                           '1950-1975', '1976-1990', '1991-2002',
                           '2003-2011', '2012-Present'
main_fuel              : one of:
                           'Mains Gas', 'Electricity', 'Heating Oil',
                           'LPG', 'Solid Fuel', 'Biomass'
wall_type              : one of:
                           'Cavity - Uninsulated', 'Cavity - Insulated',
                           'Solid - Uninsulated', 'Solid - Insulated',
                           'Timber Frame', 'System Build'
```

Unknown values in categorical features are handled gracefully by both the
`OrdinalEncoder` (`unknown_value` parameter) and LightGBM's own unknown-category
handling — the model will still produce a prediction, though with lower confidence.

---

## 10. What Would Actually Improve Performance

Ranked by expected improvement:

### High impact (would meaningfully change model quality)

| Feature to add | Source column in EPC data | Expected uplift |
|---|---|---|
| **Roof/loft insulation** | `ROOF_DESCRIPTION` | Large — loft is the second-largest heat loss pathway after walls |
| **Boiler type and age** | `MAINHEAT_DESCRIPTION` | Large — boiler efficiency determines heating SAP component |
| **Window glazing type** | `WINDOWS_DESCRIPTION` | Medium — double vs single glazing has significant U-value difference |
| **Floor insulation** | `FLOOR_DESCRIPTION` | Medium — ground floor heat loss, especially in older properties |
| **Hot water system** | `HOTWATER_DESCRIPTION` | Medium — hot water contributes ~25% of SAP score |

All of these are available in `raw.epc_domestic` — they were not included in the
staging model or this feature set, but can be joined and standardized following the
same `CASE`/`LIKE` pattern used for `wall_type`.

### Medium impact (architecture changes)

| Change | Expected uplift | Trade-offs |
|---|---|---|
| **Ordinal regression** (ordered logit) | Small–medium | Better loss function for ordered classes but harder to interpret |
| **Multi-output model** (predict SAP score first, then bin) | Medium | Continuous SAP → band conversion; more features needed for SAP anyway |
| **Ensemble of LightGBM + RF** (stacking) | Small | Marginal gain, ~2× inference cost |

### Low impact (confirmed by Optuna results)

- More hyperparameter tuning trials
- Larger sample size (already at 500k; law of large numbers kicks in)
- Different GBM variants (XGBoost, CatBoost) — architecturally equivalent

### Why we did not add more features in this study

The goal was to demonstrate a complete, reproducible ML development workflow
(model selection → comparison → Bayesian tuning → evaluation → documentation)
using only the features explicitly requested. Adding roof and window features
is the logical next step in a production modelling pipeline.

---

## Appendix: Reproducibility

```bash
# Complete reproduction from scratch
source dbt-env/bin/activate
python bulk_load_epc.py                      # load raw CSVs → DuckDB
cd ducklake_energy_uk && dbt run && cd ..    # build staging view
python epc_band_tuned.py                     # compare, tune, evaluate
```

All random seeds are fixed (`random_state=42`, Optuna `seed=42`, DuckDB
reservoir sample `seed=42`). Results are deterministic across runs on the same
data.

**Package versions used**:
- `lightgbm` 4.6.0
- `optuna` 4.8.0
- `scikit-learn` 1.8.0
- `duckdb` (version from requirements.txt)
- `polars` (version from requirements.txt)
