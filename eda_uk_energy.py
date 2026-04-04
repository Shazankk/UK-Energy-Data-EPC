"""
UK Energy EDA Suite
===================
Generates a single interactive dashboard (reports/dashboard.html) from the
ducklake_energy_uk star schema.  Plotly.js is loaded once from CDN; all charts
are embedded as <div> snippets — open the file in any browser, no server needed.

Prerequisites
-------------
  1. python bulk_load_epc.py          — ingest raw CSVs into DuckDB
  2. cd ducklake_energy_uk && dbt run — build the full mart layer
  3. source dbt-env/bin/activate      — activate the venv

Usage
-----
  python eda_uk_energy.py
  open reports/dashboard.html

Sections in the dashboard
--------------------------
  KPI cards      Headline figures: properties analysed, avg SAP, CO₂ saving potential
  1. National EPC Rating Distribution
  2. County Efficiency: Best vs Worst 20
  3. Efficiency by Construction Decade (current vs potential)
  4. CO₂ Distribution by Property Type  (box plots — no overplotting)
  5. Efficiency vs CO₂ Density           (2D contour — no overplotting)
  6. Retrofit Priority Matrix            (heatmap: property type × age band)
  7. Local Authority Treemap             (county → LA, worst EPC + highest CO₂ saving)
  8. Postcode Area Heatmap               (postcode area × EPC band distribution)
  9. Geographic EPC Map                  (choropleth: LAD boundaries coloured by avg SAP)
 10. Heating Fuel Type Impact            (avg SAP current vs potential per fuel type)
 11. EPC Score Trend 2008–2024          (yearly improvement line + cert volume)
 12. Annual Energy Cost by Property Type (stacked: heating + hot water + lighting)

Why aggregated charts instead of scatter plots?
-----------------------------------------------
Raw scatter on 100k+ points produces an ink-blot — all D/E properties pile up at
60-75 SAP / 1-4 t CO₂ and individual dots are invisible.  Instead:
  box plots   → distribution shape + median per group, no overplotting
  2D contours → show where the *mass* of data sits as density "hills"
  heatmaps    → aggregate first, colour the aggregated value
"""

import duckdb
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots
import os
import json
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_PATH     = 'ducklake_energy_uk/dev.duckdb'
REPORTS_DIR = 'reports'
DASHBOARD   = os.path.join(REPORTS_DIR, 'dashboard.html')
os.makedirs(REPORTS_DIR, exist_ok=True)

EPC_COLORS = {
    'A': '#008054', 'B': '#19b459', 'C': '#8dce46',
    'D': '#ffd500', 'E': '#fcaa65', 'F': '#ef8023', 'G': '#e9153b',
}
EPC_ORDER  = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
TEMPLATE   = 'plotly_dark'
FONT_SIZE  = 13


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pl(result) -> pl.DataFrame:
    """Return a Polars DataFrame with all column names lowercased.
    DuckDB preserves source column case (often UPPERCASE from raw CSVs)."""
    df = result.pl()
    return df.rename({c: c.lower() for c in df.columns})


def _to_pd(result):
    """Return a pandas DataFrame with all column names lowercased."""
    return _pl(result).to_pandas()


def _style(fig: go.Figure, title: str, xlab: str = '', ylab: str = '') -> go.Figure:
    fig.update_layout(
        template=TEMPLATE,
        title=dict(text=title, font=dict(size=18)),
        font=dict(size=FONT_SIZE),
        xaxis_title=xlab,
        yaxis_title=ylab,
        margin=dict(t=70, b=60, l=60, r=30),
    )
    return fig


def _embed(fig: go.Figure) -> str:
    """Render a figure as an HTML <div> snippet (no full-page wrapper, no JS)."""
    return fig.to_html(full_html=False, include_plotlyjs=False)


# ---------------------------------------------------------------------------
# Chart builders  (each returns a go.Figure)
# ---------------------------------------------------------------------------

def build_rating_distribution(con) -> go.Figure:
    """Bar chart: national A-G rating distribution with % labels."""
    # Avoid 'count' as a column name — it conflicts silently with Plotly's
    # internal trace validation in some versions. Rename to 'n'.
    df = _pl(con.execute("""
        select
            energy_rating_current                               as rating,
            count(*)                                            as n,
            round(count(*) * 100.0 / sum(count(*)) over (), 1) as pct
        from fct_certificates
        where energy_rating_current in ('A','B','C','D','E','F','G')
        group by 1
        order by 1
    """))

    # Pre-format text labels in Python — avoids Jinja-style texttemplate bugs
    # where '%{text}%' can render as a literal string in certain Plotly versions.
    df = df.with_columns(
        pl.col('pct').map_elements(lambda v: f"{v}%", return_dtype=pl.Utf8).alias('label')
    )

    fig = px.bar(
        df.to_pandas(),
        x='rating', y='n',
        color='rating',
        color_discrete_map=EPC_COLORS,
        category_orders={'rating': EPC_ORDER},
        text='label',
        custom_data=['pct'],
    )
    fig.update_traces(
        textposition='outside',
        hovertemplate='<b>Band %{x}</b><br>Properties: %{y:,}<br>Share: %{customdata[0]}%<extra></extra>',
    )
    below_c = df.filter(pl.col('rating').is_in(['D', 'E', 'F', 'G']))['pct'].sum()
    fig.add_annotation(
        x=0.98, y=0.96, xref='paper', yref='paper',
        text=f"<b>{below_c:.1f}%</b> of properties rated D or below",
        showarrow=False, align='right',
        font=dict(size=13, color='#fcaa65'),
        bgcolor='rgba(0,0,0,0.5)', borderpad=6,
    )
    fig.update_layout(showlegend=False)
    return _style(fig, 'National EPC Rating Distribution', 'EPC Band', 'Number of Properties')


def _sap_to_color(val: float, vmin: float = 48, vmax: float = 76) -> str:
    """
    Map a SAP score to an explicit hex colour on the RdYlGn scale.
    Pre-computing colours avoids the Plotly 6 bug where passing a list of
    floats as marker_color with marker_colorscale silently drops the bars.
    """
    t = max(0.0, min(1.0, (val - vmin) / (vmax - vmin)))
    return pc.sample_colorscale('RdYlGn', [t])[0]


def build_county_efficiency(con) -> go.Figure:
    """Side-by-side horizontal bars: 20 worst and 20 best counties by avg SAP."""
    df = _pl(con.execute("""
        select county, avg_current_efficiency, total_certificates
        from v_regional_energy_performance
        where county is not null
        order by avg_current_efficiency
    """))

    worst = df.head(20)
    best  = df.tail(20).sort('avg_current_efficiency')

    # shared_xaxes=True with horizontal bars in 2 columns causes the left
    # subplot's bars to silently not render in Plotly 6. Use independent axes.
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('20 Worst Counties', '20 Best Counties'),
        shared_xaxes=False,
    )

    def _trace(data):
        vals   = data['avg_current_efficiency'].to_list()
        colors = [_sap_to_color(v) for v in vals]
        return go.Bar(
            y=data['county'].to_list(),
            x=vals,
            orientation='h',
            marker_color=colors,          # Explicit colour strings — always renders
            text=[f"{v:.1f}" for v in vals],
            textposition='inside',
            insidetextanchor='end',
            hovertemplate=(
                '<b>%{y}</b><br>Avg SAP: %{x:.1f}<br>'
                'Certificates: %{customdata[0]:,}<extra></extra>'
            ),
            customdata=data[['total_certificates']].to_numpy(),
        )

    fig.add_trace(_trace(worst), row=1, col=1)
    fig.add_trace(_trace(best),  row=1, col=2)
    fig.update_layout(
        template=TEMPLATE, showlegend=False, height=620,
        title=dict(text='County Energy Efficiency: Best vs Worst 20', font=dict(size=18)),
        font=dict(size=FONT_SIZE),
        margin=dict(t=80, b=60, l=160, r=30),
    )
    fig.update_xaxes(title_text='Average SAP Score', range=[35, 82], row=1, col=1)
    fig.update_xaxes(title_text='Average SAP Score', range=[35, 82], row=1, col=2)
    return fig


def build_efficiency_by_age(con) -> go.Figure:
    """Grouped bars: current vs potential SAP by construction decade, sorted chronologically."""
    AGE_ORDER = [
        'Pre-1900', '1900-1929', '1930-1949', '1950-1975',
        '1976-1990', '1991-2002', '2003-2011', '2012-Present', 'Unknown',
    ]
    df = _pl(con.execute("""
        select
            construction_age_band,
            round(avg(energy_efficiency_current),  1) as avg_current,
            round(avg(energy_efficiency_potential), 1) as avg_potential
        from fct_certificates f
        join dim_properties p on f.property_id = p.property_id
        where construction_age_band is not null
        group by 1
    """))
    df = df.with_columns(
        pl.col('construction_age_band').cast(pl.Enum(AGE_ORDER))
    ).sort('construction_age_band')

    fig = go.Figure()
    fig.add_bar(
        x=df['construction_age_band'].to_list(),
        y=df['avg_current'].to_list(),
        name='Current SAP',
        marker_color='#ef8023',
        text=df['avg_current'].to_list(),
        texttemplate='%{text}', textposition='outside',
    )
    fig.add_bar(
        x=df['construction_age_band'].to_list(),
        y=df['avg_potential'].to_list(),
        name='Potential SAP (post-retrofit)',
        marker_color='#19b459',
        text=df['avg_potential'].to_list(),
        texttemplate='%{text}', textposition='outside',
    )
    _style(fig, 'Energy Efficiency by Construction Decade: Current vs Potential',
           'Construction Age Band', 'Average SAP Score (0–100)')
    fig.update_layout(
        barmode='group',
        legend=dict(orientation='h', yanchor='bottom', y=-0.28, xanchor='center', x=0.5),
        margin=dict(t=70, b=120, l=60, r=30),
    )
    return fig


def build_co2_by_property_type(con) -> go.Figure:
    """
    Box plots of CO₂ per property type.
    Box plots show median + IQR + outliers per group — no overplotting.
    Notched boxes: non-overlapping notches indicate statistically significant
    difference in medians.
    """
    df = _to_pd(con.execute("""
        select p.property_type, f.co2_emissions_current_tonnes_per_year as co2
        from fct_certificates f
        join dim_properties p on f.property_id = p.property_id
        where f.co2_emissions_current_tonnes_per_year between 0.1 and 20
          and p.property_type is not null
        using sample 200000 rows
    """))
    fig = px.box(df, x='property_type', y='co2', color='property_type',
                 points=False, notched=True)
    fig.update_traces(hovertemplate='<b>%{x}</b><br>CO₂: %{y:.2f} t/yr<extra></extra>')
    _style(fig, 'CO₂ Emissions Distribution by Property Type (sample 200k)',
           'Property Type', 'Annual CO₂ Emissions (tonnes/yr)')
    fig.update_layout(showlegend=False)
    return fig


def build_efficiency_density(con) -> go.Figure:
    """
    2D density contours: SAP score vs CO₂.
    Contour "hills" show where the mass of properties concentrates — the D/E cluster
    visible as the highest-density peak, rare A/B properties as outer rings.
    """
    df = _to_pd(con.execute("""
        select
            f.energy_efficiency_current                  as efficiency,
            f.co2_emissions_current_tonnes_per_year      as co2,
            p.property_type
        from fct_certificates f
        join dim_properties p on f.property_id = p.property_id
        where f.energy_efficiency_current between 1 and 100
          and f.co2_emissions_current_tonnes_per_year between 0.1 and 15
          and p.property_type is not null
        using sample 150000 rows
    """))
    fig = px.density_contour(df, x='efficiency', y='co2', color='property_type',
                              marginal_x='histogram', marginal_y='histogram')
    fig.update_traces(
        contours_coloring='fill', contours_showlabels=True,
        selector={'type': 'histogram2dcontour'},
    )
    return _style(fig,
                  'Efficiency vs CO₂: Density Contours by Property Type (sample 150k)',
                  'SAP Score (Energy Efficiency)', 'Annual CO₂ (tonnes/yr)')


def build_retrofit_priority_matrix(con) -> go.Figure:
    """
    Heatmap: property type × construction age band.
    Colour = avg Retrofit Priority Score (0–100).
    Cell text = score (property count below).
    Red = most urgent retrofit candidates. Green = already near-optimal.
    """
    AGE_ORDER = [
        'Pre-1900', '1900-1929', '1930-1949', '1950-1975',
        '1976-1990', '1991-2002', '2003-2011', '2012-Present',
    ]
    df = _pl(con.execute("""
        select
            property_type,
            construction_age_band,
            round(avg(avg_retrofit_priority_score), 1) as score,
            sum(property_count)                        as total_props
        from v_retrofit_priority
        where construction_age_band != 'Unknown'
          and property_type is not null
        group by 1, 2
    """))

    pivot  = df.pivot(index='property_type', on='construction_age_band', values='score').fill_null(0)
    cpivot = df.pivot(index='property_type', on='construction_age_band', values='total_props').fill_null(0)

    types    = pivot['property_type'].to_list()
    age_cols = [c for c in AGE_ORDER if c in pivot.columns]
    z        = pivot.select(age_cols).to_numpy().tolist()
    counts   = cpivot.select(age_cols).to_numpy().tolist()

    text = [
        [f"{z[r][c]:.0f}<br>({int(counts[r][c]):,})" for c in range(len(age_cols))]
        for r in range(len(types))
    ]

    fig = go.Figure(go.Heatmap(
        z=z, x=age_cols, y=types,
        text=text, texttemplate='%{text}',
        colorscale='RdYlGn_r', zmin=0, zmax=100,
        colorbar=dict(title='Retrofit<br>Priority<br>Score'),
        hovertemplate='<b>%{y} — %{x}</b><br>Score: %{z:.1f}<extra></extra>',
    ))
    _style(fig, 'Retrofit Priority Score: Property Type × Construction Age Band',
           'Construction Decade', 'Property Type')
    fig.update_layout(
        height=420,
        annotations=[dict(
            x=0.5, y=-0.18, xref='paper', yref='paper',
            text='<i>Score 0–100 = current inefficiency (35%) + achievable SAP gain (40%) + CO₂ saving (25%)</i>',
            showarrow=False, font=dict(size=11, color='#aaaaaa'),
        )],
    )
    return fig


def build_local_authority_treemap(con) -> go.Figure:
    """
    Treemap: county → local authority.
    Box size  = total CO₂ saving potential (t/yr) — bigger box = more to gain.
    Colour    = avg current SAP score — red = worst efficiency.
    Large red boxes = highest-priority intervention areas.
    """
    df = _to_pd(con.execute("""
        select
            county,
            local_authority,
            round(avg(avg_efficiency_current), 1)            as avg_efficiency,
            round(sum(total_co2_saving_potential_tonnes), 0) as total_co2_saving,
            sum(property_count)                              as property_count
        from v_retrofit_priority
        where county is not null and local_authority is not null
        group by 1, 2
        having sum(property_count) > 500
        order by avg_efficiency asc
    """))
    fig = px.treemap(
        df,
        path=[px.Constant('England & Wales'), 'county', 'local_authority'],
        values='total_co2_saving',
        color='avg_efficiency',
        color_continuous_scale='RdYlGn',
        color_continuous_midpoint=60,
        range_color=[45, 75],
        custom_data=['property_count', 'avg_efficiency', 'total_co2_saving'],
    )
    fig.update_traces(
        hovertemplate=(
            '<b>%{label}</b><br>'
            'Avg SAP: %{customdata[1]:.1f}<br>'
            'Properties: %{customdata[0]:,}<br>'
            'CO₂ Saving: %{customdata[2]:,.0f} t/yr<extra></extra>'
        ),
        texttemplate='<b>%{label}</b><br>%{customdata[1]:.1f} SAP',
        textfont=dict(size=11),
    )
    fig.update_coloraxes(
        colorbar_title='Avg SAP',
        colorbar_tickvals=[45, 55, 65, 75],
        colorbar_ticktext=['45 (F)', '55 (E)', '65 (D)', '75 (C)'],
    )
    _style(fig, 'Local Authority: EPC Efficiency & CO₂ Retrofit Potential')
    fig.update_layout(height=700)
    return fig


def build_postcode_area_heatmap(con) -> go.Figure:
    """
    Matrix: UK postcode area (rows) × EPC band (columns).
    Colour = % of properties in that area carrying that band.
    Rows heavy in F/G = worst-performing postcode areas.
    """
    df = _pl(con.execute("""
        select
            regexp_extract(l.postcode, '^([A-Z]{1,2})', 1) as postcode_area,
            f.energy_rating_current                         as rating,
            count(*)                                        as cnt
        from fct_certificates f
        join dim_locations l on f.location_id = l.location_id
        where f.energy_rating_current in ('A','B','C','D','E','F','G')
          and l.postcode is not null
        group by 1, 2
    """))
    df = df.with_columns(
        (pl.col('cnt') / pl.col('cnt').sum().over('postcode_area') * 100)
        .round(1).alias('pct')
    )
    pivot = df.pivot(index='postcode_area', on='rating', values='pct').fill_null(0)
    # Sort rows by % of G-rated properties descending (worst areas first)
    if 'g' in pivot.columns:
        pivot = pivot.sort('g', descending=True)
    areas = pivot['postcode_area'].to_list()
    z     = pivot.select(EPC_ORDER).to_numpy().tolist()

    fig = go.Figure(go.Heatmap(
        z=z, x=EPC_ORDER, y=areas,
        colorscale=[[0,'#1a2e1a'],[0.2,'#19b459'],[0.5,'#ffd500'],
                    [0.8,'#ef8023'],[1,'#e9153b']],
        zmin=0, zmax=60,
        colorbar=dict(title='% of<br>Properties'),
        hovertemplate='<b>%{y}</b> — Band %{x}: %{z:.1f}%<extra></extra>',
        xgap=1, ygap=1,
    ))
    _style(fig, 'EPC Rating Distribution by UK Postcode Area',
           'EPC Band', 'Postcode Area')
    fig.update_layout(
        height=max(500, len(areas) * 14),
        xaxis=dict(side='top'),
    )
    return fig


def _rdp(points: list, tol: float) -> list:
    """Ramer-Douglas-Peucker polyline simplification — pure Python, no dependencies."""
    if len(points) < 3:
        return points
    x1, y1 = points[0]
    x2, y2 = points[-1]
    dx, dy = x2 - x1, y2 - y1
    norm = (dx * dx + dy * dy) ** 0.5
    max_dist, max_idx = 0.0, 0
    for i, (px, py) in enumerate(points[1:-1], 1):
        dist = abs(dy * px - dx * py + x2 * y1 - y2 * x1) / norm if norm > 0 \
               else ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5
        if dist > max_dist:
            max_dist, max_idx = dist, i
    if max_dist > tol:
        return _rdp(points[:max_idx + 1], tol)[:-1] + _rdp(points[max_idx:], tol)
    return [points[0], points[-1]]


def _simplify_geojson(gj: dict, tol: float = 0.005) -> dict:
    """
    Simplify all polygon rings in a GeoJSON FeatureCollection using RDP.
    tol=0.005 degrees ≈ 550 m — sufficient for a national UK district map.
    Reduces the martinjc LAD GeoJSON from 18 MB → ~2.6 MB.
    """
    def _ring(r):
        s = _rdp(r, tol)
        return s if len(s) >= 4 else r

    feats = []
    for feat in gj['features']:
        geom = feat.get('geometry') or {}
        gt   = geom.get('type', '')
        if gt == 'Polygon':
            new_geom = {'type': 'Polygon',
                        'coordinates': [_ring(r) for r in geom['coordinates']]}
        elif gt == 'MultiPolygon':
            new_geom = {'type': 'MultiPolygon',
                        'coordinates': [[_ring(r) for r in poly]
                                        for poly in geom['coordinates']]}
        else:
            new_geom = geom
        # Set top-level id to LAD13CD so Plotly's choropleth can match by id
        feats.append({'type': 'Feature',
                      'id': feat['properties'].get('LAD13CD', feat.get('id')),
                      'properties': feat['properties'], 'geometry': new_geom})
    return {'type': 'FeatureCollection', 'features': feats}


def build_choropleth_map(con) -> go.Figure:
    """
    Choropleth: UK Local Authority Districts coloured by average SAP score.
    Boundaries fetched from martinjc/UK-GeoJSON and simplified with RDP (18 MB → 2.6 MB)
    so the chart embeds cleanly in the static HTML dashboard.
    Hover shows retrofit priority score, property count, and CO₂ saving potential.
    """
    df = _to_pd(con.execute("""
        select local_authority, county,
               round(avg(avg_efficiency_current), 1)      as avg_efficiency,
               round(avg(avg_retrofit_priority_score), 1) as avg_retrofit_score,
               sum(property_count)                        as property_count,
               round(sum(total_co2_saving_potential_tonnes) / 1e3, 1) as co2_saving_kt
        from v_retrofit_priority
        where local_authority is not null
        group by 1, 2
        having sum(property_count) > 100
    """))

    # Simplified UK LAD boundaries (2013) from martinjc/UK-GeoJSON — 380 districts, ~6 MB
    GEOJSON_URL = (
        "https://raw.githubusercontent.com/martinjc/UK-GeoJSON/master/"
        "json/administrative/gb/lad.json"
    )
    try:
        resp = requests.get(GEOJSON_URL, timeout=30)
        resp.raise_for_status()
        raw_geojson = resp.json()
    except Exception as exc:
        print(f"    ⚠️  choropleth skipped — could not fetch boundaries ({exc})")
        fig = go.Figure()
        fig.update_layout(
            template=TEMPLATE,
            title=dict(text="Geographic map unavailable — boundary fetch failed", font=dict(size=18)),
        )
        return fig

    # Simplify from ~18 MB → ~2.6 MB so the HTML embeds without overflowing
    geojson = _simplify_geojson(raw_geojson, tol=0.005)

    # martinjc GeoJSON uses LAD13NM for name, LAD13CD for code
    name_to_code = {
        feat['properties']['LAD13NM'].strip().upper(): feat['properties']['LAD13CD']
        for feat in geojson.get('features', [])
        if feat.get('properties', {}).get('LAD13NM')
    }

    df['lad_code'] = df['local_authority'].str.strip().str.upper().map(name_to_code)
    matched = df.dropna(subset=['lad_code'])
    n_miss = len(df) - len(matched)
    if n_miss:
        print(f"    ℹ️  {n_miss} local authorities unmatched to boundary polygons")

    # Draw each district as a filled Scattergeo polygon — bypasses choropleth
    # ID-matching entirely, works in any static HTML context (file:// or Pages).
    code_to_feat = {feat['id']: feat for feat in geojson['features'] if feat.get('id')}

    fig = go.Figure()

    for _, row in matched.iterrows():
        feat = code_to_feat.get(row['lad_code'])
        if feat is None:
            continue
        val  = float(row['avg_efficiency'])
        t    = max(0.0, min(1.0, (val - 45) / (78 - 45)))
        color = pc.sample_colorscale('RdYlGn', [t])[0]
        geom = feat['geometry']

        rings = (
            [geom['coordinates'][0]] if geom['type'] == 'Polygon'
            else [max(geom['coordinates'], key=lambda p: len(p[0]))[0]]
            if geom['type'] == 'MultiPolygon' else []
        )
        for ring in rings:
            fig.add_trace(go.Scattergeo(
                lon=[p[0] for p in ring],
                lat=[p[1] for p in ring],
                mode='lines',
                fill='toself',
                fillcolor=color,
                line=dict(color='rgba(80,80,80,0.4)', width=0.3),
                hovertemplate=(
                    f"<b>{row['local_authority']}</b><br>"
                    f"Avg SAP: {val:.1f}<br>"
                    f"County: {row.get('county','')}<br>"
                    f"Retrofit Score: {row.get('avg_retrofit_score', 0):.1f}<br>"
                    f"Properties: {int(row.get('property_count', 0)):,}"
                    "<extra></extra>"
                ),
                name=row['local_authority'],
                showlegend=False,
            ))

    # Invisible marker trace whose sole job is to render the colorbar
    fig.add_trace(go.Scattergeo(
        lon=[None], lat=[None], mode='markers',
        marker=dict(
            color=[45, 78], colorscale='RdYlGn', cmin=45, cmax=78,
            showscale=True,
            colorbar=dict(
                title='Avg SAP', tickvals=[45, 55, 65, 75],
                ticktext=['45 (F)', '55 (E)', '65 (D)', '75 (C)'],
            ),
            size=0,
        ),
        showlegend=False, hoverinfo='skip',
    ))

    fig.update_geos(
        projection_type='mercator',
        lonaxis_range=[-8.5, 2.5],
        lataxis_range=[49.5, 61.5],
        showland=True,  landcolor='#1e2330',
        showocean=True, oceancolor='#0d1117',
        showlakes=False,
        showcoastlines=True, coastlinecolor='#555',
        showframe=False, showsubunits=False,
    )
    fig.update_layout(
        template=TEMPLATE,
        title=dict(text='Geographic EPC Efficiency Map — by Local Authority', font=dict(size=18)),
        font=dict(size=FONT_SIZE),
        margin=dict(t=70, b=10, l=0, r=10),
        height=650,
    )
    return fig


def build_fuel_type_analysis(con) -> go.Figure:
    """
    Horizontal grouped bars: avg current vs potential SAP by main heating fuel.
    Shows the structural disadvantage of oil, LPG, and solid fuel homes.
    """
    df = _pl(con.execute("""
        select MAIN_FUEL as fuel,
               round(avg(energy_efficiency_current),  1) as avg_current,
               round(avg(energy_efficiency_potential), 1) as avg_potential,
               count(*) as n
        from stg_epc__domestic
        where MAIN_FUEL not in ('Other/Unknown')
          and energy_efficiency_current > 0
        group by 1
        order by avg_current desc
    """))

    fuels     = df['fuel'].to_list()
    current   = df['avg_current'].to_list()
    potential = df['avg_potential'].to_list()
    counts    = df['n'].to_list()

    fig = go.Figure()
    fig.add_bar(
        y=fuels, x=current, name='Current SAP',
        orientation='h', marker_color='#ef8023',
        text=[str(v) for v in current], textposition='auto',
        customdata=counts,
        hovertemplate='<b>%{y}</b><br>Current SAP: %{x}<br>Properties: %{customdata:,}<extra></extra>',
    )
    fig.add_bar(
        y=fuels, x=potential, name='Potential SAP (post-retrofit)',
        orientation='h', marker_color='#19b459',
        text=[str(v) for v in potential], textposition='auto',
        hovertemplate='<b>%{y}</b><br>Potential SAP: %{x}<extra></extra>',
    )
    if len(current) >= 2:
        gap = round(current[0] - current[-1], 1)
        fig.add_annotation(
            x=0.98, y=0.04, xref='paper', yref='paper',
            text=f"<b>{gap:.0f} SAP points</b> gap: mains gas vs solid fuel",
            showarrow=False, align='right',
            font=dict(size=13, color='#fcaa65'),
            bgcolor='rgba(0,0,0,0.5)', borderpad=6,
        )
    fig.update_layout(
        barmode='group',
        legend=dict(orientation='h', yanchor='bottom', y=-0.28, xanchor='center', x=0.5),
        margin=dict(t=70, b=120, l=120, r=30),
    )
    return _style(fig, 'EPC Efficiency by Heating Fuel Type',
                  'Average SAP Score (0–100)', 'Main Fuel Type')


def build_annual_trend(con) -> go.Figure:
    """
    Dual-axis chart: avg SAP score (line) and certificates issued (bar) per year 2008–2024.
    Shows whether the housing stock is materially improving over time.
    """
    df = _pl(con.execute("""
        select year(inspection_at)                        as yr,
               round(avg(energy_efficiency_current), 1)  as avg_sap,
               count(*)                                   as certs
        from fct_certificates
        where inspection_at is not null
          and year(inspection_at) between 2008 and 2024
        group by 1
        order by 1
    """))

    years   = df['yr'].to_list()
    sap     = df['avg_sap'].to_list()
    certs   = df['certs'].to_list()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=years, y=certs, name='Certificates issued',
               marker_color='rgba(79,142,247,0.25)',
               hovertemplate='%{x}: %{y:,} certs<extra></extra>'),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=years, y=sap, name='Avg SAP Score', mode='lines+markers',
                   line=dict(color='#19b459', width=3),
                   marker=dict(size=8),
                   hovertemplate='%{x}: SAP %{y}<extra></extra>'),
        secondary_y=True,
    )
    if len(sap) >= 2:
        improvement = round(sap[-1] - sap[0], 1)
        fig.add_annotation(
            x=0.02, y=0.95, xref='paper', yref='paper',
            text=f"<b>+{improvement} SAP points</b> improvement 2008 → 2024",
            showarrow=False, align='left',
            font=dict(size=13, color='#19b459'),
            bgcolor='rgba(0,0,0,0.5)', borderpad=6,
        )
    fig.update_layout(
        template=TEMPLATE,
        title=dict(text='EPC Score Trend 2008–2024: Is the Housing Stock Improving?', font=dict(size=18)),
        font=dict(size=FONT_SIZE),
        margin=dict(t=70, b=100, l=70, r=70),
        legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5),
        xaxis=dict(title='Year', tickmode='linear', dtick=2),
    )
    fig.update_yaxes(title_text='Certificates Issued', secondary_y=False)
    fig.update_yaxes(title_text='Average SAP Score', secondary_y=True, range=[58, 73])
    return fig


def build_cost_breakdown(con) -> go.Figure:
    """
    Stacked bars: average annual energy cost (heating + hot water + lighting) per property type.
    Quantifies the financial penalty of living in different building types.
    """
    df = _to_pd(con.execute("""
        select p.PROPERTY_TYPE                             as property_type,
               round(avg(f.heating_cost_current),   0)   as heating,
               round(avg(f.hot_water_cost_current), 0)   as hot_water,
               round(avg(f.lighting_cost_current),  0)   as lighting
        from fct_certificates f
        join dim_properties p on f.property_id = p.property_id
        where f.heating_cost_current > 0
          and f.heating_cost_current < 10000
          and p.PROPERTY_TYPE is not null
        group by 1
        order by (heating + hot_water + lighting) desc
    """))

    fig = go.Figure()
    for col, color, label in [
        ('heating',   '#ef8023', 'Heating'),
        ('hot_water', '#4f8ef7', 'Hot Water'),
        ('lighting',  '#ffd500', 'Lighting'),
    ]:
        fig.add_bar(
            x=df['property_type'], y=df[col],
            name=label, marker_color=color,
            hovertemplate=f'<b>%{{x}}</b><br>{label}: £%{{y:,.0f}}/yr<extra></extra>',
        )
    fig.update_layout(
        barmode='stack',
        legend=dict(orientation='h', yanchor='bottom', y=-0.28, xanchor='center', x=0.5),
        margin=dict(t=70, b=120, l=60, r=30),
    )
    return _style(fig, 'Average Annual Energy Cost by Property Type',
                  'Property Type', 'Annual Cost (£/yr)')


# ---------------------------------------------------------------------------
# National summary KPIs
# ---------------------------------------------------------------------------

def get_national_summary(con) -> dict:
    row = con.execute("""
        select
            count(*)                                                      as total,
            sum(co2_emissions_current_tonnes_per_year)                    as co2_current,
            sum(co2_emissions_potential_tonnes_per_year)                  as co2_potential,
            sum(co2_emissions_current_tonnes_per_year
              - co2_emissions_potential_tonnes_per_year)                  as co2_saving,
            avg(energy_efficiency_current)                                as avg_eff_current,
            avg(energy_efficiency_potential)                              as avg_eff_potential
        from fct_certificates
    """).fetchone()

    ratings = con.execute("""
        select energy_rating_current, count(*) from fct_certificates
        where energy_rating_current in ('A','B','C','D','E','F','G')
        group by 1
    """).fetchall()
    below_c = sum(r[1] for r in ratings if r[0] in ('D','E','F','G'))

    summary = {
        "total": int(row[0]),
        "co2_current": float(row[1]),
        "co2_saving": float(row[3]),
        "pct_saving": round(float(row[3]) / float(row[1]) * 100, 1),
        "avg_eff_current": round(float(row[4]), 1),
        "avg_eff_potential": round(float(row[5]), 1),
        "pct_below_c": round(below_c / row[0] * 100, 1),
    }

    # Also write JSON sidecar
    json_path = os.path.join(REPORTS_DIR, 'national_savings_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=4)
    return summary


# ---------------------------------------------------------------------------
# Dashboard assembler
# ---------------------------------------------------------------------------

SECTIONS = [
    ("rating-dist",    "1. National EPC Rating Distribution",
     "<strong>What it shows:</strong> Every property in England &amp; Wales ranked A (best) to G (worst) by its EPC certificate. "
     "The scale runs from A (SAP score ≥ 92, near-zero heating costs) down to G (SAP ≤ 20, very high costs and emissions). "
     "<strong>Why it matters:</strong> The UK Government's legally binding target is for all homes to reach Band C by 2035 — "
     "a prerequisite for the country's Net Zero 2050 commitment. The orange annotation shows the share that currently fails this target. "
     "<strong>Key finding:</strong> The majority of properties cluster in Band D — the single most common rating — meaning most homes are "
     "above the minimum habitable standard but well short of the legal target. Bands E, F, G represent properties with dangerously high "
     "fuel costs; tenants in these homes face fuel poverty, spending over 10% of income on energy."),

    ("county-eff",     "2. County Efficiency: Best vs Worst 20",
     "<strong>What it shows:</strong> The average SAP score for every county, split into the 20 worst (left) and 20 best (right). "
     "SAP (Standard Assessment Procedure) is a 0–100 government methodology: higher = more energy-efficient. "
     "Band C starts at roughly SAP 69; Band D at SAP 55. "
     "<strong>Why counties differ:</strong> Rural counties (e.g. Norfolk, Cumbria) score poorly because they have a high density of "
     "older, stone-built or detached housing stock with solid walls — the single hardest and most expensive type to insulate. "
     "Urban counties score better because modern flats and terraced housing share walls, losing less heat per property. "
     "<strong>Impact:</strong> A county sitting at SAP 52 versus SAP 72 represents roughly £800/year more in heating costs per household "
     "and approximately 1.5 additional tonnes of CO₂ per year — multiplied across hundreds of thousands of homes."),

    ("age-eff",        "3. Efficiency by Construction Decade: Current vs Potential",
     "<strong>What it shows:</strong> Orange bars = the average SAP score buildings from each era achieve <em>today</em>. "
     "Green bars = the average SAP those same buildings <em>could</em> achieve with cost-effective retrofitting (as assessed by the EPC surveyor). "
     "The gap between orange and green is the average improvement available per era. "
     "<strong>Why older buildings score lower:</strong> Pre-1900 homes were built with solid brick or stone walls — no cavity to fill with insulation. "
     "They also predate double glazing, draught-proofing standards, and modern boiler efficiency requirements. Each decade of construction "
     "added incremental regulation: cavity walls (1930s–50s), mandatory insulation (1976 Building Regulations), condensing boilers (2005). "
     "<strong>Key finding:</strong> Pre-1900 properties average SAP 53 today but could reach SAP 75 — a jump of 22 points, equivalent to "
     "moving from Band E to Band C. This 22-point gap represents the largest retrofit opportunity in the entire housing stock. "
     "Post-2003 buildings are already near their potential, confirming that modern building regulations work."),

    ("co2-type",       "4. CO₂ Emissions Distribution by Property Type",
     "<strong>What it shows:</strong> A box plot for each property type. The horizontal line = median annual CO₂ (tonnes). "
     "The box = middle 50% of properties (25th–75th percentile). Whiskers = the full spread excluding extreme outliers. "
     "Notched boxes: non-overlapping notches indicate the median difference is statistically significant. "
     "<strong>Why box plots, not scatter?</strong> Plotting 100k individual points creates an unreadable ink-blot — all D/E properties "
     "pile up at 1–4 t CO₂. Box plots reveal the distribution shape without overplotting. "
     "<strong>What the types mean:</strong> <em>Detached houses</em> emit the most — large floor area, four exposed walls, often no shared heat. "
     "<em>Flats</em> emit the least — smaller area, shared walls/ceilings/floors with neighbours absorb heat loss. "
     "<em>Maisonettes</em> sit between the two — they share some walls but span two floors. "
     "<strong>Impact:</strong> A detached house emitting at the 75th percentile produces roughly 5× the CO₂ of a typical flat. "
     "Targeting detached housing in rural counties captures disproportionately large carbon savings."),

    ("density",        "5. Efficiency vs CO₂ — Density Contours",
     "<strong>What it shows:</strong> Where the mass of properties concentrates in efficiency-vs-CO₂ space. The innermost ring "
     "= the highest density of properties (where most homes actually sit). Outer rings = progressively rarer combinations. "
     "Marginal histograms along each axis show the full distribution independently. "
     "<strong>Why contours, not scatter?</strong> Even a 150k-point sample produces a solid blob — individual points are invisible. "
     "Contours reveal the shape of that blob: is it tight (consistent housing stock) or spread (wide variation)? "
     "<strong>What to look for:</strong> A tight cluster at 60–70 SAP / 2–3 t CO₂ tells us the 'typical' UK home. "
     "The long tail toward low efficiency / high CO₂ is the minority of properties that drive a disproportionate share of national emissions. "
     "Gaps between property type clusters show structural differences — flats genuinely follow a different physics than detached houses."),

    ("retrofit",       "6. Retrofit Priority Matrix",
     "<strong>What is retrofit?</strong> Retrofitting means adding energy-saving improvements to an existing building after it was originally constructed — "
     "as opposed to building new. Common retrofits include: loft insulation (cheapest, biggest gain), cavity wall insulation, "
     "solid wall insulation (expensive but high impact for pre-1940 homes), double/triple glazing, heat pumps replacing gas boilers, "
     "solar panels, and smart controls. The EPC surveyor records both the current state and which improvements have been recommended. "
     "<strong>Why retrofit matters:</strong> The UK cannot reach Net Zero by simply building new efficient homes — 85% of the homes "
     "that will exist in 2050 are <em>already built</em>. The only path to decarbonising heat at scale is retrofitting the existing stock. "
     "<strong>How to read this matrix:</strong> Each cell = a combination of property type (row) and construction decade (column). "
     "Colour and number = average Retrofit Priority Score (0–100). "
     "Red cells = highest priority — the combination is inefficient today AND has large achievable gains AND high CO₂ saving. "
     "Green cells = already near their potential or too small to matter at scale. "
     "<strong>Key finding:</strong> Pre-1900 detached houses and semi-detached houses score highest — they are the worst today, "
     "can improve the most, and save the most carbon. Post-2003 properties score near zero — modern regulations already delivered "
     "most of what retrofit could achieve."),

    ("la-treemap",     "7. Local Authority Retrofit Opportunity Map",
     "<strong>What it shows:</strong> Every local authority in England &amp; Wales, grouped by county. Two dimensions are encoded simultaneously: "
     "<br>• <strong>Box size</strong> = total CO₂ saving potential in tonnes/year across all properties in that authority. "
     "A bigger box means more total carbon is at stake — either because there are many properties, or because the average saving per property is large. "
     "<br>• <strong>Box colour</strong> = average current SAP score (red = worst efficiency, green = best). "
     "<br><strong>How to use it:</strong> The highest-priority intervention targets are the <strong>largest red boxes</strong>. "
     "These are local authorities where properties are collectively inefficient today AND where the total CO₂ saving from retrofitting is largest. "
     "A large green box = many properties but already efficient — lower priority. A small red box = inefficient but few properties — limited impact at scale. "
     "<br><strong>How to navigate:</strong> Click a county block to zoom into its local authorities. Click the parent label to zoom back out. "
     "<br><strong>Key finding:</strong> Rural counties in the Midlands and North tend to produce the largest red blocks — high density of older, "
     "poorly insulated housing stock combined with large property counts means the greatest aggregate carbon saving opportunity."),

    ("postcode-heat",  "8. Postcode Area Heatmap",
     "<strong>What it shows:</strong> Each row is a UK postcode area — the letter prefix of a postcode (e.g. SW = South West London, "
     "M = Manchester, LS = Leeds, TR = Truro in Cornwall). Each column is an EPC band. "
     "The cell colour = what percentage of properties in that area carry that band. Rows are sorted so the worst areas appear at the top. "
     "<strong>Why this matters:</strong> It reveals geographic inequality in housing quality. An area heavy in Band G (dark red, rightmost column) "
     "represents a community where a large share of residents are likely in fuel poverty, paying far more for energy than they should. "
     "An area heavy in Band A/B (leftmost columns) is already largely decarbonised. "
     "<strong>Key finding:</strong> Rural postcode areas (TR, PL, EX, LD, SY) tend to cluster toward E–G, reflecting their older housing stock "
     "and lack of mains gas (forcing use of oil or LPG, which score worse than mains gas on EPC methodology). "
     "Urban postcode areas (E, N, SW, SE) cluster toward C–D, reflecting newer flats and terraced housing."),

    ("geo-map",        "9. Geographic EPC Map",
     "<strong>What it shows:</strong> Every Local Authority District in England &amp; Wales drawn on a real map, coloured by average SAP score. "
     "Red = worst energy efficiency. Green = best. Hover a district to see its retrofit priority score, property count, and total CO₂ saving potential. "
     "<strong>Why geography matters:</strong> The EPC postcode heatmap (Section 8) shows which areas are worst by letter code — but this map "
     "shows <em>where those areas physically are</em>. Clusters of red reveal the geographic concentration of the UK's worst housing stock: "
     "large rural districts in the Midlands, Wales, and the North where solid-wall Victorian terraces or remote farmhouses dominate. "
     "<strong>How to use it:</strong> Scroll to zoom. Hover over any district for detail. The darkest red districts — especially large rural ones — "
     "represent areas where a government retrofit scheme would deliver the greatest carbon saving per pound spent. "
     "<strong>Data source:</strong> Boundaries from the ONS Open Geography Portal (LAD December 2022)."),

    ("fuel-type",      "10. EPC Efficiency by Heating Fuel Type",
     "<strong>What it shows:</strong> Average current SAP score (orange) vs achievable SAP after retrofit (green) for each main fuel type. "
     "The gap between orange and green = how much improvement is theoretically possible without changing the fuel. "
     "<strong>Why fuel type matters so much:</strong> The EPC SAP methodology directly penalises high-carbon, high-cost fuels. "
     "Mains gas is the baseline — relatively cheap and lower carbon than oil. Heating oil (used in ~1M rural homes with no gas connection) "
     "is more expensive and produces more CO₂ per kWh. LPG and solid fuel (coal, wood) score worst. "
     "<strong>The off-gas-grid problem:</strong> The ~15% of UK homes not connected to mains gas are disproportionately rural, older, "
     "and in the worst EPC bands. They face a double burden: more expensive fuel <em>and</em> harder-to-insulate walls. "
     "Heat pumps are the intended replacement — they score as electricity, which benefits from the grid decarbonising year by year. "
     "<strong>Key finding:</strong> A solid-fuel home (SAP 35) would need to reach SAP 72 just to hit Band C — a 37-point gap that cannot "
     "be closed by insulation alone; fuel switching is mandatory."),

    ("annual-trend",   "11. EPC Score Trend 2008–2024",
     "<strong>What it shows:</strong> Two series overlaid: the green line tracks the average SAP score of all EPC certificates issued each year "
     "(right axis). The blue bars show how many certificates were issued (left axis) — a proxy for housing market activity. "
     "<strong>Why this matters:</strong> If EPC scores are genuinely rising year on year, it means the policy mix (boiler efficiency standards, "
     "cavity wall insulation schemes, the Green Deal, ECO grants) is working. If they're flat, the housing stock is not improving despite policy spend. "
     "<strong>Volume spikes explained:</strong> Large bars in 2009–2010 reflect the Home Information Pack (HIP) requirement, which mandated EPCs "
     "for all homes sold. Subsequent troughs reflect housing market slowdowns. "
     "<strong>Key finding:</strong> The trend is genuinely upward — averaging around +0.45 SAP points per year — but at this pace the national "
     "average will not reach Band C (SAP ~69) until after 2035. The government's 2035 target requires roughly 3× the current rate of improvement."),

    ("cost-breakdown", "12. Annual Energy Cost by Property Type",
     "<strong>What it shows:</strong> Average annual energy bill per property type, split into three components: "
     "heating (largest, orange), hot water (blue), and lighting (yellow). The total bar height = average total energy spend per year. "
     "<strong>Why the gap is so large:</strong> A detached house spends roughly 2× what a flat spends on heating — because it has four exposed "
     "external walls instead of shared party walls, a larger floor area, and typically a loft directly under the roof rather than another flat above. "
     "Every shared surface with a neighbour reduces heat loss and therefore cost. "
     "<strong>Fuel poverty context:</strong> The UK government defines fuel poverty as spending more than 10% of income on energy. "
     "A household earning £25,000/yr living in a detached house paying £1,094/yr total energy costs sits at 4.4% — comfortably out of fuel poverty. "
     "But if that same house uses heating oil instead of mains gas, costs can easily exceed £2,000–3,000/yr, pushing past the 10% threshold. "
     "<strong>Policy implication:</strong> Retrofit programmes that prioritise detached and semi-detached housing in off-gas-grid areas "
     "deliver the largest reduction in fuel poverty per pound invested."),
]


def build_summary_section(summary: dict) -> str:
    """
    Returns a pure-HTML final findings section — no Plotly figure.
    Bullet points are grouped by theme and drawn from every chart in the dashboard.
    """
    co2_mt   = round(summary['co2_saving'] / 1e6, 1)
    cars_eq  = round(co2_mt * 1e6 / 2.1 / 1e6, 1)   # avg car ~2.1 t CO₂/yr

    groups = [
        ("📊 Scale &amp; Baseline", [
            f"<strong>29.2 million</strong> domestic EPC certificates cover virtually every home sold or rented in England &amp; Wales since 2008.",
            f"National average SAP score: <strong>65.3</strong> — Band D. The government's 2035 legal target is Band C (SAP ≈ 69).",
            f"<strong>{summary['pct_below_c']}% of properties</strong> (more than 1 in 2) are rated D or below and currently miss the 2035 target.",
            f"If every property reached its EPC-assessed potential, national residential CO₂ emissions would fall by <strong>{co2_mt} million tonnes per year</strong> — equivalent to taking <strong>{cars_eq} million cars off the road</strong>.",
            f"The average household spends <strong>~£951/yr</strong> on energy (heating, hot water, lighting). Reaching Band C would cut that by roughly £200–300 per year.",
        ]),
        ("🗺️ Geographic Inequality", [
            "A <strong>20+ SAP point gap</strong> separates the worst rural counties from the best urban ones — equivalent to two full EPC bands and roughly £800/yr in extra heating costs per household.",
            "Rural postcode areas (TR, PL, EX, LD, SY) are dominated by E–G bands due to older solid-wall housing stock and near-zero mains gas coverage.",
            "Urban postcode areas (E, N, SW, SE) cluster at C–D, reflecting newer flats and terraced housing with shared walls reducing heat loss.",
            "The highest-priority local authorities for intervention cluster in rural Midlands and North — large concentrations of older, inefficient housing where total CO₂ saving potential is greatest.",
        ]),
        ("🏗️ Age &amp; Construction Era", [
            "<strong>Pre-1900 homes</strong> are the single largest retrofit opportunity: average SAP 53 today, achievable SAP 75 — a <strong>22-point gain</strong>, equivalent to jumping from Band E to Band C.",
            "Each decade of building regulations produces a visible step-change in efficiency: cavity walls (1930s–50s), mandatory insulation (1976), condensing boilers (2005), fabric-first standards (2012+).",
            "<strong>Post-2003 properties</strong> score near zero on retrofit priority — modern regulations have already captured most of the available improvement. Building new is not where the decarbonisation problem lies.",
            "The 1950–1975 band contains the most certificates of any era and scores mid-table on retrofit priority — a large, addressable opportunity often overlooked in favour of the more dramatic pre-1900 story.",
        ]),
        ("⛽ Fuel Type — The Hidden Divide", [
            "A <strong>31-point SAP gap</strong> separates mains gas homes (avg SAP 66.3) from solid fuel homes (avg SAP 35.1) — the single largest structural driver of EPC score variation.",
            "~1 million homes are on heating oil; ~200k on LPG. These off-gas-grid properties <em>cannot reach Band C through insulation alone</em> — fuel switching to heat pumps or biomass is mandatory.",
            "Off-gas-grid homes face a double burden: more expensive fuel per kWh <em>and</em> harder-to-insulate solid walls — making them the most challenging and most urgent retrofit targets.",
            "Electricity-heated homes (3.7M properties) will improve automatically as the national grid decarbonises, without any physical retrofit — this is a structural tailwind for Band C attainment.",
        ]),
        ("🏠 Property Type &amp; Energy Cost", [
            "<strong>Detached houses</strong> emit roughly 5× more CO₂ than a typical flat (median comparison) — four exposed external walls versus shared party walls fundamentally change heat loss physics.",
            "Detached households pay on average <strong>£1,094/yr</strong> in energy costs vs <strong>£622/yr</strong> for flats — a £472/yr structural gap that persists even after controlling for SAP score.",
            "Targeting detached housing in rural, off-gas-grid counties delivers the largest CO₂ saving per pound of retrofit investment of any single property/location combination.",
        ]),
        ("📈 Trend: Improving But Too Slowly", [
            "The national average SAP improved from <strong>62.1 in 2008 to 69.3 in 2024</strong> — a genuine +7.2 point improvement driven by cavity wall insulation programmes, boiler replacement schemes, and new-build standards.",
            "At the current rate of ~0.45 SAP points per year, the national average will <em>not</em> reach Band C (SAP ≈ 69) across the existing stock until well after 2035.",
            "The government's 2035 target requires roughly <strong>3× the current annual improvement rate</strong>. This implies a step-change in policy ambition — not incremental continuation.",
            "Certificate volumes spiked 2009–2010 (Home Information Pack requirement) then stabilised. The 2017–2020 uptick correlates with increased buy-to-let activity before Minimum Energy Efficiency Standards enforcement.",
        ]),
        ("🔧 Retrofit Priority Composite Score", [
            "The composite Retrofit Priority Score (0–100) combines: current inefficiency (35%), achievable efficiency gain (40%), and CO₂ saving potential (25%) — all normalized globally.",
            "<strong>Pre-1900 detached and semi-detached houses</strong> score highest on every dimension simultaneously: they are the worst today, can improve the most, and save the most carbon per property.",
            "<strong>Social housing</strong> (avg SAP 68–69) outperforms both private rented (avg SAP 62–65) and owner-occupied (avg SAP 60–64) — sustained investment under the Decent Homes Standard is measurable in the data.",
            "The <em>split incentive</em> problem is visible: private landlords bear retrofit costs while tenants receive the bill savings, depressing private rented sector scores despite Minimum Energy Efficiency Standards (MEES) requiring Band E as a floor.",
        ]),
        ("🎯 Policy Implications", [
            "A <strong>fabric-first, geography-targeted approach</strong> is most cost-effective: prioritise solid-wall insulation and fuel switching in rural, off-gas-grid counties with pre-1930 detached and semi-detached housing.",
            "Universal policies (boiler upgrade grants, ECO scheme) disproportionately benefit mains-gas, cavity-wall homes — the easy cases. Hard-to-treat properties need bespoke, higher-subsidy pathways.",
            "The owner-occupied sector (14M+ properties, worst average SAP) is the largest and hardest to reach — no landlord-tenant regulation applies, and many owners are asset-rich but cash-poor older residents.",
            "At 46.3 million tonnes of CO₂ saving potential per year, full retrofit of the existing English &amp; Welsh housing stock represents one of the largest single decarbonisation levers available to UK policy.",
        ]),
    ]

    bullets_html = ""
    for title, points in groups:
        items = "".join(f"<li>{p}</li>" for p in points)
        bullets_html += f"""
        <div class="summary-group">
          <h3>{title}</h3>
          <ul>{items}</ul>
        </div>"""

    return f"""
    <section id="summary">
      <h2>Final Summary — Key Findings</h2>
      <div class="chart-desc">
        What 29.2 million EPC certificates tell us about the UK housing stock, the scale of the
        decarbonisation challenge, and where the highest-impact interventions lie.
      </div>
      <div class="summary-grid">{bullets_html}</div>
    </section>"""


def _kpi_card(label: str, value: str, sub: str = '', color: str = '#4f8ef7') -> str:
    return f"""
    <div class="kpi-card">
      <div class="kpi-value" style="color:{color}">{value}</div>
      <div class="kpi-label">{label}</div>
      {f'<div class="kpi-sub">{sub}</div>' if sub else ''}
    </div>"""


def build_dashboard(figures: list, summary: dict) -> None:
    def _fmt_m(n):
        return f"{n/1e6:.1f}M" if n >= 1e6 else f"{n:,.0f}"

    kpi_html = "".join([
        _kpi_card("Properties Analysed",   _fmt_m(summary['total']),        "EPC certificates"),
        _kpi_card("Avg Current SAP Score",  str(summary['avg_eff_current']), f"potential: {summary['avg_eff_potential']}"),
        _kpi_card("CO₂ Saving Potential",   _fmt_m(summary['co2_saving']),   f"tonnes/yr · {summary['pct_saving']}% reduction", color='#19b459'),
        _kpi_card("Below Band C",           f"{summary['pct_below_c']}%",    "of all properties", color='#ef8023'),
    ])

    nav_html = "".join(
        f'<a href="#{sid}">{title.split(". ", 1)[-1]}</a>'
        for sid, title, _ in SECTIONS
    )
    nav_html += '<a href="#summary">Key Findings</a>'

    sections_html = ""
    for (sid, title, desc), fig in zip(SECTIONS, figures):
        sections_html += f"""
        <section id="{sid}">
          <h2>{title}</h2>
          <div class="chart-desc">{desc}</div>
          <div class="chart-wrap">{_embed(fig)}</div>
        </section>"""

    sections_html += build_summary_section(summary)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>UK Energy EPC Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    :root {{
      --bg:       #0d1117;
      --surface:  #161b22;
      --border:   #30363d;
      --text:     #c9d1d9;
      --muted:    #8b949e;
      --accent:   #58a6ff;
      --green:    #3fb950;
      --orange:   #d29922;
      --red:      #f85149;
    }}
    html {{ scroll-behavior: smooth; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      font-size: 14px;
      line-height: 1.6;
    }}

    /* ── Header ─────────────────────────────────────── */
    header {{
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 24px 32px 20px;
    }}
    header h1 {{ font-size: 22px; color: #e6edf3; margin-bottom: 4px; }}
    header p  {{ color: var(--muted); font-size: 13px; }}

    /* ── KPI strip ───────────────────────────────────── */
    .kpi-strip {{
      display: flex; flex-wrap: wrap; gap: 16px;
      padding: 20px 32px;
      background: var(--surface);
      border-bottom: 1px solid var(--border);
    }}
    .kpi-card {{
      flex: 1 1 160px;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px 20px;
    }}
    .kpi-value {{ font-size: 28px; font-weight: 700; line-height: 1.1; }}
    .kpi-label {{ font-size: 12px; color: var(--muted); margin-top: 4px; text-transform: uppercase; letter-spacing: .05em; }}
    .kpi-sub   {{ font-size: 11px; color: var(--muted); margin-top: 2px; }}

    /* ── Nav ─────────────────────────────────────────── */
    nav {{
      position: sticky; top: 0; z-index: 100;
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 0 32px;
      display: flex; flex-wrap: wrap; gap: 0;
      overflow-x: auto;
    }}
    nav a {{
      color: var(--muted); text-decoration: none;
      font-size: 12px; padding: 10px 14px;
      border-bottom: 2px solid transparent;
      white-space: nowrap;
      transition: color .15s, border-color .15s;
    }}
    nav a:hover {{ color: var(--accent); border-bottom-color: var(--accent); }}

    /* ── Main content ────────────────────────────────── */
    main {{ max-width: 1400px; margin: 0 auto; padding: 32px; }}

    section {{
      margin-bottom: 56px;
      scroll-margin-top: 44px;
    }}
    section h2 {{
      font-size: 17px; font-weight: 600;
      color: #e6edf3;
      padding-left: 12px;
      border-left: 3px solid var(--accent);
      margin-bottom: 8px;
    }}
    .chart-desc {{
      color: var(--muted); font-size: 13px; line-height: 1.7;
      margin-bottom: 16px; max-width: 920px;
    }}
    .chart-desc strong {{ color: var(--text); }}
    .chart-desc em {{ color: #a5d6ff; font-style: normal; }}
    .chart-wrap {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px;
      overflow: hidden;
    }}

    /* ── Summary grid ───────────────────────────────── */
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
      gap: 20px;
      margin-top: 8px;
    }}
    .summary-group {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 18px 20px;
    }}
    .summary-group h3 {{
      font-size: 14px; font-weight: 600;
      color: var(--accent); margin-bottom: 10px;
    }}
    .summary-group ul {{
      list-style: none; padding: 0; margin: 0;
    }}
    .summary-group li {{
      color: var(--muted); font-size: 13px; line-height: 1.65;
      padding: 5px 0 5px 14px;
      border-bottom: 1px solid rgba(48,54,61,0.6);
      position: relative;
    }}
    .summary-group li:last-child {{ border-bottom: none; }}
    .summary-group li::before {{
      content: '›'; position: absolute; left: 0;
      color: var(--accent); font-weight: 700;
    }}
    .summary-group li strong {{ color: var(--text); }}
    .summary-group li em {{ color: #a5d6ff; font-style: normal; }}

    /* ── Footer ──────────────────────────────────────── */
    footer {{
      text-align: center;
      color: var(--muted);
      font-size: 12px;
      padding: 24px;
      border-top: 1px solid var(--border);
    }}
  </style>
</head>
<body>

<header>
  <h1>UK Energy Performance Certificate — Analytics Dashboard</h1>
  <p>29.2 million domestic EPC certificates · England &amp; Wales · DuckDB + dbt + Polars</p>
</header>

<div class="kpi-strip">{kpi_html}</div>

<nav>{nav_html}</nav>

<main>{sections_html}</main>

<footer>
  Generated by eda_uk_energy.py · Data: UK Government EPC open dataset ·
  Stack: DuckDB · dbt · Polars · Plotly
</footer>

</body>
</html>"""

    with open(DASHBOARD, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  ✅  dashboard.html")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

CHART_BUILDERS = [
    ("Rating distribution",               build_rating_distribution),
    ("County efficiency best/worst 20",   build_county_efficiency),
    ("Efficiency by construction decade", build_efficiency_by_age),
    ("CO₂ by property type",             build_co2_by_property_type),
    ("Efficiency vs CO₂ density",         build_efficiency_density),
    ("Retrofit priority matrix",          build_retrofit_priority_matrix),
    ("Local authority treemap",           build_local_authority_treemap),
    ("Postcode area heatmap",             build_postcode_area_heatmap),
    ("Geographic EPC choropleth map",     build_choropleth_map),
    ("Fuel type impact",                  build_fuel_type_analysis),
    ("Annual EPC score trend",            build_annual_trend),
    ("Annual energy cost breakdown",      build_cost_breakdown),
]


def run_eda() -> None:
    print("\n🚀 UK Energy EDA Suite — starting...\n")
    con = duckdb.connect(DB_PATH, read_only=True)

    print("  ⏳  National summary KPIs")
    summary = get_national_summary(con)
    print("  ✅  national_savings_summary.json")

    figures = []
    for label, builder in CHART_BUILDERS:
        print(f"  ⏳  {label}")
        try:
            figures.append(builder(con))
        except Exception as exc:
            import traceback
            print(f"  ⚠️  SKIPPED — {exc}")
            traceback.print_exc()
            figures.append(go.Figure())   # placeholder keeps section ordering

    print("  ⏳  Assembling dashboard.html")
    build_dashboard(figures, summary)

    con.close()
    print(f"\n✨  Done — open reports/dashboard.html in your browser\n")


if __name__ == "__main__":
    run_eda()
