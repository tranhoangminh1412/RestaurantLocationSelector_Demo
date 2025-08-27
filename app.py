# paste your full Streamlit code below this line
# (everything starting with import math, import streamlit as st, etc.)



# Streamlit demo: Restaurant Location Selector
# -------------------------------------------------
# Quick, client-ready demo that scores candidate locations using demo data
# and visualizes them on an interactive map. You can later upload the firm's
# real dataset (same schema) and it will use that instead of the demo data.
# -------------------------------------------------

# !pip install streamlit pydeck pandas numpy
# # For OSM fallback:
# !pip install folium streamlit-folium

import math
import os
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
from io import StringIO

# Optional OSM fallback
try:
    import folium
    from streamlit_folium import st_folium
    _FOLIUM_OK = True
except Exception:
    _FOLIUM_OK = False

st.set_page_config(page_title="Restaurant Location Selector – Demo", layout="wide")

# -------------------------
# 1) Demo data generator
# -------------------------

def make_demo_data(seed: int = 7, n: int = 40,
                   center=(39.7684, -86.1581)) -> pd.DataFrame:
    """Generate synthetic neighborhood-level candidates around a center point.
    Columns are aligned to the expected schema for easy future replacement.
    """
    rng = np.random.default_rng(seed)
    lat0, lon0 = center
    # random offsets ~ within ~15 km box
    lat = lat0 + (rng.normal(0, 0.08, n))
    lon = lon0 + (rng.normal(0, 0.12, n))

    population_density = rng.normal(4000, 1500, n).clip(500, 15000)               # ppl/sqkm
    median_income = rng.normal(60000, 15000, n).clip(25000, 140000)                # USD
    asian_pop_share = rng.uniform(0.02, 0.35, n)                                   # 0-1 share
    foot_traffic = rng.normal(1800, 650, n).clip(200, 6000)                        # daily passerby
    competitor_count = rng.poisson(5, n) + rng.integers(0, 6, n)                   # nearby similar cuisine
    rent_per_sqft = rng.normal(28, 8, n).clip(10, 90)                              # USD/sqft/mo
    crime_index = rng.normal(45, 12, n).clip(10, 95)                               # 0(safe)-100(unsafe)
    rating = rng.normal(4.0, 0.4, n).clip(2.5, 5.0)                                # avg local ratings
    growth_index = rng.normal(1.05, 0.06, n).clip(0.9, 1.25)                       # YoY growth multiplier

    df = pd.DataFrame({
        "name": [f"Candidate #{i+1}" for i in range(n)],
        "lat": lat,
        "lon": lon,
        "population_density": population_density,
        "median_income": median_income,
        "asian_pop_share": asian_pop_share,
        "foot_traffic": foot_traffic,
        "competitor_count": competitor_count,
        "rent_per_sqft": rent_per_sqft,
        "crime_index": crime_index,
        "rating": rating,
        "growth_index": growth_index,
    })

    return df

EXPECTED_COLUMNS = [
    "name", "lat", "lon", "population_density", "median_income",
    "asian_pop_share", "foot_traffic", "competitor_count", "rent_per_sqft",
    "crime_index", "rating", "growth_index"
]

# ---------------------------------------
# 2) Load data (upload or use demo data)
# ---------------------------------------

st.sidebar.header("Data Source")
up = st.sidebar.file_uploader("Upload candidate locations CSV (optional)", type=["csv"])
if up is not None:
    try:
        data = pd.read_csv(up)
        missing = [c for c in EXPECTED_COLUMNS if c not in data.columns]
        if missing:
            st.sidebar.error(f"Missing columns: {missing}. Using demo data instead.")
            data = make_demo_data()
    except Exception as e:
        st.sidebar.error(f"Failed to read CSV: {e}. Using demo data instead.")
        data = make_demo_data()
else:
    data = make_demo_data()

st.sidebar.caption("Expected columns: " + ", ".join(EXPECTED_COLUMNS))

# Map provider settings
st.sidebar.header("Map Settings")
map_engine = st.sidebar.selectbox("Map engine", ["Mapbox (pydeck)", "OpenStreetMap (folium)"] if _FOLIUM_OK else ["Mapbox (pydeck)"])
mapbox_token_input = st.sidebar.text_input(
    "Mapbox token (optional if using OSM)",
    value=os.getenv("MAPBOX_API_KEY", ""),
    type="password",
)
if map_engine.startswith("Mapbox") and not mapbox_token_input:
    st.sidebar.warning("No Mapbox token detected. Enter one or switch to OpenStreetMap.")

# Apply Mapbox token if provided
if mapbox_token_input:
    pdk.settings.mapbox_api_key = mapbox_token_input

# ------------------------------------
# 3) Scoring configuration (weights)
# ------------------------------------

st.sidebar.header("Scoring Weights")
col1, col2 = st.sidebar.columns(2)
with col1:
    w_density = st.slider("Population density", 0.0, 3.0, 1.2, 0.1)
    w_income = st.slider("Median income", 0.0, 3.0, 0.8, 0.1)
    w_asian = st.slider("Asian pop. share", 0.0, 3.0, 1.4, 0.1)
    w_traffic = st.slider("Foot traffic", 0.0, 3.0, 1.5, 0.1)
with col2:
    w_comp = st.slider("Competition (lower is better)", 0.0, 3.0, 1.2, 0.1)
    w_rent = st.slider("Rent $/sqft (lower is better)", 0.0, 3.0, 1.0, 0.1)
    w_crime = st.slider("Crime index (lower is better)", 0.0, 3.0, 0.8, 0.1)
    w_growth = st.slider("Local growth index", 0.0, 3.0, 1.0, 0.1)

WEIGHTS = {
    "population_density": (w_density, +1),
    "median_income": (w_income, +1),
    "asian_pop_share": (w_asian, +1),
    "foot_traffic": (w_traffic, +1),
    "competitor_count": (w_comp, -1),  # lower better
    "rent_per_sqft": (w_rent, -1),     # lower better
    "crime_index": (w_crime, -1),      # lower better
    "growth_index": (w_growth, +1),
}

# optional hard filters
st.sidebar.header("Filters (Optional)")
max_rent = st.sidebar.number_input("Max rent $/sqft/mo", min_value=0.0, value=50.0)
min_income = st.sidebar.number_input("Min median income ($)", min_value=0.0, value=45000.0)
max_comp = st.sidebar.number_input("Max competitor count", min_value=0, value=12)

# ------------------------------------
# 4) Score computation helpers
# ------------------------------------

def zscore(s: pd.Series) -> pd.Series:
    mu, sigma = s.mean(), s.std(ddof=0)
    if sigma == 0:
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sigma


def compute_scores(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    df2 = df.copy()
    # Apply pre-filters
    df2 = df2[(df2["rent_per_sqft"] <= max_rent) &
              (df2["median_income"] >= min_income) &
              (df2["competitor_count"] <= max_comp)]

    # Standardize and apply direction
    score_parts = []
    for col, (w, direction) in weights.items():
        zs = zscore(df2[col]) * direction
        score_parts.append(w * zs)
        df2[f"z_{col}"] = zs
        df2[f"contrib_{col}"] = w * zs

    df2["score"] = np.sum(score_parts, axis=0)
    df2 = df2.sort_values("score", ascending=False)
    return df2

scored = compute_scores(data, WEIGHTS)

# ------------------------------------
# 5) UI: headline and explanation
# ------------------------------------

st.title("Restaurant Location Selector – Demo")
st.markdown(
    "Use the sliders to weight what matters for this concept (density, income, foot traffic, competition, etc.).\n"
    "We standardize each feature (z-scores), flip where lower is better (e.g., rent), and sum with your weights\n"
    "to produce a location score. Upload your own CSV to instantly re-run on real candidates."
)

# ------------------------------------
# 6) Map visualization (real basemap)
# ------------------------------------

TOP_N = st.slider("How many top locations to show on the map?", 5, 40, 15)

if map_engine.startswith("Mapbox"):
    view_state = pdk.ViewState(
        latitude=float(data["lat"].mean()),
        longitude=float(data["lon"].mean()),
        zoom=10,
    )
    # scale score 0..1 for radius
    score_min, score_max = float(scored["score"].min()), float(scored["score"].max())
    score_span = max(1e-6, score_max - score_min)
    scored["score_01"] = (scored["score"] - score_min) / score_span

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=scored.head(TOP_N),
        get_position="[lon, lat]",
        get_radius="100 + score_01 * 1000",
        get_fill_color="[30, 144, 255, 180]",
        pickable=True,
    )

    text_layer = pdk.Layer(
        "TextLayer",
        data=scored.head(TOP_N),
        get_position='[lon, lat]',
        get_text='name',
        get_size=14,
        get_alignment_baseline='bottom'
    )

    r = pdk.Deck(
        map_style="mapbox://styles/mapbox/streets-v12",
        layers=[layer, text_layer],
        initial_view_state=view_state,
        tooltip={"text": "{name}\nScore: {score}\nRent: ${rent_per_sqft}/sqft/mo\nIncome: ${median_income}\nTraffic: {foot_traffic}\nCompetitors: {competitor_count}"}
    )
    st.pydeck_chart(r, use_container_width=True)
else:
    if not _FOLIUM_OK:
        st.info("Install folium + streamlit-folium or provide a Mapbox token for the map: `pip install folium streamlit-folium`.")
    # Fallback to OpenStreetMap tiles using folium
    m_lat = float(data["lat"].mean())
    m_lon = float(data["lon"].mean())
    fmap = folium.Map(location=[m_lat, m_lon], zoom_start=11, tiles="OpenStreetMap")
    top_df = scored.head(TOP_N)
    for _, r in top_df.iterrows():
        popup = folium.Popup(
            f"<b>{r['name']}</b><br>Score: {r['score']:.2f}<br>Rent: ${r['rent_per_sqft']}/sqft/mo<br>Income: ${int(r['median_income']):,}<br>Traffic: {int(r['foot_traffic'])}<br>Competitors: {int(r['competitor_count'])}",
            max_width=280,
        )
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=6 + 12 * (r["score"] - scored["score"].min()) / max(1e-6, scored["score"].max() - scored["score"].min()),
            fill=True,
            fill_opacity=0.7,
            popup=popup,
        ).add_to(fmap)
    st_folium(fmap, use_container_width=True, returned_objects=[])

# ------------------------------------
# 7) Ranked table + per-location explanation
# ------------------------------------

st.subheader("Ranked candidates (top 25)")
show_cols = [
    "name", "score", "rent_per_sqft", "median_income", "population_density",
    "asian_pop_share", "foot_traffic", "competitor_count", "crime_index",
    "rating", "growth_index"
]

st.dataframe(scored.head(25)[show_cols], use_container_width=True)

# drilldown
with st.expander("Explain a score"):
    options = list(scored["name"].head(25))
    pick = st.selectbox("Choose a location", options)
    row = scored.loc[scored["name"] == pick].iloc[0]

    st.markdown(f"**{pick}** — composite score: **{row['score']:.2f}**")

    contrib_cols = [c for c in scored.columns if c.startswith("contrib_")]
    explain_df = (
        row[contrib_cols]
        .rename(lambda c: c.replace("contrib_", ""))
        .to_frame(name="weighted_z")
        .assign(weight=lambda d: d.index.map(lambda k: WEIGHTS[k][0]))
        .assign(direction=lambda d: d.index.map(lambda k: WEIGHTS[k][1]))
        .sort_values("weighted_z", ascending=False)
    )
    st.dataframe(explain_df)

# ------------------------------------
# 8) Download results
# ------------------------------------

csv = scored.to_csv(index=False).encode("utf-8")
st.download_button("Download scored candidates (CSV)", csv, file_name="scored_locations_demo.csv")

st.caption("This is demo software. For investment or site selection decisions, validate inputs with firm datasets and field checks.")
