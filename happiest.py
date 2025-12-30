import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# 1) Load data from web
# -----------------------
HAPPINESS_URL = "https://ourworldindata.org/grapher/happiness-cantril-ladder.csv"
GDP_HAPPY_URL = "https://ourworldindata.org/grapher/gdp-vs-happiness.csv"
LE_HAPPY_URL  = "https://ourworldindata.org/grapher/life-satisfaction-vs-life-expectancy.csv"

h = pd.read_csv(HAPPINESS_URL)      # columns typically: Entity, Code, Year, happiness_cantril_ladder
g = pd.read_csv(GDP_HAPPY_URL)      # columns typically: Entity, Code, Year, gdp_per_capita, life_satisfaction...
le = pd.read_csv(LE_HAPPY_URL)      # columns typically: Entity, Code, Year, life_expectancy, life_satisfaction...

# Normalize column names (robust-ish)
h = h.rename(columns={"Entity":"country", "Year":"year"})
g = g.rename(columns={"Entity":"country", "Year":"year"})
le = le.rename(columns={"Entity":"country", "Year":"year"})

# Identify the happiness column name in h (in case OWID changes it slightly)
happiness_col = [c for c in h.columns if c not in ("country","Code","year")][0]
h = h[["country","Code","year", happiness_col]].rename(columns={happiness_col:"happiness"})

# -----------------------
# 2) Latest year ranking
# -----------------------
latest_year = int(h["year"].max())
latest = (
    h[h["year"] == latest_year]
    .dropna(subset=["happiness"])
    .sort_values("happiness", ascending=False)
)

print("Latest year:", latest_year)
print(latest.head(10)[["country","happiness"]])

# Viz 1: Top 10 happiest countries (latest year)
top10 = latest.head(10).sort_values("happiness")  # sort for horizontal bar
plt.figure()
plt.barh(top10["country"], top10["happiness"])
plt.title(f"Top 10 Happiest Countries ({latest_year})")
plt.xlabel("Life satisfaction (Cantril Ladder, 0–10)")
plt.tight_layout()
plt.show()

# -----------------------
# 3) Biggest improvers since 2011 (or earliest available)
# -----------------------
base_year = int(h["year"].min())
pivot = h.pivot_table(index="country", columns="year", values="happiness", aggfunc="mean")
pivot = pivot.dropna(subset=[base_year, latest_year], how="any")
pivot["change"] = pivot[latest_year] - pivot[base_year]
improvers = pivot.sort_values("change", ascending=False).head(10)

# Viz 2: Top 10 improvers
plt.figure()
plt.barh(improvers.index[::-1], improvers["change"][::-1])
plt.title(f"Top 10 Improvers in Happiness ({base_year} → {latest_year})")
plt.xlabel("Change in life satisfaction")
plt.tight_layout()
plt.show()

# -----------------------
# 4) Merge GDP + happiness for relationship analysis
# -----------------------
# For GDP dataset, detect likely columns
gdp_cols = [c for c in g.columns if "gdp" in c.lower()]
ls_cols  = [c for c in g.columns if "life" in c.lower() and "satisf" in c.lower()]

# We'll try to find GDP per capita and life satisfaction columns
gdp_col = gdp_cols[0] if gdp_cols else None
ls_col  = ls_cols[0] if ls_cols else None

# Create a tidy df for latest year
dg = g[g["year"] == latest_year].copy()
if gdp_col and ls_col:
    dg = dg[["country","Code","year", gdp_col, ls_col]].rename(columns={gdp_col:"gdp_per_capita", ls_col:"happiness_gdp"})
else:
    # fallback: join with h on country/year only
    dg = dg[["country","Code","year"]].merge(h[h["year"] == latest_year], on=["country","Code","year"], how="left")

# Viz 3: Scatter GDP per capita vs happiness (log x for readability)
if "gdp_per_capita" in dg.columns and "happiness_gdp" in dg.columns:
    plot_df = dg.dropna(subset=["gdp_per_capita","happiness_gdp"]).copy()
    plt.figure()
    plt.scatter(np.log10(plot_df["gdp_per_capita"]), plot_df["happiness_gdp"], alpha=0.6)
    plt.title(f"Happiness vs GDP per Capita ({latest_year})")
    plt.xlabel("log10(GDP per capita)")
    plt.ylabel("Life satisfaction (0–10)")
    plt.tight_layout()
    plt.show()

# -----------------------
# 5) Life expectancy vs happiness
# -----------------------
# Detect columns in LE dataset
le_cols = [c for c in le.columns if "expect" in c.lower()]
ls2_cols = [c for c in le.columns if "life" in c.lower() and "satisf" in c.lower()]
le_col = le_cols[0] if le_cols else None
ls2_col = ls2_cols[0] if ls2_cols else None

le_latest = le[le["year"] == latest_year].copy()
if le_col and ls2_col:
    le_latest = le_latest[["country","Code","year", le_col, ls2_col]].rename(
        columns={le_col:"life_expectancy", ls2_col:"happiness_le"}
    )

    # Viz 4: Scatter Life expectancy vs happiness
    plot_df2 = le_latest.dropna(subset=["life_expectancy","happiness_le"])
    plt.figure()
    plt.scatter(plot_df2["life_expectancy"], plot_df2["happiness_le"], alpha=0.6)
    plt.title(f"Happiness vs Life Expectancy ({latest_year})")
    plt.xlabel("Life expectancy (years)")
    plt.ylabel("Life satisfaction (0–10)")
    plt.tight_layout()
    plt.show()

# -----------------------
# 6) Trend line for a few countries (your choice)
# -----------------------
countries = ["United States", "India", "Finland", "Japan"]
trend = h[h["country"].isin(countries)].dropna(subset=["happiness"])

plt.figure()
for c in countries:
    tmp = trend[trend["country"] == c].sort_values("year")
    plt.plot(tmp["year"], tmp["happiness"], label=c)
plt.title("Happiness Trends (2011–2024)")
plt.xlabel("Year")
plt.ylabel("Life satisfaction (0–10)")
plt.legend()
plt.tight_layout()
plt.show()
