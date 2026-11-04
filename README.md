# Citi Bike Fleet Availability Planner

**Machine learning · demand forecasting · geospatial analytics · operations decision support**

[**Launch the interactive Streamlit application**](https://citibike-fleet-planner-nyc.streamlit.app)

An end-to-end academic prototype for a New York City bike-share operations manager: explore historical usage, predict neighborhood-level hourly bicycle departures and returns, estimate the inventory required to protect service availability, and explore constrained bicycle relocation scenarios.

**Project:** Team 9 · AI for Business Decisions · ESSEC Business School  
**Time window:** October 2025–September 2026 · **Scale:** approximately 45 million completed ride records · **Geography:** 131 NYC NTAs with mapped stations · **Bicycle types:** classic and electric

## Why this matters

Customers move bicycles around the network. A zone can accumulate surplus bicycles while another runs short; empty docks can be scarce even where bicycle supply is adequate. A manager needs to see *where and when* imbalances might occur, and which transfers could address them within practical constraints.

## Evidence: forecast model selection

Forecasting uses two independently trained models (departures and returns) and a chronological **August–September 2026 holdout**. Results below are recorded in [`development/03_model_training/recorded_outputs/model_comparison.csv`](development/03_model_training/recorded_outputs/model_comparison.csv).

| Method | Departure MAE ↓ | Return MAE ↓ |
| --- | ---: | ---: |
| Historical hourly-average baseline | 10.554 | 10.511 |
| Poisson regression with weather | 10.030 | 10.238 |
| Gradient-boosted trees without weather | 9.646 | 8.984 |
| **Gradient-boosted trees with weather (selected)** | **7.168** | **6.878** |

Compared with the historical hourly-average baseline, the selected approach reduced holdout **MAE by 32.1% for departures and 34.6% for returns**, at the modeled neighborhood × bicycle type × hour granularity. *MAE is the average absolute count error per forecast observation; these are retrospective held-out results, not live production accuracy or an estimate of unmet demand.* Saved models are provided under `models/`.

## Application walkthrough

**1. Understand Usage** — Inspect historical departures and returns on an interactive NYC neighborhood map; filter by dates, bicycle type, member category, day and hour. Explore hourly profiles, weekday patterns, Manhattan-versus-outer-borough directional flows, and expandable fixed-reference seasonality/weather charts.

**2. Plan Availability** — Choose a future planning window and bike type. Hourly departure and return models use calendar and weather inputs. Cumulative net movements inform expected peak inventory depletion; a manager-adjustable safety buffer informs target starting stock. Available bicycles/docks come from the current retrieved GBFS feed or a labelled fallback snapshot. A rule-based, capacity-aware procedure proposes neighborhood-level transfers subject to relocation budget, maximum straight-line distance and same-borough restrictions.

The relocation algorithm is **not** an additional trained ML model and does **not** solve station-by-station routing.

## How the pipeline fits together

```text
Citi Bike monthly trip histories (~45M records)        NYC NTA geography / GBFS stations        Open-Meteo weather
                   |                                            |                                 |
                   +----------------------+---------------------+---------------------------------+
                                          |
                         Clean + validate station identities
                                          |
                        Station-hour and zone-hour aggregations
                                          |
                  Compare baseline / Poisson / boosted tree models
                                          |
                     Saved departure and return forecast models
                                          |
          Forecast hourly flows -> inventory and dock-risk calculations
                                          |
                    Rule-based neighborhood relocation scenarios
                                          |
                           Interactive Streamlit manager dashboard
```

The original historical scripts, recorded model comparisons, station-matching audit, first app, and later dashboard data-preparation tools are available in [`development/`](development/). **The ~45 million original trip rows and some intermediary station-level aggregates are intentionally not committed**, so this repo runs the saved models and demos the complete application but is not a one-command full training reproduction. See [`development/DATA_SOURCES.md`](development/DATA_SOURCES.md).

## Quick start

From the **repository root** (containing `app_v2_3.py`):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app_v2_3.py
```

Open <http://localhost:8501> if Streamlit does not open a browser automatically. Python 3.13 is recommended. On Windows, use `py -3.13 -m venv .venv` and `.venv\Scripts\python.exe` in place of `.venv/bin/python`. Internet access improves live station/weather inputs and map backgrounds; the app labels supported fallbacks.

**Run the application from the repository root, not from `development/`.** Original preprocessing/training scripts are historical source code and are **not** prerequisites for the dashboard.

## Repository map

| Path | Contents |
| --- | --- |
| `app_v2_3.py` | Streamlit user interface / two dashboards |
| `planner_core_v2.py` | Forecast feature preparation, stock and transfer calculations |
| `dashboard_additions.py` | Directional-flow / weather visualizations |
| `models/` | Saved departure and return models |
| `processed_data/` | Aggregated historical rides, mapped NYC zones, weather, charts |
| `reference_data/` | Saved GBFS station-status snapshot for fallback |
| `development/` | Original processing/training scripts, previous app, model comparison and QA |
| `docs/AI_DEVELOPMENT_LOG.md` | Retrospective AI-assisted development history and selected real prompt examples |
| `requirements.txt` | Runtime Python dependencies |

## Key technical decisions and checks

- **Station identity reconciliation:** Historical station identifiers differed from the GBFS directory. The corrected model pipeline achieved **98.25% departure** and **98.24% return** event matching (see `stage3_report.json` and `station_matching_audit.csv`). The first unsuccessful training script is kept as explicitly labelled history.
- **Chronological evaluation:** August–September 2026 were held out for comparing candidates; the chosen models were subsequently refit using the available historical training data.
- **Original/final consistency:** Offline numerical checks using matching saved inputs found identical one-day forecasts, inventory targets and relocation results in examined scenarios. The later multi-day safety-buffer method is intentionally revised. See [`development/06_validation/`](development/06_validation/).
- **Transparent scope:** Predicted rides reflect **completed rentals**, not latent/unmet requests. Current inventory is used as a hypothetical starting point for the future scenario; zone-level transfers require operational review.

## Sources and limitations

Data sources include [Citi Bike System Data](https://citibikenyc.com/system-data), Citi Bike's public GBFS feeds, [NYC Open Data NTAs](https://data.cityofnewyork.us/d/9nt8-h7nd), and [Open-Meteo](https://open-meteo.com/). Details and missing raw data are listed in [`development/DATA_SOURCES.md`](development/DATA_SOURCES.md).

Important limitations: independently produced departure/return forecasts are not physically flow-constrained; the safety buffer is a manager-defined planning assumption, **not a confidence interval**; straight-line zone-to-zone relocation feasibility is not the same as actual truck routing; live station and weather feeds may be unavailable; historical weather association charts are descriptive, not causal. This is an **academic prototype, not an official Citi Bike/Lyft product**.

## Development and Generative AI Disclosure

I independently led the end-to-end technical development of this application as part of a university group project, using ChatGPT extensively as an AI-assisted development tool.

My work included collecting and validating the data, directing the forecasting and model-comparison process, evaluating results, designing the decision-support workflow, testing the implementation, identifying inconsistencies, and iteratively refining the application.

ChatGPT assisted with generating Python code, data-processing scripts, model implementation, debugging, interface development, and documentation. I guided the development through successive prompts, critically reviewed the outputs, and made the final analytical and design decisions.

The deployed application uses trained machine-learning models and a rule-based relocation algorithm. It does not rely on generative AI during operation.

The full development history, including selected prompts and technical iterations, is documented in the [development and AI usage log](docs/AI_DEVELOPMENT_LOG.md).
