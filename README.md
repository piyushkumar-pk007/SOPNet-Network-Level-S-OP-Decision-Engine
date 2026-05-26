# SOPNet — Network-Level S&OP Decision Engine

SOPNet is a network-level supply chain planning project built around a simple idea: demand forecasting becomes much more valuable when it is directly connected to production, inventory, transportation, and service-level decisions.

The project uses public M5 retail sales data as the demand signal, then builds the rest of the planning workflow around it. Because M5 does not include real manufacturing plants, distribution centers, or transportation lanes, the project generates a realistic synthetic network and uses that as the planning layer for optimization and simulation.

**This is an independent Kaggle/public-data based prototype built to demonstrate network-level S&OP, demand forecasting, production planning, logistics optimization, and simulation.**

## What this project is

SOPNet is an end-to-end S&OP and supply chain decision engine. It combines:

- demand forecasting
- hierarchical planning
- forecast reconciliation
- production and logistics optimization
- discrete-event simulation
- scenario analysis
- dashboarding

Instead of stopping at “what demand might look like,” the project continues into “what the network should do about it.”

## Business problem

Large supply chains need to balance demand, production, transportation, inventory, and service levels. Traditional planning often works in silos. This project demonstrates how forecasting, optimization, and simulation can be combined into one decision engine for S&OP.

The system is designed to answer six planning questions:

1. What will demand look like for the next four weeks?
2. Which plant should produce which SKU or category, and how much?
3. Which warehouse should receive inventory?
4. How should inventory be allocated to retail nodes?
5. What service level should be expected once uncertainty is introduced?
6. What changes under demand spikes, transport cost inflation, or lead-time deterioration?

## Why this project matters

Forecasting alone does not create a supply plan. Planning teams still need to decide:

- what to produce
- where to produce it
- how much to ship
- how much inventory to hold
- how much service risk is acceptable

This project matters because it links those decisions together. It also mirrors the structure of real S&OP conversations, where commercial teams, planners, manufacturing teams, and logistics teams all need a single version of the plan.

## Dataset

Primary dataset:

- M5 Forecasting Accuracy dataset

Expected source files:

- `sales_train_validation.csv`
- `calendar.csv`
- `sell_prices.csv`
- `sample_submission.csv`

This repository uses a sampled subset so that it can run on a normal laptop:

- 3 stores
- 5 departments
- roughly 50 to 200 SKUs depending on the sample
- last 2 years of daily sales history where possible

Daily demand is aggregated to weekly demand because the planning use case here is S&OP rather than daily replenishment execution.

## Why a synthetic network was created

The M5 dataset is very useful for forecasting and hierarchical demand planning, but it does not contain plants, DCs, transportation lanes, setup costs, storage limits, or production capacities.

To make the project useful for network-level planning, a synthetic supply network is generated and linked to the M5 demand structure. This makes it possible to demonstrate production planning, distribution allocation, and service-level tradeoffs in a realistic way.

## Architecture

```text
Raw M5 Data
→ Data Cleaning
→ Weekly Demand Aggregation
→ Hierarchy Builder
→ Forecasting Models
→ Forecast Reconciliation
→ Synthetic Supply Network
→ MILP Optimizer
→ SimPy Simulation
→ Scenario Analysis
→ Streamlit Dashboard
```

## Main modules

### Data preprocessing
- loads the M5 sample
- converts wide daily sales into long format
- joins with calendar and price data
- aggregates to weekly demand

### Hierarchical forecasting
- forecasts weekly demand using baseline, statistical, and ML models
- supports store-category and store-item style series
- uses walk-forward validation

### Forecast reconciliation
- bottom-up
- top-down
- middle-out
- simplified MinT-style approximation

Why this matters:

In S&OP, finance may look at total demand, supply teams may look at plant/category demand, and planners may look at SKU-store demand. If these numbers do not reconcile, planning meetings become inconsistent. Hierarchical reconciliation keeps all levels aligned.

### Synthetic network generation
- creates plants
- creates DCs
- creates retail nodes
- creates transportation lanes
- classifies SKUs into ABC segments

### MILP optimization
- decides production quantity by plant
- decides plant-to-DC and DC-to-retail shipments
- tracks DC inventory
- controls unmet demand under service-level constraints

### Discrete-event simulation
- introduces uncertain demand
- introduces lead-time variability
- tests whether deterministic plans still perform under uncertainty

### Scenario analysis
- high demand
- lead-time delay
- transport cost inflation
- plant capacity shock
- premium service scenario

### Dashboard
- executive view
- forecast view
- network plan view
- optimization view
- simulation view
- scenario view

## Why these methods were chosen

### Why hierarchical forecasting
Because supply chain planning happens at multiple levels at once. Leadership may plan at total network level, planners may work at category or store level, and operations teams may still need SKU visibility.

### Why not only forecast at SKU level
SKU-level forecasting can become noisy and intermittent, especially on sampled public retail data. Aggregated levels are often more stable and more useful for S&OP decisions. This project supports granular demand views, but the default planning flow keeps the optimization laptop-friendly by working at category-store-week level.

### Why MILP
The planning problem includes both continuous decisions and binary setup decisions. That makes Mixed Integer Linear Programming a natural fit.

### Why this is not only linear programming
Production setup decisions are binary. If a plant either does or does not produce a category in a week, that choice cannot be represented well with pure continuous linear programming.

### Why simulation comes after optimization
Optimization uses expected demand and expected lead times. Real supply chains operate under uncertainty. Simulation reveals whether a plan that looks good on average still works when demand and lead time fluctuate.

## Problems faced and how they were handled

### Problem 1: M5 is a demand dataset, not a full network dataset
The dataset does not include plants, warehouses, lane capacities, or production cost.

How it was handled:

- created a synthetic but realistic network layer
- linked it directly to the M5 product and location structure
- kept all assumptions explicit in code and documentation

### Problem 2: M5 is large for a normal laptop
The full competition dataset can be heavy for portfolio execution.

How it was handled:

- sampled 3 stores
- limited the number of departments and SKUs
- aggregated from daily to weekly planning buckets
- kept the default optimization level at category-store-week

### Problem 3: hierarchy consistency matters
Forecasts at different levels can disagree.

How it was handled:

- built explicit hierarchy tables
- added multiple reconciliation methods
- saved reconciled outputs for side-by-side comparison

### Problem 4: deterministic optimization hides risk
An optimizer can look strong under average assumptions but still fail once randomness is introduced.

How it was handled:

- added SimPy-based simulation
- compared deterministic outputs with simulated service-level behavior

### Problem 5: network models can become too large
Full SKU-level MILP planning is often too large for a lightweight demonstration.

How it was handled:

- kept the code compatible with finer-grain planning
- used category-store-week as the default run mode so the model stays solvable on a laptop

## How to run

### 1. Create a virtual environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

### 3. Get the M5 files

Option A:
Use Kaggle API and set credentials in `.env`.

Option B:
Manually place these files inside `data/raw/m5/`:

- `sales_train_validation.csv`
- `calendar.csv`
- `sell_prices.csv`

If Kaggle authentication works but competition access is denied, the ingestion step will automatically fall back to a public M5 mirror and continue from there.

### 4. Run the pipeline

The easiest way is:

```bash
python run_project.py
```

By default, this runs the pipeline and then launches the Streamlit dashboard in your browser session.

If you prefer to run step by step manually:

```bash
python -m src.data_ingestion
python -m src.preprocessing
python -m src.hierarchy_builder
python -m src.feature_engineering
python -m src.forecasting
python -m src.reconciliation
python -m src.network_generator
python -m src.optimization
python -m src.simulation
python -m src.scenario_analysis
```

### 5. Launch the dashboard

```bash
python -m streamlit run app/streamlit_app.py
```

### 6. Run tests

```bash
python -m pytest
```

## Runner script

To run everything step by step with one file:

```bash
python run_project.py
```

Useful options:

```bash
python run_project.py --list-steps
python run_project.py --from-step 5
python run_project.py --to-step 8
python run_project.py --include-tests
python run_project.py --skip-dashboard
```

## Key outputs

### Processed data
- `data/processed/sales_sample.csv`
- `data/processed/calendar_clean.csv`
- `data/processed/prices_clean.csv`
- `data/processed/weekly_demand.csv`
- `data/processed/hierarchy_table.csv`
- `data/processed/hierarchical_weekly_demand.csv`

### Forecasting outputs
- `data/outputs/forecast_results.csv`
- `data/outputs/model_comparison.csv`
- `data/outputs/forecast_accuracy_by_level.csv`
- `data/outputs/forecast_plot_selected_skus.png`

### Reconciliation outputs
- `data/outputs/reconciled_forecast.csv`
- `data/outputs/reconciliation_comparison.csv`

### Network optimization outputs
- `data/outputs/optimized_production_plan.csv`
- `data/outputs/optimized_shipment_plan.csv`
- `data/outputs/inventory_plan.csv`
- `data/outputs/unmet_demand.csv`
- `data/outputs/optimization_summary.csv`
- `data/outputs/baseline_comparison.csv`

### Simulation outputs
- `data/outputs/simulation_results.csv`
- `data/outputs/service_level_distribution.csv`
- `data/outputs/stockout_distribution.csv`
- `data/outputs/simulation_summary.csv`

### Scenario outputs
- `data/outputs/scenario_comparison.csv`
- `data/outputs/scenario_service_level_chart.png`
- `data/outputs/scenario_cost_chart.png`

## Business impact story

The business value of SOPNet comes from turning demand signals into supply decisions. The project is set up to calculate:

- cost improvement versus simple replenishment baselines
- service-level tradeoffs
- network bottlenecks
- scenario resilience under shocks

When you run the full pipeline, use the generated outputs in `optimization_summary.csv`, `baseline_comparison.csv`, and `simulation_summary.csv` to explain:

- how the optimized plan compares with baseline plans
- whether service levels stay acceptable once uncertainty is introduced
- which scenario harms performance most

## Portfolio talking points

### 30-second version
I built SOPNet, a network-level S&OP decision engine using public M5 sales data and a synthetic supply network. It forecasts demand across product and location hierarchy, reconciles forecasts for planning consistency, optimizes production and shipment decisions using MILP, and stress-tests the plan using SimPy simulation. The goal is to show how data science can move beyond forecasting into actual supply-chain decision support.

### 2-minute version
Large supply chains do not struggle only with forecasting. They struggle with connecting demand forecasts to production, warehouse allocation, transportation, and service-level decisions. In SOPNet, I used M5 public retail sales data, converted it into weekly S&OP demand, and built a hierarchy from total network down to store-item level. I trained baseline, statistical, and machine-learning forecasting models using walk-forward validation and then reconciled forecasts across levels so that total, category, store, and SKU plans are consistent. After that, I generated a realistic plant-DC-retail network and built a PuLP MILP optimizer that decides production quantity, plant allocation, DC inventory, and shipments while respecting capacity, lane, storage, setup, and service-level constraints. Finally, I used SimPy simulation to test the optimized plan under stochastic demand and lead-time delays. The output is a Streamlit dashboard showing forecast accuracy, optimized network plan, service-level risk, and scenario tradeoffs. This demonstrates the full journey from data to analytics to supply-chain decision-making.

### Deep technical explanation
- Time-series forecasting is validated with walk-forward splits rather than random folds.
- Reconciliation is used to align top-level and detailed-level forecasts.
- The supply network is synthetic, but it is linked consistently to the demand structure.
- MILP is used because the problem includes both quantity decisions and binary setup decisions.
- Simulation is used because average-demand plans can hide service-level risk.

## Common questions

### Q1. Why hierarchical forecasting?
Because S&OP decisions need aligned numbers across total network, state, store, category, and SKU levels.

### Q2. Why not just forecast at SKU level?
Because SKU-level data can be intermittent and noisy. Planning is often stronger when multiple aggregation levels are available.

### Q3. Why use MILP?
Because the planning problem includes capacities, setups, minimum runs, lane limits, inventory balance, and service constraints.

### Q4. Why is this not just linear programming?
Because setup decisions are binary and require integer decision variables.

### Q5. Why do we need simulation after optimization?
Because deterministic optimization can overstate performance by ignoring uncertainty in demand and lead time.

### Q6. How does this support S&OP?
It links demand planning to production, distribution, inventory, and service-level decisions in one framework.

### Q7. How would this scale to 1000 SKUs?
Feature engineering and preprocessing could move to PySpark or distributed pipelines, while optimization could use decomposition or category-level planning for speed.

### Q8. How would this be deployed on Oracle OCI Data Science?
Use OCI Object Storage for data, OCI Data Science for notebooks and jobs, Model Catalog for registered forecasting models, scheduled inference via Functions or Data Flow, and Autonomous Database for dashboard-ready outputs.

### Q9. What are the limitations of synthetic network data?
It cannot capture the exact economics or constraints of a real company network. It is useful for demonstrating planning logic, but final business decisions would require real operational data.

### Q10. What would improve with real client data?
Real plant capacities, lane performance, setup rules, service policies, lead-time variability, and inventory targets would make both optimization and simulation more realistic.

## OCI readiness

The repository is structured to map cleanly to Oracle OCI:

- OCI Object Storage for raw and processed data
- OCI Data Science notebooks or jobs for training and scoring
- OCI Model Catalog for best forecasting model registration
- OCI Functions or OCI Data Flow for scheduled planning runs
- Autonomous Database for serving planning outputs
- Streamlit dashboard deployed separately

## What not to claim

- Do not claim this is a real client project.
- Do not claim the synthetic plant/DC network reflects a real company’s actual network.
- Do not claim scenario savings are realized business savings unless backed by real implementation data.
