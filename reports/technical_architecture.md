# Technical Architecture

## End-to-end flow
1. Ingest M5 files using Kaggle API or manual file placement.
2. Convert wide daily sales into a long format and aggregate to weekly demand.
3. Build the planning hierarchy from total network down to store-item.
4. Forecast demand at the selected planning level.
5. Reconcile forecasts across levels.
6. Generate a synthetic plant-DC-retail network because M5 does not contain a physical supply network.
7. Optimize production, distribution, and inventory decisions using PuLP MILP.
8. Stress-test the optimized plan using SimPy simulation.
9. Compare alternative scenarios and publish outputs through Streamlit.

## Architecture text diagram
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

