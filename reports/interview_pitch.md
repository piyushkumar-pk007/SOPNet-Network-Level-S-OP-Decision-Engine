# SOPNet Pitch

This is an independent Kaggle/public-data based prototype built to demonstrate network-level S&OP, demand forecasting, production planning, logistics optimization, and simulation.

## 30-second pitch
I built SOPNet, a network-level S&OP decision engine using public M5 sales data and a synthetic supply network. It forecasts demand across product and location hierarchy, reconciles forecasts for planning consistency, optimizes production and shipment decisions using MILP, and stress-tests the plan using SimPy simulation. The goal is to show how data science can move beyond forecasting into actual supply-chain decision support.

## 2-minute pitch
Large supply chains do not struggle only with forecasting. They struggle with connecting demand forecasts to production, warehouse allocation, transportation, and service-level decisions. In SOPNet, I used M5 public retail sales data, converted it into weekly S&OP demand, and built a hierarchy from total network down to store-item level. I trained baseline, statistical, and machine-learning forecasting models using walk-forward validation and then reconciled forecasts across levels so that total, category, store, and SKU plans are consistent. After that, I generated a realistic plant-DC-retail network and built a PuLP MILP optimizer that decides production quantity, plant allocation, DC inventory, and shipments while respecting capacity, lane, storage, setup, and service-level constraints. Finally, I used SimPy simulation to test the optimized plan under stochastic demand and lead-time delays. The output is a Streamlit dashboard showing forecast accuracy, optimized network plan, service-level risk, and scenario tradeoffs. This demonstrates the full journey from data to analytics to supply-chain decision-making.

## Deep technical explanation
- Demand is converted from daily retail sales to weekly planning buckets because S&OP decisions typically operate on weekly cadence rather than daily execution cadence.
- Forecasting is evaluated with walk-forward validation to avoid temporal leakage.
- Reconciliation keeps total, state, store, category, and item views aligned so planning discussions stay consistent.
- The optimization model is mixed-integer rather than purely linear because setup decisions are binary.
- Simulation is used after optimization because deterministic plans can look strong on average but still fail under uncertainty.

