from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard_data import executive_metrics, load_dashboard_datasets


st.set_page_config(page_title="SOPNet", layout="wide")


def fmt_currency(value: float) -> str:
    return f"${value:,.0f}"


def render_missing() -> None:
    st.info(
        "Run the pipeline first: `python -m src.data_ingestion`, `python -m src.preprocessing`, "
        "`python -m src.hierarchy_builder`, `python -m src.forecasting`, `python -m src.reconciliation`, "
        "`python -m src.network_generator`, `python -m src.optimization`, `python -m src.simulation`, "
        "and `python -m src.scenario_analysis`."
    )


datasets = load_dashboard_datasets()
metrics = executive_metrics(datasets)

st.title("SOPNet — Network-Level S&OP Decision Engine")
st.caption(
    "This is an independent Kaggle/public-data based prototype built to demonstrate network-level S&OP, demand forecasting, production planning, logistics optimization, and simulation."
)

page = st.sidebar.radio(
    "Navigate",
    [
        "Executive Overview",
        "Demand Forecasting",
        "Network Plan",
        "Optimization Results",
        "Simulation",
        "Scenario Analysis",
    ],
)

if page == "Executive Overview":
    cols = st.columns(6)
    cols[0].metric("Total Forecasted Demand", f"{metrics['total_forecast_demand']:,.0f}")
    cols[1].metric("Optimized Total Cost", fmt_currency(metrics["optimized_total_cost"]))
    cols[2].metric("Expected Service Level", f"{metrics['expected_service_level']:.1%}")
    cols[3].metric("Stockout Risk", f"{metrics['stockout_risk']:,.0f}")
    cols[4].metric("Saving vs Baseline", fmt_currency(metrics["cost_saving_vs_baseline"]))
    cols[5].metric("Top Constrained Plant", metrics["top_constrained_plant"] or "N/A")

    if datasets["forecast_results"].empty:
        render_missing()
    else:
        st.subheader("Forecast Horizon View")
        fc = datasets["forecast_results"].groupby("week_id", as_index=False)["forecast"].sum()
        st.plotly_chart(px.bar(fc, x="week_id", y="forecast"), use_container_width=True)

        if not datasets["production_plan"].empty:
            st.subheader("Production by Plant")
            plant_df = datasets["production_plan"].groupby("plant_id", as_index=False)["produce_qty"].sum()
            st.plotly_chart(px.bar(plant_df, x="plant_id", y="produce_qty", color="plant_id"), use_container_width=True)

if page == "Demand Forecasting":
    forecast = datasets["forecast_results"]
    model_cmp = datasets["model_comparison"]
    reconciled = datasets["reconciled_forecast"]
    if forecast.empty:
        render_missing()
    else:
        selected_series = st.selectbox("Select series", sorted(forecast["series_id"].dropna().unique()))
        filtered = forecast[forecast["series_id"] == selected_series].sort_values("date")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filtered["date"], y=filtered["actual"], mode="lines+markers", name="Actual"))
        fig.add_trace(go.Scatter(x=filtered["date"], y=filtered["forecast"], mode="lines+markers", name="Forecast"))
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Model Comparison")
        st.dataframe(model_cmp[model_cmp["series_id"] == selected_series].sort_values("wape"), use_container_width=True)
        st.subheader("Reconciled Forecast View")
        if not reconciled.empty:
            st.dataframe(reconciled.head(50), use_container_width=True)

if page == "Network Plan":
    production = datasets["production_plan"]
    shipments = datasets["shipment_plan"]
    inventory = datasets["inventory_plan"]
    if production.empty:
        render_missing()
    else:
        st.subheader("Production Plan by Plant")
        st.dataframe(production, use_container_width=True)
        st.subheader("Shipments")
        st.dataframe(shipments, use_container_width=True)
        st.subheader("Inventory Position by DC")
        st.dataframe(inventory, use_container_width=True)

if page == "Optimization Results":
    baseline = datasets["baseline_comparison"]
    unmet = datasets["unmet_demand"]
    if baseline.empty:
        render_missing()
    else:
        st.subheader("Baseline vs Optimized")
        st.dataframe(baseline, use_container_width=True)
        st.plotly_chart(px.bar(baseline, x="strategy", y="total_cost", color="strategy"), use_container_width=True)
        if not unmet.empty:
            st.subheader("Unmet Demand")
            st.dataframe(unmet, use_container_width=True)

if page == "Simulation":
    sim_results = datasets["simulation_results"]
    sim_summary = datasets["simulation_summary"]
    if sim_results.empty:
        render_missing()
    else:
        if not sim_summary.empty:
            st.subheader("Simulation Summary")
            st.dataframe(sim_summary, use_container_width=True)
        st.plotly_chart(px.histogram(sim_results, x="avg_service_level", nbins=25), use_container_width=True)
        st.plotly_chart(px.histogram(sim_results, x="stockout_units", nbins=25), use_container_width=True)

if page == "Scenario Analysis":
    scenario = datasets["scenario_comparison"]
    if scenario.empty:
        render_missing()
    else:
        st.subheader("Scenario Comparison")
        st.dataframe(scenario, use_container_width=True)
        st.plotly_chart(px.bar(scenario, x="scenario", y="total_cost", color="scenario"), use_container_width=True)
        st.plotly_chart(px.bar(scenario, x="scenario", y="simulation_avg_service_level", color="scenario"), use_container_width=True)

