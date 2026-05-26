# OCI Deployment Plan

This is an independent Kaggle/public-data based prototype built to demonstrate network-level S&OP, demand forecasting, production planning, logistics optimization, and simulation.

## 1. OCI Object Storage
- Store raw M5 files
- Store processed weekly demand outputs
- Store forecast, optimization, and simulation result files

## 2. OCI Data Science Notebook Sessions
- Run feature engineering
- Train forecasting models
- Track experiments and compare validation results

## 3. OCI Model Catalog
- Register the best demand forecasting model
- Register packaged scoring logic used in planning jobs

## 4. OCI Functions or OCI Data Flow
- Schedule weekly forecast inference
- Schedule optimization and scenario-analysis jobs

## 5. Oracle Autonomous Database
- Store production plans
- Store shipment plans
- Store simulation and scenario outputs for reporting

## 6. Dashboard deployment
- Deploy Streamlit on a VM or container service
- Connect the dashboard to database tables or object storage outputs

## Implementation note
My production experience may be stronger on AWS, but the project is designed to be OCI-native for the role and can map cleanly to OCI-managed data science and analytics services.

