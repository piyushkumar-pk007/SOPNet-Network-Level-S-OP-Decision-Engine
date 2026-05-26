# SOPNet — Network-Level S&OP Decision Engine

SOPNet is a network-level planning project I built to connect demand forecasting with the decisions that normally follow it in an S&OP cycle: what to produce, where to hold inventory, how to route product, and what service level is actually achievable once uncertainty shows up. I used the M5 retail dataset for the demand signal, then built a synthetic supply network on top of it so the analysis could move beyond forecasting and into production planning, distribution, and scenario testing.

I built it because a lot of supply chain data science work stops at demand prediction. That is useful, but it leaves the rest of the planning conversation unresolved. Once a forecast exists, someone still has to decide how plants, DCs, and retail nodes should respond. I wanted this repo to show that the interesting part of supply chain analytics is the connection between those steps, not the steps in isolation.

**This is an independent Kaggle/public-data based prototype built to demonstrate network-level S&OP, demand forecasting, production planning, logistics optimization, and simulation.**

## Why I chose this framing

The M5 data is strong at the retail-demand level and weak everywhere else. It captures sales, calendar effects, and prices, but it does not say anything about manufacturing plants, lane capacities, setup costs, storage constraints, or service policies. Rather than treat that as a problem to hide, I made it part of the project design. The demand signal is public and real. The network is synthetic and explicit. That split keeps the project honest while still allowing it to answer planning questions that matter.

## The data and its limits

I used the M5 Forecasting Accuracy dataset, specifically `sales_train_validation.csv`, `calendar.csv`, and `sell_prices.csv`. Because the full competition data is heavier than I wanted for a laptop-friendly project, I sampled three stores, a limited set of departments, and a manageable SKU count, then aggregated daily demand to weekly demand. That shift from daily to weekly is intentional. The problem here is S&OP and network planning, not short-cycle store replenishment.

That same choice creates the main limitation. This is not a real network model driven by operational master data. Plant capacities, lane costs, DC storage, and setup logic are synthetic assumptions built to be plausible and inspectable. They are good enough to demonstrate planning mechanics, but they are not substitutes for client data.

## How the workflow fits together

The project starts by reshaping M5 from wide daily sales into a weekly planning table joined to calendar events and prices. From there I built a hierarchy that runs from total network down to item level. That hierarchy matters because the rest of the workflow depends on it: the forecasting layer needs it to compare planning levels, reconciliation needs it to keep totals aligned, and the optimizer needs a planning grain that is accurate enough to be useful but still small enough to solve on a normal machine.

Forecasting comes next. I compared seasonal naive, moving average, SARIMA, Prophet when available, and a lag-based tree model. Walk-forward validation was the only sensible choice because random folds would leak future information. In the current run, the tree-based model was consistently the strongest at the store-category level. For `CA_1 | FOODS`, for example, it finished with a WAPE around `24.36`, while the nearest classical benchmark was closer to `32.30`. That gap is meaningful because it carried through to the planning inputs rather than living only in a notebook metric table.

Once the forecasts existed, I reconciled them across levels. That piece matters more in planning than it does in modeling. Finance may care about the network total, operations may care about category-store demand, and planners may still want item detail. If those views disagree, the meeting gets harder before the plan gets better. I kept bottom-up, top-down, middle-out, and a simplified MinT-style blend so the project shows the trade-off between granular detail and top-level consistency.

After reconciliation, I generated the synthetic network: three plants, five DCs, lane costs and capacities, retail nodes derived from the sampled stores, and an ABC-style SKU master. That feeds the MILP model built in PuLP. I used mixed-integer optimization because the problem is not just about volumes. Once production setup decisions enter the picture, pure continuous LP stops being an honest representation of the planning choice.

The optimizer decides production, plant-to-DC shipments, DC-to-retail shipments, DC inventory, and unmet demand subject to plant capacity, minimum runs, lane limits, storage limits, and service-level constraints. In the current run, the optimized plan came out at about `84.2k` in modeled total cost, compared with `175.8k` for a replenish-to-par baseline and `159.8k` for a cheapest-transport baseline. It also held deterministic service at `1.0` with zero unmet demand. That sounds strong, but deterministic optimization always looks cleaner than the real world, which is why the next step matters.

I added SimPy simulation to test the optimized plan under demand variability and lead-time shocks. The simulation results were more conservative and, to me, more believable. Average service across replications was about `0.956`, with a 5th percentile near `0.935`, and expected stockout cost around `26.1k`. That gap between deterministic service and simulated service is one of the main reasons I built the simulation layer at all. It exposes the risk hidden by average-case planning.

The scenario results pushed that one step further. Transport cost inflation raised total cost as expected, high demand increased both cost and simulated stockout exposure, and the premium-service scenario slightly improved simulated service while holding the deterministic plan unchanged. Those outputs are less about “winning” a scenario and more about giving planners a way to talk through the cost-service tradeoff with something concrete in front of them.

## How to run it

I kept the local setup straightforward:

```powershell
cd "f:\Portfolio\Supply Chain - DS Project\sopnet-network-sop-engine"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

You can either put Kaggle credentials in `.env` or place `sales_train_validation.csv`, `calendar.csv`, and `sell_prices.csv` in `data/raw/m5/`. If Kaggle authentication works but the competition download is blocked, the ingestion step falls back to a public mirror automatically.

I usually run the full workflow through the wrapper:

```powershell
python run_project.py
```

That runs the pipeline and opens the Streamlit app at the end. If you want the modules directly, they are still available:

```powershell
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

The dashboard can also be launched on its own:

```powershell
python -m streamlit run app\streamlit_app.py
```

And the tests are standard:

```powershell
python -m pytest
```

## What the results actually mean

I would not frame the optimized cost reduction as a business promise. It is a modeled improvement over simple baselines inside a synthetic network. What I do think is credible is the directional lesson: once demand, production, transport, inventory, and service constraints are linked, the cheapest local choice is rarely the best network choice.

I would say the same about the simulation results. The exact service level is not the point. The point is that the deterministic plan and the stochastic plan are not the same, and that gap is exactly what planners need to understand before they trust a network recommendation.

## Honest limitations

The biggest limitation is that the network is synthetic. The plants, DCs, lane costs, and capacities are plausible, but they are not calibrated to a real company. With real operational data, the model would become much more specific about bottlenecks, setup trade-offs, and service targets.

The second limitation is aggregation. The code can support finer-grain planning, but the default run uses a category-store-week planning grain because that is the right trade-off for a laptop-scale demo. Full SKU-level MILP planning is possible in principle, but it would need either decomposition or a heavier solver setup to stay practical.

The third limitation is that M5 is a retail-demand dataset, not an end-to-end S&OP dataset. It gives a credible demand signal and a useful hierarchy, but it does not give production logic or network economics directly. I filled that gap deliberately and transparently rather than pretending it was already present.

## Closing thought

What I wanted from SOPNet was not a prettier forecast chart. I wanted a project that shows how forecasting, reconciliation, optimization, and simulation fit together in the same planning conversation. That is the part of supply chain data science I find most interesting, and it is the part this repo tries to make concrete.
