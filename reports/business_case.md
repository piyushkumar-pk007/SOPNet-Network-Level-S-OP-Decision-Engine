# Business Case

I framed SOPNet around a planning problem that most supply chains run into sooner or later: demand, production, inventory, and transport are managed by different teams, but the consequences are shared. A strong demand forecast is useful, but it does not tell a plant what to make, a DC what to hold, or a planner what service risk they are really accepting.

That is why the project starts with demand but does not stop there. I used M5 as the demand signal because it has enough history and hierarchy to support serious forecasting work. From that point, I built a synthetic network so the analysis could move into production and distribution decisions. The synthetic piece is a limitation, but it is also what makes the project a network-planning exercise rather than a retail forecast notebook.

The central business question is not just “what will demand be next month?” It is “given that demand outlook, what should the network do?” In practice that means deciding which plant should supply which categories, how much inventory should move into each DC, how retail demand should be covered, and what happens to cost and service when demand or lead time shifts.

The case for the project rests on that integration. Forecasting alone gives a better estimate of the future. Optimization turns that estimate into a plan. Simulation shows whether the plan still holds up when the future arrives less neatly than expected. When those three pieces are linked, the output starts to look like something a planning team could debate and refine, rather than just inspect.
