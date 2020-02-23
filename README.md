# Counterfactual Regret Minimization for Integrated Cyber and Air Defense
This repository includes algorithms and data for Monte Carlo counterfactual regret minimization and several variants, including discounted, data-biased, and constrained CFR. It also includes linear programming formulations for best response, Nash equilibrium, and robust best response. 

## Source Code
The src file includes all development and analysis code.

The key files for each variant of CFR are:
* `cfrops_solve.jl`: Counterfactual regret minimization.
* `ccfrops_solve.jl`: Constrained counterfactual regret minimization. 
* `dcfrops_solve.jl`: Discounted counterfactual regret minimization.
* `dbr_cfrops_solve.jl`: Data-biased response CFR variant. 

The key file for the linear and robust programming approaches are:
* `optops_solve.jl`: Includes exact best response, Nash equilibrium, and robust linear programs. 

The are several files for defining the game, helper functions, and analyzing results.
* `ops_build.jl`: Builds the game structure
* `ops_methods.jl`: Lower-level functions used by CFR algorithm.
* `ops_results.jl`: Analyzes and plots the resulting strategies
* `cfrops_collect.jl`: Collects results comparing different CFR variants for standard game variants.
* `robust_collect.jl`: Collects results comparing different CFR variants for robust game variants.
* `ops_utility.jl`: Helper functions for small calculations.

Lastly there are a few development files that are no longer relevant: 
* `cfrkuhn.jl`: Hard-coded Kuhn Poker CFR solver (`cfrkuhn_old.jl` is an older version)
* `kuhnopt.jl`: Hard-coded Kuhn Poker LP solver
* `counterfactual_regret.jl`: Basic CFR methods.
* `gamedef.jl`: Basic game size methods.
* `scratch.jl`: Working development file. 
* `nodedef.jl`: Structure for defining a node.

## Citations
If this work is useful to you, please consider citing my dissertation, available [here](https://scholar.afit.edu/etd/2464/). Chapter VI provides the necessary details and reviews the CFR and extensive-form game literature that informs this code.  
