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
If this work is useful to you, please cite the following paper:
```
@article{KEITH2020,
title = "Counterfactual regret minimization for integrated cyber and air defense resource allocation",
journal = "European Journal of Operational Research",
year = "2020",
issn = "0377-2217",
doi = "https://doi.org/10.1016/j.ejor.2020.10.015",
url = "http://www.sciencedirect.com/science/article/pii/S0377221720308912",
author = "Andrew Keith and Darryl Ahner",
keywords = "OR in defence, regret, exploitation, robust, cybersecurity",
abstract = "This research presents a new application of optimal and approximate solution techniques to solve resource allocation problems with imperfect information in the cyber and air-defense domains. We develop a two-player, zero-sum, extensive-form game to model attacker and defender roles in both physical and cyber space. We reformulate the problem to find a Nash equilibrium using an efficient, sequence-form linear program. Solving this linear program produces optimal defender strategies for the multi-domain security game. We address large problem instances with an application of the approximate counterfactual regret minimization algorithm. This approximation reduces computation time by 95% while maintaining an optimality gap of less than 3%. Our application of discounted counterfactual regret results in a further 36% reduction in computation time from the base algorithm. We develop domain insights through a designed experiment to explore the parameter space of the problem and algorithm. We also address robust opponent exploitation by combining existing techniques to extend the counterfactual regret algorithm to include a discounted, constrained variant. A comparison of robust linear programming, data-biased response, and constrained counterfactual regret approaches clarifies trade-offs between exploitation and exploitability for each method. The robust linear programming approach is the most effective, producing an exploitation to exploitability ratio of 10.8 to 1."
}
```
