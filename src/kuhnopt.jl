# ] activate "I:\My Documents\00 AFIT\Research\Julia Projects\StrategyGames"
# Sequence-form Kuhn Poker
# G = (N, Σ, g, C)

# Useful extensive-form websites:
# https://cw.fel.cvut.cz/old/_media/courses/be4m36mas/6.pdf

using Pkg; Pkg.activate(pwd())
using JuMP, Clp
using LinearAlgebra


# using Convex, SCS
# const cvx = Convex
# vals = [5.0, 3.0, 1.0]
# q = [0.1, 0.2, 0.7]
# del = 0.2

# take = cvx.Variable(3)
# p = cvx.Variable(3)
# obj = sum(vals[i] * take[i] * p[i] for i = 1:3)
# c1 = sum(p) == 1
# c2 = p >= 0
# c3 = take >= 0
# c4 = take <= 1
# c5 = -sum(q[i] * log(p[i] / q[i]) for i in 1:length(q)) <= del
# constraints = [c1, c2, c3, c4, c5]
# problem = maximize(obj, constraints)
# solve!(problem)
# problem.status
# problem.optval
#
# m = Model(solver = SCSSolver())
#
# @variable(m, 0 <= take[1:3] <= 1)
# @variable(m, p[1:3] >= 0)
# @objective(m, Max, sum(vals[i] * take[i] * p[i] for i = 1:3))
# @constraint(m, sum(p) == 1)
# @constraint(m, -sum(q[i] * log(p[i] / q[i]) for i in 1:length(q)) <= del)
# solve(m)
# println(getvalue(take))

#################################
# Game Definition

# N : player indices
# Two-player, nature not included
N = [1, 2]

# I : information sets
I1 = ["JX", "QX", "KX",
      "JX-check1-bet2", "QX-check1-bet2", "KX-check1-bet2"]
I11 = view(I1, 1:3)
I12 = view(I1, 4:6)
I2 = ["XJ-check1", "XJ-bet1", "XQ-check1", "XQ-bet1", "XK-check1", "XK-bet1"]
I = (I1, I2)

# Σ : sequences
Σ1 = ["∅",
      "JX-check1", "JX-bet1", "QX-check1", "QX-bet1", "KX-check1", "KX-bet1",
      "JX-check1-fold1", "JX-check1-call1", "QX-check1-fold1", "QX-check1-call1", "KX-check1-fold1", "KX-check1-call1"]
Σ2 = ["∅",
      "XJ-check2", "XJ-bet2", "XJ-fold2", "XJ-call2",
      "XQ-check2", "XQ-bet2", "XQ-fold2", "XQ-call2",
      "XK-check2", "XK-bet2", "XK-fold2", "XK-call2",]
Σ = (Σ1, Σ2)

# g : payoff function
g = [0	0	0	0	0	0	0	0	0	0	0	0	0 ;
     0	0	0	0	0	-1	0	0	0	-1	0	0	0;
     0	0	0	0	0	0	0	1	-2	0	0	1	-2;
     0	1	0	0	0	0	0	0	0	-1	0	0	0;
     0	0	0	1	2	0	0	0	0	0	0	1	-2;
     0	1	0	0	0	1	0	0	0	0	0	0	0;
     0	0	0	1	2	0	0	1	2	0	0	0	0;
     0	0	0	0	0	0	-1	0	0	0	-1	0	0;
     0	0	0	0	0	0	-2	0	0	0	-2	0	0;
     0	0	-1	0	0	0	0	0	0	0	-1	0	0;
     0	0	2	0	0	0	0	0	0	0	-2	0	0;
     0	0	-1	0	0	0	-1	0	0	0	0	0	0;
     0	0	2	0	0	0	2	0	0	0	0	0	0]

 # probability of each branch
 p = 1/6

 #################################
 # Best Response

# r2 : player 2 model
r2 = [1.0,
      2/3, 1/3, 1, 0,
      1, 0, 2/3, 1/3,
      0, 1, 0, 1]

# best response model
ns1 = length(Σ1)
m = Model(solver = ClpSolver())
@variable(m, r[1:ns1] >= 0)
@objective(m, Max, sum(sum(g[s1, s2] * p * r2[s2] for s2 = 1:13) * r[s1] for s1 = 1:13))
@constraint(m, r[1] == 1.0)
@constraint(m, r[2] + r[3] == r[1])
@constraint(m, r[4] + r[5] == r[1])
@constraint(m, r[6] + r[7] == r[1])
@constraint(m, r[8] + r[9] == r[2])
@constraint(m, r[10] + r[11] == r[4])
@constraint(m, r[12] + r[13] == r[6])
# print(m)
status = solve(m)
println("Objective value: ", getobjectivevalue(m))
println("r = ", getvalue(r))

# above answer: α = 1.0
# correct ansewr: α ∈ [0, 1/3]

strategy1 = hcat(Σ1, getvalue(r))
strategy2 = hcat(Σ2, r2)

val = dot(fill(p, 6), [0, -2, 1/3, -1, 1, 4/3])


gval(strat::Vector{Float64}) = sum(sum(g[s1, s2] * (1/6) * r2[s2] for s2 = 1:13) * strat[s1] for s1 = 1:13)

α = 1/3
rNE = [1.0, 1 - α, α, 1, 0, 1 - 3α, 3α, 2/3, 0, 2/3 - α, 1/3 + α, 0, 0]
strategyNE = hcat(Σ1, rNE)
gval(rNE)
gval(getvalue(r))

tab = hcat(1:13,strategyNE,1:13,strategy1,1:13,strategy2)

# strategy 1 is a best response but not a NE (i.e. multiple player 1 strategies
# have the same value, but only some of those are NE)
# 13×3 Array{Any,2}:
#   1  "∅"                 1.0
#   2  "JX-check1"        -0.0
#   3  "JX-bet1"           1.0
#   4  "QX-check1"         1.0
#   5  "QX-bet1"           0.0
#   6  "KX-check1"        -0.0
#   7  "KX-bet1"           1.0
#   8  "JX-check1-fold1"   0.0
#   9  "JX-check1-call1"   0.0
#  10  "QX-check1-fold1"   1.0
#  11  "QX-check1-call1"   0.0
#  12  "KX-check1-fold1"   0.0
#  13  "KX-check1-call1"   0.0


#################################
# Nash  Equilibrium

# NE model
ns2 = length(Σ2)
ni1 = length(I1)
mne = Model(solver = ClpSolver())
@variable(mne, r2[1:ns2] >= 0)
@variable(mne, v1[1:ni1])
@variable(mne, v0)
@objective(mne, Min, v0)
@constraint(mne, v0 - (v1[1] + v1[2] + v1[3]) >= sum(g[1, s2] * r2[s2] for s2 = 1:ns2))
@constraint(mne, v1[1] - v1[4] >= sum(g[2, s2] * r2[s2] for s2 = 1:ns2))
@constraint(mne, v1[1] >= sum(g[3, s2] * r2[s2] for s2 = 1:ns2))
@constraint(mne, v1[2] - v1[5] >= sum(g[4, s2] * r2[s2] for s2 = 1:ns2))
@constraint(mne, v1[2] >= sum(g[5, s2] * r2[s2] for s2 = 1:ns2))
@constraint(mne, v1[3] - v1[6] >= sum(g[6, s2] * r2[s2] for s2 = 1:ns2))
@constraint(mne, v1[3] >= sum(g[7, s2] * r2[s2] for s2 = 1:ns2))
@constraint(mne, v1[4] >= sum(g[8, s2] * r2[s2] for s2 = 1:ns2))
@constraint(mne, v1[4] >= sum(g[9, s2] * r2[s2] for s2 = 1:ns2))
@constraint(mne, v1[5] >= sum(g[10, s2] * r2[s2] for s2 = 1:ns2))
@constraint(mne, v1[5] >= sum(g[11, s2] * r2[s2] for s2 = 1:ns2))
@constraint(mne, v1[6] >= sum(g[12, s2] * r2[s2] for s2 = 1:ns2))
@constraint(mne, v1[6] >= sum(g[13, s2] * r2[s2] for s2 = 1:ns2))
@constraint(mne, r2[1] == 1.0)
@constraint(mne, r2[2] + r2[3] == r2[1])
@constraint(mne, r2[4] + r2[5] == r2[1])
@constraint(mne, r2[6] + r2[7] == r2[1])
@constraint(mne, r2[8] + r2[9] == r2[1])
@constraint(mne, r2[10] + r2[11] == r2[1])
@constraint(mne, r2[12] + r2[13] == r2[1])
print(mne)
status = solve(mne)

using Clp.ClpCInterface
r1 = dual_row_solution(getrawsolver(mne))[1:13] # player 1 strategy is the dual of player 1 values for each sequence constraint

println("Objective value: ", getobjectivevalue(mne))
println("r2 = ", getvalue(r2))
println("v0 = ", getvalue(v0))
println("v1 = ", getvalue(v1))

strategy1 = hcat(Σ1, r1)
strategy2 = hcat(Σ2, getvalue(r2))
α = 1/3
r1NE = [1.0, 1 - α, α, 1, 0, 1 - 3α, 3α, 2/3, 0, 2/3 - α, 1/3 + α, 0, 0]
strategy1NE = hcat(Σ1, r1NE)
gval(r1)

tab = hcat(1:13,strategy1NE,1:13,strategy1,1:13,strategy2)

# strategies for player 1 and player 2 match NE
