
## load packages
using JuMP, Clp

## Game Definition: Kuhn Poker
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
g = [0 0 0 0 0 0 0 0 0 0 0 0 0;
     0 0 0 0 0 -1 0 0 0 -1 0 0 0;
     0 0 0 0 0 0 0 1 -2 0 0 1 -2;
     0 1 0 0 0 0 0 0 0 -1 0 0 0;
     0 0 0 1 2 0 0 0 0 0 0 1 -2;
     0 1 0 0 0 1 0 0 0 0 0 0 0;
     0 0 0 1 2 0 0 1 2 0 0 0 0;
     0 0 0 0 0 0 -1 0 0 0 -1 0 0;
     0 0 0 0 0 0 -2 0 0 0 -2 0 0;
     0 0 -1 0 0 0 0 0 0 0 -1 0 0;
     0 0 2 0 0 0 0 0 0 0 -2 0 0;
     0 0 -1 0 0 0 -1 0 0 0 0 0 0;
     0 0 2 0 0 0 2 0 0 0 0 0 0]

 # p: probability of each branch
 p = 1/6

## Nash Equilibrium Solver
ns2 = length(Σ2)
ni1 = length(I1)
mne = Model(Clp.Optimizer)
@variable(mne, r2[1:ns2] >= 0)
@variable(mne, v1[1:ni1])
@variable(mne, v0)
@objective(mne, Min, v0)
@constraint(mne, val1, v0 - (v1[1] + v1[2] + v1[3]) >= sum(g[1, s2] * r2[s2] for s2 = 1:ns2))
@constraint(mne, val2, v1[1] - v1[4] >= sum(g[2, s2] * r2[s2] for s2 = 1:ns2))
@constraint(mne, val3, v1[1] >= sum(g[3, s2] * r2[s2] for s2 = 1:ns2))
@constraint(mne, val4, v1[2] - v1[5] >= sum(g[4, s2] * r2[s2] for s2 = 1:ns2))
@constraint(mne, val5, v1[2] >= sum(g[5, s2] * r2[s2] for s2 = 1:ns2))
@constraint(mne, val6, v1[3] - v1[6] >= sum(g[6, s2] * r2[s2] for s2 = 1:ns2))
@constraint(mne, val7, v1[3] >= sum(g[7, s2] * r2[s2] for s2 = 1:ns2))
@constraint(mne, val8, v1[4] >= sum(g[8, s2] * r2[s2] for s2 = 1:ns2))
@constraint(mne, val9, v1[4] >= sum(g[9, s2] * r2[s2] for s2 = 1:ns2))
@constraint(mne, val10, v1[5] >= sum(g[10, s2] * r2[s2] for s2 = 1:ns2))
@constraint(mne, val11, v1[5] >= sum(g[11, s2] * r2[s2] for s2 = 1:ns2))
@constraint(mne, val12, v1[6] >= sum(g[12, s2] * r2[s2] for s2 = 1:ns2))
@constraint(mne, val13, v1[6] >= sum(g[13, s2] * r2[s2] for s2 = 1:ns2))
@constraint(mne, r2[1] == 1.0)
@constraint(mne, r2[2] + r2[3] == r2[1])
@constraint(mne, r2[4] + r2[5] == r2[1])
@constraint(mne, r2[6] + r2[7] == r2[1])
@constraint(mne, r2[8] + r2[9] == r2[1])
@constraint(mne, r2[10] + r2[11] == r2[1])
@constraint(mne, r2[12] + r2[13] == r2[1])

## Solve 
optimize!(mne)
val_constraints = [val1,
                    val2, 
                    val3, 
                    val4, 
                    val5, 
                    val6, 
                    val7, 
                    val8, 
                    val9, 
                    val10, 
                    val11, 
                    val12, 
                    val13]

## Values and Duals
r1 = dual.(val_constraints)