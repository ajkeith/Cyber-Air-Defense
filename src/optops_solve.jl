# Sequence-form integrated cyber air defense, G = (N, Σ, g, C)
using Pkg; Pkg.activate("I:\\My Documents\\00 AFIT\\Research\\Julia Projects\\StrategyGames")
using Revise, LinearAlgebra, FileIO, JLD2, ProgressMeter
using JuMP, Clp
using Clp: ClpCInterface

# load functions
dir = "I:\\My Documents\\00 AFIT\\Research\\Julia Projects\\StrategyGames"
include(joinpath(dir, "src\\ops_utility.jl"))
include(joinpath(dir, "src\\ops_build.jl"))
include(joinpath(dir, "src\\ops_methods.jl"))

#############
# best response
#############

# fix player 2's realization plan as uniformly random
# how to make better realization plan? Maybe pick a few pure strategies?
# greedy plan: defender assigns cyber assets to IADS with most covered value
function build_rp2(gos::GameOptSet)
    ns2, ns2_stage, na_stage = gos.ns2, gos.ns2_stage, gos.na_stage
    rp2 = zeros(Float64, ns2)
    ns_cumsum = cumsum(ns2_stage)
    p0 = 1.0
    p3 = p0 / na_stage[3]
    p6 = p3 / na_stage[6]
    rp2[1] = p0
    rp2[(ns_cumsum[1] + 1):ns_cumsum[2]] .= p3
    rp2[(ns_cumsum[2] + 1):ns_cumsum[3]] .= p6
    return rp2
end

function build_rp1(gos::GameOptSet)
    ns1, ns1_stage, na_stage = gos.ns1, gos.ns1_stage, gos.na_stage
    rp = zeros(Float64, ns1)
    ns_cumsum = cumsum(ns1_stage)
    p0 = 1.0
    p5 = p0 / na_stage[5]
    rp[1] = p0
    rp[(ns_cumsum[1] + 1):ns_cumsum[2]] .= p5
    return rp
end

function lp_best_response(gos::GameOptSet, rfixed::Vector{Float64}; fixedplayer::Int64 = 2)
    ns1, ns2, ni1, ni2, reward = gos.ns1, gos.ns2, gos.ni1, gos.ni2, gos.reward
    seqI1, seqI2, nextseqI1, nextseqI2 = gos.seqI1, gos.seqI2, gos.nextseqI1, gos.nextseqI2
    ind_nz = findall(!iszero, gos.reward)
    nsi, nso = fixedplayer == 1 ? (ns2, ns1) : (ns1, ns2)
    nii, nio = fixedplayer == 1 ? (ni2, ni1) : (ni1, ni2)
    seqIi, seqIo = fixedplayer == 1 ? (seqI2, seqI1) : (seqI1, seqI2)
    nextseqIi, nextseqIo = fixedplayer == 1 ? (nextseqI2, nextseqI1) : (nextseqI1, nextseqI2)
    println("Building model...")
    m = Model(solver = ClpSolver())
    @variable(m, r[1:nsi] >= 0)
    if fixedplayer == 1 # reward is player 1's reward(seq1, seq2) so we can't switch the indexes
        @objective(m, Max, sum(-1 * gos.reward[I] * rfixed[I[1]] * r[I[2]] for I in ind_nz))
    else
        @objective(m, Max, sum(gos.reward[I] * r[I[1]] * rfixed[I[2]] for I in ind_nz))
    end
    @constraint(m, r[1] == 1.0)
    # @progress for i in 1:nii
    for i in 1:nii
        @constraint(m, sum(r[si] for si in nextseqIi[i]) == r[seqIi[i]])
    end
    println("Solving model...")
    @show m
    status = solve(m)
    return status, getobjectivevalue(m), getvalue(r)
end

#############
# Nash Equilibrium
#############

# TODO: make building the objective function faster in lp_nash
function lp_nash(gos::GameOptSet; timelimit = 5 * 60)
    ns2, ni1, ni2, reward = gos.ns2, gos.ni1, gos.ni2, gos.reward
    ns1_stage, na_stage = gos.ns1_stage, gos.na_stage
    seqI2, nextseqI2 = gos.seqI2, gos.nextseqI2
    println("Building model...")
    mne = Model(solver = ClpSolver(MaximumSeconds = timelimit))
    @variable(mne, rp2[1:ns2] >= 0)
    @variable(mne, v1[1:ni1])
    @variable(mne, v0)
    @objective(mne, Min, v0)
    @constraint(mne, v0 - sum(v1[j] for j in 1:ni1) >= sum(reward[1, s2] * rp2[s2] for s2 in 1:ns2))
    nvconstraint = 1
    # @progress for s1 in (1 + 1):(1 + ns1_stage[2])
    for s1 in (1 + 1):(1 + ns1_stage[2])
        iind = ceil(Int, (s1 - 1) / na_stage[5]) # Σ1 is in blocks na_stage[5] actions for each infoset
        @constraint(mne, v1[iind] >= sum(reward[s1, s2] * rp2[s2] for s2 in 1:ns2))
        nvconstraint += 1
    end
    @constraint(mne, rp2[1] == 1)
    # @progress for i in 1:ni2
    for i in 1:ni2
        @constraint(mne, sum(rp2[s2] for s2 in nextseqI2[i]) == rp2[seqI2[i]])
    end
    @show mne
    println("Solving model...")
    status, t_solve = @timed solve(mne)
    println("obj value: $(getobjectivevalue(mne)), status: $status")
    rp1 = ClpCInterface.dual_row_solution(getrawsolver(mne))[1:nvconstraint] # player 1 strategy is the dual of player 1 values for each sequence constraint
    return status, getobjectivevalue(mne), rp1, getvalue(rp2), t_solve
end

#############
# Robust Best Response
#############

function lp_robust_br(gos::GameOptSet, rp2lb, rp2ub; timelimit = 5 * 60)
    ns2, ni1, ni2, reward = gos.ns2, gos.ni1, gos.ni2, gos.reward
    ns1_stage, na_stage = gos.ns1_stage, gos.na_stage
    seqI2, nextseqI2 = gos.seqI2, gos.nextseqI2
    println("Building model...")
    mr = Model(solver = ClpSolver(MaximumSeconds = timelimit))
    @variable(mr, rp2[1:ns2])
    @variable(mr, v1[1:ni1])
    @variable(mr, v0)
    @objective(mr, Min, v0)
    @constraint(mr, v0 - sum(v1[j] for j in 1:ni1) >= sum(reward[1, s2] * rp2[s2] for s2 in 1:ns2))
    # @progress for s1 in (1 + 1):(1 + ns1_stage[2])
    nvconstraint = 1
    for s1 in (1 + 1):(1 + ns1_stage[2])
        iind = ceil(Int, (s1 - 1) / na_stage[5]) # Σ1 is in blocks na_stage[5] actions for each infoset
        @constraint(mr, v1[iind] >= sum(reward[s1, s2] * rp2[s2] for s2 in 1:ns2))
        nvconstraint += 1
    end
    @constraint(mr, rp2[1] == 1)
    # @progress for i in 1:ni2
    for i in 1:ni2
        @constraint(mr, sum(rp2[s2] for s2 in nextseqI2[i]) == rp2[seqI2[i]])
    end
    @constraint(mr, rp2 .>= rp2lb)
    @constraint(mr, rp2 .<= rp2ub)
    @show mr
    println("Solving model...")
    status, t_solve = @timed solve(mr)
    rp1 = dual_row_solution(getrawsolver(mr))[1:nvconstraint] # player 1 strategy is the dual of player 1 values for each sequence constraint
    return status, getobjectivevalue(mr), rp1, getvalue(rp2), t_solve
end
