using Pkg; Pkg.activate("I:\\My Documents\\00 AFIT\\Research\\Julia Projects\\StrategyGames")
using Revise
using Test, LinearAlgebra, Random, Statistics
using FileIO, JLD2
dir = "I:\\My Documents\\00 AFIT\\Research\\Julia Projects\\StrategyGames"
include(joinpath(dir, "src\\ops_utility.jl"))
include(joinpath(dir, "src\\ops_build.jl"))
include(joinpath(dir, "src\\ops_methods.jl"))
include(joinpath(dir, "src\\cfrops_solve.jl"))
include(joinpath(dir, "src\\dcfrops_solve.jl"))
include(joinpath(dir, "src\\optops_solve.jl"))
include(joinpath(dir, "src\\ops_results.jl"))


##########################################################
# build sequence-form reward
# CAUTION: takes 1 hr+ to build sequence-form reward for 10 city default
ncity = 10
g = AirDefenseGame(ncity, 6, 3, 4, 4, 2, 2)
A, An, na_stage = build_action(g)
ni, ni1, ni2, ni_stage = build_info(g)
ns1, ns2, ns1_stage, ns2_stage = build_nseq(g, na_stage, ni_stage)
Σ, seqn, seqactions = build_Σset(g, A, ns1, ns2)
(U, z), uh_time = @timed build_utility_hist(g, A, An)
reward_exp2, runtime2, nalloc, gc_time, misc = @timed build_utility_seq(g, gs, seqn, seqactions, expected = true)

dir = "I:\\My Documents\\00 AFIT\\Research\\Julia Projects\\StrategyGames"
fn = joinpath(dir, "data\\vars_opt_rewardfunction_expected_15city_temp.jld2")
# @save fn g U z reward_exp Σ seqn seqactions runtime nalloc
@show runtime
# # load(fn)

gs = GameSet(g)
gos = GameOptSet(gs, reward_exp)
(status, u_ne, r_ne, t_solve), t_total = @timed lp_nash(gos)
lp_best_response(gos, r_ne, fixedplayer = 2)

##########################################################
# Data-biased response and robust best response
g = AirDefenseGame(10,
        reshape([0.03674972063074833,0.7471235124528937,0.8006182686839263,0.1134119509104532,0.3720810993450727,0.6452465195265953,0.9927983579441313,0.8545575129265128,0.29473362663716407,0.9808676508834953,0.6242283949611096,0.6173536676330247,0.4739652533798455,0.6677615345281047,0.6338233451643072,0.8231268712815414,0.011816562466650193,0.048514560425303443,0.7974455234366102,0.22065505430418475], 10, 2),
        [0.47973468500756566,0.1596483573101528,0.7116043145363229,0.969005005082471,0.7139100481289631,0.4747849042596852,0.5066187533113924,0.7074385618615113,0.21009348198668265,0.29520436695319363],
        0.3,
        6,
        3,
        4,
        4,
        2,
        2,
        0.9,
        0.8,
        0.7,
        0.2,
        [1, 3, 4, 5, 8, 9])
gs = GameSet(g)
A, An, na_stage = build_action(g)
ni, ni1, ni2, ni_stage = build_info(g)
ns1, ns2, ns1_stage, ns2_stage = build_nseq(g, na_stage, ni_stage)
Σ, seqn, seqactions = build_Σset(g, A, ns1, ns2)
(U, z), uh_time = @timed build_utility_hist(g, A, An)
(reward_exp, reward_complete), runtime, _, _, _ = @timed build_utility_seq(g, gs, (ns1, ns2), seqn, seqactions, expected = true)

dir = pwd()
fn = joinpath(dir, "data\\vars_opt_rewardfunction_expected_15city_flipped_temp.jld2")
# @save fn g U z reward_exp Σ seqn seqactions runtime
@show runtime
# load(fn)

dir = "I:\\My Documents\\00 AFIT\\Research\\Julia Projects\\StrategyGames"
include(joinpath(dir, "src\\dbr_cfrops_solve.jl"))
using Plots; gr()
utemp, rtemp, stemp, σtemp = cfr(1_000, g, gs)
σfix = copy(σtemp)
pdbr = vcat(fill(0.95, gs.ni_stage[3]), fill(0.0, gs.ni_stage[5]), fill(0.85, gs.ni_stage[6]))
fn = "data\\vars_cfr_dbr_attackerfixed_temp.jld2"
# @save joinpath(dir, fn) σfix pdbr

T = 3_000

# NE for g
ucfr, _, _, σcfr = cfr(T, g, gs)
mean(ucfr)
plot(cumsum(ucfr) ./ collect(1:T))

# NE for g using cfr_dbr (PConf = 0 and σfix = arbitrary)
u2, _, _, σ2 = cfr_dbr(T, g, gs, zeros(length(pdbr)), σcfr .* 100)
mean(u2)
all(sum(i) ≈ 1 for i in σ2)

# NE for g using cfr_dbr (PConf = 1 and σfix = σNE)
u3, _, _, _ = cfr_dbr(T, g, gs, ones(length(pdbr)), σcfr)
mean(u3)

# calculate defender's robust CFR BR to σfix at 100% confidence
# this matches lp_best_response (at least sometimes...)
T = 3000
u1, _, _, _ = cfr_dbr(T, g, gs, ones(length(pdbr)), σfix)
mean(u1)
plot(cumsum(u1) ./ collect(1:T))

# calculate defender's BR to σfix
gos = GameOptSet(g, gs, reward_exp)
status_br, u_br, r_br = lp_best_response(gos, real_strat(σfix, gs, gos)[2], fixedplayer = 2)
u_br

status_brne, u_brne, r_brne = lp_best_response(gos, real_strat(σcfr, gs, gos)[2], fixedplayer = 2)
u_brne

st_ne, u_ne, r_ne, t_ne = lp_nash(gos, 5 * 60)
u_ne

status_brne, u_brne, r_brne = lp_best_response(gos, r_ne, fixedplayer = 2)
u_brne

T = 20_000
ucfr, _, _, σcfr = cfr(T, g, gs)
mean(ucfr)
status_brne, u_brne_1, r_brne = lp_best_response(gos, real_strat(σcfr, gs, gos)[2], fixedplayer = 2)
u_brne_1
status_brne, u_brne_2, r_brne = lp_best_response(gos, real_strat(σcfr, gs, gos)[1], fixedplayer = 1)
u_brne_2

# calculate defender's robust CFR BR to σfix at pdbr% confidence
u, _, _, _ = cfr_dbr(T, g, gs, pdbr, σfix)
mean(u)
plot(cumsum(u) ./ collect(1:T))


# calculate defender's exact robust BR to σfix
# TBD - need to code up robust BR...from overleaf

##########################################################
# deprecate once GameOptSet constructor works
# build AirDefenseGame, GameSet, and GameOptSet for both optimization and CFR
g = AirDefenseGame()
gs = GameSet(g)
A, An, na_stage = build_action(g)
ni, ni1, ni2, ni_stage = build_info(g)
ns1, ns2, ns1_stage, ns2_stage = build_nseq(g, na_stage, ni_stage)
Σ, seqn, seqactions = build_Σset(g, A, ns1, ns2)
Iset, (I1set, I2set), (seqI1, seqI2), (nextseqI1, nextseqI2) = build_Iset(g, A, na_stage, ns1_stage, ns2_stage, ni1, ni2, ns1, ns2)
# load expensive reward function
fn = joinpath(dir, "data\\vars_opt_rewardfunction.jld2")
@load fn reward
fn2 = joinpath(dir, "data\\vars_opt_rewardfunction_expected.jld2")
@load fn2 reward_exp
# Next line defaults to expected reward
gos = GameOptSet(ns1, ns2, ni1, ni2, ns1_stage, ns2_stage, reward_exp,
    gs.na_stage, gs.sensors, seqI1, seqI2, nextseqI1, nextseqI2)


##########################################################
# CFR
Random.seed!(579843)
# g = AirDefenseGame(0.95)
# gamesize(AirDefenseGame(15,8,5,8,8,3,3))
# g = AirDefenseGame(13,6,4,6,6,3,3)
# gamesize(g)
# gamesize(AirDefenseGame())
# g = AirDefenseGame(10,8,4,8,8,3,3)
g = AirDefenseGame()


g = AirDefenseGame(10,
        reshape([0.03674972063074833,0.7471235124528937,0.8006182686839263,0.1134119509104532,0.3720810993450727,0.6452465195265953,0.9927983579441313,0.8545575129265128,0.29473362663716407,0.9808676508834953,0.6242283949611096,0.6173536676330247,0.4739652533798455,0.6677615345281047,0.6338233451643072,0.8231268712815414,0.011816562466650193,0.048514560425303443,0.7974455234366102,0.22065505430418475], 10, 2),
        [0.47973468500756566,0.1596483573101528,0.7116043145363229,0.969005005082471,0.7139100481289631,0.4747849042596852,0.5066187533113924,0.7074385618615113,0.21009348198668265,0.29520436695319363],
        0.3,
        6,
        3,
        4,
        4,
        2,
        2,
        0.9,
        0.8,
        0.7,
        0.2,
        [1, 3, 4, 5, 8, 9])
gs = GameSet(g)
# gos = GameOptSet(gs, reward_exp)
T = 5_000
@time u, r, s, σ = cfr(T, g, gs) # 45 ms => 45 sec per 1k iterations
mean(u)

@time u1, r, s, σ, σ1, converged, iter = dcfr_full(g, gs, iterlimit = T)
mean(u1)

# psc = 0.55, pac = 0.8, T = 5_000: utility = -1.2374797
# psc = 0.95, pac = 0.8, T = 5_000: utility = -1.2249

##########################################################
# CFR Stopping Rule and Step size
using Plots; gr()
using Distributed
nprocs() == 1 && addprocs()
@everywhere using Pkg
@everywhere Pkg.activate("I:\\My Documents\\00 AFIT\\Research\\Julia Projects\\StrategyGames")
@everywhere(using Revise, BenchmarkTools, Printf, ProgressMeter, FileIO, JLD2)
@everywhere(using Random, LinearAlgebra, Statistics, Distances, IterTools, DataFrames, DataFramesMeta)
@everywhere(using StaticArrays, SparseArrays)
@everywhere(using JuMP, Clp)
@everywhere(using Clp: ClpCInterface)
@everywhere dir = "I:\\My Documents\\00 AFIT\\Research\\Julia Projects\\StrategyGames"
@everywhere include(joinpath(dir, "src\\ops_utility.jl"))
@everywhere include(joinpath(dir, "src\\ops_build.jl"))
@everywhere include(joinpath(dir, "src\\ops_methods.jl"))
@everywhere include(joinpath(dir, "src\\cfrops_solve.jl"))
@everywhere include(joinpath(dir, "src\\dcfrops_solve.jl"))
@everywhere include(joinpath(dir, "src\\optops_solve.jl"))
@everywhere include(joinpath(dir, "src\\ops_results.jl"))

Random.seed!(579843)
g = AirDefenseGame(10,
        reshape([0.03674972063074833,0.7471235124528937,0.8006182686839263,0.1134119509104532,0.3720810993450727,0.6452465195265953,0.9927983579441313,0.8545575129265128,0.29473362663716407,0.9808676508834953,0.6242283949611096,0.6173536676330247,0.4739652533798455,0.6677615345281047,0.6338233451643072,0.8231268712815414,0.011816562466650193,0.048514560425303443,0.7974455234366102,0.22065505430418475], 10, 2),
        [0.47973468500756566,0.1596483573101528,0.7116043145363229,0.969005005082471,0.7139100481289631,0.4747849042596852,0.5066187533113924,0.7074385618615113,0.21009348198668265,0.29520436695319363],
        0.3,
        6,
        3,
        4,
        4,
        2,
        2,
        0.9,
        0.8,
        0.7,
        0.2,
        [1, 3, 4, 5, 8, 9])
gs = GameSet(g)
# @load joinpath(dir, "data\\vars_opt_rewardfunction_expected.jld2") reward_exp
# gos = GameOptSet(gs, reward_exp)

# @time u, r, s, σ, σ1, converged = cfr_stoprule(5, g, gs, tolerance = 5e-5)
# mean(u)
# um = cumsum(u) ./ collect(1:length(u))
# status, u_br, r_br = lp_best_response(gos, real_strat(σ, gs, gos)[1], fixedplayer = 1)
# plot(1:length(um), um, xlim = (0,1000))
# ud, rd, sd, σd, σ1d, converged_d = dcfr_full(g, gs, timelimit = 2, tol = 5e-5,
#                                                 α = 1.5, β = 0.5, γ = 2.0, discounted = true)
# non-discounted: mean(ud) = -1.245077
# @show mean(ud)
# umd = cumsum(ud) ./ collect(1:length(ud))
# plot!(1:length(umd), umd)
# plot(1:length(ud), ud)
# statusd, u_brd, r_brd = lp_best_response(gos, real_strat(σd, gs, gos)[1], fixedplayer = 1)
# rerr(-u_brd, -1.2595387377)

#
# alphas = [0.0, 0.0, 0.0, 0.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 1.5]
# betas = [-8.0, -8.0, 8.0, 8.0, -8.0, -8.0, 8.0, 8.0, 8.0, -8.0, 0.5]
# gammas = [0.0, 8.0, 0.0, 8.0, 0.0, 8.0, 0.0, 8.0, 0.0, 2.0, 2.0]
reps = 50
alphas = repeat([8.0, 8.0, 1.5], inner = reps)
betas = repeat([8.0, -8.0, 0.5], inner = reps)
gammas = repeat([0.0, 2.0, 2.0], inner = reps)
doe_discount = DataFrame(alpha = alphas, beta = betas, gamma = gammas)
results = pmap((a,b,c) -> dcfr_full(g, gs, timelimit = 10, tol = 5e-5,
                            α = a, β = b, γ = c, discounted = true), alphas, betas, gammas)
nsteps = 12_000
ums = [cumsum(r[1]) ./ collect(1:length(r[1])) for r in results]
ums_means = [[mean(ums[i][j] for i = (k * reps + 1):((k + 1) * reps)) for j in 1:nsteps] for k in 0:2]
ums_stds = [[std(ums[i][j] for i = (k * reps + 1):((k + 1) * reps)) for j in 1:nsteps] for k in 0:2]
# us = [r[1] for r in results]
# us_means = [[mean(us[i][j] for i = (k * reps + 1):((k + 1) * reps)) for j in 1:nsteps] for k in 0:2]
# us_stds = [[std(us[i][j] for i = (k * reps + 1):((k + 1) * reps)) for j in 1:nsteps] for k in 0:2]
pcolors = [:steelblue, :tomato, :mediumseagreen]
pstyles = [:solid, :dash, :dashdot]
fig = plot(xlabel = "Iterations", ylabel = "Defender Utility",
    xlims = (0, nsteps), ylims = (-1.27, -1.2))
# fig = plot(xlabel = "Iterations", ylabel = "Defender Utility",
#     xlims = (0, 2_000), ylims = (-1.5, -0.5))
for i in 1:3
    plot!(fig, 1:nsteps, ums_means[i], ribbon = 2.009 * ums_stds[i] ./ sqrt(reps), fillalpha = 0.2, linecolor = pcolors[i],
        label = string(alphas[i*reps], ", ", betas[i*reps], ", ", gammas[i*reps]),
        linestyle = pstyles[i])
    # plot!(fig, 1:nsteps, us_means[i], alpha = 0.5, seriestype = :line,
    #     color = pcolors[i])
    # plot!(fig, 1:nsteps, ums_means[i] .- ums_stds[i], linecolor = pcolors[i], linestyle = :dot)
    # plot!(fig, 1:nsteps, ums_means[i] .+ ums_stds[i], linecolor = pcolors[i], linestyle = :dot)
end
plot!(fig, 1:nsteps, fill(-1.2595387377, nsteps), label = "Nash Equilibrium",
    linestyle = :dot, color = :black)

dir = "I:\\My Documents\\00 AFIT\\Research\\Julia Projects\\StrategyGames"
fn = joinpath(dir, "data\\plot_stepsize_v4_temp.pdf")
# savefig(fn)
fn = joinpath(dir, "data\\vars_stepsize_fig_temp.jld2")
# @save fn results reps g doe_discount

##########################################################
# CFR DOE
using Pkg; Pkg.activate("I:\\My Documents\\00 AFIT\\Research\\Julia Projects\\StrategyGames")
using Distributed, SharedArrays, Statistics, Dates
using CSV
# using Plots; gr()
nprocs() == 1 && addprocs()
@everywhere using Pkg
@everywhere Pkg.activate("I:\\My Documents\\00 AFIT\\Research\\Julia Projects\\StrategyGames")
@everywhere(using Revise, BenchmarkTools, Printf, ProgressMeter, FileIO, JLD2)
@everywhere(using Random, LinearAlgebra, Statistics, Distances, IterTools, DataFrames, DataFramesMeta)
@everywhere(using StaticArrays, SparseArrays)
@everywhere(using JuMP, Clp)
@everywhere(using Clp: ClpCInterface)
@everywhere dir = "I:\\My Documents\\00 AFIT\\Research\\Julia Projects\\StrategyGames"
@everywhere include(joinpath(dir, "src\\ops_utility.jl"))
@everywhere include(joinpath(dir, "src\\ops_build.jl"))
@everywhere include(joinpath(dir, "src\\ops_methods.jl"))
@everywhere include(joinpath(dir, "src\\cfrops_solve.jl"))
@everywhere include(joinpath(dir, "src\\dcfrops_solve.jl"))
@everywhere include(joinpath(dir, "src\\optops_solve.jl"))
@everywhere include(joinpath(dir, "src\\ops_results.jl"))

# see scratch for screening design and RSM design
@everywhere function build_design(fn::String)
    factor_names = ["pp", "psc", "pdc",
                    "alpha", "beta", "gamma"]
    col_types = [Float64, Float64, Float64,
                    Float64, Float64, Float64]
    const_names = ["radius", "pac",
                    "ncity", "ndp", "nap",
                    "ndpdc", "ndpac", "ndc", "nac"]
    const_types = [Float64, Float64,
                    Int64, Int64, Int64,
                    Int64, Int64, Int64, Int64]
    const_values = [0.3, 0.2,
                    13, 6, 4,
                    6, 6, 3, 3]
    design = CSV.read(joinpath(dir, fn), types = col_types)
    rename!(design, old => new for (old, new) = zip(names(design), Symbol.(factor_names)))
    l, w = size(design)
    for i in 1:2
        design[Symbol(const_names[i])] = fill(const_values[i], l)
    end
    for i in 3:length(const_names)
        design[Symbol(const_names[i])] = fill(Int64(const_values[i]), l)
    end
    return design
end

@everywhere function build_constants(design)
    nruns = size(design, 1)
    Xmax = rand(maximum(design.ncity), 2)
    vmax = rand(maximum(design.ncity))
    Xs = [Xmax[1:nc, :] for nc in design.ncity]
    vs = [vmax[1:nc] for nc in design.ncity]
    iadstable = Vector{Any}[]
    for nc in unique(design.ncity), nndp in unique(design.ndp)
        push!(iadstable, [nc, nndp, sort!(sample(1:nc, nndp, replace = false))])
    end
    iadss = [iadstable[findfirst(x -> x[1] == design.ncity[i] && x[2] == design.ndp[i],
                                    iadstable)][3] for i in 1:nruns]
    return Xs, vs, iadss
end

@everywhere function build_doegames(design, Xs, vs, iadss)
    nruns = size(design, 1)
    gvec = Array{AirDefenseGame}(undef, nruns)
    gsvec = Array{GameSet}(undef, nruns)
    nnodevec = Array{String}(undef, nruns)
    for i in 1:nruns
        ncity = design.ncity[i][1]
        radius = design.radius[i][1]
        ndp = design.ndp[i][1]
        nap = design.nap[i][1]
        ndpdc = design.ndpdc[i][1]
        ndpac = design.ndpac[i][1]
        ndc = design.ndc[i][1]
        nac = design.nac[i][1]
        pp = design.pp[i][1]
        psc = design.psc[i][1]
        pdc = design.pdc[i][1]
        pac = design.pac[i][1]
        a = design.alpha[i][1]
        b = design.beta[i][1]
        c = design.gamma[i][1]
        gvec[i] = AirDefenseGame(ncity, Xs[i], vs[i], radius, ndp, nap, ndpdc, ndpac, ndc, nac,
                            pp, psc, pdc, pac, iadss[i])
        gsvec[i] = GameSet(gvec[i])
        nnodevec[i] = gamesize(gvec[i])[4]
    end
    return gvec, gsvec, nnodevec
end

@everywhere function calc_run(df_row, g, gs, runnum)
    a = df_row.alpha[1]
    b = df_row.beta[1]
    c = df_row.gamma[1]
    println("----------- Start Run $runnum -----------")
    (u, _, _, _, _, converged, iter), runtime = @timed dcfr_full(g, gs,
                iterlimit = 100_000, timelimit = 240, tol = 5e-5,
                α = a, β = b, γ = c, discounted = true)
    println("------##### End Run $runnum #####------")
    return mean(u), converged, iter, runtime
end

# design = build_design("data\\design_frac_207.csv")
design = build_design("data\\design_frac_207_augment.csv")
nruns = size(design, 1)
Xs, vs, iadss = build_constants(design)
gvec, gsvec, nnodevec = build_doegames(design, Xs, vs, iadss)
parse.(Float64, nnodevec) |> findmax

results_doe, t_doe = @timed pmap(x -> calc_run(design[x, :], gvec[x], gsvec[x], x), 1:nruns)
fn = joinpath(dir, "data\\results_doe_frac207augment_temp.jld2")
# @save fn results_doe, t_doe, gvec

dfr = hcat(design, DataFrame(utility = [ri[1] for ri in results_doe],
                converged = [ri[2] for ri in results_doe],
                iterations = [ri[3] for ri in results_doe],
                time = [ri[4] for ri in results_doe]))

# CSV.write(joinpath(dir, "data\\results_doe_frac207augment_temp.csv"), dfr)

##########################################################
# CFR Large

using Pkg; Pkg.activate("I:\\My Documents\\00 AFIT\\Research\\Julia Projects\\StrategyGames")
using Revise
using Distributed, SharedArrays, Statistics, Dates
using CSV
# using Plots; gr()
nprocs() == 1 && addprocs()
@everywhere using Pkg
@everywhere Pkg.activate("I:\\My Documents\\00 AFIT\\Research\\Julia Projects\\StrategyGames")
@everywhere(using Revise, BenchmarkTools, Printf, ProgressMeter, FileIO, JLD2)
@everywhere(using Random, LinearAlgebra, StatsBase, Distances, IterTools, DataFrames, DataFramesMeta)
@everywhere(using StaticArrays, SparseArrays)
@everywhere(using JuMP, Clp)
@everywhere(using Clp: ClpCInterface)
@everywhere dir = "I:\\My Documents\\00 AFIT\\Research\\Julia Projects\\StrategyGames"
@everywhere include(joinpath(dir, "src\\ops_utility.jl"))
@everywhere include(joinpath(dir, "src\\ops_build.jl"))
@everywhere include(joinpath(dir, "src\\ops_methods.jl"))
@everywhere include(joinpath(dir, "src\\cfrops_solve.jl"))
@everywhere include(joinpath(dir, "src\\dcfrops_solve.jl"))
@everywhere include(joinpath(dir, "src\\optops_solve.jl"))
@everywhere include(joinpath(dir, "src\\ops_results.jl"))

# collect and load reward function data
@everywhere function collect_sizedata_seqreward(ncity, tlim)
    g = AirDefenseGame(ncity, 6, 3, 4, 4, 2, 2)
    gs = GameSet(g)
    A, An, na_stage = build_action(g)
    ni, ni1, ni2, ni_stage = build_info(g)
    ns1, ns2, ns1_stage, ns2_stage = build_nseq(g, na_stage, ni_stage)
    Σ, seqn, seqactions = build_Σset(g, A, ns1, ns2)
    U, z = build_utility_hist(g, A, An)
    (reward_exp, completed), runtime = @timed build_utility_seq(g, gs, (ns1, ns2), seqn, seqactions,
                                                    expected = true, timelimit = tlim)
    return g, U, z, reward_exp, completed, Σ, seqn, seqactions, runtime
end
Random.seed!(579843)
ncities = [7, 8, 9, 10, 11, 12, 13, 14, 15] # number of cities (all other game params fixed)
n = length(ncities)
println("Start Time: $(now())")
results = pmap((a, b) -> collect_sizedata_seqreward(a, b), ncities, fill(12 * 60, length(ncities)))
dirlocal = "C:\\Users\\AKeith\\JuliaProjectsLocal\\StrategyGamesLocal"
fnlocal = joinpath(dirlocal, "data\\vars_opt_rewardfunction_expected_7to15city_temp.jld2")
# # @save fnlocal results
# fn = joinpath(dir, "data\\vars_opt_rewardfunction_expected_7to15city_temp.jld2") # 12 hr time limit
# # @save fn results
# @load joinpath(dirlocal, "data\\vars_opt_rewardfunction_expected_7to15city.jld2") results # approx 5 min
gvec = [ri[1] for ri in results]
gsvec = GameSet.(gvec)
gosvec = [GameOptSet(gvec[i], gsvec[i], rexps[i]) for i in 1:n]
rexps = [ri[4] for ri in results]
gsizes = [gamesize(gi)[4] for gi in gvec]
seqsizes = [size(ri[7][1])[1] * size(ri[7][2])[1] for ri in results]
statuses = [ri[5] for ri in results]
runtimes = [ri[end] for ri in results] ./ (60 * 60) # in hours

# collect and load LP
unes = SharedArray{Float64}(n)
tnes = SharedArray{Float64}(n)
statusnes = SharedArray{Bool}(n)
println("LP NE start time for $n city sizes: $(now())")
@distributed for i in 1:n
    g = gvec[i]
    gs = gsvec[i]
    gos = gosvec[i]
    maxseconds = 10 * 60
    (status_ne, u_ne, r_ne, t_solve), t_ne = @timed lp_nash(gos, maxseconds)
    unes[i] = u_ne
    tnes[i] = t_ne
    (status_ne == :Optimal) && (statusnes[i] = true)
end
fn = joinpath(dir, "results_size_lpne_7to15city_10min_temp.jld2")
# @save fn unes tnes statusnes
# @load joinpath(dir, "results_size_lpne_15city_10min.jld2")  unes tnes statusnes

# collect and load CFR
@everywhere function collect_sizedata_cfr(g::AirDefenseGame, gs::GameSet)
    (u, r, s, σ, σs, converged), runtime = @timed dcfr_full(g, gs, timelimit = 10, tol = 5e-5, α = 1.5, β = 0.0, γ = 2.0, discounted = false)
    return u, r, s, σ, σs, converged, runtime
end
Random.seed!(748685)
cfr_params = Dict("timelimit" => 10, "tol" => 5e-5, "α" => 1.5, "β" => 0.0, "γ" => 2.0, "discounted" => false)
println("Start Time: $(now())")
results_cfr = pmap((a, b) -> collect_sizedata_cfr(a, b), gvec, gsvec)
fn = joinpath(dir, "data\\results_size_cfr_7to15city_10min_temp.jld2")
# @save fn results_cfr gvec cfr_params
# @load joinpath(dir, "data\\results_size_cfr_15city_5min.jld2") results ncities T
ucfrs = [ri[1] for ri in results_cfr]
umcfrs = [cumsum(ri[1]) ./ collect(1:length(ri[1])) for ri in results_cfr]
umeancfrs = [mean(ri[1]) for ri in results_cfr]
tcfrs = [ri[7] for ri in results_cfr]
σcfrs = [ri[4] for ri in results_cfr]
convergecfrs = [ri[6] for ri in results_cfr]

# collect and load best responses
@everywhere fbr(g, gs, gos, σ) = lp_best_response(gos, real_strat(σ, gs, gos)[1], fixedplayer = 1)
results_br = pmap(fbr, gvec, gsvec, gosvec, σcfrs) # status_br, u_br, r_br
fn = joinpath(dir, "data\\results_size_br_7to15city_10min_temp.jld2")
# @save fn results_br gvec
ubrs = [-1 * ri[2] for ri in results_br]
uexploits = rerr.(ubrs, unes)

# make size summary data frame
df_size = DataFrame(ncity = ncities, nnodes = gsizes, nseqcombos = seqsizes,
                        rewardtime = runtimes, rewardcomplete = statuses,
                        lptime = tnes, lputility = unes, lpstatus = statusnes,
                        cfrtime = tcfrs, cfrutility = umeancfrs, cfrconverged = convergecfrs,
                        brutility = ubrs, exploitability = uexploits)
# CSV.write(joinpath(dir, "data\\results_size_7to15_temp.csv"), df_size)


# plots
using DataFrames, DataFramesMeta

dfr = DataFrame(ncity = Int64[], iter = Int64[], utility = Float64[], um = Float64[], converged = Bool[], runtime = Float64[])
for i in eachindex(us)
    for j in eachindex(us[i])
        push!(dfr, [ncities[i], j, us[i][j], ums[i][j]])
    end
end

dfx = @linq dfr |>
    where(:ncity .== 10) |>
    select(:iter, :um)

fig = plot(dfx[:iter], dfx[:um], xlabel = "Iterations", ylabel = "Defender Utility")
for city in ncities
    dfi = @linq dfr |>
        where(:ncity .== city) |>
        select(:iter, :um)
    plot!(fig, dfi[:iter], dfi[:um], label = "|M|=$city")
end
fig





######################################################################################
# Analyze default scenario
# Output: Defender utility by iteration plot, relative exploitability by iteration plot
include(joinpath(dir, "src\\optops_solve.jl"))
using Plots; gr()
Random.seed!(579843)
g = AirDefenseGame()
gs = GameSet(g)
@load joinpath(dir, "data\\vars_opt_rewardfunction_expected.jld2") reward_exp
gos = GameOptSet(g, gs, reward_exp)
T = 1_000
@time u, r, s, σ, u_br = cfr_exploit(T, g, gs, gos)
(status, u_ne, r1_ne, r2_ne, t_solve), t_total = @timed lp_nash(gos)
println("Status: $status, Obj Value: $u_ne, Solve Time: $t_solve, Build Time: $(t_total - t_solve)")

@show u_ne # -1.2595387377
um = cumsum(u) ./ collect(1:T) # incremental utility
u_rerr = rerr.(u_br, u_ne) # relative error calc of br to ne

# title = "Defensive utility raw"
plot(1:T, um, legend = false, ylims = (-1.26, -1.2),
    ylabel = "Defender Utility", xlabel = "Iterations")

# title = "CFR approximate relative error"
plot(1:T, [rerr.(um, u_true)] * 100, legend = false, ylims = (0.0,15),
    ylabel = "Relative Error (%)", xlabel = "Iterations")

# title = "CFR exploitability"
plot(T ÷ 10:T ÷ 10:T, u_rerr * 100, legend = false, ylims = (0.0,2),
    seriestype = :line, markerstrokealpha = 0,
    ylabel = "Exploitability (% Nash Equilibrium Utility)", xlabel = "Iterations")
plot!(T ÷ 10:T ÷ 10:T, u_rerr * 100, seriestype = :scatter, markerstrokealpha = 0,
    markercolor = :steelblue)
plot!(1:T, fill(0.5, T), linecolor = :red)
annotate!(T, 0.6, text("0.5% Exploitability", 10, :red, :right))
dir = "I:\\My Documents\\00 AFIT\\Research\\Julia Projects\\StrategyGames"
fn = joinpath(dir, "data\\plot_default_temp.pdf")
# savefig(fn)

function h_mle(σ, g, gs; chance_ind = [1,1,0,1,0,0])
    h = SVector(0, 0, 0, 0, 0, 0)
    depth = 1
    for i = 1:6
        if getplayer(depth) == 3
            h = setindex(h, chance_ind[depth], depth)
            depth += 1
        else
            maxval, maxind = findmax(σ[infoset(h, depth, g, gs)])
            h = setindex(h, maxind, depth)
            depth += 1
        end
    end
    h
end

function getutility(h, U, z)
    U[findfirst(x -> x == h, z)]
end

g = AirDefenseGame()
gs = GameSet(g)
U, z = build_utility_hist(g, gs.A, gs.An)
h1 = h_mle(σ, g, gs, chance_ind = [1,1,0,4,0,0])
h3 = SVector(1, 1, 1, 4, 1, 1)
dp3 = getlaydown(h3, g, coverage(g), gs.A)
plot_laydown(dp3)
dp1 = getlaydown(h1, g, coverage(g), gs.A)
plot_laydown(dp1)
dir = "I:\\My Documents\\00 AFIT\\Research\\Julia Projects\\StrategyGames"
fn = joinpath(dir, "data\\plot_default_[1,1,1,1,6,39].pdf")
savefig(fn)
h2 = h_mle(σ, g, gs, chance_ind = [1,1,0,4,0,0])
dp2 = getlaydown(h2, g, coverage(g), gs.A)
plot_laydown(dp2)
getutility(h1, U, z)
getutility(h2, U, z) # this should be higher since we have cyber sa

######################################################
# Distributed Pdetect investigation

using Distributed, Statistics, Plots
gr()
addprocs(10); nprocs()
@everywhere using Pkg
@everywhere Pkg.activate("I:\\My Documents\\00 AFIT\\Research\\Julia Projects\\StrategyGames")
@everywhere(using Revise, BenchmarkTools, Printf, ProgressMeter, FileIO, JLD2)
@everywhere(using Random, LinearAlgebra, StatsBase, Distances, IterTools, DataFrames, DataFramesMeta)
@everywhere(using StaticArrays, SparseArrays)
@everywhere(using JuMP, Clp)
@everywhere(using Clp: ClpCInterface)
@everywhere dir = "I:\\My Documents\\00 AFIT\\Research\\Julia Projects\\StrategyGames"
@everywhere include(joinpath(dir, "src\\ops_utility.jl"))
@everywhere include(joinpath(dir, "src\\ops_build.jl"))
@everywhere include(joinpath(dir, "src\\ops_methods.jl"))
@everywhere include(joinpath(dir, "src\\cfrops_solve.jl"))
@everywhere include(joinpath(dir, "src\\dcfrops_solve.jl"))
@everywhere include(joinpath(dir, "src\\optops_solve.jl"))
@everywhere include(joinpath(dir, "src\\ops_results.jl"))


@everywhere function sensor_eval(p::Float64)
    Random.seed!(579843)
    g = AirDefenseGame(p)
    gs = GameSet(g)
    (u, r, s, σ, σs, converged), runtime = @timed dcfr_full(g, gs, timelimit = 10,
        tol = 5e-5, discounted = false)
    return u, σ, converged, runtime
end

ps = 0.5:0.05:1.0
results = pmap(sensor_eval, ps)
us = [mean(r[1]) for r in results]
plot(ps, us, xlabel = "Probability of Detection", ylabel = "Defender Utility",
    legend = false, seriestype = :line)
plot!(ps, us, seriestype = :scatter, markerstrokealpha = 0, markercolor = :steelblue)
dir = "I:\\My Documents\\00 AFIT\\Research\\Julia Projects\\StrategyGames"
fn = joinpath(dir, "data\\plot_pdetect_stoprule_temp.pdf")
# savefig(fn)

rmprocs(getindex(procs(), procs() .> 1))

gtoy = AirDefenseGame(2, [1.0 1;2 1], [1.0, 1], 0.2, 2, 1, 2, 2, 1, 1, 0.9, 0.99, 0.7, 0.8, [1,2])
gstoy = GameSet(gtoy)
T = 15_000
(utoy, r, s, σtoy), t_cfr = @timed cfr(T, gtoy, gstoy)
mean(utoy)
c1 = [chance(4, i, gtoy, gstoy) for i in 1:2]
s1 = σtoy[infoset(SVector(1,1,1,1,0,0), 5, gtoy, gstoy)]
s2 = σtoy[infoset(SVector(1,1,1,2,0,0), 5, gtoy, gstoy)]
plot_laydown(getlaydown(SVector(1,1,1,1,2,1), gtoy, coverage(gtoy), gstoy.A))
leafutility(SVector(1,1,1,1,1,1), 1, 1.0, 1.0, gtoy, gstoy)
leafutility(SVector(1,1,1,1,2,1), 1, 1.0, 1.0, gtoy, gstoy)
sh1 = h_mle(σtoy, gtoy, gstoy, chance_ind = [1,1,0,1,0,0])
sh2 = h_mle(σtoy, gtoy, gstoy, chance_ind = [1,1,0,2,0,0])

gtoy = AirDefenseGame(2, [1.0 1;2 1], [1.0, 1], 0.2, 2, 1, 2, 2, 1, 1, 0.9, 0.5, 0.7, 0.8, [1,2])
gstoy = GameSet(gtoy)
T = 15_000
(utoy, r, s, σtoy), t_cfr = @timed cfr(T, gtoy, gstoy)
mean(utoy)
c1 = [chance(4, i, gtoy, gstoy) for i in 1:2]
s1 = σtoy[infoset(SVector(1,1,1,1,0,0), 5, gtoy, gstoy)]
s2 = σtoy[infoset(SVector(1,1,1,2,0,0), 5, gtoy, gstoy)]
plot_laydown(getlaydown(SVector(1,1,1,1,2,1), gtoy, coverage(gtoy), gstoy.A))
leafutility(SVector(1,1,1,1,1,1), 1, 1.0, 1.0, gtoy, gstoy)
leafutility(SVector(1,1,1,1,2,1), 1, 1.0, 1.0, gtoy, gstoy)
sh1 = h_mle(σtoy, gtoy, gstoy, chance_ind = [1,1,0,1,0,0])
sh2 = h_mle(σtoy, gtoy, gstoy, chance_ind = [1,1,0,2,0,0])

g1 = AirDefenseGame(0.5)
gs1 = GameSet(g1)
@show c1 = [chance(4, i, g1, gs1) for i in 1:4]
leafutility(SVector(1,1,1,4,1,1), 1, 1.0, 1.0, g1, gs1)
leafutility(SVector(1,1,1,4,6,1), 1, 1.0, 1.0, g1, gs1)
T = 15_000
(u1, r, s, σ1), t_cfr = @timed cfr(T, g1, gs1)
mean(u1)
s11 = σ1[infoset(SVector(1,1,1,1,0,0), 5, g1, gs9)]
s12 = σ1[infoset(SVector(1,1,1,4,0,0), 5, g1, gs9)]
s1h1 = h_mle(σ1, g1, gs1, chance_ind = [1,1,0,1,0,0])
s1h2 = h_mle(σ1, g1, gs1, chance_ind = [1,1,0,4,0,0])

g9 = AirDefenseGame(0.99)
gs9 = GameSet(g9)
@show c9 = [chance(4, i, g9, gs9) for i in 1:4]
leafutility(SVector(1,1,1,4,1,1), 1, 1.0, 1.0, g9, gs9)
leafutility(SVector(1,1,1,4,6,1), 1, 1.0, 1.0, g9, gs9)
T = 15_000
(u9, r, s, σ9), t_cfr = @timed cfr(T, g9, gs9)
mean(u9)
s9h1 = h_mle(σ9, g9, gs9, chance_ind = [1,1,0,1,0,0])
s9h2 = h_mle(σ9, g9, gs9, chance_ind = [1,1,0,4,0,0])


sensor_eval(0.1, 10)
sensor_eval(0.9, 10)

function sensor_eval_toy(p::Float64, T::Int64)
    Random.seed!(579843)
    g = AirDefenseGame(2, [1.0 1;2 1], [1.0, 1], 0.2, 2, 1, 2, 2, 1, 1, 0.9, p, 0.7, 0.8, [1,2])
    @show g.v
    gs = GameSet(g)
    (u, r, s, σ), t_cfr = @timed cfr(T, g, gs)
    return u, r, s, σ, t_cfr
end

ps = 0.55:0.05:0.95
evals = [mean(sensor_eval_toy(p, 15000)[1]) for p in ps]

T = 15_000
results = pmap((a, b) -> sensor_eval(a, b), ps, fill(T, length(ps)))
us = [mean(r[1]) for r in results]
plot(ps, us, xlabel = "Probability of Detection", ylabel = "Defender Utility")
