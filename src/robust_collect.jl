## packages
using Revise
using LinearAlgebra, Random, Statistics
using Distributions
using FileIO, JLD2, CSV
using Plots
gr()
dir = pwd()

## parallel packages
using Distributed
nworkers() == 1 && addprocs()
@everywhere using Pkg
@everywhere Pkg.activate(pwd())
@everywhere using Revise
@everywhere dir = pwd()
@everywhere include(joinpath(dir, "src\\ops_utility.jl"))
@everywhere include(joinpath(dir, "src\\ops_build.jl"))
@everywhere include(joinpath(dir, "src\\ops_methods.jl"))
@everywhere include(joinpath(dir, "src\\cfrops_solve.jl"))
@everywhere include(joinpath(dir, "src\\dbr_cfrops_solve.jl"))
@everywhere include(joinpath(dir, "src\\ccfrops_solve.jl"))
@everywhere include(joinpath(dir, "src\\optops_solve.jl"))
@everywhere include(joinpath(dir, "src\\ops_results.jl"))

## build g, gs, gos
function build_robust_game_scratch!()
    g = AirDefenseGame(10,
    reshape([0.03674972063074833,0.7471235124528937,0.8006182686839263,0.1134119509104532,0.3720810993450727,0.6452465195265953,0.9927983579441313,0.8545575129265128,0.29473362663716407,0.9808676508834953,0.6242283949611096,0.6173536676330247,0.4739652533798455,0.6677615345281047,0.6338233451643072,0.8231268712815414,0.011816562466650193,0.048514560425303443,0.7974455234366102,0.22065505430418475], 10, 2),
    [0.47973468500756566,0.1596483573101528,0.7116043145363229,0.969005005082471,0.7139100481289631,0.4747849042596852,0.5066187533113924,0.7074385618615113,0.21009348198668265,0.29520436695319363],
    0.3,
    5,
    3,
    4,
    4,
    2,
    2,
    0.9,
    0.8,
    0.7,
    0.2,
    [1, 3, 4, 5, 9])
    # approx 1.2M nodes
    gs = GameSet(g)
    A, An, na_stage = build_action(g)
    ni, ni1, ni2, ni_stage = build_info(g)
    ns1, ns2, ns1_stage, ns2_stage = build_nseq(g, na_stage, ni_stage)
    Σ, seqn, seqactions = build_Σset(g, A, ns1, ns2)
    (U, z), uh_time = @timed build_utility_hist(g, A, An)
    (reward_exp, reward_complete), runtime, _, _, _ = @timed build_utility_seq(g, gs, (ns1, ns2), seqn, seqactions, expected = true)
    fn = joinpath(pwd(), "data\\vars_opt_rewardfunction_expected_10city_flippedpac.jld2")
    @save fn g U z reward_exp Σ seqn seqactions runtime
    @show runtime
end

function build_robust_game()
    println("Building robust game...")
    fn = joinpath(dir, "data\\vars_opt_rewardfunction_expected_10city_flippedpac.jld2")
    fload = load(fn) # loads g and reward_exp
    g, reward_exp = fload["g"], fload["reward_exp"]
    gs = GameSet(g)
    gos = GameOptSet(g, gs, reward_exp)
    return g, gs, gos
end

# build opponent strategy and uncertainty set
@everywhere function build_opponent_strat(g, gs, gos, σfix, conf::Float64, sd::Float64)
    noise = Normal(0, sd)
    hw = quantile(noise, 1 - (1 - conf) / 2)
    σlb = [max.(σi .- hw, 0.0) for σi in σfix]
    σub = [min.(σi .+ hw, 1.0) for σi in σfix]
    mean(mean.(σub[i] - σlb[i] for i in 1:length(σlb)))
    rp2fix = real_strat(σfix, gs, gos)[2]
    rp2lb = real_strat(σlb, gs, gos)[2]
    rp2ub = real_strat(σub, gs, gos)[2]
    rp2lb_ccfr = real_strat_struct_combined(σlb, g)
    rp2ub_ccfr = real_strat_struct_combined(σub, g)
    !all(rp2lb .<= rp2fix .<= rp2ub) && @warn "Negative realizaiton probabilities."
    !all(rp2lb_ccfr .<= rp2ub_ccfr) && @warn "Negative realizaiton probabilities."
    return rp2fix, rp2lb, rp2ub, rp2lb_ccfr, rp2ub_ccfr, noise
end

@everywhere function calc_robust_strat(g, gs, gos, rp2fix, rp2lb, rp2ub, rp2lb_ccfr, rp2ub_ccfr, σfix, pconf)
    T_min = 30
    T_sec = T_min * 60
    println("Calculating NE LP strategy...")
    status_ne, u_ne, r1_ne, r2_ne, t_ne = lp_nash(gos, timelimit = T_sec)
    println("Calculating BR LP strategy...")
    status_br, u_br, r1_br = lp_best_response(gos, rp2fix, fixedplayer = 2)
    println("Calculating Robust BR LP strategy...")
    status_r, u_r, r1_r, r2_r, t_r = lp_robust_br(gos, rp2lb, rp2ub, timelimit = T_sec)
    println("Calculating Robust DBR CFR strategy...")
    (u_dbr, _, _, σ_dbr, status_dbr), t_dbr = @timed cfr_dbr(g, gs, pconf, σfix,
            tol = 5e-9, timelimit = T_min, iterlimit = 25_000)
    r1_dbr = real_strat(σ_dbr, gs, gos)[1]
    println("Calculating Robust Constrained CFR strategy...")
    (u_ccfr, _, _, σ_ccfr, _, status_ccfr, _), t_ccfr = @timed ccfr_full(g, gs, rp2lb_ccfr, rp2ub_ccfr,
            timelimit = T_min, tol = 5e-9, iterlimit = 25_000,
            α = 7.4, β = -2.9, γ = 6.3, discounted = true,
            λmax = 2000, λscale = 5)
    r1_ccfr = real_strat(σ_ccfr, gs, gos)[1]
    return (r1_ne, r1_br, r1_r, r1_dbr, r1_ccfr), (t_ne, 0.0, t_r, t_dbr, t_ccfr),
        (status_ne, status_br, status_r, status_dbr, status_ccfr)
end

@everywhere function build_robust(g, gs, gos, conf::Float64, sd::Float64, opplevel::Int)
    pconf = fill(conf, gs.ni)
    _, _, _, σfix = cfr(opplevel, g, gs)
    σfix .= [zeros(length(x)) for x in σfix]
    for i = 1:length(σfix)
        σfix[i][1] = 1.0
    end
    println("Building opponent strategies...")
    rp2fix, rp2lb, rp2ub, rp2lb_ccfr, rp2ub_ccfr, noise = build_opponent_strat(g, gs, gos, σfix, conf, sd)
    r1s, ts, statuses = calc_robust_strat(g, gs, gos, rp2fix, rp2lb, rp2ub, rp2lb_ccfr, rp2ub_ccfr, σfix, pconf)
    return r1s, ts, statuses, σfix, noise
end

@everywhere function simrobust(σfix_rand, gs, gos, r1s, ind_nz; reps = 10)
    r1_ne, r1_br, r1_r, r1_dbr, r1_ccfr = r1s
    r2_rand = real_strat(σfix_rand, gs, gos)[2]
    u1_ne = sum(gos.reward[I] * r1_ne[I[1]] * r2_rand[I[2]] for I in ind_nz)
    u1_br = sum(gos.reward[I] * r1_br[I[1]] * r2_rand[I[2]] for I in ind_nz)
    u1_r = sum(gos.reward[I] * r1_r[I[1]] * r2_rand[I[2]] for I in ind_nz)
    u1_dbr = sum(gos.reward[I] * r1_dbr[I[1]] * r2_rand[I[2]] for I in ind_nz)
    u1_ccfr = sum(gos.reward[I] * r1_ccfr[I[1]] * r2_rand[I[2]] for I in ind_nz)
    return u1_ne, u1_br, u1_r, u1_dbr, u1_ccfr
end

@everywhere function calc_robust(gs, gos, r1s, σfix, noise, nreps)
    println("Building noisy opponent strategies...")
    σfix_rands = [[LinearAlgebra.normalize(clamp.(x .+ rand(noise, length(x)), 0.0, 1.0), 1) for x in σfix] for _ in 1:nreps]
    !all(all(all(y .>= 0) for y in x) for x in σfix_rands) && @warn "Negative probabilities."
    println("Calculating utility of robust strategies...")
    ind_nz = findall(!iszero, gos.reward)
    u1s = pmap(x -> simrobust(x, gs, gos, r1s, ind_nz, reps = nreps), σfix_rands)
    u1_nes = [x[1] for x in u1s]
    u1_brs = [x[2] for x in u1s]
    u1_rs = [x[3] for x in u1s]
    u1_dbrs = [x[4] for x in u1s]
    u1_ccfrs = [x[5] for x in u1s]
    return (u1_nes, u1_brs, u1_rs, u1_dbrs, u1_ccfrs)
end

function plot_robust(u1tuple)
    u1_nes, u1_brs, u1_rs, u1_dbrs, u1_ccfrs = u1tuple
    fig = histogram(u1_nes, bins=:auto, color=:mediumseagreen, alpha=0.2,
                xlabel = "Defender Utility", ylabel = "Count",
                label = "Nash Equilibrium Strategy")
                # xlims = (-0.420, -0.390))
    # histogram!(fig, u1_brs, bins=:auto, color=:tomato, alpha=0.2,
    #             label = "Best Response Strategy")
    histogram!(fig, u1_rs, bins=:auto, color=:tomato, alpha=0.95,
                label = "Robust LP Strategy")
    histogram!(fig, u1_dbrs, bins=:auto, color=:steelblue, alpha=0.4,
                label = "Data Biased CFR Strategy")
    histogram!(fig, u1_ccfrs, bins=:auto, color=:gold, alpha=0.6,
                label = "Constrained CFR Strategy")
    fig
end

@everywhere signerr(u::Number, u_true::Number) = (u - u_true) / abs(u_true)

@everywhere function process_robust(u1tuple, r1s, gos, nreps)
    means = mean.(u1tuple)
    stds = std.(u1tuple)
    tstar = 1.98
    confintervals = (means .- tstar .* stds ./ sqrt(nreps), means .+ tstar .* stds ./ sqrt(nreps))
    u1_nes, u1_brs, u1_rs, u1_dbrs, u1_ccfrs = u1tuple
    df_robust = DataFrame(NE = u1_nes, RLP = u1_rs, DBR = u1_dbrs, CCFR = u1_ccfrs)
    r1_ne, r1_br, r1_r, r1_dbr, r1_ccfr = r1s
    br_results = pmap(x -> lp_best_response(gos, x, fixedplayer = 1), [r1_ne, r1_r, r1_dbr, r1_ccfr])
    br_ne, br_r, br_dbr, br_ccfr = collect(x[2] for x in br_results)
    # exploitability
    exp_neg = zeros(3)
    exp_neg[1] = signerr(br_r, br_ne)
    exp_neg[2] = signerr(br_dbr, br_ne)
    exp_neg[3] = signerr(br_ccfr, br_ne)
    # exploitation
    exp_pos = zeros(3)
    exp_pos[1] = signerr(means[3], means[1]) # r to ne
    exp_pos[2] = signerr(means[4], means[1]) # dbr to ne
    exp_pos[3] = signerr(means[5], means[1]) # ccfr to ne
    return means[3], means[4], means[5], exp_neg[1], exp_neg[2], exp_neg[3],
        exp_pos[1], exp_pos[2], exp_pos[3]
end

@everywhere function size_experiment_vector(gvec, gsvec, gosvec)
    nruns = length(gvec)
    df_size = DataFrame(size = Int[],
        status_r = Symbol[], status_dbr = Bool[], status_ccfr = Bool[],
        t_r = Float64[], t_dbr = Float64[], t_ccfr = Float64[],
        u_r = Float64[], u_dbr = Float64[], u_ccfr = Float64[],
        exp_neg_r = Float64[], exp_neg_dbr = Float64[], exp_neg_ccfr = Float64[],
        exp_pos_r = Float64[], exp_pos_dbr = Float64[], exp_pos_ccfr = Float64[])
    conf = 0.99
    sd = 0.02
    opplevel = 1
    nreps = 180
    for i in 1:length(gvec)
        print("Run $i:                                   ")
        g = gvec[i]
        gs = gsvec[i]
        gos = gosvec[i]
        print("\rRun $i: Building strategies...")
        (r1s, ts, statuses, σfix, noise), buildtime = @timed build_robust(g, gs, gos, conf, sd, opplevel)
        print("\rRun $i: Simulating value...")
        u1tuple, calctime = @timed calc_robust(gs, gos, r1s, σfix, noise, nreps)
        summary = vcat(i + 6, statuses[3:5]..., ts[3:5]..., process_robust(u1tuple, r1s, gos, nreps)...)
        push!(df_size, summary)
    end
    return df_size
end

@everywhere function size_experiment(g, gs, gos)
    conf = 0.99
    sd = 0.02
    opplevel = 1
    nreps = 180
    print("\rBuilding strategies...")
    (r1s, ts, statuses, σfix, noise), buildtime = @timed build_robust(g, gs, gos, conf, sd, opplevel)
    print("\rSimulating value...")
    u1tuple, calctime = @timed calc_robust(gs, gos, r1s, σfix, noise, nreps)
    return vcat(i + 6, statuses[3:5]..., ts[3:5]..., process_robust(u1tuple, r1s, gos, nreps)...)
end

##########################################################
# Single game comparison

conf = 0.99
sd = 0.02
opplevel = 1
nreps = 180

# build_robust
# g, gs, gos = build_robust_game()
# # build_robust takes about 20 min
# (r1s, ts, statuses, σfix, noise), buildtime = @timed build_robust(g, gs, gos, conf, sd, opplevel)
# fn = joinpath(pwd(), "data\\results_robust_10city_flippedpac.jld2")
# @save fn g gos r1s ts statuses σfix noise 

# load build_robust outputs
fn = joinpath(pwd(), "data\\results_robust_10city_flippedpac.jld2")
fload = load(fn) 
g, gos, r1s, σfix, noise = fload["g"], fload["gos"], fload["r1s"], fload["σfix"], fload["noise"]
gs = GameSet(g)

# calc_robust
u1tuple, calctime = @timed calc_robust(gs, gos, r1s, σfix, noise, nreps)

# process_robust 
ts, statuses = fload["ts"], fload["statuses"]
results_robust = process_robust(u1tuple, r1s, gos, nreps)
plot_robust(u1tuple)

##########################################################
# Size Experiment

dirlocal = "C:\\Users\\AKeith\\JuliaProjectsLocal\\StrategyGamesLocal"
fnlocal = joinpath(dirlocal, "data\\vars_opt_rewardfunction_expected_7to15city_temp.jld2")
@load joinpath(dirlocal, "data\\vars_opt_rewardfunction_expected_7to15city.jld2") results # approx 5 min
ncities = [7, 8, 9, 10, 11, 12, 13, 14, 15] # number of cities (all other game params fixed)
n = length(ncities)
gvec = [ri[1] for ri in results]
gsvec = GameSet.(gvec)
rexps = [ri[4] for ri in results]
gosvec = [GameOptSet(gvec[i], gsvec[i], rexps[i]) for i in 1:n]
gsizes = [gamesize(gi)[4] for gi in gvec]
seqsizes = [size(ri[7][1])[1] * size(ri[7][2])[1] for ri in results]
statuses = [ri[5] for ri in results]
runtimes = [ri[end] for ri in results] ./ (60 * 60) # in hours

# result_size = pmap(x -> size_experiment(gvec[x], gsvec[x], gosvec[x]), 1:2)
df_size, t_exp = @timed size_experiment_vector(gvec, gsvec, gosvec)
println("Robust Experiment Total Time: $t_exp")
fn = "data\\robust_results_temp.csv"
CSV.write(joinpath(dir, fn), df_size)

# dfs = df_size[1:8, :]
dfs = df_size[1:8, :]
function plot_rel_robust(dir)
    plot(dfs[:size], signerr.(dfs[:u_r], dfs[:u_r]) .* 100,
        seriestype = :line, linecolor = :black, linestyle = :dash,
        legend = :bottomright, label = "Robust LP", ylim = (-6, 1),
        xlabel = "Problem Size (Number of Cities)", ylabel = "Relative Utility (Percent)")
    plot!(dfs[:size], signerr.(dfs[:u_dbr], dfs[:u_r]) .* 100,
        label = "Data-Biased CFR", linecolor = :lightblue)
    plot!(dfs[:size], signerr.(dfs[:u_dbr], dfs[:u_r]) .* 100,
        lab = "",
        seriestype = :scatter, markercolor = :lightblue, markerstrokealpha = 0.0)
    plot!(dfs[:size], signerr.(dfs[:u_ccfr], dfs[:u_r]) .* 100,
        label = "Constrained CFR", linecolor = :steelblue)
    plot!(dfs[:size], signerr.(dfs[:u_ccfr], dfs[:u_r]) .* 100,
        lab = "",
        seriestype = :scatter, markercolor = :steelblue, markerstrokealpha = 0.0)
    fn = "data\\plot_rel_robust_temp.pdf"
    savefig(joinpath(dir, fn))
end
plot_rel_robust(dir)

plot(dfr[:u_r] - dfr[:u_ccfr], label = "CCFR")
plot!(dfr[:u_r] - dfr[:u_dbr], label = "DBR")

plot(dfr[:size], dfr[:u_r], label = "Robust LP",
    ylim = (-1.0, 0.0), palette = :viridis)
plot!(dfr[:size], dfr[:u_dbr], label = "Data-Biased CFR")
plot!(dfr[:size], dfr[:u_ccfr], label = "Constrained CFR")

plot(dfr[:exp_neg_r] - dfr[:exp_neg_ccfr], label = "CCFR")
plot!(dfr[:exp_neg_r] - dfr[:exp_neg_dbr], label = "DBR")

plot(dfr[:exp_pos_r] - dfr[:exp_pos_ccfr], label = "CCFR")
plot!(dfr[:exp_pos_r] - dfr[:exp_pos_dbr], label = "DBR")

plot(dfr[:exp_pos_r] ./ dfr[:exp_neg_r], label = "R")
plot!(dfr[:exp_pos_dbr] ./ dfr[:exp_neg_dbr], label = "DBR")
plot!(dfr[:exp_pos_ccfr] ./ dfr[:exp_neg_ccfr], label = "CCFR")

##########################################
# old processing
conf = 0.99
sd = 0.02
opplevel = 1
nreps = 180
# (g, gs, gos), gametime = @timed build_robust_game()
(r1s, ts, statuses, σfix, noise), buildtime = @timed build_robust(g, gs, gos, conf, sd, opplevel)
@show (buildtime / 60)
# fn = "data\\vars_robust_builds_temp.pdf"
# @save joinpath(dir, fn) r1s, σfix, noise, buildtime

u1tuple, calctime = @timed calc_robust(gs, gos, r1s, σfix, noise, nreps)
@show (calctime / 60)
# fn = "data\\results_robust_calc_temp.pdf"
# @save joinpath(dir, fn) u1tuple, calctime

means = mean.(u1tuple)
stds = std.(u1tuple)
tstar = 1.98
confintervals = (means .- tstar .* stds ./ sqrt(nreps), means .+ tstar .* stds ./ sqrt(nreps))
minimum.(u1tuple)
u1_nes, u1_brs, u1_rs, u1_dbrs, u1_ccfrs = u1tuple
df_robust = DataFrame(NE = u1_nes, RLP = u1_rs, DBR = u1_dbrs, CCFR = u1_ccfrs)
fn = "data\\results_robust_data.csv"
using CSV
CSV.write(joinpath(dir, fn), df_robust)
plot_robust(u1tuple)
fn = "data\\plot_robust_hist_v2_temp.pdf"
# savefig(joinpath(dir, fn))

# _, br_ne, _ = lp_best_response(gos, r1_ne, fixedplayer = 1)
# _, br_r, _ = lp_best_response(gos, r1_r, fixedplayer = 1)
# _, br_dbr, _ = lp_best_response(gos, r1_dbr, fixedplayer = 1)
# _, br_ccfr, _ = lp_best_response(gos, r1_ccfr, fixedplayer = 1)
r1_ne, r1_br, r1_r, r1_dbr, r1_ccfr = r1s
br_results = pmap(x -> lp_best_response(gos, x, fixedplayer = 1), [r1_ne, r1_r, r1_dbr, r1_ccfr])
br_ne, br_r, br_dbr, br_ccfr = collect(x[2] for x in br_results)

# exploitability
rerr(br_r, br_ne)
rerr(br_dbr, br_ne)
rerr(br_ccfr, br_ne)

# exploitation
rerr(means[3], means[1]) # r to ne
rerr(means[4], means[1]) # dbr to ne
rerr(means[5], means[1]) # ccfr to ne


##########################################################
# Robust CFR DOE

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
@everywhere include(joinpath(dir, "src\\ccfrops_solve.jl"))
@everywhere include(joinpath(dir, "src\\optops_solve.jl"))
@everywhere include(joinpath(dir, "src\\ops_results.jl"))

# see scratch for screening design and RSM design
@everywhere function build_design(fn::String)
    factor_names = ["alpha", "beta", "gamma", "lambda_max", "lambda_scale"]
    col_types = [Int64, Int64, Int64, Int64, Int64]
    design = CSV.read(joinpath(dir, fn), types = col_types)
    rename!(design, old => new for (old, new) = zip(names(design), Symbol.(factor_names)))
    return design
end

@everywhere function build_robust_doe(simdata)
    g, gs, gos = build_robust_game()
    conf, sd, nreps = simdata
    opplevel = 1
    _, _, _, σfix = cfr(opplevel, g, gs)
    σfix .= [zeros(length(x)) for x in σfix]
    for i = 1:length(σfix)
        σfix[i][1] = 1.0
    end
    _, _, _, rp2lb, rp2ub, noise = build_opponent_strat(g, gs, gos, σfix, conf, sd)
    gamedata = (g, gs, gos, σfix, rp2lb, rp2ub, noise)
    return gamedata
end

@everywhere function strategy_doe(gamedata, simdata, params)
    g, gs, gos, σfix, rp2lb, rp2ub, noise = gamedata
    conf, sd, nreps = simdata
    a, b, c, lmax, lscale = params
    status_ne, u_ne, r1_ne, r2_ne, t_ne = lp_nash(gos, timelimit = 5 * 60)
    (u_ccfr, _, _, σ_ccfr, _, status_ccfr, iter_ccfr), t_cfr = @timed ccfr_full(g, gs, rp2lb, rp2ub,
            timelimit = 20, tol = 5e-9, α = a, β = b, γ = c, discounted = true,
            λmax = lmax, λscale = lscale)
    r1_ccfr = real_strat(σ_ccfr, gs, gos)[1]
    return (u_ne, u_ccfr), (r1_ne, r1_ccfr), (status_ccfr, iter_ccfr, t_cfr)
end

@everywhere function simrobust_doe(r1s, gamedata, simdata, ind_nz)
    r1_ne, r1_ccfr = r1s
    _, gs, gos, σfix, _, _, noise = gamedata
    _, _, nreps = simdata
    u1_ne = zeros(nreps)
    u1_ccfr = zeros(nreps)
    for i = 1:nreps
        σfix_rand = [normalize(clamp.(x .+ rand(noise, length(x)), 0.0, 1.0), 1) for x in σfix]
        r2_rand = real_strat(σfix_rand, gs, gos)[2]
        u1_ne[i] = sum(gos.reward[I] * r1_ne[I[1]] * r2_rand[I[2]] for I in ind_nz)
        u1_ccfr[i] = sum(gos.reward[I] * r1_ccfr[I[1]] * r2_rand[I[2]] for I in ind_nz)
    end
    return u1_ne, u1_ccfr
end

@everywhere function process_doe(u1s, r1s, gamedata, simdata)
    g, gs, gos, σfix, rp2lb, rp2ub, noise = gamedata
    conf, sd, nreps = simdata
    u1_ne, u1_ccfr = u1s
    r1_ne, r1_ccfr = r1s
    means = mean.(u1s)
    stds = std.(u1s)
    tstar = 1.98
    confintervals = (means .- tstar .* stds ./ sqrt(nreps), means .+ tstar .* stds ./ sqrt(nreps))
    df_robust = DataFrame(NE = u1_ne, CCFR = u1_ccfr)
    _, br_ne, _ = lp_best_response(gos, r1_ne, fixedplayer = 1)
    _, br_ccfr, _ = lp_best_response(gos, r1_ccfr, fixedplayer = 1)
    exp_neg = signerr(br_ccfr, br_ne) # exploitability
    exp_pos = signerr(means[2], means[1]) # exploitation
    return means[2], confintervals[2][1], confintervals[2][2], exp_neg, exp_pos
end

@everywhere function calc_run(df_row, gamedata, simdata, runnum)
    g, gs, gos, σfix, rp2lb, rp2ub = gamedata
    conf, sd, nreps = simdata
    a = df_row.alpha[1]
    b = df_row.beta[1]
    c = df_row.gamma[1]
    lmax = df_row.lambda_max[1]
    lscale = df_row.lambda_scale[1]
    params = (a, b, c, lmax, lscale)
    println("----------- Start Run $runnum -----------")
    uraw, r1s, (status_ccfr, iter_ccfr, t_ccfr) = strategy_doe(gamedata, simdata, params)
    ind_nz = findall(!iszero, gos.reward)
    println("----------- Sim Robust $runnum -----------")
    u1s = simrobust_doe(r1s, gamedata, simdata, ind_nz)
    umean, ulb, uub, exp_neg, exp_pos = process_doe(u1s, r1s, gamedata, simdata)
    println("------##### End Run $runnum #####------")
    return umean, ulb, uub, exp_neg, exp_pos, status_ccfr, iter_ccfr, t_ccfr
end

design = build_design("data\\design_robust.csv")
nruns = size(design, 1)
simdata = 0.99, 0.02, 90
gamedata = build_robust_doe(simdata)
results_robust_doe, t_robust_doe = @timed pmap(i -> calc_run(design[i:i, :], gamedata, simdata, i), 1:nruns)
fn = joinpath(dir, "data\\results_robust_doe_temp.jld2")
@save fn results_robust_doe, t_robust_doe, gamedata

dfr = hcat(design, DataFrame(utility = [ri[1] for ri in results_robust_doe],
                ulb = [ri[2] for ri in results_robust_doe],
                uub = [ri[3] for ri in results_robust_doe],
                exploitability = [ri[4] for ri in results_robust_doe],
                exploitation = [ri[5] for ri in results_robust_doe],
                status = [ri[6] for ri in results_robust_doe],
                iter = [ri[7] for ri in results_robust_doe],
                time = [ri[8] for ri in results_robust_doe]))
fn = "data\\results_robust_doe_temp.csv"
CSV.write(joinpath(dir, fn), dfr)
