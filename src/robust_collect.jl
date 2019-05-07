# build reward, g, gs, and gos
using Pkg; Pkg.activate("I:\\My Documents\\00 AFIT\\Research\\Julia Projects\\StrategyGames")
using Revise
using LinearAlgebra, Random, Statistics, Distributions
using FileIO, JLD2
using Plots; gr()
dir = "I:\\My Documents\\00 AFIT\\Research\\Julia Projects\\StrategyGames"
include(joinpath(dir, "src\\ops_utility.jl"))
include(joinpath(dir, "src\\ops_build.jl"))
include(joinpath(dir, "src\\ops_methods.jl"))
include(joinpath(dir, "src\\cfrops_solve.jl"))
include(joinpath(dir, "src\\dbr_cfrops_solve.jl"))
include(joinpath(dir, "src\\optops_solve.jl"))
include(joinpath(dir, "src\\ops_results.jl"))

using Distributed
nworkers() == 1 && addprocs()
@everywhere using Pkg
@everywhere Pkg.activate("I:\\My Documents\\00 AFIT\\Research\\Julia Projects\\StrategyGames")
@everywhere using Revise
@everywhere dir = "I:\\My Documents\\00 AFIT\\Research\\Julia Projects\\StrategyGames"
@everywhere include(joinpath(dir, "src\\ops_utility.jl"))
@everywhere include(joinpath(dir, "src\\ops_build.jl"))
@everywhere include(joinpath(dir, "src\\ops_methods.jl"))

# build g, gs, gos
function build_robust_game()
    println("Building robust game...")
    fn = joinpath(dir, "data\\vars_opt_rewardfunction_expected_15city_flipped.jld2")
    fload = load(fn) # loads g and reward_exp
    g, (reward_exp, _) = fload["g"], fload["reward_exp"]
    gs = GameSet(g)
    gos = GameOptSet(g, gs, reward_exp)
    return g, gs, gos
end

# # build nominal fixed attacker strategy (NE)
# pdbr = vcat(fill(0.95, gs.ni_stage[3]), fill(0.0, gs.ni_stage[5]), fill(0.85, gs.ni_stage[6]))
# _, _, _, σNE = cfr(25_000, g, gs)

# # check that DBR matches LP BR
# T = 3000
# u_dbr, _, _, _ = cfr_dbr(T, g, gs, ones(length(pconf)), σfix)
# _, u_br, _ = lp_best_response(gos, real_strat(σfix, gs, gos)[2], fixedplayer = 2)
# isapprox(mean(u_dbr), u_br, atol = 0.05)

# build opponent strategy and uncertainty set
function build_opponent_strat(gs, gos, σfix, conf::Float64, sd::Float64)
    noise = Normal(0, sd)
    hw = quantile(noise, 1 - (1 - conf) / 2)
    σlb = [max.(σi .- hw, 0.0) for σi in σfix]
    σub = [min.(σi .+ hw, 1.0) for σi in σfix]
    mean(mean.(σub[i] - σlb[i] for i in 1:length(σlb)))
    rp2fix = real_strat(σfix, gs, gos)[2]
    rp2lb = real_strat(σlb, gs, gos)[2]
    rp2ub = real_strat(σub, gs, gos)[2]
    !all(rp2lb .<= rp2fix .<= rp2ub) && @warn "Negative realizaiton probabilities."
    return rp2fix, rp2lb, rp2ub, noise
end

function calc_robust_strat(g, gs, gos, rp2fix, rp2lb, rp2ub, σfix, pconf)
    println("Calculating NE LP strategy...")
    status_ne, u_ne, r1_ne, r2_ne, t_ne = lp_nash(gos, timelimit = 5 * 60)
    println("Calculating BR LP strategy...")
    status_br, u_br, r1_br = lp_best_response(gos, rp2fix, fixedplayer = 2)
    println("Calculating Robust BR LP strategy...")
    status_r, u_r, r1_r, r2_r, t_r = lp_robust_br(gos, rp2lb, rp2ub, timelimit = 5 * 60)
    println("Calculating Robust BR CFR strategy...")
    u_dbr, _, _, σ_dbr = cfr_dbr(5_000, g, gs, pconf, σfix)
    r1_dbr = real_strat(σ_dbr, gs, gos)[1]
    return (r1_ne, r1_br, r1_r, r1_dbr)
end

function build_robust(g, gs, gos, conf::Float64, sd::Float64, opplevel::Int)
    pconf = fill(conf, gs.ni)
    _, _, _, σfix = cfr(opplevel, g, gs)
    σfix .= [zeros(length(x)) for x in σfix]
    for i = 1:length(σfix)
        σfix[i][1] = 1.0
    end
    println("Building opponent strategies...")
    rp2fix, rp2lb, rp2ub, noise = build_opponent_strat(gs, gos, σfix, conf, sd)
    r1s = calc_robust_strat(g, gs, gos, rp2fix, rp2lb, rp2ub, σfix, pconf)
    return r1s, σfix, noise
end

@everywhere function simrobust(σfix_rand, gs, gos, r1s, ind_nz; reps = 10)
    r1_ne, r1_br, r1_r, r1_dbr = r1s
    r2_rand = real_strat(σfix_rand, gs, gos)[2]
    u1_ne = sum(gos.reward[I] * r1_ne[I[1]] * r2_rand[I[2]] for I in ind_nz)
    u1_br = sum(gos.reward[I] * r1_br[I[1]] * r2_rand[I[2]] for I in ind_nz)
    u1_r = sum(gos.reward[I] * r1_r[I[1]] * r2_rand[I[2]] for I in ind_nz)
    u1_dbr = sum(gos.reward[I] * r1_dbr[I[1]] * r2_rand[I[2]] for I in ind_nz)
    return u1_ne, u1_br, u1_r, u1_dbr
end

function calc_robust(gs, gos, r1s, σfix, noise, nreps)
    println("Building noisy opponent strategies...")
    σfix_rands = [[normalize(clamp.(x .+ rand(noise, length(x)), 0.0, 1.0), 1) for x in σfix] for _ in 1:nreps]
    !all(all(all(y .>= 0) for y in x) for x in σfix_rands) && @warn "Negative probabilities."
    println("Calculating utility of robust strategies...")
    ind_nz = findall(!iszero, gos.reward)
    u1s = pmap(x -> simrobust(x, gs, gos, r1s, ind_nz, reps = nreps), σfix_rands)
    u1_nes = [x[1] for x in u1s]
    u1_brs = [x[2] for x in u1s]
    u1_rs = [x[3] for x in u1s]
    u1_dbrs = [x[4] for x in u1s]
    return (u1_nes, u1_brs, u1_rs, u1_dbrs)
end

function plot_robust(u1tuple)
    u1_nes, u1_brs, u1_rs, u1_dbrs = u1tuple
    fig = histogram(u1_nes, bins=:auto, color=:mediumseagreen, alpha=0.2,
                xlabel = "Defender Utility", ylabel = "Count",
                label = "Nash Equilibrium Strategy",
                xlims = (-0.420, -0.390))
    # histogram!(fig, u1_brs, bins=:auto, color=:tomato, alpha=0.2,
    #             label = "Best Response Strategy")
    histogram!(fig, u1_rs, bins=:auto, color=:tomato, alpha=0.95,
                label = "Robust LP Strategy")
    histogram!(fig, u1_dbrs, bins=:auto, color=:steelblue, alpha=0.4,
                label = "Robust CFR Strategy")
    fig
end
plot_robust(u1tuple)

conf = 0.99
sd = 0.02
opplevel = 1
nreps = 180
# (g, gs, gos), gametime = @timed build_robust_game()
(r1s, σfix, noise), buildtime = @timed build_robust(g, gs, gos, conf, sd, opplevel)
@show (buildtime / 60)
u1tuple, calctime = @timed calc_robust(gs, gos, r1s, σfix, noise, nreps)
@show (calctime / 60)
means = mean.(u1tuple)
stds = std.(u1tuple)
tstar = 1.98
confintervals = (means .- tstar .* stds ./ sqrt(nreps), means .+ tstar .* stds ./ sqrt(nreps))
minimum.(u1tuple)
plot_robust(u1tuple)
fn = "data\\plot_robust_hist_temp.pdf"
# savefig(joinpath(dir, fn))

_, br_ne, _ = lp_best_response(gos, r1_ne, fixedplayer = 1)
_, br_r, _ = lp_best_response(gos, r1_r, fixedplayer = 1)
_, br_dbr, _ = lp_best_response(gos, r1_dbr, fixedplayer = 1)

# exploitability
rerr(br_r, br_ne)
rerr(br_dbr, br_ne)

# exploitation
rerr(means[3], means[1]) # r to ne
rerr(means[4], means[1]) # dbr to ne
