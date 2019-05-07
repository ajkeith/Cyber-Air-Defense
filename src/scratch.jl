# ] activate "I:\My Documents\00 AFIT\Research\Julia Projects\StrategyGames"
using Pkg; Pkg.activate("I:\\My Documents\\00 AFIT\\Research\\Julia Projects\\StrategyGames")


############################
#
# Scratch work for results collection
#
###########################



## DOE design with some set factors and some constant factors

@everywhere function build_design(fn::String)
    # screening design
    # factor_names = ["radius",
    #                 "ncity", "ndp", "nap",
    #                 "ndpdc_p", "ndpac_p", "ndc_p", "nac_p",
    #                 "pp", "psc", "pdc", "pac",
    #                 "alpha", "beta", "gamma"]
    # col_types = [Float64,
    #                 Int64, Int64, Int64,
    #                 Float64, Float64, Float64, Float64,
    #                 Float64, Float64, Float64, Float64,
    #                 Float64, Float64, Float64]
    # factor_limits = [(0.2, 0.4),
    #                 (9, 15), (4, 8), (2, 6),
    #                 (0.2, 1.0), (0.2, 1.0), (0.2, 0.8), (0.2, 0.8),
    #                 (0.6, 0.95), (0.6, 0.95), (0.6, 0.95), (0.6, 0.95),
    #                 (1.0, 8.0), (-8.0, 8.0), (0.0, 8.0)]
    factor_names = ["size",
                    "pp", "psc", "pdc",
                    "alpha", "beta", "gamma"]
    col_types = [Int64,
                    Float64, Float64, Float64,
                    Float64, Float64, Float64]
    factor_limits = [(9, 15),
                    (0.55, 0.95), (0.55, 0.95), (0.55, 0.95),
                    (0.5, 4.0), (-4.0, 4.0), (0.0, 4.0)]
    const_names = ["radius", "pac",
                    "ncity", "ndp", "nap",
                    "ndpdc", "ndpac", "ndc", "nac"]
    const_types = [Float64, Float64,
                    Int64, Int64, Int64,
                    Int64, Int64, Int64, Int64]
    const_limits = [(0.3, 0.3), (0.0, 0.0),
                    (9, 15), (4,8), (2,6),
                    (4,8), (4,8), (2,6), (2,6)]
    design = CSV.read(joinpath(dir, fn), types = col_types)
    rename!(design, old => new for (old, new) = zip(names(design), Symbol.(factor_names)))
    l, w = size(design)
    for j in 1:w, i in 1:l
        if design[i, j] == -1
            design[i, j] = factor_limits[j][1]
        elseif design[i, j] == 1
            design[i, j] = factor_limits[j][2]
        elseif design[i, j] == 0
            design[i, j] = mean(factor_limits[j])
        else
            @warn "Unexpected factor design value at $i, $j."
        end
    end
    for j in 1:length(const_names)
        design[Symbol(const_names[j])] = Vector{const_types[j]}(undef, l)
    end
    for j in 1:length(const_names), i in 1:l
        if design.size[i] == factor_limits[1][1]
            design[i, j + w] = const_limits[j][1]
        elseif design.size[i] == factor_limits[1][2]
            design[i, j + w] = const_limits[j][2]
        elseif design.size[i] == mean(factor_limits[1])
            design[i, j + w] = mean(const_limits[j])
        else
            @warn "Unexpected const design value at $i, $j."
            return design
        end
    end
    return design
end

##

# SharedArray version of DOE results collection
# runs each run twice when I only want to run each run once...not sure what's
# going on

nruns = size(design, 1)
u_doe = SharedArray{Float64}(nruns)
converge_doe = SharedArray{Bool}(nruns)
iter_doe = SharedArray{Int64}(nruns)
t_doe = SharedArray{Float64}(nruns)

@distributed for i in 1:3
    println("Start $i")
    ui, convergei, iteri, ti = calc_run(design[i,:], gvec[i], gsvec[i], i)
    prtinln("Complete $i")
    @show ui, convergei, iteri, ti
    u_doe[i] = ui
    converge_doe[i] = convergei
    iter_doe[i] = iteri
    t_doe[i] = ti
end

##


gvec = [AirDefenseGame(i, 6, 3, 4, 4, 2, 2) for i in 7:15]
seqsizes = Vector{Int64}(undef, 9)
for (i, gi) in enumerate(gvec)
    A, An, na_stage = build_action(gi)
    ni, ni1, ni2, ni_stage = build_info(gi)
    ns1, ns2, ns1_stage, ns2_stage = build_nseq(gi, na_stage, ni_stage)
    seqsizes[i] = ns1 * ns2
end

## CFR means are not close to LP NE means.
# investigate n = 7
Random.seed!(579843)
ncity = 10
g = rexps["results"][ncity - 6][1]
gs = GameSet(g)
(u, r, s, σ, σs, converged), runtime = @timed dcfr_full(g, gs, iterlimit = 5_000, tol = 5e-5, α = 1.5, β = 0.0, γ = 2.0, discounted = false)
mean(u)
status, u_br, r_br = lp_best_response(gos, real_strat(σ, gs, gos)[1], fixedplayer = 1)


Random.seed!(579843)
ncity = 10
g = rexps["results"][ncity - 6][1]
gs = GameSet(g)
u, r, s, σ = cfr(5000, g, gs)
mean(u)

Random.seed!(579843)
ncity = 10
g = AirDefenseGame(ncity, 6, 3, 4, 4, 2, 2)
gs = GameSet(g)
gos = GameOptSet(g, gs, rexps["results"][ncity - 6][4])
status_ne, u_ne, r_ne, t_ne = lp_nash(gos)
u_ne

ind1 = 43
ind2 = 62491
h = [1, 6, 5, 1, 6, 120] # h associated with seq1, seq2
indu = findfirst(x -> x == h, z)
@test U[indu] == reward[ind1, ind2]


g = AirDefenseGame()
gs = GameSet(g)
A, An, na_stage = build_action(g)
ni, ni1, ni2, ni_stage = build_info(g)
ns1, ns2, ns1_stage, ns2_stage = build_seq(g, na_stage, ni_stage)
Σ1, Σ2, seq1defended_iads, seq2attacked_iads, seq2defended_iads, seq2attacked_cities = build_Σset(g, A, ns1, ns2)
Iset, I1set, I2set, seqI1, seqI2, nextseqI1, nextseqI2 = build_Iset(g, A, na_stage, ns1_stage, ns2_stage, ni1, ni2, ns1, ns2)
g.iads
i = 289
I1set[i]
Σ1[seqI1[i]]
nextseqI1[i]
i = 16
I2set[i]
Σ2[seqI2[i]]
nextseqI2[i]
ub = 100
ftab(ub) = hcat([I2set[i] for i = (ub - 25):ub], [Σ2[seqI2[i]] for i = (ub - 25):ub])

g = AirDefenseGame()
gs = GameSet(g)
A, An, na_stage = build_action(g)
ni, ni1, ni2, ni_stage = build_info(g)
ns1, ns2, ns1_stage, ns2_stage = build_seq(g, na_stage, ni_stage)
Σ1, Σ2, seqhist1, seqhist2 = build_Σset(g, A, ns1, ns2)
reward = build_utility_lp(g, gs, ns1, ns2, seq1defended_iads, seq2attacked_iads, seq2defended_iads, seq2attacked_cities)

using JuMP, Clp, SparseArrays
m = Model(solver = ClpSolver())
X = sparse([1,2,3],[1,2,3], [0,2,0])
t = [1, 10, 100]
t' * X
@variable(m, r[1:3] >= 0)
@objective(m, Min, sum(r' * X))


using CSV, DataFrames
dir = "I:\\My Documents\\00 AFIT\\Research\\Julia Projects\\StrategyGames"
# fn = joinpath(dir, "data\\utility_fixed_default_100000.csv")
# CSV.write(fn, DataFrame(utility = u))

using Pkg; Pkg.activate("I:\\My Documents\\00 AFIT\\Research\\Julia Projects\\StrategyGames")
using FileIO, JLD2
dir = "I:\\My Documents\\00 AFIT\\Research\\Julia Projects\\StrategyGames"
fn = joinpath(dir, "data\\vars_fixed_default_100000.jld2")
# # @ save fn u r s σ
# # @ load fn u r s σ

function rerr_approx(u)
    lb = round(Int, 0.9 * length(u))
    u_approx = mean(u[lb:end])
    um = cumsum(u) ./ collect(1:T)
    u_rerr_approx = abs.(um .- u_approx) ./ abs.(u_approx)
    u_rerr_approx
end


function rerr_approx(u)
    lb = round(Int, 0.9 * length(u))
    u_approx = mean(u[lb:end])
    um = cumsum(u) ./ collect(1:T)
    u_rerr_approx = abs.(um .- u_approx) ./ abs.(u_approx)
    return u_rerr_approx
end



############################
#
# Scratch work for comparison between history-form utility and sequence-form utility
#
###########################

s1_ind = 43
s2_ind = 62491
reward[s1_ind, s2_ind]
Σ[1][s1_ind]
Σ[2][s2_ind]
c1 = findfirst(x -> x == [1,2,3,4], A[1])
c2 = findfirst(x -> x == [1,2,5,6], A[2])
ac = findfirst(x -> x == [2,4], A[3])
c4 = 1
dc = findfirst(x -> x == [3,4], A[5])
ap = findfirst(x -> x == [8,9,10], A[6])
h = [c1, c2, ac, 1, dc, ap]
uind = findfirst(x -> x == h, z)
U[uind] == reward[s1_ind, s2_ind]

get_attcity(h, g, gs) = gs.A[6][h[6]]
get_defiads(h, g, gs) = getindex(g.iads, getindex(gs.A[1][h[1]], gs.A[5][h[5]]))
get_attiads(h, g, gs) = getindex(g.iads, getindex(gs.A[2][h[2]], gs.A[3][h[3]]))

# check that sequnce-form actions match history-form actions
seqactions[3][s2_ind, :] == get_attcity(h, g, gs)# attacked cities
seqactions[1][s1_ind, :] == get_defiads(h, g, gs)# defended iads
seqactions[2][s2_ind, :] == get_attiads(h, g, gs) # attacked iads
seqactions[3][s2_ind, :] # attacked cities
seqactions[1][s1_ind, :] # defended iads
seqactions[2][s2_ind, :]  # attacked iads
attcity = get_attcity(h, g, gs)
defiads = get_defiads(h, g, gs)
attiads = get_attiads(h, g, gs)
utility_defender(g, coverage(g), attcity, defiads, attiads)

function ft(seq1defended_iads, seq2attacked_iads, seq2attacked_cities)
    return seq1defended_iads, seq2attacked_iads, seq2attacked_cities
end

seq1defended_iads, seq2attacked_iads, seq2attacked_cities = ft(seqactions...)
utility_defender(g, coverage(g), seq2attacked_cities[s2_ind, :], seq1defended_iads[s1_ind, :], seq2attacked_iads[s2_ind, :])
utility_defender(g, coverage(g), seq2attacked_cities[s2_ind], seq1defended_iads[s1_ind], seq2attacked_iads[s2_ind])

reward, reward_timing, bytes, gct, memallocs = @timed reward_seq(g, A, seqn..., seqactions...)
using FileIO, JLD2
dir = "I:\\My Documents\\00 AFIT\\Research\\Julia Projects\\StrategyGames"
# fn = joinpath(dir, "data\\temp.jld2")
# @save fn g reward reward_timing

# Lookup is much slower than inplace calculation
# function leafutility_lookup(h::Array{Int64}, player::Int64, πi::Float64, πo::Float64, gs::GameSet)
#     U, z = gs.U, gs.z
#     ind = findfirst(x -> x == h, z)
#     u = player == 1 ? U[ind] * πo : -1.0 * U[ind] * πo
#     return u
# end

# Reduce code repitition by doing the same logic but with utility_defender
# function leafutility_rawcalc(h::Array{Int64}, player::Int64, πi::Float64, πo::Float64, g::AirDefenseGame, gs::GameSet)
#     c1, c2, ac, c4, dc, ap = h
#     attacked_cities = @view gs.A[6][h[6]][:]
#     defended_iads = getindex(g.iads, getindex(gs.A[1][h[1]], gs.A[5][h[5]]))
#     attacked_iads = getindex(g.iads, getindex(gs.A[2][h[2]], gs.A[3][h[3]]))
#     leaf_value = 0.0
#     for city in attacked_cities
#         q = 1.0
#         for defender in g.iads
#             if gs.C[defender, city] == 0
#                 q *= (1 - 0.0)
#             elseif defender ∈ attacked_iads && defender ∉ defended_iads
#                 q *= (1 - 0.0)
#             elseif defender ∉ attacked_iads
#                 q *= (1 - g.pp)
#             elseif defender ∈ attacked_iads && defender ∈ defended_iads
#                 q *= (1 - g.pp * (g.pdc * g.pac)) # if a defender is cyber attacked, it has pdc * pac of being cyver-defended
#             else
#                 @warn "Unexpected defense layout."
#             end
#         end
#         leaf_value += -1.0 * g.v[city] * q # defender's expected utility for a city
#     end
#     u = player == 1 ? leaf_value * πo : -1.0 * leaf_value * πo
#     return u
# end

############################################





using Distributed
addprocs(20); nprocs()
@everywhere using Pkg
@everywhere Pkg.activate("I:\\My Documents\\00 AFIT\\Research\\Julia Projects\\StrategyGames")
@everywhere using JuMP, Clp
using BenchmarkTools

@everywhere function runm(ub::Float64)
    m = Model(solver = ClpSolver())
    @variable(m, 0 <= x <= 2 )
    @variable(m, 0 <= y <= 30 )
    @objective(m, Max, 5x + 3*y )
    @constraint(m, 1x + 5y <= ub )
    # print(m)
    status = solve(m)
    # println("Objective value: ", getobjectivevalue(m))
    # println("x = ", getvalue(x))
    # println("y = ", getvalue(y))
    return getobjectivevalue(m)
end

runm(3.2)

N = 1_000
sumobj = @distributed (+) for i = 1:N
    runm(3.0)
end

meanobj = sumobj / N

@everywhere function pc()
    nheads = @distributed (+) for i = 1:200_000_000_000
        Int(rand(Bool))
    end
    nheads
end

@everywhere f(x) = Int(rand(Bool))

@btime nheads = @distributed (+) for i = 1:200_000_000
    f(1)
end

a = zeros(10)
nheads = Threads.@threads for i = 1:10
    a[i] = i
end

#########################################

using Distributed
addprocs(3)
@everywhere using Pkg
@everywhere Pkg.activate("I:\\My Documents\\00 AFIT\\Research\\Julia Projects\\StrategyGames")  # required
@everywhere using LightGraphs
@everywhere using JuMP
@everywhere using Clp


@everywhere function runm(ub::Float64)
    m = Model(solver = ClpSolver())
    @variable(m, 0 <= x <= 2 )
    @variable(m, 0 <= y <= 30 )

    @objective(m, Max, 5x + 3*y )
    @constraint(m, 1x + 5y <= ub )

    # print(m)

    status = solve(m)

    # println("Objective value: ", getobjectivevalue(m))
    # println("x = ", getvalue(x))
    # println("y = ", getvalue(y))
    return getobjectivevalue(m)
end


using BenchmarkTools
nprocs()
N = 1_000_000

# single processor
# N = 1_000, 160 ms
@btime [runm(3.0) for _ in 1:N]

# 21 processors
# N = 1_000, 7.6 ms
@btime sumobj = @distributed (+) for i = 1:N
    runm(3.0)
end

meanobj = sumobj / N
runm(3.0)

using Plots; gr()

plot(rand(5))

using Gadfly, RDatasets

Gadfly.push_theme(:default)
iris = dataset("datasets", "iris")
Gadfly.plot(iris, x=:SepalLength, y=:SepalWidth)


######## Game Size ########

using Pkg; Pkg.activate("I:\\My Documents\\00 AFIT\\Research\\Julia Projects\\StrategyGames")
using Revise, BenchmarkTools
using Printf


function psize_full(ncity, ndp, nap, ndc, nac)
    npatch = 2
    nexploit = 2
    nn = binomial(ncity, ndp) * npatch * nexploit * binomial(ndp, nac) *
        2^nac * binomial(ndp, ndc) * 2^nac * binomial(ncity, nap)
    @sprintf("%.2E", nn)
end

function psize_rep(ncity, ndp, nap, ndc, nac)
    npatch = 2
    nexploit = 2
    nn = 1 * npatch * nexploit * binomial(ndp, nac) *
        2^nac * binomial(ndp, ndc) * 2^nac * 1
    @sprintf("%.2E", nn)
end

function gsize(ncity, ndp, nap, ndc, nac)
    nsig1 = binomial(ncity, ndp) * binomial(ndp, ndc)
    nsig2 = binomial(ndp, nac) * binomial(ncity, nap)
    ng = nsig1 * nsig2
    @sprintf("%.2E", ng)
end

function nI(ncity, ndp, nap, ndc, nac)
    nI11 = 1
    nI13 = nI11 * binomial(ncity, ndp) * 2 * 2^nac
    nI1 = @sprintf("%.2E", nI11 + nI13)
    nI22 = binomial(ncity, ndp) * 2
    nI24 = nI22 * binomial(ndp, nac) * binomial(ndp, ndc)
    nI2 = @sprintf("%.2E", nI22 + nI24)
    (nI1, nI2)
end

function nΣ(ncity, ndp, nap, ndc, nac)
    nΣ11 = binomial(ncity, ndp)
    nΣ13 = binomial(ndp, ndc)
    nΣ1 = @sprintf("%.2E", nΣ11 * nΣ13)
    nΣ22 = binomial(ndp, nac)
    nΣ24 = binomial(ncity, nap)
    nΣ2 = @sprintf("%.2E", nΣ22 * nΣ24)
    (nΣ1, nΣ2)
end

# small size
ncity = 6
ndp = 4
nap = 3
ndc = 2
nac = 2
psize_full(ncity, ndp, nap, ndc, nac) # 7E+05
psize_rep(ncity, ndp, nap, ndc, nac) # 2E+03
gsize(ncity, ndp, nap, ndc, nac)
nI(ncity, ndp, nap, ndc, nac)
nΣ(ncity, ndp, nap, ndc, nac)

# large size
ncity = 25
ndp = 3
nap = 2
ndc = 2
nac = 2
psize_full(ncity, ndp, nap, ndc, nac)
psize_rep(ncity, ndp, nap, ndc, nac)
gsize(ncity, ndp, nap, ndc, nac)
nI(ncity, ndp, nap, ndc, nac)
nΣ(ncity, ndp, nap, ndc, nac)

ncity = 35
nnd = binomial(ncity, 1) + binomial(ncity, 2) + binomial(ncity, 3)
nnt = nnd * binomial(ncity, 5)
@sprintf("%.2E", nnt)


# Calculate coverage matrix
using Random, Distances
n = 55
r = 0.3
x = rand(2, n)
dist = pairwise(Euclidean(), x)
A = [dist[i,j] < r ? 1 : 0 for i = 1:n, j = 1:n]

using Plots; gr()
labels = string.(A[2,:])
plot(x[1,:], x[2,:], seriestype = :scatter,
    series_annotations = labels)


# Distributed scratch
using Distributed
addprocs(10)

@everywhere using LinearAlgebra

@everywhere function fsvd(k::Int64)
    svd(rand(k,k)).U |> sum
end

function fast()
    responses = Vector{Float64}(undef, nworkers())
    @sync begin
        for (idx, pid) in enumerate(workers())
            @async responses[idx] = remotecall_fetch(fsvd, pid, 1_000)
        end
    end
end

function slow()
    responses = Vector{Float64}(undef, nworkers())
    for (idx, pid) in enumerate(workers())
        responses[idx] = fsvd(1_000)
    end
end

@btime fast()
@btime slow()

########## Large file manipulation

# JuliaDB Example from https://juliacomputing.com/blog/2019/02/27/juliadb.html
using Pkg
Pkg.activate("I:\\My Documents\\00 AFIT\\Research\\Julia Projects\\StrategyGames")
using Distributed
addprocs(12)
@everywhere using Pkg
@everywhere Pkg.activate("I:\\My Documents\\00 AFIT\\Research\\Julia Projects\\StrategyGames")
@everywhere using JuliaDB, OnlineStats

cd("C:\\Users\\AKeith\\JuliaProjectsLocal\\temp")
datadir = "data\\Stocks"

files = glob("*.txt", datadir)
t = loadtable(files, filenamecol = :Stocks)
groupreduce(Mean(), t, :Stock, select = :Volume)

###############################

using BenchmarkTools
const arr = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
const d = Dict(arr[i] => i for i in 1:length(arr))

function ff(s::String)
    findfirst(arr .== s)
end

function fdict(s::String)
    d[s]
end

@btime ff("a")
@btime fdict("a")
@btime ff("i")
@btime fdict("i")
