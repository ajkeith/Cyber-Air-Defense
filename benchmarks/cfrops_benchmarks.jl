using Pkg; Pkg.activate(pwd())
using Revise
using Test, LinearAlgebra, Random, StaticArrays
dir = "I:\\My Documents\\00 AFIT\\Research\\Julia Projects\\StrategyGames\\src"
include(joinpath(dir, "ops_build.jl"))
include(joinpath(dir, "cfrops_solve.jl"))

Random.seed!(579843)
g = AirDefenseGame()
gs = GameSet(g)
ni, ni_stage, na_stage, players = gs.ni, gs.ni_stage, gs.na_stage, gs.players
r = [zeros(numactions(i, ni_stage, na_stage)) for i in 1:ni] # regret
s = [zeros(numactions(i, ni_stage, na_stage)) for i in 1:ni] # cumulative strategies
u = 0.0
T = 100
u1 = Vector{Float64}(undef, T)


# Benchmark to find slow functions
hs = SVector(11, 14, 5, 4, 6, 110)
h = [11, 14, 5, 4, 6]
h2 = [11, 14, 5, 4, 6]
@btime vcat([1,2,3], 4) # 2.4 μs # replace with constant size h vectors and keep track of depth? (drops to ~50 ns)
@btime hs2 = setindex(hs, 10, 4) # 30 ns
@btime leafutility(hs, 1, 0.9, 0.7, $g, $gs) # 800 ns
@btime terminal($h) # 2 ns
@btime getplayer($h) # 2 ns
@btime actions($h, $gs.A) # 2 ns
@btime chance([1, 1, 3], 4, $g, $gs) # 500 ns
@btime sample(pweights([0.1, 0.2, 0.3, 0.4])) # 50 ns
@btime infoset($h, $g, $gs) # 200 ns
@btime matchregret(1, $r, $gs) # 220 ns
@btime Vector{Float64}(undef, 100) # 40 ns
@btime U = gs.U # 35 ns

# Benchmark full CFR algorithm
# with max performance lefutility (rand) we have 26 ms => 28 sec per 1k iterations on default AirDefenseGame
# with leafutility = leafutilty_sub, we have 0.08 sec => 65 sec per 1k iterations on default AirDefenseGame
# with static h node and leafutility_sub, we have 11 ms => 11 sec per 1k iterations on default AirDefenseGame
Random.seed!(579843)
g = AirDefenseGame()
gs = GameSet(g)
T = 10_000
cfr(1, g, gs)
@btime cfr(1, g, gs)
@time u, r, s, σ = cfr(T, g, gs)
sum(u) == -83.9854058521706 # check correctness

# push h: checksum passes, 38 ms per iter, 4.4 sec per 100 iter
# static h: checksum passes exactly, 11 ms per iter, 1.1 sec per 100 iter,

# Benchmark fixes to leafutility
function leafutility_rand(h::Array{Int64}, player::Int64, πi::Float64, πo::Float64, gs::GameSet)
    # U, z = gs.U, gs.z
    # ind = findfirst(x -> x == h, z)
    # u = player == 1 ? U[ind] * πo : -1.0 * U[ind] * πo
    val = rand()
    u = player == 1 ? -val : val
    return u
end

function leafutility_lookup(h::Array{Int64}, player::Int64, πi::Float64, πo::Float64, gs::GameSet)
    U, z = gs.U, gs.z
    ind = findfirst(x -> x == h, z)
    u = player == 1 ? U[ind] * πo : -1.0 * U[ind] * πo
    return u
end

function leafutility_rawcalc(h::Array{Int64}, player::Int64, πi::Float64, πo::Float64, g::AirDefenseGame, gs::GameSet)
    c1, c2, ac, c4, dc, ap = h
    attacked_cities = @view gs.A[6][h[6]][:]
    defended_iads = getindex(g.iads, getindex(gs.A[1][h[1]], gs.A[5][h[5]]))
    attacked_iads = getindex(g.iads, getindex(gs.A[2][h[2]], gs.A[3][h[3]]))
    leaf_value = 0.0
    for city in attacked_cities
        q = 1.0
        for defender in g.iads
            if gs.C[defender, city] == 0
                q *= (1 - 0.0)
            elseif defender ∈ attacked_iads && defender ∉ defended_iads
                q *= (1 - 0.0)
            elseif defender ∉ attacked_iads
                q *= (1 - g.pp)
            elseif defender ∈ attacked_iads && defender ∈ defended_iads
                q *= (1 - g.pp * (g.pdc * g.pac)) # if a defender is cyber attacked, it has pdc * pac of being cyver-defended
            else
                @warn "unexpected defense layout"
            end
        end
        leaf_value += -1.0 * g.v[city] * q # defender's expected utility for a city
    end
    u = player == 1 ? leaf_value * πo : -1.0 * leaf_value * πo
    return u
end

# this is the version used in the actual leafutility function
function leafutility_sub(h::Array{Int64}, player::Int64, πi::Float64, πo::Float64, g::AirDefenseGame, gs::GameSet)
    c1, c2, ac, c4, dc, ap = h
    attacked_cities = @view gs.A[6][h[6]][:]
    defended_iads = getindex(g.iads, getindex(gs.A[1][h[1]], gs.A[5][h[5]]))
    attacked_iads = getindex(g.iads, getindex(gs.A[2][h[2]], gs.A[3][h[3]]))
    leaf_value = utility_defender(g, gs, attacked_cities, defended_iads, attacked_iads)
    u = player == 1 ? leaf_value * πo : -1.0 * leaf_value * πo
    return u
end

@btime leafutility_rand([11, 14, 5, 4, 6, 110], 1, 0.9, 0.7, $gs) # 40 ns
@btime leafutility_lookup([11, 14, 5, 4, 6, 9], 1, 0.9, 0.7, $gs) # 66 ms
@btime leafutility_rawcalc([11, 14, 5, 4, 6, 9], 1, 0.9, 0.7, $g, $gs) # 820 ns
@btime leafutility_sub([11, 14, 5, 4, 6, 9], 1, 0.9, 0.7, $g, $gs) # 850 ns
