# Chance Sampled Monte Carlo Counterfactual Regret Minimization
# Based on Johanson et al 2012 Algorithm 1
# http://modelai.gettysburg.edu/2013/cfr/cfr.pdf page 12 is also useful
# https://justinsermeno.com/posts/cfr/ is also useful for vanilla CFR

# h is represented by just a string, not an entire node structure
# this lets us avoid building the entire game tree and instead only have
# memory structures in the size of the info sets

# TODO: Dict is approx 10x faster than findfirst...restructure code?

# add packages
using Pkg
Pkg.activate("I:\\My Documents\\00 AFIT\\Research\\Julia Projects\\StrategyGames")
using Revise, Random, StatsBase

const N = 1:2 # set of players
const cards = ["J","Q","K"] # cards in the deck
const hands = ["JQ", "JK", "QJ", "QK", "KJ", "KQ"]
const acts = ["p","b"]
const ni_stage = [1, 3, 6, 3] # ni_stage[j] is num of infosets at stage j
const ni = sum(ni_stage) # total number of infosets
const na = vcat(6, fill(2, ni - 1)) # na[j] is number of actions at info set j
const AI = vcat([hands], fill(acts, ni - 1)) # AI[i][j] is the jth action at infoset i
const leafplays = ["pp", "pbp", "pbb", "bp", "bb"]
const leafnodes = reshape([string(i, j) for j in leafplays, i in hands], 30)
const leafvals = [-1.0, -1, -2, 1, -2,
            -1, -1, -2, 1, -2,
            1, -1, 2, 1, 2,
            -1, -1, -2, 1, -2,
            1, -1, 2, 1, 2,
            1, -1, 2, 1, 2]
const Uz = [leafvals, -leafvals]
const Iset = ["",
    "J", "Q", "K",
    "Jp", "Qp", "Kp", "Jb", "Qb", "Kb",
    "Jpb", "Qpb", "Kpb"]

actions(h::String) = length(h) < 1 ? hands : ["p", "b"] # node actions

function player(h::String)
    len = length(h)
    if len < 2
        return 3
    else
        return isodd(length(h) - 2) ? 2 : 1
    end
end

function infoset(h::String)
    depth = length(h)
    if depth < 2
        return 1 # null infoset
    elseif depth == 2
        return findfirst(cards .== h[1:1])[1] + ni_stage[1]
    elseif depth == 3
        bet = h[3:3] == "b" ? 3 : 0
        return findfirst(cards .== h[2:2])[1] + bet + sum(ni_stage[1:2])
    else
        return findfirst(cards .== h[1:1])[1] + sum(ni_stage[1:3])
    end
end

function terminal(h::String)
    depth = length(h)
    if depth > 4
        return true
    elseif depth < 4
        return false
    elseif h[3:4] == "pb"
        return false
    else
        return true
    end
end

function chance(h::String, a::String)
    1 / length(actions(h)) # assumes constant f_c(a|h)
end

function leafutility(h::String, i::Int64, πi::Float64, πo::Float64)
    ind = findfirst(leafnodes .== h)
    u = Uz[i][ind] * πo
    # println("h:$h, πo:$πo, u:$u ")
    return u
end

function matchregret(I::Int64, r::Vector{Vector{Float64}})
    σ = Vector{Float64}(undef, na[I])
    for a in 1:na[I]
        denom = sum(max(r[I][b], 0.0) for b in 1:na[I])
        if denom > 0.0
            σ[a] = max(r[I][a], 0.0) / denom
        else
            σ[a] = 1.0 / na[I]
        end
    end
    σ
end

function updateutility!(i::Int64, I::Int64, h::String, σ::Vector{Float64},
        u::Float64, πi::Float64, πo::Float64,
        r::Vector{Vector{Float64}}, s::Vector{Vector{Float64}})
    m = Vector{Float64}(undef, na[I])
    nextactions = actions(h)
    for a in 1:na[I]
        ha = string(h, nextactions[a])
        if player(h) == i
            πip = σ[a] * πi
            up = updatetree!(ha, i, πip, πo, r, s)
            m[a] = up
            u = u + σ[a] * up
        else
            πop = σ[a] * πo
            up = updatetree!(ha, i, πi, πop, r, s)
            u = u + up
        end
    end
    m, u
end

function updateregret!(i::Int64, I::Int64, h::String, σ::Vector{Float64},
        u::Float64, m::Vector{Float64}, πi::Float64,
        r::Vector{Vector{Float64}}, s::Vector{Vector{Float64}})
    if player(h) == i
        for a in 1:na[I]
            r[I][a] = r[I][a] + m[a] - u
            s[I][a] = s[I][a] + πi * σ[a]
        end
    end
end

function updatetree!(h::String, i::Int64, πi::Float64, πo::Float64,
    r::Vector{Vector{Float64}}, s::Vector{Vector{Float64}})
    # πo is short for \vec{π}_{-i}
    (terminal(h)) && (return leafutility(h, i, πi, πo))
    if player(h) == 3
        as = actions(h)
        chance_weights = pweights([chance(h, a) for a in as])
        a = as[sample(chance_weights)]
        ha = string(h, a)
        return updatetree!(ha, i, πi, πo, r, s)
    end
    I = infoset(h)
    u = 0.0
    σ = matchregret(I, r)
    m, u = updateutility!(i, I, h, σ, u, πi, πo, r, s)
    updateregret!(i, I, h, σ, u, m, πi, r, s)
    u
end

function cfr(T::Int64)
    r = [zeros(na[i]) for i in 1:ni] # regret
    s = [zeros(na[i]) for i in 1:ni] # cumulative strategies
    u = 0.0
    u1 = Vector{Float64}(undef, T)
    println("Starting $T iterations...")
    for t in 1:T
        for i in N
            u = updatetree!("", i, 1.0, 1.0, r, s) # utility
            (i == 1) && (u1[t] = u)
        end
        t % 1_000 == 0 && print("\rIteration $t complete. Utility: $u")
    end
    denoms = [sum(s[i][:]) for i in 1:ni]
    σ = [s[i][:] ./ denoms[i] for i in 1:ni]
    u1, r, s, σ #-u for player 1's utility since we end on player 2
end
