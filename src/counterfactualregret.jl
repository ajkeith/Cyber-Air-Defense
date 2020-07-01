# Chance Sampled Monte Carlo Counterfactual Regret Minimization
# Johanson et al 2012 Algorithm 1
# http://modelai.gettysburg.edu/2013/cfr/cfr.pdf page 12 is also useful

# set up distributed processing

# add packages
using Pkg
Pkg.activate("I:\\My Documents\\00 AFIT\\Research\\Julia Projects\\StrategyGames")
using Distributed
@everywhere using Random, StatsBase

# include other files
include("helper.jl")

# Define variables
N = 1:2 # number of players
I = 1:10 # information sets
A = [[1,2,3], [1,2], [1,2,3,4], [1,3,5], [1,2,6], [3,4,5], [4,6], [6], [4,5,6], [6]] # actions
actor = [1,2,1,2,1,2,1,1,1,1] # decision maker at each info set
ni = size(I,1) # number of info sets
na = sum.(A) # number of actions at each info set
regret = [zeros(na[i]) for i in 1:ni] # player 1 regret
strategy = [zeros(na[i]) for i in 1:ni] # cumulative strategies
σ1 = [fill(1 / na[i], na[i]) for i in 1:ni] # player 1 strategy
σ2 = [fill(1 / na[i], na[i]) for i in 1:ni] # player 2 strategy
σ = (σ1, σ2)

# sample tree
leaf1 = Node(1, [])
leaf2 = Node(3, [])
leaf3 = Node(4, [])
n = Node(2, [leaf2, leaf3])
head = Node(0, [leaf1, n])
nh = nnodes(head)
H, Z = gametree()

# constants
H, Z, C, P # all history nodes, terminal nodes, chance nodes, public nodes
S = [S[i] for i in N] # chance nodes only observable to player i
O = [O[i] for i in N] # chance nodes only observable to opponent of player i
# note: C = S1 ∪ O1 ∪ P and also C = S2 ∪ O2 ∪ P
Πi = [π1, π2] # contribution of player i to prob(h)
Πo = [πo1, πo2] # contribution of all other players than i to prob(h)
I1 = Vector{Vector{Node}}(undef, ni)
I2 = Vector{Vector{Node}}(undef, ni)
Iset = [I1, I2] # info sets
AI # dict from info set to action
Uz = [Uz1, Uz2] # utility of each leaf node for each player

function gametree
    # build h nodes into tree
    # record Z while doing this
    H, Z
end

function chance(h::Node, a::Int)
    1 / length(children(h)) # need to make this actually match public chance
end

function leafutility(h::Node, i::Int)
    Uz[i][idnode(h)]
end

function sampleoutcome(h::Node, πi::Float64, πo::Float64)
    if h ∈ C
        nextnodes = children(h)
        chance_weights = [chance(h, a) for a in actions]
        ha = sample(nextnodes, chance_weights)
        return updatetree!(ha, i, πi, πo)
    end
end

function infoset(h::Node)
    idinfo(h), Iset[idactor(h)][idinfo(h)]
end

function matchregret(I::Vector{Node})
    ind = findfirst(isequal(I), Iset)
    na = length(AI[ind])
    σ = Vector{Float64}(undef, na)
    for a in AI[ind]
        denom = sum(b * max(r[ind][b], 0) for b in AI[ind])
        if denom > 0
            σ[a] = max(r[ind][a], 0) / denom
        else
            σ[a] = 1 / length(AI[ind])
        end
    end
    σ
end

function updateutility!(i::Int, ind::Int, h::Node, σ::Vector{Float64},
        u::Float64, πi::Float64, πo::Float64)
    na = length(AI[ind])
    m = Vector{Float64}(undef, na)
    for a in AI[ind]
        ind_ha = findfirst(x -> x == a, nextactions(h))
        ha = children(h)[ind_ha]
        if idactor(h) == i
            πip = σ[a] * πi
            up = updatetree(ha, i, πip, πo)
            m[a] = up
            u = u + σ[a] * up
        else
            πop = σ[a] * πo
            up = updatetree(ha, i, πi, πop)
            u = u + up
        end
    end
    m
end

function updateregret!(i::Int, ind::Int, σ::Vector{Float64},
        u::Float64, m::Vector{Float64}, πi::Float64,
        r::Vector{Vector{Float64}}, s::Vector{Vector{Float64}})
    if idactor(h) == i
        for a in AI[ind]
            r[ind][a] = r[ind][a] + m[a] - u
            s[ind][a] = s[ind][a] + πi * σ[a]
        end
    end
end

function updatetree(h::Node, i::Int, πi::Float64, πo::Float64)
    # πo is short for \vec{π}_{-i}
    h ∈ Z && return leafutility(h, i)
    sampleoutcome(h, πi, πo)
    indI, I = infoset(h)
    u = 0
    σ = matchregret(I)
    m = updateutility!(i, indI, h, σ, u, πi, πo)
    updateregret!(i, indI, σ, m, πi, r, s)
    u
end

function cfr(T::Int, ni::Int)
    r = [zeros(na[i]) for i in 1:ni] # regret
    s = [zeros(na[i]) for i in 1:ni] # cumulative strategies
    for t in 1:T
        for i in N
            u = updatetree!([], i, 1, 1, r, s) # utility
        end
    end
    u, r, s
end
