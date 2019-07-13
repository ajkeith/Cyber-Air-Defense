using Pkg; Pkg.activate("I:\\My Documents\\00 AFIT\\Research\\Julia Projects\\StrategyGames")
using Revise, BenchmarkTools, Printf, ProgressMeter, Dates
using Random, StatsBase, Distributions, Distances, IterTools, DataFrames, DataFramesMeta
using StaticArrays, SparseArrays

#################################
# Build AirDefenseGame and GameSet

# Air defense game parameters
struct AirDefenseGame
    ncity::Int64 # number of cities
    X::Array{Float64,2} # city coordinates
    v::Array{Float64} # city populations (millions of people)
    r::Float64 # coverage radius of each air defense
    ndp::Int64 # number of air defense systems
    nap::Int64 # number of air attack targets
    ndpdc::Int64 # number of cyber-defendable defense systems
    ndpac::Int64 # number of cyber-attackable defense systems
    ndc::Int64 # number of cyber defense teams
    nac::Int64 # number of cyber attack teams
    pp::Float64 # physical defense effectiveness
    psc::Float64 # cyber sensor effectiveness
    pdc::Float64 # cyber defense effectiveness
    pac::Float64 # cyber attack effectiveness
    iads::Array{Int64} # sorted list of cities with IADS
end

# Generic game structure (except sensors and iadsdefenses pending refactoring)
struct GameSet
    players::UnitRange{Int64} # set of player indices
    A::Vector{Vector{Vector{Int64}}} # actions by stage
    An::Vector{UnitRange{Int64}} # action indices by stage
    na_stage::Vector{Int64} # number of actions by stage
    Iset::Vector{Any} # infosets
    ni_stage::Vector{Int64} # number of infosets by stage
    ni::Int64 # number of total infosets
    C::Array{Int64, 2} # coverage matrix
    sensors::Vector{Vector{Int64}} # combinations of sensed attacks
    iadsdefenses::Vector{Vector{Int64}} # combinations of defended iads
end

# data structure for optimization problem
struct GameOptSet
    ns1::Int64 # number of player 1 sequences
    ns2::Int64 # number of player 2 sequences
    ni1::Int64 # number of player 1 infosets
    ni2::Int64 # number of player 2 infosets
    ns1_stage::Vector{Int64} # number of player 1 sequences by stage
    ns2_stage::Vector{Int64} # number of player 2 sequences by stage
    reward # player 1 utility by seq1, seq2 (default is sparse array)
    na_stage::Vector{Int64} # number of actions by stage
    sensors::Vector{Vector{Int64}} # combinations of sensed attacks
    seqI1::Vector{Int64} # player 1: info set to sequence arriving at info set
    seqI2::Vector{Int64} # player 2: info set to sequence arriving at info set
    nextseqI1::Vector{UnitRange{Int64}} # player 1: info set to all sequences available from infoset
    nextseqI2::Vector{UnitRange{Int64}} # player 2: info set to all sequences available from infoset
end

AirDefenseGame() = AirDefenseGame(10,
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
        0.8,
        [1, 3, 4, 5, 8, 9])

AirDefenseGame(ncity, ndp, nap, ndpdc, ndpac, ndc, nac) = AirDefenseGame(ncity,
        rand(ncity, 2), rand(ncity), 0.3, ndp, nap, ndpdc, ndpac, ndc, nac,
        0.9, 0.8, 0.7, 0.8, sort!(sample(1:ncity, ndp, replace = false)))

AirDefenseGame(ncity, v, ndp, nap, ndpdc, ndpac, ndc, nac) = AirDefenseGame(ncity,
        rand(ncity, 2), v, 0.3, ndp, nap, ndpdc, ndpac, ndc, nac,
        0.9, 0.8, 0.7, 0.8, sort!(sample(1:ncity, ndp, replace = false)))

# AirDefenseGame(psc) = AirDefenseGame(10,
#         rand(10, 2), rand(10), 0.3, 6, 3, 4, 4, 2, 2,
#         0.9, psc, 0.7, 0.8, sort!(sample(1:10, 6, replace = false)))

AirDefenseGame(psc) = AirDefenseGame(10,
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
        psc,
        0.7,
        0.8,
        [1, 3, 4, 5, 8, 9])

function AirDefenseGame(ncity, X, v, r, ndp, nap, ndpdc, ndpac, ndc, nac,
        pp, psc, pdc, pac, iads)
    if any((pp, psc, pdc, pac) .< 0)
        @warn "Negative probabilities. Returning standard AirDefenseGame."
        return AirDefenseGame()
    elseif any((pp, psc, pdc, pac) .> 1)
        @warn "Probability > 1. Returning standard AirDefenseGame."
        return AirDefenseGame()
    elseif any((ndp, nap) .> ncity)
        @warn "Too many ndp or nap. Check ncity. Returning standard AirDefenseGame."
        return AirDefenseGame()
    elseif iads != sort(iads)
        @warn "IADS list must be sorted in increasing order. Returning standard AirDefenseGame."
        return AirDefenseGame()
    else
        return AirDefenseGame(ncity, X, v, r, ndp, nap, ndpdc, ndpac, ndc, nac,
                pp, psc, pdc, pac, iads)
    end
end

# ```
# coverage(g::AirDefenseGame)
#
# Compute the coverage of air defenses located in certain cities. C[i,j] is 1
# if city j is covered by defender i and 0 otherwise.
# ```
function coverage(g::AirDefenseGame)
    X, r, ncity = g.X, g.r, g.ncity
    dist = pairwise(Euclidean(), X')
    C = [dist[i,j] < r ? 1 : 0 for i = 1:ncity, j = 1:ncity]
    C
end

function build_action(g::AirDefenseGame)
    ncity, ndp, nap, ndpdc, ndpac, ndc, nac = g.ncity, g.ndp, g.nap, g.ndpdc, g.ndpac, g.ndc, g.nac
    a1c = collect(subsets(1:ndp, ndpdc)) # defensive cyber defendable na1c = binomial(ndp, ndpdc)
    a2c = collect(subsets(1:ndp, ndpac)) # offensive cyber attackable na2c = binomial(ndp, ndpac)
    a3a = collect(subsets(1:ndpac, nac)) # cyber attacks na3a = binomial(ndpac, nac)
    a4c = reshape_product(collect(Iterators.product(fill(1:2, nac)...))) # defensive cyber SA na4c = 2 ^ nac
    a5d = collect(subsets(1:ndpdc, ndc)) # defensive cyber response na5d = binomial(ndpdc, ndc)
    a6a = collect(subsets(1:ncity, nap)) # physical attcks  na6a = binomial(ncity, nap)
    A = [a1c, a2c, a3a, a4c, a5d, a6a]
    na_stage = length.(A)
    An = [1:na for na in na_stage]
    A, An, na_stage
end

function build_info(g::AirDefenseGame)
    ncity, ndp, nap, ndpdc, ndpac, ndc, nac = g.ncity, g.ndp, g.nap, g.ndpdc, g.ndpac, g.ndc, g.nac
    ni3a = binomial(ndp, ndpac)
    nsensor = sum(binomial(ndp, i) for i = 0:nac)
    ni5d = binomial(ndp, ndpdc) * nsensor
    ni6a = ni3a * binomial(ndpac, nac) * binomial(ndp, ndc)
    ni_stage = [0, 0, ni3a, 0, ni5d, ni6a] # ni_stage[j] is num of infosets at stage j
    ni = sum(ni_stage) # total number of infosets
    ni1 = ni5d
    ni2 = ni3a + ni6a
    ni, ni1, ni2, ni_stage
end

# the same physical action at a different info set has a different action name (see ref in kuhnopt.jl)
function build_nseq(g::AirDefenseGame, na_stage, ni_stage)
    ns1_stage = [1, ni_stage[5] * na_stage[5]]
    ns2_stage = [1, ni_stage[3] * na_stage[3], ni_stage[6] * na_stage[6]]
    ns1, ns2 = sum(ns1_stage), sum(ns2_stage)
    ns1, ns2, ns1_stage, ns2_stage
end

function build_Σset(g::AirDefenseGame, A, ns1, ns2)
    Σ1 = Vector{String}(undef, ns1)
    Σ2 = Vector{String}(undef, ns2)
    seqn1 = Array{Int64, 2}(undef, ns1, length(A))
    seqn2 = Array{Int64, 2}(undef, ns2, length(A))
    seq1defended_iads = Array{Int64, 2}(undef, ns1, g.ndc)
    seq2attacked_iads = Array{Int64, 2}(undef, ns2, g.nac)
    seq2defended_iads = Array{Int64, 2}(undef, ns2, g.ndc)
    seq2attacked_cities = Array{Int64, 2}(undef, ns2, g.nap)
    sensors = Vector{Int64}[]
    for i = 0:g.nac
        push!(sensors, collect(subsets(1:g.ndp, i))...) # by index
    end
    iadsdefenses = collect(subsets(1:g.ndp, g.ndc))
    Σ1[1], Σ2[1] = "∅", "∅"
    seqn1[1,:], seqn2[1,:] = zeros(Int64, length(A)), zeros(Int64, length(A))
    seq1defended_iads[1, :] = zeros(Int64, g.ndc)
    seq2attacked_iads[1, :] = zeros(Int64, g.nac)
    seq2defended_iads[1, :] = zeros(Int64, g.ndc)
    seq2attacked_cities[1, :] = zeros(Int64, g.nap)
    k = 1
    for i = 1:length(A[1]), j = 1:length(sensors), l = 1:length(A[5])
        k += 1
        Σ1[k] = string("c:", A[1][i], " c:", sensors[j], " d:", A[5][l])
        seqn1[k, :] = [i, 0, 0, j, l, 0]
        iads_ind = getindex(A[1][i], A[5][l])
        seq1defended_iads[k, :] = getindex(g.iads, iads_ind)
    end
    k = 1
    for i = 1:length(A[2]), j = 1:length(A[3])
        k += 1
        Σ2[k] = string("c:", A[2][i], " a:", A[3][j])
        seqn2[k, :] = [0, i, j, 0, 0, 0]
        seq2attacked_iads[k, :] = getindex(g.iads, getindex(A[2][i], A[3][j]))
    end
    for i = 1:length(A[2]), j = 1:length(A[3]), l = 1:length(iadsdefenses), m = 1:length(A[6])
        k += 1
        Σ2[k] = string("c:", A[2][i], " a:", A[3][j], " d:", iadsdefenses[l], " a:", A[6][m])
        seqn2[k, :] = [0, i, j, 0, l, m]
        attack_ind = getindex(A[2][i], A[3][j])
        seq2attacked_iads[k, :] = getindex(g.iads, attack_ind)
        seq2attacked_cities[k, :] = A[6][m]
    end
    (Σ1, Σ2), (seqn1, seqn2), (seq1defended_iads, seq2attacked_iads, seq2attacked_cities)
end

function build_Iset(g::AirDefenseGame, A, na_stage, ns1_stage, ns2_stage, ni1, ni2, ns1, ns2)
    ndp, ndc, nac, iads = g.ndp, g.ndc, g.nac, g.iads
    Iset = Vector{String}(undef, ni1 + ni2)
    I1set = Vector{String}(undef, ni1)
    I2set = Vector{String}(undef, ni2)
    seqI1 = Vector{Int64}(undef, ni1)
    seqI2 = Vector{Int64}(undef, ni2)
    nextseqI1 = Vector{UnitRange{Int64}}(undef, ni1)
    nextseqI2 = Vector{UnitRange{Int64}}(undef, ni2)
    ki, ki1, ki2, ksi1, ksi2, knsi1, knsi2 = 1, 1, 1, 1, 1, 1, 1
    lbnext = ns2_stage[1] + 1
    for a in A[2] # offensive cyber attackable
        targetable = getindex(iads, a)
        Iset[ki] = "Attacker: targetable $targetable"
        ki += 1
        I2set[ki2] = "Attacker: targetable $targetable"
        ki2 += 1
        seqI2[ksi2] = 1
        ksi2 += 1
        nextseqI2[knsi2] = lbnext:(lbnext + na_stage[3] - 1)
        knsi2 += 1
        lbnext += na_stage[3]
    end
    sensors = Vector{Int64}[]
    for i = 0:nac
        push!(sensors, collect(subsets(iads, i))...)
    end
    lbnext = ns1_stage[1] + 1
    for a in A[1] # defensive cyber defendable
        defendable = getindex(iads, a)
        for s in sensors
            Iset[ki] = "Defender: defendable $defendable, sensed $s"
            ki += 1
            I1set[ki1] = "Defender: defendable $defendable, sensed $s"
            ki1 += 1
            seqI1[ksi1] = 1
            ksi1 += 1
            nextseqI1[knsi1] = lbnext:(lbnext + na_stage[5] - 1)
            knsi1 += 1
            lbnext += na_stage[5]
        end
    end
    lbnext = sum(ns2_stage[1:2]) + 1
    for (a1i, a1) in enumerate(A[2]) # offensive cyber capes
        targetable = getindex(iads, a1)
        for (a2i, a2) in enumerate(A[3])  # cyber attcks
            attacked = getindex(iads, getindex(a1, a2))
            seq2 = 1 + (a1i - 1) * length(A[3]) + a2i
            for defended in collect(subsets(iads, ndc))
                Iset[ki] = "Attacker: targetable $targetable, cyberattack $attacked, cyberdefend $defended"
                ki += 1
                I2set[ki2] = "Attacker: targetable $targetable, cyberattack $attacked, cyberdefend $defended"
                ki2 += 1
                seqI2[ksi2] = seq2
                ksi2 += 1
                nextseqI2[knsi2] = lbnext:(lbnext + na_stage[6] - 1)
                knsi2 += 1
                lbnext += na_stage[6]
            end
        end
    end
    Iset, (I1set, I2set), (seqI1, seqI2), (nextseqI1, nextseqI2)
end

# defender's utility for a combination of player actions
function utility(g::AirDefenseGame, C, defended_iads, attacked_iads, attacked_cities)
    u = 0.0
    for city in attacked_cities
        q = 1.0
        for defender in g.iads
            if C[defender, city] == 0
                q *= (1 - 0.0)
            elseif defender ∈ attacked_iads && defender ∉ defended_iads
                q *= (1 - 0.0)
            elseif defender ∉ attacked_iads
                q *= (1 - g.pp)
            elseif defender ∈ attacked_iads && defender ∈ defended_iads
                q *= (1 - g.pp * g.pdc * (1 - g.pac)) # if a defender is cyber attacked, it has pdc * pac of being cyber-defended
            else
                @warn "Unexpected defense layout."
            end
        end
        u += -1.0 * g.v[city] * q # defender's expected utility for a city
    end
    u
end

function build_utility_hist(g::AirDefenseGame, A, An)
    nz = prod(An[i][end] for i in 1:length(An))
    leafnodes = Vector{Vector{Int64}}(undef, nz)
    U = Vector{Float64}(undef, nz)
    k = 0
    C = coverage(g)
    println("Building history-form utilities...")
    # @progress for c1 in An[1], c2 in An[2], ac in An[3], c4 in An[4], dc in An[5], ap in An[6]
    for c1 in An[1], c2 in An[2], ac in An[3], c4 in An[4], dc in An[5], ap in An[6]
        k += 1
        h = [c1, c2, ac, c4, dc, ap]
        leafnodes[k] = h
        attacked_cities = @view A[6][h[6]][:]
        defended_iads = getindex(g.iads, getindex(A[1][h[1]], A[5][h[5]]))
        attacked_iads = getindex(g.iads, getindex(A[2][h[2]], A[3][h[3]]))
        U[k] = utility(g, C, defended_iads, attacked_iads, attacked_cities) # defender's expected utility for a leaf node
    end
    U, leafnodes
end

# Expected = true, sets the utliity to the expected utility over the sequence given the chance actions
function build_utility_seq(g::AirDefenseGame, gs::GameSet, nss, seqn, seqactions;
                            expected::Bool = false, timelimit::Int = 90)
    A = gs.A
    ns1, ns2 = nss
    seqn1, seqn2 = seqn
    seq1defended_iads, seq2attacked_iads, seq2attacked_cities = seqactions
    nstage = length(A)
    iseqn1 = zeros(Int64, nstage)
    iseqn2 = zeros(Int64, nstage)
    reward = spzeros(ns1, ns2)
    C = coverage(g)
    sensors = Vector{Int64}[]
    for i = 0:g.nac
        push!(sensors, collect(subsets(1:g.ndp, i))...) # by index
    end
    iadsdefenses = collect(subsets(1:g.ndp, g.ndc)) # by index
    probs = [(1 / 15) * (1 / 15) * chance(4, c4, g, gs) for c4 in gs.An[4]] # assumes equal prob in first two rounds
    p = 1.0 # default value if expected = false
    k = 0
    println("Building sequence-form utilities, expected = $expected...")
    starttime = now()
    # @progress for (ic1, c1) in enumerate(A[1])
    for (ic1, c1) in enumerate(A[1])
        iseqn1[1] = ic1
        for (ic2, c2) in enumerate(A[2])
            iseqn2[2] = ic2
            for (iac, ac) in enumerate(A[3])
                iseqn2[3] = iac
                for (ic4, c4) in enumerate(A[4])
                    sensed = getindex(c2, getindex(ac, c4 .== 2)) # ndp id's of sensed attacks
                    isensor = findfirst(x -> x == sensed, sensors)
                    iseqn1[4] = isensor
                    expected && (p = probs[ic4])
                    for (idc, dc) in enumerate(A[5])
                        iseqn1[5] = idc
                        idefended = findfirst(x -> x == getindex(c1, dc), iadsdefenses)
                        iseqn2[5] = idefended
                        ind1 = matchrow(iseqn1, seqn1)
                        for (iap, ap) in enumerate(A[6])
                            iseqn2[6] = iap
                            ind2 = matchrow(iseqn2, seqn2)
                            reward[ind1, ind2] = p * utility(g, C,
                                seq1defended_iads[ind1, :], seq2attacked_iads[ind2, :], seq2attacked_cities[ind2, :])
                            k += 1
                            k % 1_000 == 0 && print("\rNodes (M): $(k/1_000_000)")
                            if k % 10_000 == 0
                                (tominute(now() - starttime) > timelimit) && (return reward, false)
                            end
                        end
                    end
                end
            end
        end
    end
    return reward, true
end

function GameSet(g::AirDefenseGame)
    players = 1:2
    A, An, na_stage = build_action(g)
    ni, ni1, ni2, ni_stage = build_info(g)
    ns1, ns2, ns1_stage, ns2_stage = build_nseq(g, na_stage, ni_stage)
    Iset, (I1set, I2set), (seqI1, seqI2), (nextseqI1, nextseqI2) = build_Iset(g, A, na_stage, ns1_stage, ns2_stage, ni1, ni2, ns1, ns2)
    C = coverage(g)
    sensors = Vector{Int64}[]
    for i = 0:g.nac
        push!(sensors, collect(subsets(1:g.ndp, i))...)
    end
    iadsdefenses = collect(subsets(1:g.ndp, g.ndc))
    GameSet(players, A, An, na_stage, Iset, ni_stage, ni, C, sensors, iadsdefenses)
end

function GameOptSet(g::AirDefenseGame, gs::GameSet, reward)
    A, An, na_stage = gs.A, gs.An, gs.na_stage
    ni, ni1, ni2, ni_stage = build_info(g)
    ns1, ns2, ns1_stage, ns2_stage = build_nseq(g, na_stage, ni_stage)
    Σ, seqn, seqactions = build_Σset(g, A, ns1, ns2)
    Iset, (I1set, I2set), (seqI1, seqI2), (nextseqI1, nextseqI2) = build_Iset(g, A, na_stage, ns1_stage, ns2_stage, ni1, ni2, ns1, ns2)
    return GameOptSet(ns1, ns2, ni1, ni2, ns1_stage, ns2_stage, reward,
        gs.na_stage, gs.sensors, seqI1, seqI2, nextseqI1, nextseqI2)
end
