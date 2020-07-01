using BenchmarkTools, Printf, ProgressMeter, Dates

function gamesize(g::AirDefenseGame)
    ncity, ndp, nap, ndpdc, ndpac, ndc, nac = g.ncity, g.ndp, g.nap, g.ndpdc, g.ndpac, g.ndc, g.nac
    nIc1 = 1
    nIa2 = binomial(ndp, ndpac)
    nIa5 = nIa2 * binomial(ndp, nac) * binomial(ndp, ndc)
    nIc3 = 1
    nId4 = binomial(ndp, ndpdc) * (2 ^ ndp)
    nIa = @sprintf("%.2E", nIa2 + nIa5)
    nId = @sprintf("%.2E", nId4)
    na1c = binomial(ndp, ndpdc)
    na2c = binomial(ndp, ndpac)
    na3a = binomial(ndpac, nac)
    na4c = 2 ^ nac
    na5d = binomial(ndpdc, ndc)
    na6a = binomial(ncity, nap)
    nas = [na1c, na2c, na3a, na4c, na5d, na6a]
    leafnodes = @sprintf("%.2E", prod(nas))
    nnodes = @sprintf("%.2E", sum(cumprod(nas)))
    (nIa, nId, leafnodes, nnodes)
end

# N : player indices (nature not included)
getplayers(g::AirDefenseGame) = 1:2

# Player index of current node (note: this is distinct from which player is currently updating CFR)
function getplayer(depth::Int64)
    (depth == 6 || depth == 3) ? 2 : (depth == 5) ? 1 : 3
    # 1 = defender, 2 = attacker, 3 = nature
end

actions(depth::Int64, An::Vector{UnitRange{Int64}}) = An[depth]
actions(depth::Int64, A::Vector{Vector{Vector{Int64}}}) = 1:length(A[depth])

function numactions(I::Int64, ni_stage, na_stage)
    if I in 1:ni_stage[3]
        return na_stage[3] # na3a
    elseif I in (ni_stage[3] + 1):(ni_stage[3] + ni_stage[5])
        return na_stage[5] # na5d
    elseif I in (ni_stage[3] + ni_stage[5] + 1):(ni_stage[3] + ni_stage[5] + ni_stage[6])
        return na_stage[6] # na5a
    else
        @warn "Infoset $I not found."
        return 0
    end
end

terminal(depth::Int64) = depth == 7

function details(h::SVector, A)
    len = findfirst(x -> x == 0, h) - 1
    (len == nothing) && (len = length(h))
    df = DataFrame(Stage = Int[], Domain = String[], Player = String[], ActionNumber = Int[], Action = Array[])
    for i = 1:len
        stage = i
        domain = (i == 6) ? "Physical" : "Cyber"
        p = getplayer(h[1:i])
        pstring = p == 2 ? "Attacker" : p == 1 ? "Defender" : "Chance"
        push!(df, (stage, domain, pstring, h[i], A[i][h[i]]))
    end
    df
end

function infoset(h::SVector, depth::Int64, g::AirDefenseGame, gs::GameSet) # btime: ~150 ns
    # findfirst benchmarks faster than numerically calculating infoset (see earlier commit)
    A, ni_stage, na_stage, sensors, iadsdefenses = gs.A, gs.ni_stage, gs.na_stage, gs.sensors, gs.iadsdefenses
    cyber_attacks, cyber_defenses = A[3], A[5]
    ndp, nac = g.ndp, g.nac
    len = depth - 1
    if len == 5
        defended_iads = getindex(gs.A[1][h[1]], gs.A[5][h[5]])
        idc = findfirst(x -> x == defended_iads, iadsdefenses)
        niadsdefenses = length(gs.iadsdefenses)
        return sum(ni_stage[1:5]) + (h[2] - 1) * na_stage[3] * niadsdefenses + (h[3] - 1) * niadsdefenses + idc
    elseif len == 4
        nsensor = length(sensors) # number of combinations of sensed cyber attacks
        sensed = getindex(gs.A[2][h[2]], getindex(A[3][h[3]], A[4][h[4]] .== 2)) # ndp id's of sensed attacks
        isensor = findfirst(x -> x == sensed, sensors) # infoset id of that combination
        return sum(ni_stage[1:4]) + (h[1] - 1) * nsensor + isensor
    elseif len == 2
        return h[2]
    else
        player = getplayer(h)
        @warn "No infoset found. Chance node? h = $h, player = $player."
        return 0
    end
end

#TODO: Make these probabilities more rational?
function chance(depth::Int64, a::Int64, g::AirDefenseGame, gs::GameSet)
    len = depth - 1
    if len == 3 # nature determines defense cyber SA
        nsensed = sum(gs.A[4][a] .== 2) # number of sensed attacks
        return (g.psc ^ nsensed) * (1 - g.psc)^(g.nac - nsensed)
    elseif len == 1 # nature deals offensive capes
        return 1.0 / gs.na_stage[2]
    elseif len == 0 # nature deals defensive capes
        return 1.0 / gs.na_stage[1]
    else
        @warn "Node at depth $depth not a chance node."
        return 0.0
    end
end

function leafutility(h::SVector, player::Int64, πi::Float64, πo::Float64, g::AirDefenseGame, gs::GameSet)
    c1, c2, ac, c4, dc, ap = h
    attacked_cities = @view gs.A[6][h[6]][:]
    defended_iads = getindex(g.iads, getindex(gs.A[1][h[1]], gs.A[5][h[5]]))
    attacked_iads = getindex(g.iads, getindex(gs.A[2][h[2]], gs.A[3][h[3]]))
    leaf_value = utility(g, gs.C, defended_iads, attacked_iads, attacked_cities)
    u = player == 1 ? leaf_value * πo : -1.0 * leaf_value * πo
    return u
end

# convert behavioral strategy to realization strategy (i.e. sequence form?)
function real_strat_calc(σ, ni, ni_stage, ns1, ns2, seqI1, seqI2, nextseqI1, nextseqI2)
    length(σ) != ni && @warn "Unexpected behavioral strategy length: $(length(σ))."
    σ1 = σ[(ni_stage[3] + 1):sum(ni_stage[1:5])]
    σ2 = vcat(σ[1:ni_stage[3]], σ[(sum(ni_stage[1:5]) + 1):end])
    rp1 = zeros(Float64, ns1)
    rp2 = zeros(Float64, ns2)
    rp1[1], rp2[1] = 1.0, 1.0
    for (i, s1) in enumerate(σ1)
        rp1[nextseqI1[i]] .= rp1[seqI1[i]] .* s1
    end
    for (i, s2) in enumerate(σ2)
        rp2[nextseqI2[i]] .= rp2[seqI2[i]] .* s2
    end
    return (rp1, rp2)
end

function real_strat_struct_calc(σ, ni, ni_stage, ns1, ns2, seqI1, seqI2, nextseqI1, nextseqI2)
    length(σ) != ni && @warn "Unexpected behavioral strategy length: $(length(σ))."
    σ1 = σ[(ni_stage[3] + 1):sum(ni_stage[1:5])]
    σ2 = vcat(σ[1:ni_stage[3]], σ[(sum(ni_stage[1:5]) + 1):end])
    rp1_struct, rp2_struct = similar(σ1), similar(σ2)
    rp1, rp2 = real_strat_calc(σ, ni, ni_stage, ns1, ns2, seqI1, seqI2, nextseqI1, nextseqI2)
    for (i, s1) in enumerate(σ1)
        rp1_struct[i] = rp1[nextseqI1[i]]
    end
    for (i, s2) in enumerate(σ2)
        rp2_struct[i] = rp2[nextseqI2[i]]
    end
    return (rp1_struct, rp2_struct)
end

function real_strat_struct_combined_calc(σ, ni, ni_stage, ns1, ns2, seqI1, seqI2, nextseqI1, nextseqI2)
    σ1 = σ[(ni_stage[3] + 1):sum(ni_stage[1:5])]
    σ2 = vcat(σ[1:ni_stage[3]], σ[(sum(ni_stage[1:5]) + 1):end])
    rp1_struct, rp2_struct = similar(σ1), similar(σ2)
    rp1, rp2 = real_strat_calc(σ, ni, ni_stage, ns1, ns2, seqI1, seqI2, nextseqI1, nextseqI2)
    for (i, s1) in enumerate(σ1)
        rp1_struct[i] = rp1[nextseqI1[i]]
    end
    for (i, s2) in enumerate(σ2)
        rp2_struct[i] = rp2[nextseqI2[i]]
    end
    rp_struct = similar(σ)
    rp_struct[1:ni_stage[3]] .= rp2_struct[1:ni_stage[3]]
    rp_struct[(ni_stage[3] + 1):sum(ni_stage[1:5])] .= rp1_struct
    rp_struct[(sum(ni_stage[1:5]) + 1):end] .= rp2_struct[(ni_stage[3] + 1):end]
    return rp_struct
end

function real_strat(σ::Vector{Vector{Float64}}, gs::GameSet, gos::GameOptSet)
    ni, ni_stage, ns1, ns2 = gs.ni, gs.ni_stage, gos.ns1, gos.ns2
    seqI1, seqI2, nextseqI1, nextseqI2 = gos.seqI1, gos.seqI2, gos.nextseqI1, gos.nextseqI2
    return real_strat_calc(σ, ni, ni_stage, ns1, ns2, seqI1, seqI2, nextseqI1, nextseqI2)
end

function real_strat(σ::Vector{Vector{Float64}}, g::AirDefenseGame)
    A, _, na_stage = build_action(g)
    ni, ni1, ni2, ni_stage = build_info(g)
    ns1, ns2, ns1_stage, ns2_stage = build_nseq(g, na_stage, ni_stage)
    _, _, (seqI1, seqI2), (nextseqI1, nextseqI2) = build_Iset(g, A, na_stage, ns1_stage, ns2_stage, ni1, ni2, ns1, ns2)
    return real_strat_calc(σ, ni, ni_stage, ns1, ns2, seqI1, seqI2, nextseqI1, nextseqI2)
end

function real_strat_struct(σ::Vector{Vector{Float64}}, g::AirDefenseGame)
    A, _, na_stage = build_action(g)
    ni, ni1, ni2, ni_stage = build_info(g)
    ns1, ns2, ns1_stage, ns2_stage = build_nseq(g, na_stage, ni_stage)
    _, _, (seqI1, seqI2), (nextseqI1, nextseqI2) = build_Iset(g, A, na_stage, ns1_stage, ns2_stage, ni1, ni2, ns1, ns2)
    return real_strat_struct_calc(σ, ni, ni_stage, ns1, ns2, seqI1, seqI2, nextseqI1, nextseqI2)
end

function real_strat_struct_combined(σ::Vector{Vector{Float64}}, g::AirDefenseGame)
    A, _, na_stage = build_action(g)
    ni, ni1, ni2, ni_stage = build_info(g)
    ns1, ns2, ns1_stage, ns2_stage = build_nseq(g, na_stage, ni_stage)
    _, _, (seqI1, seqI2), (nextseqI1, nextseqI2) = build_Iset(g, A, na_stage, ns1_stage, ns2_stage, ni1, ni2, ns1, ns2)
    return real_strat_struct_combined_calc(σ, ni, ni_stage, ns1, ns2, seqI1, seqI2, nextseqI1, nextseqI2)
end

function real_strat_struct_combined(σ::Vector{Vector{Float64}}, gs::GameSet, seqvals)
    ni, ni_stage = gs.ni, gs.ni_stage
    ns1, ns2, seqI1, seqI2, nextseqI1, nextseqI2 = seqvals
    return real_strat_struct_combined_calc(σ, ni, ni_stage, ns1, ns2, seqI1, seqI2, nextseqI1, nextseqI2)
end

# build consraint set on sequence-form strategy for CCFR
function build_ccfr_constraints(g, σfix, conf::Float64, sd::Float64)
    noise = Normal(0, sd)
    hw = quantile(noise, 1 - (1 - conf) / 2)
    σlb = [max.(σi .- hw, 0.0) for σi in σfix]
    σub = [min.(σi .+ hw, 1.0) for σi in σfix]
    rp2fix = real_strat_struct(σfix, g)[2]
    rp2lb = real_strat_struct(σlb, g)[2]
    rp2ub = real_strat_struct(σub, g)[2]
    !all(rp2lb .<= rp2fix .<= rp2ub) && @warn "Sequence-form strategy out of bounds."
    constraint_coeff = [[1, -1 .* rp2lb[i][j]] for i in 1:length(rp2lb), j in 1:length(rep2lb[i])]
    constraint_coeff = vcat(hcat(ones(length(rp2fix)), -1 .* rp2lb), hcat(-1 .* ones(length(rp2fix)), rp2ub))
    @show length(constraint_coeff), length(constraint_coeff[:,1])
    grad_constraint_coeff = hcat(constraint_coeff[:,1], zeros(length(constraint_coeff[:,1])))
    return constraint_coeff, grad_constraint_coeff
end
