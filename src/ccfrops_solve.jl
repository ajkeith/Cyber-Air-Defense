function cmatchregret(I::Int64, r::Vector{Vector{Float64}}, gs::GameSet)
    ni_stage, na_stage = gs.ni_stage, gs.na_stage
    na = numactions(I, ni_stage, na_stage)
    σ = Vector{Float64}(undef, na)
    for a in 1:na
        denom = sum(max(r[I][b], 0.0) for b in 1:na)
        if denom > 0.0
            σ[a] = max(r[I][a], 0.0) / denom
        else
            σ[a] = 1.0 / na
        end
    end
    return σ
end

function cupdateutility!(player::Int64, I::Int64, h::SVector, σ::Vector{Float64},
        u::Float64, πi::Float64, πo::Float64,
        r::Vector{Vector{Float64}}, s::Vector{Vector{Float64}},
        g::AirDefenseGame, gs::GameSet, depth::Int64,
        α::Number, β::Number, γ::Number, iter::Int, siter::Vector{Int}, discounted::Bool,
        λl::Vector{Vector{Float64}}, λu::Vector{Vector{Float64}})
    A, ni_stage, na_stage = gs.A, gs.ni_stage, gs.na_stage
    na = numactions(I, ni_stage, na_stage)
    m = Vector{Float64}(undef, na)
    nextactions = actions(depth, A)
    # @show na[I]
    for a in 1:na
        ha = setindex(h, nextactions[a], depth)
        if getplayer(depth) == player
            πip = σ[a] * πi
            up = cupdatetree!(ha, player, πip, πo, r, s, g, gs, depth + 1, α, β, γ, iter, siter, discounted, λl, λu)
            if getplayer(depth) == 2 # constrained player
                lagrange = λl[I][a] * -1 + λu[I][a] * 1 # assumes indpendent constraints
            else
                lagrange = 0.0
            end
            m[a] = up - lagrange
            u = u + σ[a] * up
        else
            πop = σ[a] * πo
            up = cupdatetree!(ha, player, πi, πop, r, s, g, gs, depth + 1, α, β, γ, iter, siter, discounted, λl, λu)
            u = u + up # possibly need a lagrange adjustment here? but I don't think so
        end
    end
    return m, u
end

function cupdateregret!(player::Int64, I::Int64, h::SVector, σ::Vector{Float64},
        u::Float64, m::Vector{Float64}, πi::Float64,
        r::Vector{Vector{Float64}}, s::Vector{Vector{Float64}}, gs::GameSet, depth ::Int64,
        α::Number, β::Number, γ::Number, iter::Int, siter::Vector{Int}, discounted::Bool)
    ni_stage, na_stage = gs.ni_stage, gs.na_stage
    if getplayer(depth) == player
        siter[I] += 1
        for a in 1:numactions(I, ni_stage, na_stage)
            rnew = r[I][a] + m[a] - u
            if discounted
                # r[I][a] = rnew >= 0 ? rnew * (iter^α / (iter^α + 1)) : rnew * (iter^β / (iter^β + 1))
                # s[I][a] = (s[I][a] + πi * σ[a]) * (iter / (iter + 1))^γ
                if rnew >= 0
                    r[I][a] = rnew * (siter[I]^α / (siter[I]^α + 1))
                else
                    r[I][a] = rnew * (float(siter[I])^β / (float(siter[I])^β + 1))
                end
                s[I][a] = (s[I][a] + πi * σ[a]) * (siter[I] / (siter[I] + 1))^γ
            else
                r[I][a] = rnew
                s[I][a] = s[I][a] + πi * σ[a]
            end
        end
    end
end

function cupdatetree!(h::SVector, player::Int64, πi::Float64, πo::Float64,
    r::Vector{Vector{Float64}}, s::Vector{Vector{Float64}}, g::AirDefenseGame,
    gs::GameSet, depth::Int64, α::Number, β::Number, γ::Number, iter::Int, siter::Vector{Int}, discounted::Bool,
    λl::Vector{Vector{Float64}}, λu::Vector{Vector{Float64}})
    # πo is short for \vec{π}_{-i}
    A = gs.A
    (terminal(depth)) && (return leafutility(h, player, πi, πo, g, gs))
    if getplayer(depth) == 3
        as = actions(depth, A)
        chance_weights = pweights([chance(depth, a, g, gs) for a in as])
        a = as[sample(chance_weights)]
        ha = setindex(h, a, depth)
        return cupdatetree!(ha, player, πi, πo, r, s, g, gs, depth + 1, α, β, γ, iter, siter, discounted, λl, λu)
    end
    I = infoset(h, depth, g, gs)
    u = 0.0
    σ = cmatchregret(I, r, gs)
    m, u = cupdateutility!(player, I, h, σ, u, πi, πo, r, s, g, gs, depth, α, β, γ, iter, siter, discounted, λl, λu)
    cupdateregret!(player, I, h, σ, u, m, πi, r, s, gs, depth, α, β, γ, iter, siter, discounted)
    return u
end

function strat_ccfr(s::Vector{Vector{Float64}})
    σ = [similar(si) for si in s]
    for i in eachindex(s)
        denom = sum(s[i])
        if denom == 0
            len = length(s[i])
            σ[i] .= fill(1 / len, len)
        else
            σ[i] .= s[i] ./ denom
        end
    end
    return σ
end

function ccfr_full(g::AirDefenseGame, gs::GameSet, rp2lb, rp2ub;
        iterlimit::Int = 100_000,
        timelimit::Number = 600, tol::Float64 = 5e-5,
        α::Number = 1.5, β::Number = 0.5, γ::Number = 2, discounted::Bool = false,
        λmax::Number = 1_000, λscale::Number = 4)
    ni, ni_stage, na_stage, players = gs.ni, gs.ni_stage, gs.na_stage, gs.players
    A, _, na_stage = build_action(g)
    ni, ni1, ni2, ni_stage = build_info(g)
    ns1, ns2, ns1_stage, ns2_stage = build_nseq(g, na_stage, ni_stage)
    _, _, (seqI1, seqI2), (nextseqI1, nextseqI2) = build_Iset(g, A, na_stage, ns1_stage, ns2_stage, ni1, ni2, ns1, ns2)
    seqvals = (ns1, ns2, seqI1, seqI2, nextseqI1, nextseqI2)
    r = [zeros(numactions(i, ni_stage, na_stage)) for i in 1:ni] # regret
    s = [zeros(numactions(i, ni_stage, na_stage)) for i in 1:ni] # cumulative strategies
    λl = [zeros(numactions(i, ni_stage, na_stage)) for i in 1:ni]
    λu = [zeros(numactions(i, ni_stage, na_stage)) for i in 1:ni]
    λmean = zeros(iterlimit ÷ 10)
    siter = zeros(Int64, ni) # infoset visit counts
    u = 0.0
    n_hist = 10 # number of intermediate strategies to record (for best responses)
    Tx = round(Int64, timelimit * 60 / n_hist) # frequency of recording history
    tx = 1 # index for sigma calculations
    iter = 1 # index for number of iter
    lind = 1 # index for lambda tracking
    u1deriv = 0
    u1prev = 0
    converged = false
    u1 = Float64[]
    σ1 = Vector{Vector{Vector{Float64}}}(undef, n_hist)
    println("CCFR: start $timelimit min or $iterlimit iter at $(now())...")
    tstart, t_hist = now(), now()
    # lamcheck = []
    while !converged && tominute(now() - tstart) < timelimit && iter < iterlimit
        ξ = (λmax ^ λscale) / (1 * sqrt(iter)) # λ learning rate β / (G √T)
        Ψ = real_strat_struct_combined(strat_ccfr(s), gs, seqvals) # could be faster if we only update for player 2
        for I in 1:size(rp2lb, 1)
            for a in 1:length(rp2lb[I])
                λl[I][a] = clamp(λl[I][a] + ξ * (-Ψ[I][a] + rp2lb[I][a]), 0.0, λmax)
                λu[I][a] = clamp(λu[I][a] + ξ * (Ψ[I][a] - rp2ub[I][a]), 0.0, λmax)
            end
        end
        # push!(lamcheck, deepcopy(λl))
        # iter % 100 == 0 && @show λl[1], λu[1]
        for player in players
            h0 = SVector(0, 0, 0, 0, 0, 0)
            u = cupdatetree!(h0, player, 1.0, 1.0, r, s, g, gs, 1, α, β, γ, iter, siter, discounted, λl, λu)
            (player == 1) && push!(u1, u)
        end
        if tosecond(now() - t_hist) > Tx # calculate sigma every Tx seconds
            σ1[tx] = strat_ccfr(s)
            tx += 1
            t_hist = now()
        end
        u1deriv = u1deriv * (1 - 1 / iter) + (u1[iter] - u1prev) * (1 / iter)
        converged = abs(u1deriv) < tol
        u1prev = u1[iter]
        # if iter % 10 == 0
        #     lammean = mean(mean.(λu))
        #     lamcount = sum(sum(x .> 0) for x in λu)
        #     λmean[lind] = lammean
        #     print("\rIter: $iter, λ mean: $(round(lammean, digits = 6)), λ count: $lamcount")
        #     lind += 1
        # end
        iter % 10 == 0 && print("\rIter: $iter")
        iter += 1
    end
    println("")
    σ = strat_ccfr(s)
    # return u1, r, s, σ, σ1, converged, iter, λmean
    return u1, r, s, σ, σ1, converged, iter
end
