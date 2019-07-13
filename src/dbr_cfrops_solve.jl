function matchregret_dbr(I::Int64, r::Vector{Vector{Float64}}, gs::GameSet,
                            pdbr::Vector{Float64}, σfix::Vector{Vector{Float64}})
    ni_stage, na_stage = gs.ni_stage, gs.na_stage
    na = numactions(I, ni_stage, na_stage)
    σ = Vector{Float64}(undef, na)
    if I <= ni_stage[3] || I > sum(ni_stage[1:5]) # attacker infoset
        for a in 1:na
            denom = sum(max(r[I][b], 0.0) for b in 1:na)
            if denom > 0.0
                σ[a] = pdbr[I] * σfix[I][a] + (1 - pdbr[I]) * (max(r[I][a], 0.0) / denom)
            else
                σ[a] = pdbr[I] * σfix[I][a] + (1 - pdbr[I]) * (1.0 / na)
            end
        end
    else # defender infoset
        for a in 1:na
            denom = sum(max(r[I][b], 0.0) for b in 1:na)
            if denom > 0.0
                σ[a] = max(r[I][a], 0.0) / denom
            else
                σ[a] = 1.0 / na
            end
        end
    end
    return σ
end

function updateutility_dbr!(player::Int64, I::Int64, h::SVector, σ::Vector{Float64},
        u::Float64, πi::Float64, πo::Float64,
        r::Vector{Vector{Float64}}, s::Vector{Vector{Float64}},
        g::AirDefenseGame, gs::GameSet, depth::Int64, pdbr::Vector{Float64}, σfix::Vector{Vector{Float64}})
    A, ni_stage, na_stage = gs.A, gs.ni_stage, gs.na_stage
    na = numactions(I, ni_stage, na_stage)
    m = Vector{Float64}(undef, na)
    nextactions = actions(depth, A)
    # @show na[I]
    for a in 1:na
        ha = setindex(h, nextactions[a], depth)
        if getplayer(depth) == player
            πip = σ[a] * πi
            # up = updatetree_dbr!(ha, player, πip, πo, r, s, g, gs)
            up = updatetree_dbr!(ha, player, πip, πo, r, s, g, gs, depth + 1, pdbr, σfix)
            m[a] = up
            u = u + σ[a] * up
        else
            πop = σ[a] * πo
            # up = updatetree_dbr!(ha, player, πi, πop, r, s, g, gs)
            up = updatetree_dbr!(ha, player, πi, πop, r, s, g, gs, depth + 1, pdbr, σfix)
            u = u + up
        end
    end
    return m, u
end

function updateregret_dbr!(player::Int64, I::Int64, h::SVector, σ::Vector{Float64},
        u::Float64, m::Vector{Float64}, πi::Float64,
        r::Vector{Vector{Float64}}, s::Vector{Vector{Float64}}, gs::GameSet, depth ::Int64)
    ni_stage, na_stage = gs.ni_stage, gs.na_stage
    if getplayer(depth) == player
        for a in 1:numactions(I, ni_stage, na_stage)
            r[I][a] = r[I][a] + m[a] - u
            s[I][a] = s[I][a] + πi * σ[a]
        end
    end
end

function updatetree_dbr!(h::SVector, player::Int64, πi::Float64, πo::Float64,
    r::Vector{Vector{Float64}}, s::Vector{Vector{Float64}}, g::AirDefenseGame,
    gs::GameSet, depth::Int64, pdbr::Vector{Float64}, σfix::Vector{Vector{Float64}})
    # πo is short for \vec{π}_{-i}
    A = gs.A
    (terminal(depth)) && (return leafutility(h, player, πi, πo, g, gs))
    if getplayer(depth) == 3
        as = actions(depth, A)
        chance_weights = pweights([chance(depth, a, g, gs) for a in as])
        a = as[sample(chance_weights)]
        ha = setindex(h, a, depth)
        # return updatetree_dbr!(ha, player, πi, πo, r, s, g, gs)
        return updatetree_dbr!(ha, player, πi, πo, r, s, g, gs, depth + 1, pdbr, σfix)
    end
    I = infoset(h, depth, g, gs)
    u = 0.0
    σ = matchregret_dbr(I, r, gs, pdbr, σfix)
    m, u = updateutility_dbr!(player, I, h, σ, u, πi, πo, r, s, g, gs, depth, pdbr, σfix)
    updateregret_dbr!(player, I, h, σ, u, m, πi, r, s, gs, depth)
    return u
end

function cfr_dbr(g::AirDefenseGame, gs::GameSet, pdbr::Vector{Float64}, σfix::Vector{Vector{Float64}};
        iterlimit::Int = 100_000, timelimit::Number = 600, tol::Float64 = 5e-5)
    ni, ni_stage, na_stage, players = gs.ni, gs.ni_stage, gs.na_stage, gs.players
    r = [zeros(numactions(i, ni_stage, na_stage)) for i in 1:ni] # regret
    s = [zeros(numactions(i, ni_stage, na_stage)) for i in 1:ni] # cumulative strategies
    u = 0.0
    u1 = Float64[]
    u1deriv = 0
    u1prev = 0
    converged = false
    iter = 1
    tstart = now()
    while !converged && tominute(now() - tstart) < timelimit && iter < iterlimit
        for player in players
            h0 = SVector(0, 0, 0, 0, 0, 0)
            u = updatetree_dbr!(h0, player, 1.0, 1.0, r, s, g, gs, 1, pdbr, σfix)
            (player == 1) && push!(u1, u)
        end
        u1deriv = u1deriv * (1 - 1 / iter) + (u1[iter] - u1prev) * (1 / iter)
        converged = abs(u1deriv) < tol
        u1prev = u1[iter]
        iter % 10 == 0 && print("\rIter: $iter")
        iter += 1
    end
    denoms = [sum(i) for i in s]
    σ = Vector{Vector{Float64}}(undef, gs.ni)
    for i in eachindex(s)
        if denoms[i] == 0
            σ[i] = fill(1 / length(s[i][:]), length(s[i][:]))
        else
            σ[i] = s[i][:] ./ denoms[i]
        end
    end
    return u1, r, s, σ, converged
end

function strat_cfr_dbr(s::Vector{Vector{Float64}})
    denoms = [sum(i) for i in s]
    σ = Vector{Vector{Float64}}(undef, length(s))
    for i in eachindex(s)
        if denoms[i] == 0
            σ[i] = fill(1 / length(s[i][:]), length(s[i][:]))
        else
            σ[i] = s[i][:] ./ denoms[i]
        end
    end
    return σ
end
