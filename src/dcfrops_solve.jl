function dmatchregret(I::Int64, r::Vector{Vector{Float64}}, gs::GameSet)
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

function dupdateutility!(player::Int64, I::Int64, h::SVector, σ::Vector{Float64},
        u::Float64, πi::Float64, πo::Float64,
        r::Vector{Vector{Float64}}, s::Vector{Vector{Float64}},
        g::AirDefenseGame, gs::GameSet, depth::Int64,
        α::Number, β::Number, γ::Number, iter::Int, siter::Vector{Int}, discounted::Bool)
    A, ni_stage, na_stage = gs.A, gs.ni_stage, gs.na_stage
    na = numactions(I, ni_stage, na_stage)
    m = Vector{Float64}(undef, na)
    nextactions = actions(depth, A)
    # @show na[I]
    for a in 1:na
        ha = setindex(h, nextactions[a], depth)
        if getplayer(depth) == player
            πip = σ[a] * πi
            up = dupdatetree!(ha, player, πip, πo, r, s, g, gs, depth + 1, α, β, γ, iter, siter, discounted)
            m[a] = up
            u = u + σ[a] * up
        else
            πop = σ[a] * πo
            up = dupdatetree!(ha, player, πi, πop, r, s, g, gs, depth + 1, α, β, γ, iter, siter, discounted)
            u = u + up
        end
    end
    return m, u
end

# function dupdateregret!(player::Int64, I::Int64, h::SVector, σ::Vector{Float64},
#         u::Float64, m::Vector{Float64}, πi::Float64,
#         r::Vector{Vector{Float64}}, s::Vector{Vector{Float64}}, gs::GameSet, depth ::Int64,
#         α::Number, β::Number, γ::Number, iter::Int, siter::Vector{Int}, discounted::Bool)
#     ni_stage, na_stage = gs.ni_stage, gs.na_stage
#     if getplayer(depth) == player
#         siter[I] += 1
#         for a in 1:numactions(I, ni_stage, na_stage)
#             rnew = r[I][a] + m[a] - u
#             if discounted
#                 # r[I][a] = rnew >= 0 ? rnew * (iter^α / (iter^α + 1)) : rnew * (iter^β / (iter^β + 1))
#                 # s[I][a] = (s[I][a] + πi * σ[a]) * (iter / (iter + 1))^γ
#                 if rnew >= 0
#                     r[I][a] = rnew * (siter[I]^α / (siter[I]^α + 1))
#                 else
#                     r[I][a] = rnew * (siter[I]^β / (siter[I]^β + 1))
#                 end
#                 s[I][a] = (s[I][a] + πi * σ[a]) * (siter[I] / (siter[I] + 1))^γ
#             else
#                 r[I][a] = rnew
#                 s[I][a] = s[I][a] + πi * σ[a]
#             end
#         end
#     end
# end

function dupdateregret!(player::Int64, I::Int64, h::SVector, σ::Vector{Float64},
        u::Float64, m::Vector{Float64}, πi::Float64,
        r::Vector{Vector{Float64}}, s::Vector{Vector{Float64}}, gs::GameSet, depth ::Int64,
        α::Number, β::Number, γ::Number, iter::Int, siter::Vector{Int}, discounted::Bool)
    ni_stage, na_stage = gs.ni_stage, gs.na_stage
    if getplayer(depth) == player
        siter[I] += 1
        for a in 1:numactions(I, ni_stage, na_stage)
            rnew = m[a] - u
            if discounted
                # r[I][a] = rnew >= 0 ? rnew * (iter^α / (iter^α + 1)) : rnew * (iter^β / (iter^β + 1))
                # s[I][a] = (s[I][a] + πi * σ[a]) * (iter / (iter + 1))^γ
                if rnew >= 0
                    r[I][a] = r[I][a] + rnew * (siter[I]^α / (siter[I]^α + 1))
                else
                    r[I][a] = r[I][a] + rnew * (siter[I]^β / (siter[I]^β + 1))
                end
                s[I][a] = s[I][a] + (πi * σ[a]) * (siter[I] / (siter[I] + 1))^γ
            else
                r[I][a] = r[I][a] + rnew
                s[I][a] = s[I][a] + πi * σ[a]
            end
        end
    end
end

function dupdatetree!(h::SVector, player::Int64, πi::Float64, πo::Float64,
    r::Vector{Vector{Float64}}, s::Vector{Vector{Float64}}, g::AirDefenseGame,
    gs::GameSet, depth::Int64, α::Number, β::Number, γ::Number, iter::Int, siter::Vector{Int}, discounted::Bool)
    # πo is short for \vec{π}_{-i}
    A = gs.A
    (terminal(depth)) && (return leafutility(h, player, πi, πo, g, gs))
    if getplayer(depth) == 3
        as = actions(depth, A)
        chance_weights = pweights([chance(depth, a, g, gs) for a in as])
        a = as[sample(chance_weights)]
        ha = setindex(h, a, depth)
        return dupdatetree!(ha, player, πi, πo, r, s, g, gs, depth + 1, α, β, γ, iter, siter, discounted)
    end
    I = infoset(h, depth, g, gs)
    u = 0.0
    σ = dmatchregret(I, r, gs)
    m, u = dupdateutility!(player, I, h, σ, u, πi, πo, r, s, g, gs, depth, α, β, γ, iter, siter, discounted)
    dupdateregret!(player, I, h, σ, u, m, πi, r, s, gs, depth, α, β, γ, iter, siter, discounted)
    return u
end

function strat_dcfr(s::Vector{Vector{Float64}})
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

# function dcfr_exploit(T::Int64, g::AirDefenseGame, gs::GameSet, gos::GameOptSet)
#     ni, ni_stage, na_stage, players = gs.ni, gs.ni_stage, gs.na_stage, gs.players
#     r = [zeros(numactions(i, ni_stage, na_stage)) for i in 1:ni] # regret
#     s = [zeros(numactions(i, ni_stage, na_stage)) for i in 1:ni] # cumulative strategies
#     u = 0.0
#     Tx = T ÷ 10 # iterations for best response calculations
#     tx = 1 # index for best respoinse calculations
#     u1 = Vector{Float64}(undef, T)
#     u_br = Vector{Float64}(undef, 10)
#     println("MCCFR: starting $T iterations...")
#     # @progress for t in 1:T
#     for t in 1:T
#         for player in players
#             h0 = SVector(0, 0, 0, 0, 0, 0)
#             # u = dupdatetree!(h0, player, 1.0, 1.0, r, s, g, gs)
#             u = dupdatetree!(h0, player, 1.0, 1.0, r, s, g, gs, 1)
#             (player == 1) && (u1[t] = u)
#         end
#         if t % Tx == 0 # calculate best response
#             println("")
#             σ = strat_dcfr(s)
#             rc1, _ = real_strat(σ, gs, gos)
#             status, u_br[tx], _ = lp_best_response(gos, rc1, fixedplayer = 1)
#             tx += 1
#             status != :Optimal && @warn "Best response LP not optimal in iter $t."
#         end
#         t % 10 == 0 && print("\rIteration $t complete. Utility: $u")
#     end
#     σ = strat_dcfr(s)
#     println("...$T iterations complete.")
#     return u1, r, s, σ, -u_br
# end

function dcfr_full(g::AirDefenseGame, gs::GameSet;
        iterlimit::Int = 100_000,
        timelimit::Number = 600, tol::Float64 = 5e-5,
        α::Number = 1.5, β::Number = 0.5, γ::Number = 2, discounted::Bool = false)
    ni, ni_stage, na_stage, players = gs.ni, gs.ni_stage, gs.na_stage, gs.players
    r = [zeros(numactions(i, ni_stage, na_stage)) for i in 1:ni] # regret
    s = [zeros(numactions(i, ni_stage, na_stage)) for i in 1:ni] # cumulative strategies
    siter = zeros(Int64, ni) # infoset visit counts
    u = 0.0
    n_hist = 10 # number of intermediate strategies to record (for best responses)
    Tx = round(Int64, timelimit * 60 / n_hist) # frequency of recording history
    tx = 1 # index for sigma calculations
    iter = 1 # index for number of iter
    u1deriv = 0
    u1prev = 0
    converged = false
    u1 = Float64[]
    σ1 = Vector{Vector{Vector{Float64}}}(undef, n_hist)
    println("MCCFR: start $timelimit min or $iterlimit iter at $(now())...")
    tstart, t_hist = now(), now(), now()
    while !converged && tominute(now() - tstart) < timelimit && iter < iterlimit
        for player in players
            h0 = SVector(0, 0, 0, 0, 0, 0)
            u = dupdatetree!(h0, player, 1.0, 1.0, r, s, g, gs, 1, α, β, γ, iter, siter, discounted)
            (player == 1) && push!(u1, u)
        end
        if tosecond(now() - t_hist) > Tx # calculate sigma every Tx seconds
            σ1[tx] = strat_dcfr(s)
            tx += 1
            t_hist = now()
        end
        u1deriv = u1deriv * (1 - 1 / iter) + (u1[iter] - u1prev) * (1 / iter)
        converged = abs(u1deriv) < tol
        u1prev = u1[iter]
        iter += 1
        iter % 10 == 0 && print("\rIter: $iter")
    end
    println("")
    σ = strat_dcfr(s)
    return u1, r, s, σ, σ1, converged, iter
end
