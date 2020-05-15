using Pkg; Pkg.activate(pwd())
using Test, LinearAlgebra, Random, JLD2

include(joinpath(pwd(), "src\\ops_utility.jl"))
include(joinpath(pwd(), "src\\ops_build.jl"))
include(joinpath(pwd(), "src\\ops_methods.jl"))
include(joinpath(pwd(), "src\\cfrops_solve.jl"))

@testset "Build Ops Game" begin
    # Build game and infosets Random.seed!(579843)
    g = AirDefenseGame()
    A, An, na_stage = build_action(g)
    ni, ni1, ni2, ni_stage = build_info(g)
    ns1, ns2, ns1_stage, ns2_stage = build_nseq(g, na_stage, ni_stage)
    Σ, seqn, seqactions = build_Σset(g, A, ns1, ns2)
    Iset, (I1set, I2set), (seqI1, seqI2), (nextseqI1, nextseqI2) = build_Iset(g, A, na_stage, ns1_stage, ns2_stage, ni1, ni2, ns1, ns2)
    @test length(Iset) == 1695
    gs = GameSet(g)
    # @test seqI2[631] == 7 # sequence [3,4] => [5,8] for Attacker: targetable [1, 4, 5, 8], cyberattack [5, 8], cyberdefend [1, 3]

    # History to infoset
    ndp, nac, ndc = g.ndp, g.nac, g.ndc
    id1 = infoset(SVector(2, 2, 2, 4, 0, 0), 5, g, gs)
    @test Iset[id1] == "Defender: defendable [1, 3, 4, 8], sensed [1, 4]"
    id2 = infoset(SVector(15, 3, 5, 4, 0, 0), 5, g, gs)
    @test Iset[id2] == "Defender: defendable [4, 5, 8, 9], sensed [3, 9]"
    id3 = infoset(SVector(10, 4, 6, 4, 1, 0), 6, g, gs)
    @test Iset[id3] == "Attacker: targetable [1, 3, 5, 8], cyberattack [5, 8], cyberdefend [1, 5]"

    # chance distributions: stage 1
    An = gs.An
    h0 = SVector(0, 0, 0, 0, 0, 0)
    @test all(isapprox(sum(chance(1, a, g, gs) for a in An[1]), 1.0, atol = 0.00001))
    @test all(chance(1, a, g, gs) >= 0.0 for a in An[1])
    # chance distributions: stage 2
    @test all(isapprox(sum(chance(2, a, g, gs) for a in An[2]), 1.0, atol = 0.00001))
    @test all(chance(2, a, g, gs) >= 0.0 for a in An[2], h1 in An[1])
    # chance distributions: stage 4
    @test all(isapprox(sum(chance(4, a, g, gs) for a in An[4]), 1.0, atol = 0.00001))
    @test all(chance(4, a, g, gs) >= 0.0 for a in An[4])
end

@testset "Build Ops Utility Tests (Loads Local Files)" begin
    # Build utility
    g = AirDefenseGame()
    gs = GameSet(g)
    A, An, na_stage = build_action(g)
    ni, ni1, ni2, ni_stage = build_info(g)
    ns1, ns2, ns1_stage, ns2_stage = build_nseq(g, na_stage, ni_stage)
    Σ, seqn, seqactions = build_Σset(g, A, ns1, ns2)
    # U, z = build_utility_hist(g, A, An)
    # reward = build_utility_seq(g, gs, seqn, seqactions) # takes an 1 hr to build...
    dir = "I:\\My Documents\\00 AFIT\\Research\\Julia Projects\\StrategyGames"
    fn = joinpath(dir, "data\\vars_opt_rewardfunction.jld2")
    @load fn U z reward
    @test length(U) == 3_888_000
    @test length(U) == nnz(reward)

    ind1 = 43
    ind2 = 62491
    h = [1, 6, 5, 1, 6, 120] # h associated with seq1, seq2
    indu = findfirst(x -> x == h, z)
    @test U[indu] == reward[ind1, ind2]
end

@testset "Ops CFR" begin
    # Build game and infosets Random.seed!(579843)
    g = AirDefenseGame()
    gs = GameSet(g)

    # check that strategy builds (still need correctness test)
    T = 100
    u, r, s, σ = cfr(T, g, gs)
    @test all(all(isnan.(σi)) || all(σi .>= 0.0) for σi in σ)
    @test all(all(isnan.(σi)) || isapprox(sum(σi), 1.0, atol = 0.001) for σi in σ)

    gtoy = AirDefenseGame(2, [1.0 1;2 1], [1.0, 1], 0.2, 2, 1, 2, 2, 1, 1, 0.9, 0.5, 0.7, 0.8, [1,2])
    gstoy = GameSet(gtoy)
    T = 15_000
    (utoy, r, s, σtoy), t_cfr = @timed cfr(T, gtoy, gstoy)
    @test isapprox(σtoy[infoset(SVector(1,1,1,1,0,0), 5, gtoy, gstoy)][1], 0.5, atol = 0.1)
    @test isapprox(σtoy[infoset(SVector(1,1,1,2,0,0), 5, gtoy, gstoy)][1], 1.0, atol = 0.001)
end
