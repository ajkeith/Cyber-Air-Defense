using Test, LinearAlgebra, Random, Statistics

dir = "I:\\My Documents\\00 AFIT\\Research\\Julia Projects\\StrategyGames\\src"
include(joinpath(dir, "cfrkuhn.jl"))

@testset "Kuhn CFR" begin
    Random.seed!(23408)
    @test actions("JQp") == ["p", "b"]
    @test infoset("KQpb") == 13
    @test player("") == 3
    @test player("JQp") == 2

    h = "KJpp"
    i = player(h)
    πi = 1.0
    πo = 1.0
    leafutility(h, i, πi, πo)

    u, r, s, σ = cfr(200_000)
    α = σ[2][2] # α = p(b | J)
    optstrategy_p = [1 - α, 1.0, 1 - 3α, 2/3, 1.0, 0.0, 1.0, 2/3, 0.0, 1.0, 2/3 - α, 0.0] # optimal p(pass) for each info set
    cfrstrategy_p = [σ[i][1] for i = 2:13] # calculated p(pass) for each infoset
    @test isapprox(cfrstrategy_p, optstrategy_p, atol = .05)
    @test mean(u) ≈ -0.055 atol = 0.01
end
#
# using Test, LinearAlgebra, Random, Statistics
#
# dir = "I:\\My Documents\\00 AFIT\\Research\\Julia Projects\\StrategyGames\\src"
# include(joinpath(dir, "cfrkuhn.jl"))
#
Random.seed!(23408)
T = 100_000
u, r, s, σ = cfr(T)
um = [sum(u[1:i]) / i for i in 1:T]
ustar = -1/18
rerr = abs(mean(u[(end - Int(T * 0.1)):end]) - ustar) / abs(ustar)

using Plots; gr()
plot(1:T, um)
plot!(1:T, fill(ustar, T))
