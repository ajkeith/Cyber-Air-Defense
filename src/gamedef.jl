######## Game Size ########
using Pkg; Pkg.activate("I:\\My Documents\\00 AFIT\\Research\\Julia Projects\\StrategyGames")
using Revise, BenchmarkTools
using Printf

######################################
# Non-fixed IADS

# function psize_full(ncity, ndp, nap, ndc, nac)
#     npatch = 2
#     nexploit = 2
#     nn = binomial(ncity, ndp) * npatch * nexploit * binomial(ndp, nac) *
#         2^nac * binomial(ndp, ndc) * 2^nac * binomial(ncity, nap)
#     @sprintf("%.2E", nn)
# end
#
# function psize_rep(ncity, ndp, nap, ndc, nac)
#     npatch = 2
#     nexploit = 2
#     nn = 1 * npatch * nexploit * binomial(ndp, nac) *
#         2^nac * binomial(ndp, ndc) * 2^nac * 1
#     @sprintf("%.2E", nn)
# end
#
# function gsize(ncity, ndp, nap, ndc, nac)
#     nsig1 = binomial(ncity, ndp) * binomial(ndp, ndc)
#     nsig2 = binomial(ndp, nac) * binomial(ncity, nap)
#     ng = nsig1 * nsig2
#     @sprintf("%.2E", ng)
# end
#
# function nI(ncity, ndp, nap, ndc, nac)
#     nI11 = 1
#     nI13 = nI11 * binomial(ncity, ndp) * 2 * 2^nac
#     nI1 = @sprintf("%.2E", nI11 + nI13)
#     nI22 = binomial(ncity, ndp) * 2
#     nI24 = nI22 * binomial(ndp, nac) * binomial(ndp, ndc)
#     nI2 = @sprintf("%.2E", nI22 + nI24)
#     (nI1, nI2)
# end
#
# function nΣ(ncity, ndp, nap, ndc, nac)
#     nΣ11 = binomial(ncity, ndp)
#     nΣ13 = binomial(ndp, ndc)
#     nΣ1 = @sprintf("%.2E", nΣ11 + nΣ11 * nΣ13)
#     nΣ22 = binomial(ndp, nac)
#     nΣ24 = binomial(ncity, nap)
#     nΣ2 = @sprintf("%.2E", nΣ22 + nΣ22 * nΣ24)
#     (nΣ1, nΣ2)
# end
#
# # small size
# ncity = 6
# ndp = 4
# nap = 3
# ndc = 2
# nac = 2
# psize_full(ncity, ndp, nap, ndc, nac) # 7E+05
# psize_rep(ncity, ndp, nap, ndc, nac) # 2E+03
# gsize(ncity, ndp, nap, ndc, nac)
# nI(ncity, ndp, nap, ndc, nac)
# nΣ(ncity, ndp, nap, ndc, nac)
#
# # large size
# ncity = 25
# ndp = 4
# nap = 3
# ndc = 2
# nac = 2
# psize_full(ncity, ndp, nap, ndc, nac)
# psize_rep(ncity, ndp, nap, ndc, nac)
# gsize(ncity, ndp, nap, ndc, nac)
# nI(ncity, ndp, nap, ndc, nac)
# nΣ(ncity, ndp, nap, ndc, nac)
#
# ncity = 35
# nnd = binomial(ncity, 1) + binomial(ncity, 2) + binomial(ncity, 3)
# nnt = nnd * binomial(ncity, 5)
# @sprintf("%.2E", nnt)
#
# # Calculate coverage matrix
# using Random, Distances
# n = 55
# r = 0.3
# x = rand(2, n)
# dist = pairwise(Euclidean(), x)
# C = [dist[i,j] < r ? 1 : 0 for i = 1:n, j = 1:n]

######################################

# Fixed defense physical assets

function nnodes_approx(ncity, ndp, nap, ndpdc, ndpac, ndc, nac)
    c1d = binomial(ndp, ndpdc)
    c1a = binomial(ndp, ndpac)
    c1 = binomial(ndp, ndpdc) * binomial(ndp, ndpac)
    a2 = binomial(ndpac, nac)
    c3 = 2 ^ ndp
    d4 = binomial(ndpdc, ndc)
    a5 = binomial(ncity, nac)
    napprox = c1 * a2 * c3 * d4 * a5
    @sprintf("%.2E", napprox) # number of naive nodes
end

function nI_approx(ncity, ndp, nap, ndpdc, ndpac, ndc, nac)
    nIc1 = 1
    nIa2 = binomial(ndp, ndpac)
    nIa5 = nIa2 * binomial(ndp, nac) * binomial(ndp, ndc)
    nIc3 = 1
    nId4 = binomial(ndp, ndpdc) * (2 ^ ndp)
    nIa = @sprintf("%.2E", nIa2 + nIa5)
    nId = @sprintf("%.2E", nId4)
    (nIa, nId)
end

ncity = 10
ndp = 5
nap = 3
ndpdc = 4
ndpac = 4
ndc = 2
nac = 2

nnodes_approx(ncity, ndp, nap, ndpdc, ndpac, ndc, nac)
nI_approx(ncity, ndp, nap, ndpdc, ndpac, ndc, nac)
