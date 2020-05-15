using Pkg; Pkg.activate(pwd())
using Revise, BenchmarkTools, Printf
using Random, Distances

const PASS = 1
const BET = 2
const NUM_ACTIONS = 2

mutable struct Knode
    infoset::String
    regretsum::Vector{Float64}
    strategy::Vector{Float64}
    strategysum::Vector{Float64}
end

Knode(i::String) = Knode(i, zeros(NUM_ACTIONS), fill(1 / NUM_ACTIONS, NUM_ACTIONS), zeros(NUM_ACTIONS))

function getstrategy!(n::Knode, weight::Float64)
    normalsum = 0.0
    for a = 1:NUM_ACTIONS
        n.strategy[a] = n.regretsum[a] > 0 ? n.regretsum[a] : 0.0
        normalsum += n.strategy[a]
    end
    for a = 1:NUM_ACTIONS
        if normalsum > 0
            n.strategy[a] /= normalsum
        else
            n.strategy[a] = 1.0 / NUM_ACTIONS
        end
        n.strategysum[a] += weight * n.strategy[a]
    end
    n.strategy
end

function getstrategy_average(n::Knode)
    avgstrategy = Vector{Float64}(undef, NUM_ACTIONS)
    normalsum = 0.0
    for a = 1:NUM_ACTIONS
        normalsum += n.strategysum[a]
    end
    for a = 1:NUM_ACTIONS
        if normalsum > 0
            avgstrategy[a] = n.strategysum[a] / normalsum
        else
            avgstrategy[a] = 1.0 / NUM_ACTIONS
        end
    end
    avgstrategy
end

function infostring(n::Knode)
    @sprintf("%4s: %s", n.infoset, string(getstrategy_average(n)))
end

function nextnode!(nodedict::Dict{String, Knode}, infoset::String)
    if haskey(nodedict, infoset)
        n = nodedict[infoset]
    else
        n = Knode(infoset)
        nodedict[infoset] = n
    end
    n
end


function cfr!(cards::Vector{Int64}, h::String, p1::Float64, p2::Float64, nodedict::Dict{String,Knode})
    plays = length(h)
    player = rem(plays, 2) + 1
    opponent = 2 - player + 1
    if plays > 1
        terminal = (h[plays] == 'p')
        doublebet = h[(plays - 1):plays] == "bb"
        higher = cards[player] > cards[opponent]
        if terminal
            if h == "pp"
                payoff = higher ? 1 : -1
                # println("h: $h ", cards[1:2], " Player $player gets $payoff")
                return payoff
            else
                payoff = 1
                # println("h: $h ", cards[1:2], " Player $player gets $payoff")
                return payoff
            end
        elseif doublebet
            payoff = higher ? 2 : -2
            # println("h: $h ", cards[1:2], " Player $player gets $payoff")
            return payoff
        end
    end
    infoset = string(cards[player], h)
    n = nextnode!(nodedict, infoset)
    strategy = getstrategy!(n, player == 1 ? p1 : p2)
    # println("regretsum a: ", n.infoset, n.regretsum)
    util = Vector{Float64}(undef, NUM_ACTIONS)
    nodeutil = 0.0
    for a = 1:NUM_ACTIONS # recur cfr
        hnext = string(h, a == 1 ? "p" : "b")
        if player == 1
            util[a] = -1 * cfr!(cards, hnext, p1 * strategy[a], p2, nodedict)
        else
            util[a] = -1 * cfr!(cards, hnext, p1, p2 * strategy[a], nodedict)
        end
        nodeutil += strategy[a] * util[a]
    end
    for a = 1:NUM_ACTIONS # regret
        temp = util[a]
        # println(n.infoset, " util = $temp, nodeutil = $nodeutil")
        regret = util[a] - nodeutil
        n.regretsum[a] += (player == 1 ? p1 : p2) * regret
        # @show strategy[a]
    end
    # println("regretsum b: ", n.infoset, n.regretsum)
    return nodeutil
end

function train(iter::Int64)
    println("STARTING...")
    cards = [1, 2, 3]
    util = 0.0
    nodedict = Dict{String, Knode}()
    vals = Vector{Float64}(undef, iter)
    for i = 1:iter
        print("\rITER $i")
        shuffle!(cards)
        vals[i] = cfr!(cards, "", 1.0, 1.0, nodedict)
        util += vals[i]
    end
    println("Average game value: ", util / iter)
    for n in values(nodedict)
        println(infostring(n))
    end
end

function kuhncfr(reps::Int64)
    Random.seed!(23894719)
    train(reps)
end


@btime kuhncfr(10_000)
