using StatsBase
import StatsBase: sample
using Random
using Random: AbstractRNG

struct BranchingNode{T}
    j::Int # event number
    bweights::Vector{T}
    BranchingNode{T}(j, bweights) where {T} = length(bweights) == j ? new(j, bweights) : error("Weights vector must be equal in length to number of events")
end

BranchingNode(j, bweights::Vector{T}) where {T} = BranchingNode{T}(j, bweights)
BranchingNode(::Type{T}, j) where {T} = BranchingNode(j, ones(T, j))

function update_weights!(b::BranchingNode{T}, catalog::Catalog{T}, μ::Function, K, α, c, p) where {T}
    b.bweights[1] = μ(catalog.t[b.j]) # probability of being a background event
    @. b.bweights[2:end] = κ(catalog.ΔM[1:(b.j-1)], K, α)*h(catalog.t[b.j], catalog.t[1:(b.j-1)], c, p)
    b.bweights ./= sum(b.bweights) # statsbase automatically normalizes so lets not do this
end

function StatsBase.sample(rng::AbstractRNG, b::BranchingNode)
    wv = aweights(b.bweights)
    return sample(rng, wv) 
end

StatsBase.sample(b::BranchingNode) = sample(Random.GLOBAL_RNG, b)

struct BranchingProcess{T}
    n::Int # total events
    bnodes::Vector{BranchingNode{T}}
    BranchingProcess{T}(n, bnodes) where {T} = length(bnodes) == n ? new(n, bnodes) : error("Branching Nodes vector must be equal in length to the number of events")
end

BranchingProcess(n, bnodes::Vector{BranchingNode{T}}) where {T} = BranchingProcess{T}(n, bnodes)

BranchingProcess(::Type{T}, n) where {T} = BranchingProcess(n, [BranchingNode(T, j) for j = 1:n])

function update_weights!(b::BranchingProcess{T}, catalog::Catalog{T},  μ::Function, K, α, c, p) where {T}
    for i = 1:b.n
        update_weights!(b.bnodes[i], catalog, μ, K, α, c, p)
    end
end

function StatsBase.sample!(rng::AbstractRNG, b::BranchingProcess, x::Vector{Int})
    for i = 1:b.n
        x[i] = sample(rng, b.bnodes[i])
    end
end

StatsBase.sample!(b::BranchingProcess, x::Vector{Int}) = sample!(Random.GLOBAL_RNG, b, x)

function StatsBase.sample(rng::AbstractRNG, b::BranchingProcess)
    x = zeros(Int, b.n)
    sample!(rng, b, x)
    return x
end

StatsBase.sample(b::BranchingProcess) = sample(Random.GLOBAL_RNG, b)