using Distributions: MersenneTwister, Exponential
using Dates: DateTime
import Base: length, maximum, minimum

export AbstractCatalog, Catalog, NormalizedCatalog
export ETASInhomogeneousPP
export simulate_ETAS

abstract type AbstractCatalog end

Base.length(cat :: AbstractCatalog) = Base.length(cat.t)
Base.maximum(cat :: AbstractCatalog) = Base.maximum(cat.M)
Base.minimum(cat :: AbstractCatalog) = Base.minimum(cat.M)

struct Catalog{T<:Real} <: AbstractCatalog
    t::Vector{T}
    M::Vector{T}
    ΔM::Vector{T}
    M₀::T
    start_time::Union{DateTime, Missing}
    elapsed_time::T
    # Δt::Vector{T}
    # ΔT::Matrix{T}
    # ΔTf::Matrix{T}
    # function Catalog{T}(t, M, M₀, start_time, elapsed_time) where {T<:Real}
    #     @assert length(t) == length(M)
    #     Δt = vcat(t[2:end] .- t[1:end-1], elapsed_time - t[end])
    #     ΔT = UpperTriangular([t[i]-t[j] for i=1:length(t), j=1:length(t)])
    #     ΔTf = UpperTriangular([i==j ? zero(T) : one(T) for i=1:length(t), j=1:length(t)])
    #     return new(t, M, M.-M₀, M₀, start_time, elapsed_time, Δt, ΔT, ΔTf)
    # end
    function Catalog{T}(t, M, M₀, start_time, elapsed_time) where {T<:Real}
        @assert length(t) == length(M)
        return new(t, M, M.-M₀, M₀, start_time, elapsed_time)
    end
end

Catalog(t::Vector{T}, M::Vector{T}, M₀::T, start_time::Union{DateTime, Missing}, elapsed_time::T) where {T<:Real} = Catalog{T}(t, M, M₀, start_time, elapsed_time)

struct NormalizedCatalog{T<:Real} <: AbstractCatalog
    t::Vector{T}
    M::Vector{T}
    function NormalizedCatalog{T}(t, M) where {T<:Real}
        @assert length(t) == length(M)
        return new(t, M)
    end
end

NormalizedCatalog(t::Vector{T}, M::Vector{T}) where {T<:Real} = NormalizedCatalog{T}(t, M)

function normalize(cat::Catalog)
    #sort catalog and normalize times to the interval [0,1]
    t = cat.t
    M = cat.M
    tsp = sortperm(t)
    tn = maximum(t) - minimum(t)
    tnorm = (t .- minimum(t)) ./ tn
    return NormalizedCatalog(tnorm[tsp], M[tsp])
end

struct ETASInhomogeneousPP{T <: Real}
    μ :: Function
    K::T
    α::T
    p::T
    c::T
end

function simulate_ETAS(model::ETASInhomogeneousPP{T}, t_end::T, b::T, rng = MersenneTwister(43771120)) where {T <: Real}
    # Simulates an ETAS IPP catalog using Ogata's thinning algorithm
    t = Array{T,1}()
    M = Array{T,1}() 
    t_now = zero(T)
    β = b * convert(T, log(10))
    scale = one(T) / β
    μ = model.μ
    K = model.K
    α = model.α
    c = model.c
    p = model.p
    while t_now < t_end
        if length(t) > 0
            ν = K * sum(exp.(α.*M) .* (t_now + c .- t).^-p) 
        else
            ν = zero(T)
        end
        λ̄ = μ(t_now) + ν
        u = rand(rng, T, 1)[1]
        w = -log(u)/λ̄
        t_now = t_now  + w
        D = rand(rng, T, 1)[1]
        if length(t) > 0
            ν = K * sum(exp.(α.*M) .* (t_now + c .- t).^-p) 
        else
            ν = zero(T)
        end
        λ = μ(t_now) + ν 
        if D*λ̄ <= λ
            if t_now > t_end
                break
            else
                push!(M, rand(rng, Exponential(scale)))
                push!(t, t_now)
            end
        end
    end
    return Catalog(t, M, zero(T), missing, t_end) 
end