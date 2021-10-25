struct ETASPriors
    Kprior
    αprior
    cprior
    p̃prior
end

struct ScalarSPDELayerPriors
    μprior
    lprior
    σprior
end

struct VectorSPDELayerPriors
   μprior
   σprior
end

function hatsum(fx::Vector{T}) where T
    s = zero(T)
    s += fx[1] / 2
    for i = 2:(length(fx)-1)
        s += fx[i]
    end
    s += fx[end] / 2
    return s
end

function hatinterpolationmatrix(x::AbstractRange, xi::Vector{T}) where T
    I = Int[]
    J = Int[]
    V = T[]
    for i = 1:length(xi)
        if xi[i] == x[1]
            push!(I, i)
            push!(J, 1)
            push!(V, one(T))
        elseif xi[i] == x[end]
            push!(I, i)
            push!(J, length(x))
            push!(V, one(T))
        else
            iu = findfirst(x.>=xi[i]) 
            il = iu - 1
            ru = (xi[i]-x[il]) / (x[iu]-x[il])
            ri = one(T) - ru
            push!(I, i, i)
            push!(J, il, iu)
            push!(V, ri, ru)
        end
    end
    return sparse(I, J, V, length(xi), length(x))
end

function count1(x)
    s = 0
    for xx in x
        if xx == 1
            s += 1
        end
    end
    return s
end

function logsum(μ::Vector{T}) where T
    s = zero(T)
    @turbo for i ∈ eachindex(μ)
        s += log(μ[i])
    end
    return s
end

function logsum1(μ::Vector{T}, x) where T
    s = zero(T)
    n = length(μ)
    @inbounds for i ∈ 1:n
        s += (x[i] == 1 ? log(μ[i]) : zero(T))
    end
    return s
end

function smoothclamp(x, low, high)
    r = high - low
    x = clamp((x-low) / r, 0, 1)
    xi = x * x * (3 - 2 * x)
    return xi * r + low
end

sigmoid(x) = 1/(1 + exp(-x))
isigmoid(x) = log(x/(1-x))

function clamp_w_scale(τ, lrange, lmin, lscale)
	lfactor = isigmoid((lscale-lmin)/(lmax-lmin))
	l = lrange * sigmoid(τ+lfactor) + lmin 
	return l
end

function sigmoid_clamp(τ, lrange, lmin)
	l = lrange * sigmoid(τ) + lmin 
	return l
end

function etas_log_likelihood(K::T, α::T, c::T, p::T, x, catalog, Tspan) where {T<:Real}
    cx = counts(x, length(x)+1)
    nS = @views cx[2:end]
    etasloglik = zero(T)
    for i in 1:length(catalog)
        j = x[i]-1
        κm = κ(catalog.ΔM[i], K, α)
        etasloglik -= κm*H(Tspan, catalog.t[i], c, p) 
        etasloglik += nS[i]*log(κm) 
        if j != 0 # don't add this next term for events with the background as the parent
            etasloglik += log(h(catalog.t[i], catalog.t[j], c, p))
        end
    end
    return etasloglik
end
