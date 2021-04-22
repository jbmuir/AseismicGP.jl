struct ETASPriors
    Kprior
    αprior
    cprior
    p̃prior
end

struct SPDELayerPriors
    μ₀prior
    lprior
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