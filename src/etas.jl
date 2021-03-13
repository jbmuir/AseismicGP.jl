using Distributions

# Δmᵢ is mᵢ - M₀; i.e. relative to cutoff magnitude choice
κ(Δmᵢ, K, α) = K*exp(α*Δmᵢ)
# p > 1! 
h(t, tⱼ, c, p) = (p-1)*c^(p-1) / (t-tⱼ+c)^p
H(t, tⱼ, c, p) = 1 - c^(p-1) / (t-tⱼ+c)^(p-1)

function Kα_log_likelihood(K, α, c, p, nS, catalog, Tspan)
    if (K < 0) || (α < 0)
        return -Inf
    else
        κvec = κ.(catalog.ΔM, K, α)
        pvec = -κvec.*H.(Tspan,catalog.t,c,p)
        pvec .+= nS.*log.(κvec)
        return sum(pvec)
    end
end

function cp_log_likelihood(c, p, x, K, α, catalog, Tspan)
    if (c < 0) || (p < 1)
        return -Inf
    else
        κvec = κ.(catalog.ΔM, K, α)
        pvec = -κvec.*H.(Tspan,catalog.t,c,p)
        for i in eachindex(x)
            j = x[i]
            pvec[j] += log(h(catalog.t[i], catalog.t[j], c, p))
        end
        return sum(pvec)
    end
end

function etas_log_likelihood(K, α, c, p, x, nS, catalog, Tspan)
    if (K < 0) || (α < 0) || (c < 0) || (p < 1)
        return -Inf
    else
        κvec = κ.(catalog.ΔM, K, α)
        pvec = -κvec.*H.(Tspan,catalog.t,c,p)
        pvec .+= nS.*log.(κvec)
        for i in eachindex(x)
            j = x[i]
            pvec[j] += log(h(catalog.t[i], catalog.t[j], c, p))
        end
        return sum(pvec)
    end
end