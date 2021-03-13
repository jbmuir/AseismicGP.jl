using OnlineStats
using Distributions

function mh_step!(ar::S, p::Vector{T}, proposal, logp; β = one(T), is_symmetric_proposal=false) where {T<:Real, S<:OnlineStat}
    q = p .+ β.*rand(proposal) # we will use a very simple tuning scheme with only one parameter...
    if is_symmetric_proposal
        mh = logp(q) - logp(p)
    else
        mh = logp(q) + logpdf(proposal, p.-q) - logp(p) - logpdf(proposal, q.-p)
    end
    if log(rand(T)) < mh
        fit!(ar, 1)
        return q
    else
        fit!(ar, 0)
        return p
    end
end

#This is a simple proportional controller for β
function tune_β(β::T, ar::S, ar_target::T, η::T, lb::T, ub::T) where {T<:Real, S<:OnlineStat}
	@assert lb <= β <= ub
    @assert 0 <= η <= 1 # so that 0<=s<=1
	δa = value(ar) - ar_target
	s = δa * η # +ve step = acceptance too high 
	if s >= 0 
		return β + s * (ub - β) # increase step length (+ * +) to reduce acceptance
	else 
		return β + s * (β - lb) # decrease step length (- * *) to increase acceptance
	end
end

function draw_constant_μ_posterior(α::T, β::T, nS₀::Int, Tspan::T) where {T<:Real}
    posterior = Gamma(α + nS₀, 1 / (β+Tspan))
    rand(posterior)
end

function gamma_moment_tuner(μ, σ)
    # gives parameters for a Gamma with given mean and standard deviation
    a = (μ/σ)^2
    b = μ/σ^2
    θ = 1/b
    return (a, b, θ)
end