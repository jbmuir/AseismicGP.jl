module AseismicGP

export ConstantRateParameters
export etas_chain
export gamma_moment_tuner
export TuringConstantRateParameters
export etas_turing

using StatsBase
using OnlineStats
using MCMCChains
using Turing

include("catalog.jl")
include("etas.jl")
include("branching_process.jl")
include("samplers.jl")

abstract type RateParameters end

struct ConstantRateParameters{T<:Real} <: RateParameters
    Tspan::T
    α::T
    β::T
    Kαprior
    cpprior
    Kαproposal
    cpproposal
end

function update_state!(b::BranchingProcess{T}, state, tuning, catalog::Catalog{T}, params::ConstantRateParameters{T}) where {T<:Real}
    x, μ, Kα, cp = state
    β1, ar1, β2, ar2 = tuning
    # step 1 of Ross algorithm (constant μ)
    update_weights!(b, catalog, t->μ, Kα[1], Kα[2], cp[1], cp[2])
    xn = sample(b)
    cxn = counts(xn, length(xn)+1)
    # step 2 of Ross algorithm (constant μ)
    nS₀ = cxn[1]
    μn = draw_constant_μ_posterior(params.α, params.β, nS₀, params.Tspan)
    # step 3 of Ross algorithm
    nS = @views cxn[2:end]
    Kαn = mh_step!(ar1, Kα, params.Kαproposal, (Kαp -> Kα_log_likelihood(Kαp[1], Kαp[2], cp[1], cp[2], nS, catalog, params.Tspan) + logpdf(params.Kαprior, Kαp)); β = β1, is_symmetric_proposal=true)
    #β1 = tune_β(β1, ar1, 0.67, 0.2, 0.0, 1.0)
    # step 4 of Ross algorithm
    cpn = mh_step!(ar2, cp, params.cpproposal, (cpp -> cp_log_likelihood(cpp[1], cpp[2], x, Kαn[1], Kαn[2], catalog, params.Tspan) + logpdf(params.cpprior, cpp)); β = β2, is_symmetric_proposal=true)
    #β2 = tune_β(β2, ar2, 0.67, 0.2, 0.0, 1.0)

    return ((xn, μn, Kαn, cpn), (β1, ar1, β2, ar2))

end

function etas_chain(nsteps, catalog::Catalog{T}, params::S) where {T<:Real, S<:RateParameters}
    #initialization
    b = BranchingProcess(T, length(catalog.t))
    x = ones(Int, length(catalog.t))
    μ = rand(Gamma(params.α, 1 / params.β))
    Kα = rand(params.Kαprior)
    cp = rand(params.cpprior)
    state = (x, μ, Kα, cp)
    β1 = one(T)
    ar1 = Mean()
    β2 = one(T)
    ar2 = Mean()
    tuning = (β1, ar1, β2, ar2)
    
    xc = zeros(Int, (length(x), nsteps))
    etasc = zeros(T, (5,nsteps))

    for i = 1:nsteps
        state, tuning = update_state!(b, state, tuning, catalog, params)
        x, μ, Kα, cp = state
        xc[:,i] .= x
        etasc[1,i] = μ
        etasc[2,i] = Kα[1]
        etasc[3,i] = Kα[2]
        etasc[4,i] = cp[1]
        etasc[5,i] = cp[2]
    end
    return (Chains(etasc', [:μ,:K,:α,:c,:p]), Chains(xc'), tuning)
end

struct TuringConstantRateParameters{T<:Real} <: RateParameters
    Tspan::T
    μα
    μβ
    Kprior
    αprior
    cprior
    pprior
end

function etas_turing(nsteps, nchains, catalog::Catalog{T}, params::TuringConstantRateParameters{T}) where {T<:Real}

    b = BranchingProcess(T, length(catalog.t))

    function count1(x)
        s = 0
        for xx in x
            if xx == 1
                s+=1
            end
        end
        return s
    end

    @model ETASModel(catalog) = begin
        # assign dummy priors for x and μ so that Turing knows they exist
        x ~ Product([Categorical(inv(i) .* ones(i)) for i = 1:length(catalog.t)]) 
        μ ~ Gamma(params.μα, inv(params.μβ))
        # assign priors for ETAS parameters
        K ~ params.Kprior
        α ~ params.αprior
        c ~ params.cprior
        p ~ params.pprior

        # setup latent branching process
        
        # this is given by direct gibbs sampling in the conditional gibbs function below

        #estimate background weight
 
        # this is also given by direct gibbs sampling in another conditional gibbs function

        # log-likelihood for etas parameters
        cx = counts(x, length(x)+1)
        nS = @views cx[2:end]
        Turing.@addlogprob! etas_log_likelihood(K, α, c, p, x, nS, catalog, params.Tspan)
    end

    function cond_x_b_c(c, b::BranchingProcess{T}, catalog::Catalog{T}) where T
        update_weights!(b, catalog, t->c.μ, c.K, c.α, c.c, c.p)
        return Product([Categorical(b.bnodes[i].bweights) for i = 1:length(catalog.t)])
    end
    
    cond_x(c) = cond_x_b_c(c, b, catalog)

    function cond_μ(c)
        nS₀ = count1(c.x)
        Gamma(params.μα + nS₀, inv(params.μβ+params.Tspan))
    end

    etas_model = ETASModel(catalog)
    etas_sampler = Gibbs(GibbsConditional(:x, cond_x), 
                         GibbsConditional(:μ, cond_μ),
                         NUTS(1000, 0.65, :K, :α, :c, :p))
                         
    
    #etas_chain = mapreduce(c -> sample(etas_model, etas_sampler, nsteps), chainscat, 1:nchains)
    etas_chain = sample(etas_model, etas_sampler, MCMCThreads(), nsteps, nchains)
    etas_chain = set_section(etas_chain,  Dict(:internals => [Symbol("x[$i]") for i in 1:length(catalog.t)]))
    return etas_chain

end

end