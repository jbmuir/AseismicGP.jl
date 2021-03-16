module AseismicGP

export TuringConstantRateParameters
export etas_turing

using StatsBase
using OnlineStats
using MCMCChains
using Turing

include("catalog.jl")
include("etas.jl")
include("branching_process.jl")

abstract type RateParameters end

struct TuringConstantRateParameters{T<:Real} <: RateParameters
    Tspan::T
    μα
    μβ
    Kprior
    αprior
    cprior
    pprior
end

function etas_turing(ntune, nsteps, nchains, catalog::Catalog{T}, params::TuringConstantRateParameters{T}; accept=0.65) where {T<:Real}

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
                         NUTS(ntune, accept, :K, :α, :c, :p))
                         
    
    #etas_chain = mapreduce(c -> sample(etas_model, etas_sampler, nsteps), chainscat, 1:nchains)
    etas_chain = sample(etas_model, etas_sampler, MCMCThreads(), nsteps, nchains)
    etas_chain = set_section(etas_chain,  Dict(:internals => [Symbol("x[$i]") for i in 1:length(catalog.t)]))
    return etas_chain

end

end