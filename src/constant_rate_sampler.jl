struct ConstantRateParameters{T<:Real}
    Tspan::T
    μα::T
    μβ::T
    etaspriors::ETASPriors
end

function etas_sampling(nsteps, nchains, catalog::Catalog{T}, params::ConstantRateParameters{T}; threads=false, init_theta=nothing) where {T<:Real}

    b = BranchingProcess(T, length(catalog.t))
    xp_dummy = Product([Categorical(inv(i) .* ones(i)) for i = 1:length(catalog.t)]) 
    μp_dummy = Gamma(params.μα, inv(params.μβ))

    @model ETASModel(catalog, Kprior, αprior, cprior, p̃prior, xp_dummy, μp_dummy, Tspan) = begin
        # assign dummy priors for x and μ so that Turing knows they exist
        x ~ xp_dummy
        μ ~ μp_dummy
        # assign priors for ETAS parameters
        K ~ Kprior
        α ~ αprior
        c ~ cprior
        p̃ ~ p̃prior

        p = p̃ + 1

        #=
        setup latent branching process:
        - this is given by direct gibbs sampling in the conditional gibbs function below
        estimate background weight:
        - this is also given by direct gibbs sampling in another conditional gibbs function
        log-likelihood for etas parameters:
        =#
        Turing.@addlogprob! etas_log_likelihood(K, α, c, p, x, catalog, Tspan)

        return μ
    end

    function cond_x_b_c(c, b::BranchingProcess{T}, catalog::Catalog{T}) where T
        update_weights!(b, catalog, c.μ, c.K, c.α, c.c, c.p̃+1)
        return Product([Categorical(b.bnodes[i].bweights) for i = 1:length(catalog.t)])
    end
    
    cond_x(c) = cond_x_b_c(c, b, catalog)

    function cond_μ(c)
        nS₀ = count1(c.x)
        Gamma(params.μα + nS₀, inv(params.μβ+params.Tspan))
    end

    etas_model = ETASModel(catalog, 
                           params.etaspriors.Kprior, 
                           params.etaspriors.αprior, 
                           params.etaspriors.cprior, 
                           params.etaspriors.p̃prior, 
                           xp_dummy, 
                           μp_dummy, 
                           params.Tspan)

    etas_sampler = Gibbs(GibbsConditional(:μ, cond_μ),
                         GibbsConditional(:x, cond_x), 
                         HMC{Turing.ForwardDiffAD{4}}(0.01, 10, :K, :α, :c, :p̃))

    if init_theta !== nothing
        varinfo = Turing.VarInfo(etas_model);
        etas_model(varinfo, Turing.SampleFromPrior(), Turing.PriorContext(init_theta));
        init_theta_arr = varinfo[Turing.SampleFromPrior()]
        if threads
            etas_chain = sample(etas_model, etas_sampler, MCMCThreads(), nsteps, nchains, init_params = init_theta_arr)
        else
            etas_chain = mapreduce(c -> sample(etas_model, etas_sampler, nsteps), chainscat, 1:nchains, init_params = init_theta_arr)
        end
    else
        if threads
            etas_chain = sample(etas_model, etas_sampler, MCMCThreads(), nsteps, nchains)
        else
            etas_chain = mapreduce(c -> sample(etas_model, etas_sampler, nsteps), chainscat)
        end
    end

    etas_chain = set_section(etas_chain,  Dict(:parameters => [:K,:α,:c,:p̃,:μ], 
                                               :internals => [Symbol("x[$i]") for i in 1:length(catalog.t)], 
                                               :logposterior => [:lp]))
    return (etas_model, etas_chain)
end

function ipp_sampling(nsteps, nchains, catalog::Catalog{T}, params::ConstantRateParameters{T}; threads=false, init_theta=nothing) where {T<:Real}
    nev = length(catalog)
    Tspan = params.Tspan
    λ_obs = nev/Tspan
    μp_dummy = Gamma(params.μα, inv(params.μβ))


    @model IPPModel() = begin #note that even though we call this an IPP model for consistancy with more layers, it is in fact a homogenous model
        μ ~ μp_dummy
        λ_obs ~ Poisson(μ)
        return μ
    end

    ipp_model =  IPPModel()

    ipp_sampler = HMC{Turing.ForwardDiffAD{1}}(0.01, 10, :μ)

    if init_theta !== nothing
        varinfo = Turing.VarInfo(ipp_model);
        ipp_model(varinfo, Turing.SampleFromPrior(), Turing.PriorContext(init_theta));
        init_theta_arr = varinfo[Turing.SampleFromPrior()]
        if threads
            ipp_chain = sample(ipp_model, ipp_sampler, MCMCThreads(), nsteps, nchains, init_params = init_theta_arr)
        else
            ipp_chain = mapreduce(c -> sample(ipp_model, ipp_sampler, nsteps), chainscat, 1:nchains, init_params = init_theta_arr)
        end
    else
        if threads
            ipp_chain = sample(ipp_model, ipp_sampler, MCMCThreads(), nsteps, nchains)
        else
            ipp_chain = mapreduce(c -> sample(ipp_model, ipp_sampler, nsteps), chainscat, 1:nchains)
        end
    end

    ipp_chain = set_section(ipp_chain,  Dict(:parameters => [:μ], :logposterior => [:lp]))
    return (ipp_model, ipp_chain)
end