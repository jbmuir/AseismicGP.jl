struct OneLayerRateParameters{T<:Real}
    Tspan::T
    M::MaternSPDE{T}
    spdepriors::ScalarSPDELayerPriors
    etaspriors::ETASPriors
end

function OneLayerRateParameters(Tspan::T, 
                                N::Integer, 
                                spdepriors::ScalarSPDELayerPriors,       
                                etaspriors::ETASPriors) where T
    @assert N >= 2
    d = 1 #only one time dimension
    h = Tspan / (N-1)
    D = Tridiagonal(vcat(ones(T,N-2),2), -2*ones(T,N), vcat(2,ones(T,N-2))) ./ h^2
    M = MaternSPDE(d, N, h, D, Diagonal(ones(T, N)))
    return OneLayerRateParameters(Tspan, M, spdepriors, etaspriors)
end

function etas_sampling(nsteps, nchains, catalog::Catalog{T}, params::OneLayerRateParameters{T}; threads=false, init_theta=nothing) where {T<:Real}

    ETASPriors = params.etaspriors
    L₁priors = params.spdepriors
    Tspan = params.Tspan
    M = params.M
    b = BranchingProcess(T, length(catalog.t))
    xp_dummy = Product([Categorical(inv(i) .* ones(i)) for i = 1:length(catalog.t)]) 
    S = hatinterpolationmatrix(zero(T):params.M.h:params.Tspan, catalog.t)
    wprior = MvNormal(params.M.N, one(T))
    lmin = params.M.h
    lmax = params.Tspan

    @model ETASModel() = begin
        # assign dummy priors for x so that Turing knows it exists
        x ~ xp_dummy
        #assign priors for the SPDE layer
        μ₁ ~ L₁priors.μprior
        l₁ ~ L₁priors.lprior
        σ₁ ~ L₁priors.σprior
        w₁ ~ wprior
        # assign priors for ETAS parameters
        K ~ ETASPriors.Kprior
        α ~ ETASPriors.αprior
        c ~ ETASPriors.cprior
        p̃ ~ ETASPriors.p̃prior
        p = p̃ + 1

        #=         
        estimate background weight:
        =#

        μ = μ₁ .* exp.(M(smoothclamp(l₁, lmin, lmax), σ₁, w₁))
        μi = S*μ
        Turing.@addlogprob! logsum1(μi,x) - hatsum(μ)*params.M.h

        #=
        setup latent branching process:
        - this is given by direct gibbs sampling in the conditional gibbs function below
        =#

        #=
        log-likelihood for etas parameters:
        =#

        Turing.@addlogprob! etas_log_likelihood(K, α, c, p, x, catalog, Tspan)

        return μ
    end

    function cond_x(c)
        μ = c.μ₁ .* exp.(M(smoothclamp(c.l₁, lmin, lmax), c.σ₁, c.w₁))
        μi = S*μ
        update_weights!(b, catalog, μi, c.K, c.α, c.c, c.p̃+1)
        return Product([Categorical(b.bnodes[i].bweights) for i = 1:length(catalog.t)])
    end
    
    etas_model =  ETASModel()

    etas_sampler = Gibbs(Turing.ESS(:w₁),
                         HMC{Turing.ForwardDiffAD{3}}(0.01, 10, :μ₁, :l₁, :σ₁),
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

    etas_chain = set_section(etas_chain,  Dict(:parameters => [:K,:α,:c,:p̃,:μ₁,:l₁,:σ₁],
                                               :spdelatent => [Symbol("w₁[$i]") for i in 1:params.M.N], 
                                               :internals => [Symbol("x[$i]") for i in 1:length(catalog.t)], 
                                               :logposterior => [:lp]))
    return (etas_model, etas_chain)
end


function ipp_sampling(nsteps, nchains, catalog::Catalog{T}, params::OneLayerRateParameters{T}; threads=false, init_theta=nothing) where {T<:Real}

    L₁priors = params.spdepriors
    M = params.M
    S = hatinterpolationmatrix(zero(T):params.M.h:params.Tspan, catalog.t)
    wprior = MvNormal(params.M.N, one(T))
    lmin = params.M.h
    lmax = params.Tspan

    @model IPPModel() = begin

        #assign priors for the SPDE layer
        μ₁ ~ L₁priors.μprior
        l₁ ~ L₁priors.lprior
        σ₁ ~ L₁priors.σprior
        w₁ ~ wprior

        μ = μ₁ .* exp.(M(smoothclamp(l₁, lmin, lmax), σ₁, w₁))
        μi = S*μ
        Turing.@addlogprob! logsum(μi) - hatsum(μ)*params.M.h

        return μ
    end

    ipp_model =  IPPModel()

    ipp_sampler = Gibbs(ESS(:w₁), HMC{Turing.ForwardDiffAD{3}}(0.01, 10, :μ₁, :l₁, :σ₁))

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

    ipp_chain = set_section(ipp_chain,  Dict(:parameters => [:μ₁,:l₁,:σ₁],
                                               :spdelatent => [Symbol("w₁[$i]") for i in 1:params.M.N],
                                               :logposterior => [:lp]))
    return (ipp_model, ipp_chain)
end