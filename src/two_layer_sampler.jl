struct TwoLayerRateParameters{T<:Real}
    Tspan::T
    M::MaternSPDE{T}
    spde1priors::ScalarSPDELayerPriors
    spde2priors::VectorSPDELayerPriors
    etaspriors::ETASPriors
end

function TwoLayerRateParameters(Tspan::T, 
                                N::Integer, 
                                spde1priors::ScalarSPDELayerPriors,
                                spde2priors::VectorSPDELayerPriors,
                                etaspriors::ETASPriors) where T
    @assert N >= 2
    d = 1 #only one time dimension
    h = Tspan / (N-1)
    D = Tridiagonal(vcat(ones(T,N-2),2), -2*ones(T,N), vcat(2,ones(T,N-2))) ./ h^2
    M = MaternSPDE(d, N, h, D, Diagonal(ones(T, N)))
    return TwoLayerRateParameters(Tspan, M, spde1priors, spde2priors, etaspriors)
end

function etas_sampling(nsteps, nchains, catalog::Catalog{T}, params::TwoLayerRateParameters{T}; threads=false) where {T<:Real}

    ETASPriors = params.etaspriors
    L₁Priors = params.spde1priors
    L₂Priors = params.spde2priors
    Tspan = params.Tspan
    M = params.M
    b = BranchingProcess(T, length(catalog.t))
    xp_dummy = Product([Categorical(inv(i) .* ones(i)) for i = 1:length(catalog.t)]) 
    S = hatinterpolationmatrix(zero(T):params.M.h:params.Tspan, catalog.t)
    wprior = MvNormal(params.M.N, one(T))
    lmin = params.M.h
    lmax = params.Tspan
    clamp(l) = sigmoid_clamp(l, lmax-lmin, lmin)

    @model ETASModel() = begin
        # assign dummy priors for x so that Turing knows it exists
        x ~ xp_dummy
        #assign priors for the SPDE layers
        μ₁ ~ L₁Priors.μprior
        l₁ ~ L₁Priors.lprior
        σ₁ ~ L₁Priors.σprior
        w₁ ~ wprior
        μ₂ ~ L₂Priors.μprior
        σ₂ ~ L₂Priors.σprior
        w₂ ~ wprior
        # assign priors for ETAS parameters
        K ~ ETASPriors.Kprior
        α ~ ETASPriors.αprior
        c ~ ETASPriors.cprior
        p̃ ~ ETASPriors.p̃prior
        p = p̃ + 1

        #=         
        estimate background weight:
        =#
        l₂ = μ₁ .* exp.(M(smoothclamp(l₁, lmin, lmax), σ₁, w₁))
        μ = μ₂ .* exp.(M(smoothclamp.(l₂, lmin, lmax), σ₂, w₂))
        μi = S*μ
        Turing.@addlogprob! sum(log.(μi[x.==1])) - hatsum(μ)*params.M.h

        #=
        setup latent branching process:
        - this is given by direct gibbs sampling in the conditional gibbs function below
        =#

        #=
        log-likelihood for etas parameters:
        =#

        Turing.@addlogprob! etas_log_likelihood(K, α, c, p, x, catalog, Tspan)

        return (μ, l₂)
    end

    function cond_x(c)
        l₂ = c.μ₁ .* exp.(M(smoothclamp(c.l₁, lmin, lmax), c.σ₁, c.w₁))
        μ = c.μ₂ .* exp.(M(smoothclamp.(l₂, lmin, lmax), c.σ₂, c.w₂))
        μi = S*μ
        update_weights!(b, catalog, μi, c.K, c.α, c.c, c.p̃+1)
        return Product([Categorical(b.bnodes[i].bweights) for i = 1:length(catalog.t)])
    end
    
    etas_model =  ETASModel()

    etas_sampler = Gibbs(DynamicNUTS{Turing.ForwardDiffAD{params.M.N}}(:w₁),
                         DynamicNUTS{Turing.ForwardDiffAD{3}}(:μ₁, :l₁, :σ₁),
                         DynamicNUTS{Turing.ForwardDiffAD{params.M.N}}(:w₂),
                         DynamicNUTS{Turing.ForwardDiffAD{2}}(:μ₂, :σ₂),
                         GibbsConditional(:x, cond_x), 
                         DynamicNUTS{Turing.ForwardDiffAD{4}}(:K, :α, :c, :p̃))

    if threads
        etas_chain = sample(etas_model, etas_sampler, MCMCThreads(), nsteps, nchains)
    else
        etas_chain = mapreduce(c -> sample(etas_model, etas_sampler, nsteps), chainscat, 1:nchains)
    end
    etas_chain = set_section(etas_chain,  Dict(:parameters => [:K,:α,:c,:p̃,:μ₁,:l₁,:σ₁,:μ₂,:σ₂],
                                               :spdelatent1 => [Symbol("w₁[$i]") for i in 1:params.M.N], 
                                               :spdelatent2 => [Symbol("w₂[$i]") for i in 1:params.M.N], 
                                               :internals => [Symbol("x[$i]") for i in 1:length(catalog.t)], 
                                               :logposterior => [:lp]))
    return (etas_model, etas_chain)
end