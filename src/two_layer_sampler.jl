struct TwoLayerRateParameters{T<:Real}
    Tspan::T
    M::MaternSPDE{T}
    l₁scale::T
    l₂scale::T
    μ₀prior    
    etaspriors::ETASPriors
end

function TwoLayerRateParameters(Tspan::T, 
                                N::Integer, 
                                l₁scale::T,
                                l₂scale::T,
                                μ₀prior, 
                                etaspriors::ETASPriors) where T
    @assert N >= 2
    d = 1 #only one time dimension
    h = Tspan / (N-1)
    D = Tridiagonal(vcat(ones(T,N-2),2), -2*ones(T,N), vcat(2,ones(T,N-2))) ./ h^2
    M = MaternSPDE(d, N, h, D, Diagonal(ones(T, N)))
    return TwoLayerRateParameters(Tspan, M, l₁scale, l₂scale, μ₀prior, etaspriors)
end

function etas_sampling(nsteps, nchains, catalog::Catalog{T}, params::TwoLayerRateParameters{T}; threads=false) where {T<:Real}

    b = BranchingProcess(T, length(catalog.t))
    xp_dummy = Product([Categorical(inv(i) .* ones(i)) for i = 1:length(catalog.t)]) 

    S = hatinterpolationmatrix(zero(T):params.M.h:params.Tspan, catalog.t)
    wprior = MvNormal(params.M.N, one(T))
    λprior = Normal(zero(T), one(T))

    @model ETASModel(catalog, Kprior, αprior, cprior, p̃prior, μ₀prior, λ₁prior, w₁prior, w₂prior, xp_dummy, Tspan, M, l₁scale, l₂scale, lmin, lmax) = begin
        # assign dummy priors for x so that Turing knows it exists
        x ~ xp_dummy
        #assign priors for the SPDE layer
        μ₀ ~ μ₀prior
        λ₁ ~ λ₁prior
        w₁ ~ w₁prior
        w₂ ~ w₂prior
        # assign priors for ETAS parameters
        K ~ Kprior
        α ~ αprior
        c ~ cprior
        p̃ ~ p̃prior

        p = p̃ + 1

        #=         
        estimate background weight:
        =#
        l₁ = smoothclamp(l₁scale * exp(λ₁), lmin, lmax)
        λ₂ = M(l₁, one(T), w₁)
        l₂ = @. smoothclamp(l₂scale * exp(λ₂), lmin, lmax)
        λ₃ = M(l₂, one(T), w₂)
        μ = @. μ₀ * exp(λ₃)
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

        return (μ, l₂, l₁)
    end

    function cond_x_b_c(c, b::BranchingProcess{T}, catalog::Catalog{T}, M::MaternSPDE, l₁scale, l₂scale, lmin, lmax) where T
        l₁ = smoothclamp(l₁scale * exp(c.λ₁), lmin, lmax)
        λ₂ = M(l₁, one(T), c.w₁)
        l₂ = @. smoothclamp(l₂scale * exp(λ₂), lmin, lmax)
        λ₃ = M(l₂, one(T), c.w₂)
        μ = @. c.μ₀ * exp(λ₃)
        μi = S*μ
        update_weights!(b, catalog, μi, c.K, c.α, c.c, c.p̃+1)
        return Product([Categorical(b.bnodes[i].bweights) for i = 1:length(catalog.t)])
    end
    
    cond_x(c) = cond_x_b_c(c, b, catalog, params.M, params.l₁scale, params.l₂scale, params.M.h, params.Tspan)

    etas_model =  ETASModel(catalog, 
                            params.etaspriors.Kprior, 
                            params.etaspriors.αprior, 
                            params.etaspriors.cprior, 
                            params.etaspriors.p̃prior, 
                            params.μ₀prior,
                            λprior,
                            wprior,
                            wprior,
                            xp_dummy, 
                            params.Tspan, 
                            params.M,
                            params.l₁scale,
                            params.l₂scale,
                            params.M.h,
                            params.Tspan)

    etas_sampler = Gibbs(DynamicNUTS{Turing.ForwardDiffAD{params.M.N+1}}(:w₁, :λ₁),
                         DynamicNUTS{Turing.ForwardDiffAD{params.M.N}}(:w₂),
                         DynamicNUTS{Turing.ForwardDiffAD{1}}(:μ₀),
                         GibbsConditional(:x, cond_x), 
                         DynamicNUTS{Turing.ForwardDiffAD{4}}(:K, :α, :c, :p̃))

    if threads
        etas_chain = sample(etas_model, etas_sampler, MCMCThreads(), nsteps, nchains)
    else
        etas_chain = mapreduce(c -> sample(etas_model, etas_sampler, nsteps), chainscat, 1:nchains)
    end
    etas_chain = set_section(etas_chain,  Dict(:parameters => [:K,:α,:c,:p̃,:μ₀,:λ₁],
                                               :spdelatent1 => [Symbol("w₁[$i]") for i in 1:params.M.N], 
                                               :spdelatent2 => [Symbol("w₂[$i]") for i in 1:params.M.N], 
                                               :internals => [Symbol("x[$i]") for i in 1:length(catalog.t)], 
                                               :logposterior => [:lp]))
    return (etas_model, etas_chain)
end