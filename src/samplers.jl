using StatsBase
using MCMCChains
using Turing
using DynamicHMC
using DeepGaussianSPDEProcesses
using LinearAlgebra
using SparseArrays

export ETASPriors, ConstantRateParameters
export SPDELayerPriors, OneLayerRateParameters
export etas_sampling

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

struct ConstantRateParameters{T<:Real}
    Tspan::T
    μα::T
    μβ::T
    etaspriors::ETASPriors
end

struct OneLayerRateParameters{T<:Real}
    Tspan::T
    M::MaternSPDE{T}
    spdepriors::SPDELayerPriors
    etaspriors::ETASPriors
end

function OneLayerRateParameters(Tspan::T, 
                                N::Integer, 
                                spdepriors::SPDELayerPriors, 
                                etaspriors::ETASPriors) where T
    @assert N >= 2
    d = 1 #only one time dimension
    h = Tspan / (N-1)
    D = Tridiagonal(vcat(ones(T,N-2),2), -2*ones(T,N), vcat(2,ones(T,N-2))) ./ h^2
    M = MaternSPDE(d, N, h, D, Diagonal(ones(T, N)))
    return OneLayerRateParameters(Tspan, M, spdepriors, etaspriors)
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
        if j != 0
            etasloglik += log(h(catalog.t[i], catalog.t[j], c, p))
        end
    end
    return etasloglik
end

function etas_sampling(nsteps, nchains, catalog::Catalog{T}, params::ConstantRateParameters{T}; threads=false) where {T<:Real}

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
                         DynamicNUTS{Turing.ForwardDiffAD{4}}(:K, :α, :c, :p̃))


    if threads
        etas_chain = sample(etas_model, etas_sampler, MCMCThreads(), nsteps, nchains)
    else
        etas_chain = mapreduce(c -> sample(etas_model, etas_sampler, nsteps), chainscat, 1:nchains)
    end
    etas_chain = set_section(etas_chain,  Dict(:parameters => [:K,:α,:c,:p̃,:μ], 
                                               :internals => [Symbol("x[$i]") for i in 1:length(catalog.t)], 
                                               :logposterior => [:lp]))
    return (etas_model, etas_chain)
end

function etas_sampling(nsteps, nchains, catalog::Catalog{T}, params::OneLayerRateParameters{T}; threads=false) where {T<:Real}

    b = BranchingProcess(T, length(catalog.t))
    xp_dummy = Product([Categorical(inv(i) .* ones(i)) for i = 1:length(catalog.t)]) 

    S = hatinterpolationmatrix(0:params.M.h:params.Tspan, catalog.t)
    wprior = MvNormal(params.M.N, one(T))

    @model ETASModel(catalog, Kprior, αprior, cprior, p̃prior, μ₀prior, lprior, σprior, wprior, xp_dummy, Tspan, M) = begin
        # assign dummy priors for x so that Turing knows it exists
        x ~ xp_dummy
        #assign priors for the SPDE layer
        μ₀  ~ μ₀prior

        l ~ lprior
        σ ~ σprior
        w ~ wprior
        # assign priors for ETAS parameters
        K ~ Kprior
        α ~ αprior
        c ~ cprior
        p̃ ~ p̃prior

        p = p̃ + 1

        #=         
        estimate background weight:
        =#
        μ = μ₀ .* exp.(M(l, σ, w))
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

        return μ
    end

    function cond_x_b_c(c, b::BranchingProcess{T}, catalog::Catalog{T}, M::MaternSPDE) where T
        μ = c.μ₀.* exp.(M(c.l, c.σ, c.w))
        μi = S*μ
        update_weights!(b, catalog, μi, c.K, c.α, c.c, c.p̃+1)
        return Product([Categorical(b.bnodes[i].bweights) for i = 1:length(catalog.t)])
    end
    
    cond_x(c) = cond_x_b_c(c, b, catalog, params.M)

    etas_model =  ETASModel(catalog, 
                            params.etaspriors.Kprior, 
                            params.etaspriors.αprior, 
                            params.etaspriors.cprior, 
                            params.etaspriors.p̃prior, 
                            params.spdepriors.μ₀prior,
                            params.spdepriors.lprior,
                            params.spdepriors.σprior,
                            wprior,
                            xp_dummy, 
                            params.Tspan, 
                            params.M)

    etas_sampler = Gibbs(DynamicNUTS{Turing.ForwardDiffAD{3+params.M.N}}(:μ₀, :l, :σ, :w),
                         GibbsConditional(:x, cond_x), 
                         DynamicNUTS{Turing.ForwardDiffAD{4}}(:K, :α, :c, :p̃))

    if threads
        etas_chain = sample(etas_model, etas_sampler, MCMCThreads(), nsteps, nchains)
    else
        etas_chain = mapreduce(c -> sample(etas_model, etas_sampler, nsteps), chainscat, 1:nchains)
    end
    etas_chain = set_section(etas_chain,  Dict(:parameters => [:K,:α,:c,:p̃,:μ₀,:l,:σ],
                                               :spdelatent => [Symbol("w[$i]") for i in 1:params.M.N], 
                                               :internals => [Symbol("x[$i]") for i in 1:length(catalog.t)], 
                                               :logposterior => [:lp]))
    return (etas_model, etas_chain)
end