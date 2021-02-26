using OnlineStats

function mhstep(p::Vector{T}, proposal, logp, ar::S) where {T<:Real, S<:OnlineStat}
    q = p .+ rand(a)
    if log(rand(T)) < (logp(q) - logp(p))
        fit!(ar, 1)
        return q
    else
        fit!(ar, 0)
        return p
    end
end