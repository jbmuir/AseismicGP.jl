using Distributions

# Δmᵢ is mᵢ - M₀; i.e. relative to cutoff magnitude choice
κ(Δmᵢ, K, α) = K*exp(α*Δmᵢ)
# p > 1! 
h(t, tⱼ, c, p) = (p-1)*c^(p-1) / (t-tⱼ+c)^p
H(t, tⱼ, c, p) = 1 - c^(p-1) / (t-tⱼ+c)^(p-1)