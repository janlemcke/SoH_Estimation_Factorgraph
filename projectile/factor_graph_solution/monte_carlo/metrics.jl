"""
    gelman_rubin(samples::Vector{Vector{ProjectileState}})

Calculate the Estimated Potential Scale Reduction (EPSR) for multiple MCMC chains.
The input `samples` is a vector of vectors, where each inner vector contains samples from one MCMC chain.
"""
function gelman_rubin(chains::Vector{Vector{ProjectileState}})
    M = length(chains)
    N = length(chains[1])

    avg_per_chain = [mean(getfield.(chain, :x)) for chain in chains]
    avg_over_chains = mean(avg_per_chain)

    # Bishop 12.97
    B = N / (M - 1) * sum([(avg - avg_over_chains) ^ 2 for avg in avg_per_chain])

    # Bishop 12.98
    W = 1 / M * sum([1 / (N - 1) * sum([(getfield(state, :x) - avg) ^ 2 for (state, avg) in zip(chain, avg_per_chain)]) for chain in chains])

    # Bishop 12.99
    var_hat = (N - 1) / N * W + B / N

    # Bishop 12.100
    R = sqrt(var_hat / W)

    return R
end