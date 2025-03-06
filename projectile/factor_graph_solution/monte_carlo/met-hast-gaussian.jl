using Distributions
using StatsBase
using Plots

function metropolis_hastings(;truedist::Distribution=Normal(), initialguess::Float64=2.0, n::Int=1_000_000, a::Float64=0.1)

    samples = [initialguess]

    for _ in 1:n
        current = samples[end]
        proposed = rand(Uniform(current - a, current + a))

        acceptance = min(1, pdf(truedist, proposed) / pdf(truedist, current))

        if rand() < acceptance
            push!(samples, proposed)
        else
            push!(samples, current)
        end
    end


    mean, var = StatsBase.mean_and_var(samples)
    println("sample mean: ", mean)
    println("sample std: ", sqrt(var))
    p = histogram(samples, title="Metropolis-Hastings samples", label="metropolis-hastings")
    histogram!([rand(truedist) for _ in 1:n], label="true distribution", alpha=0.2)
    display(p)
    return mean, var
end

metropolis_hastings(;
    truedist=Normal(0, 1),
    initialguess=2.0,
    n=1_000_000,
    a=0.1
)