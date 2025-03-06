include("../../../lib/gaussian.jl")
include("../../../lib/utils.jl")
include("generate_data_gaussian_mean_factor.jl")
include("plot_nn_results.jl")

using Random
using Test
using StatsBase
using Distributions
using Flux
using Plots
using LinearAlgebra
using Hyperopt
using Optim

# TODO: remove testing framework?

scale(v::AbstractVector) = (v .- minimum(v)) ./ (maximum(v) - minimum(v))

function test_sample(input; marginal_output=false, kwargs...)
    analytical_marginal, analytical_msgto = update_marginals(input...; natural_parameters=false), update_messages_to_variables(input...; natural_parameters=false)
    analytical_output = marginal_output ? analytical_marginal : analytical_msgto

    try
        sampled_msgto, sampled_marginal = generate_output_gaussian_mean_factor(input; debug=true, kwargs...)

        sampled_output = marginal_output ? sampled_marginal : sampled_msgto

        mean_kldiv = StatsBase.mean([kldivergence(Normal(mean(v, sampled_output), variance(v, sampled_output)), Normal(mean(v, analytical_output), variance(v, analytical_output))) for v in instances(Variable)])

        return (
            marginal_output=marginal_output,
            mse=Flux.mse(analytical_output, sampled_output),
            mae=Flux.mae(analytical_output, sampled_output),
            kldivergence=mean_kldiv,
            #ämre=Utils.mean_relative_error(analytical_output, sampled_output),
            input=input,
            sampled_msgto=sampled_msgto,
            sampled_marginal=sampled_marginal,
            analytical_msgto=analytical_msgto,
            analytical_marginal=analytical_marginal,
            sampled_output=sampled_output,
            analytical_output=analytical_output,
            error=nothing,
        )
    catch e
        if typeof(e) == DomainError
            rethrow(e)
        end
        return (
            marginal_output=marginal_output,
            mse=NaN,
            mae=NaN,
            kldivergence=NaN,
            mre=NaN,
            input=input,
            sampled_msgto=NaN,
            sampled_marginal=NaN,
            analytical_marginal=analytical_marginal,
            analytical_msgto=analytical_msgto,
            analytical_output=analytical_output,
            sampled_output=NaN,
            error=e
        )
    end
end

function test_datagen(;
    seed=rand(1:1000000),
    n=1000,
    variable_mean_dist::Distribution=Truncated(Normal(0, 10), -20, 20),
    variable_std_dist::Distribution=Truncated(Normal(2.0, 0.5), 1.0, 3.0),
    factor_dist::Distribution=Truncated(Normal(0, 2.5), 0, 5),
    log_weighting=true,
    variance_epsilon=1e-1,
    marginal_output=false,
    samples_per_input=1_000_000,
    std_magnitude_factor=3.0,
    normalize_results=false,
    metrics=[:mae, :mse, :kldivergence],
    colors=[:blue, :red, :green, :orange, :purple, :yellow, :black, :pink, :brown, :cyan],
    positions=[(1.2, 0.9), (1.2, 0.8), (1.2, 0.7), (1.2, 0.6), (1.2, 0.5)],
    kwargs...
)
    @assert min(length(colors), length(positions)) > length(metrics) "too many metrics, not enough colors or legend spaces"
    Random.seed!(seed)

    inputs = []
    results = []
    for _ in ProgressBar(1:n)
        
        sample = generate_input_gaussian_mean_factor(;
            variable_mean_dist=variable_mean_dist,
            variable_std_dist=variable_std_dist,
            factor_dist=factor_dist,
            std_magnitude_factor=std_magnitude_factor,
        )
        try
            
            result = test_sample(sample;
                log_weighting=log_weighting,
                variance_epsilon=variance_epsilon,
                marginal_output=marginal_output,
                samples_per_input=samples_per_input,
                kwargs...
            )
            push!(inputs, sample)
            push!(results, result)
        catch e
            println(e.msg)
        end
        
    end

    error_counts = Dict{Any,Int}()
    for r in results
        if !isnothing(r.error)
            if haskey(error_counts, r.error)
                error_counts[r.error] += 1
            else
                error_counts[r.error] = 1
            end
        end
    end
    println("Error counts: ", error_counts)

    filtered_results = filter(r -> !isnan(r.mae), results)
    println(length(filtered_results), " out of ", n, " samples are valid")
    if length(filtered_results) == 0
        return results
    end

    metric_results = Dict{Symbol,Vector{Float64}}()

    for metric in metrics
        metric_results[metric] = normalize_results ? scale([r[metric] for r in filtered_results]) : [r[metric] for r in filtered_results]
    end

    p1 = plot(1:length(filtered_results), metric_results[metrics[1]], label=string(metrics[1]), legend=positions[1], color=colors[1], ytickfontcolor=colors[1], xlabel="Sample Index", ylabel="error", right_margin=30Plots.mm, title="normalized: $normalize_results, marginals: $marginal_output, log_weighs: $log_weighting")
    for (i, m) in enumerate(metrics[2:end])
        plot!(twinx(), 1:length(filtered_results), metric_results[m], label=string(m), color=colors[i+1], legend=positions[i+1], ytickfontcolor=colors[i+1], foreground_color_guide=colors[i+1])
    end
    display(p1)

    println("##############")
    println("Seed: ", seed, ", Log Weighting: ", log_weighting, ", Marginals as Labels: ", marginal_output)

    q3 = quantile([r.mae for r in filtered_results], 0.75)
    q1 = quantile([r.mae for r in filtered_results], 0.25)
    iqr = q3 - q1
    for r in [r for r in results if r.mae > (1.5 * iqr + q3) && !isnan(r.mae)]
        println("---------------------------")
        for (k, v) in pairs(r)
            println(k, ": ", v)
        end
        println("---------------------------")
    end
    return results
end

"""
n = 100
seed = rand(1:1000000)
samples_per_input = 100_000
normalize_results = false
metrics = [:mae,]
variance_epsilon = 1e-1
std_magnitude_factor = 2

test_datagen(
        seed=seed,
        n=n,
        log_weighting=true,
        marginal_output=false,
        samples_per_input=samples_per_input,
        normalize_results=normalize_results,
        metrics=metrics,
        variance_epsilon=variance_epsilon,
        std_magnitude_factor=std_magnitude_factor,
    )


"""


seed = rand(1:1000000)
metrics = [:mae, :kldivergence,]
normalize_results = false

function objective(n, samples_per_input, variance_epsilon, std_magnitude_factor, var_mean_mean, var_mean_std, var_std_mean, var_std_std, factor_mean, factor_std)
    
    # check if std are positive
    if var_mean_std < 0 || var_std_std < 0 || factor_std < 0
        return Inf
    end

    results = test_datagen(
        seed=seed,
        n=n,
        log_weighting=true,
        marginal_output=false,
        samples_per_input=samples_per_input,
        normalize_results=normalize_results,
        metrics=metrics,
        variance_epsilon=variance_epsilon,
        variable_mean_dist=Truncated(Normal(var_mean_mean, var_mean_std), 0, 50),
        variable_std_dist=Truncated(Normal(var_std_mean, var_std_std), sqrt(variance_epsilon) + 1e-3, 50),
        factor_dist=Truncated(Normal(factor_mean, factor_std), 0, 50),
        std_magnitude_factor=std_magnitude_factor,
    )

    # Extract valid (non-NaN) MAE values
    valid_mae = [r.mae for r in results if !isnan(r.mae)]
    
    # Check if there are valid entries, return Inf if none
    if isempty(valid_mae)
        return Inf
    else
        return StatsBase.mean(valid_mae)
    end
end



bohb = @hyperopt for i=50, sampler=Hyperband(R=50, η=3,
                                            inner=BOHB(dims=[Hyperopt.Continuous(), Hyperopt.Continuous(), Hyperopt.Continuous(), Hyperopt.Continuous(), Hyperopt.Continuous(), Hyperopt.Continuous(), Hyperopt.Continuous(), Hyperopt.Continuous()])),
                                            variance_epsilon=LinRange(0.01, 1, 100),
                                            std_magnitude_factor=LinRange(1, 5, 50),
                                            var_mean_mean=LinRange(10, 50, 50),
                                            var_mean_std=LinRange(5, 20, 50),
                                            var_std_mean=LinRange(5, 20, 50),
                                            var_std_std=LinRange(5, 10, 50),
                                            factor_mean=LinRange(10, 50, 50),
                                            factor_std=LinRange(5, 20, 50)
    if state !== nothing
        variance_epsilon, std_magnitude_factor, var_mean_mean, var_mean_std, var_std_mean, var_std_std, factor_mean, factor_std = state
    end
    res = Optim.optimize(x->objective(200, 100_000, x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]), [variance_epsilon, std_magnitude_factor, var_mean_mean, var_mean_std, var_std_mean, var_std_std, factor_mean, factor_std], NelderMead(), Optim.Options(f_calls_limit=round(Int, i)))
    Optim.minimum(res), Optim.minimizer(res)
end

# write results to file
open("SoHEstimation/approximate_message_passing/gaussian_mean_factor/results_with_normals_constraint_xs.txt", "w") do io
    println(io, "Best result: ", bohb)
    println(io, "Seed:", seed)
end