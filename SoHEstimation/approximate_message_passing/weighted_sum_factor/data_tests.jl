include("../../../lib/gaussian.jl")
include("../../../lib/utils.jl")
include("generate_data_weighted_sum_factor.jl")
include("plot_nn_results.jl")

using Test
using StatsBase
using Distributions
using Flux
using Plots
using LinearAlgebra
using Hyperopt
using Optim
using ProgressMeter

scale(v::AbstractVector) = (v .- minimum(v)) ./ (maximum(v) - minimum(v))

function test_sample(input; marginal_output=true, kwargs...)
    analytical_marginal, analytical_msgto = update_marginals(to_mean_variance(input)...; natural_parameters=true), update_messages_to_variables(to_mean_variance(input)...; natural_parameters=true)
    analytical_output = marginal_output ? analytical_marginal : analytical_msgto

    try
        sampled_marginal, sampled_msgto = generate_output_weighted_sum_factor(input; debug=true, kwargs...)

        sampled_output = marginal_output ? sampled_marginal : sampled_msgto

        # filter out messages with zero tau and rho in the analytical output, since they will produce NaN/Inf in mean and variance which will produce a NaN in the KL divergence
        mean_kldiv = StatsBase.mean([kldivergence(Normal(mean(v, analytical_output), std(v, analytical_output)), Normal(mean(v, sampled_output), std(v, sampled_output))) for v in instances(Variable) if marginal_output || !any(==(0), [tau(v, analytical_output), rho(v, analytical_output)])])

        return (
            marginal_output=marginal_output,
            mse=Flux.mse(analytical_output, sampled_output),
            mae=Flux.mae(analytical_output, sampled_output),
            kldivergence=mean_kldiv,
            # mre=Utils.mean_relative_error(analytical_output, sampled_output),
            input=input,
            sampled_msgto=sampled_msgto,
            sampled_marginal=sampled_marginal,
            analytical_msgto=analytical_msgto,
            analytical_marginal=analytical_marginal,
            sampled_output=sampled_output,
            analytical_output=analytical_output,
            input_mean_var=to_mean_variance(input),
            sampled_msgto_mean_var=to_mean_variance(sampled_msgto),
            sampled_marginal_mean_var=to_mean_variance(sampled_marginal),
            analytical_msgto_mean_var=to_mean_variance(analytical_msgto),
            analytical_marginal_mean_var=to_mean_variance(analytical_marginal),
            sampled_output_mean_var=to_mean_variance(sampled_output),
            analytical_output_mean_var=to_mean_variance(analytical_output),
            std_ax=std(X, input) * a(input),
            std_by=std(Y, input) * b(input),
            std_z=std(Z, input),
            std_axby=sqrt(variance(X, input) * a(input)^2 + variance(Y, input) * b(input)^2),
            std_axbyz=sqrt(variance(X, input) * a(input)^2 + variance(Y, input) * b(input)^2 + variance(Z, input)),
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
            # mre=NaN,
            input=input,
            sampled_msgto=NaN,
            sampled_marginal=NaN,
            analytical_marginal=analytical_marginal,
            analytical_msgto=analytical_msgto,
            analytical_output=analytical_output,
            sampled_output=NaN,
            std_ax=std(X, input) * a(input),
            std_by=std(Y, input) * b(input),
            std_z=std(Z, input),
            std_axby=sqrt(variance(X, input) * a(input)^2 + variance(Y, input) * b(input)^2),
            std_axbyz=sqrt(variance(X, input) * a(input)^2 + variance(Y, input) * b(input)^2 + variance(Z, input)),
            implied_mean_axby=mean(X, input) * a(input) + mean(Y, input) * b(input),
            error=e
        )
    end
end

function test_datagen(;
    seed=rand(1:1000000),
    n=100,
    variable_mean_dist::Distribution=Truncated(Normal(0, 5), -10, 10),
    variable_std_dist::Distribution=Truncated(Normal(2.0, 0.5), 1.0, 3.0),
    factor_dist::Distribution=Truncated(Normal(0, 2.5), -5, 5),
    bias_dist::Distribution=Truncated(Normal(0, 2.5), -5, 5),
    log_weighting=true,
    variance_relative_epsilon=1e-1,
    marginal_output=false,
    samples_per_input=1_000_000,
    normalize_results=false,
    metrics=[:mae, :mse, :kldivergence], #, :mre],
    std_magnitude_factor=3.0,
    colors=[:blue, :red, :green, :orange, :purple, :yellow, :black, :pink, :brown, :cyan],
    positions=[(1.2, 0.9), (1.2, 0.8), (1.2, 0.7), (1.2, 0.6), (1.2, 0.5)],
    implied_z_max_error=15.0,
    uniform_quota=0.05,
    kwargs...
)
    @assert min(length(colors), length(positions)) > length(metrics) "too many metrics, not enough colors or legend spaces"
    Random.seed!(seed)

    inputs = []
    results = []
for _ in ProgressBar(1:n)
        sample = generate_input_weighted_sum_factor(;
            variable_mean_dist=variable_mean_dist,
            variable_std_dist=variable_std_dist,
            factor_dist=factor_dist,
            bias_dist=bias_dist,
            std_magnitude_factor=std_magnitude_factor,
            implied_z_max_error=implied_z_max_error,
            uniform_quota=uniform_quota,
        )

        result = test_sample(sample;
            log_weighting=log_weighting,
            variance_relative_epsilon=variance_relative_epsilon,
            marginal_output=marginal_output,
            samples_per_input=samples_per_input,
            kwargs...
        )
        push!(inputs, sample)
        push!(results, result)
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
    # println("Error counts: ", error_counts)

    filtered_results = filter(r -> isnothing(r.error), results)
    println(length(filtered_results), " out of ", n, " samples are valid")
    if length(filtered_results) == 0
        return results
    end

    if !all(isfinite, [r.kldivergence for r in filtered_results])
        @warn "Some KL divergences are not finite despite the output generation not throwing an error."
        return results, filtered_results
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

    q3 = quantile([r.kldivergence for r in filtered_results], 0.75)
    q1 = quantile([r.kldivergence for r in filtered_results], 0.25)
    iqr = q3 - q1
    for r in [r for r in filtered_results if r.kldivergence > (1.5 * iqr + q3)]
        println("---------------------------")
        for (k, v) in pairs(r)
            println(k, ": ", v)
        end
        println("---------------------------")
    end
    
    return results, filtered_results
end


n = 100
seed = rand(1:1000000)
samples_per_input = 1_000_000
normalize_results = false
metrics = [:mae, :kldivergence] # ,:mre]
variance_relative_epsilon = 1e-2
dirac_std = 0.1
std_magnitude_factor = 200 # 114.0 # 
implied_z_max_error =50 # 38.0 # 
uniform_quota=0.2

# variable_tau_dist=Truncated(Normal(0,25), -100, 100)
# variable_rho_dist=Truncated(Beta(0.12, 1.0)*1000 ,0.001, 100.0)
# factor_dist=Uniform(-10, 10)
# bias_dist=Uniform(-20, 20)

variable_mean_dist = Uniform(-100, 100)
variable_std_dist = Truncated(MixtureModel([Uniform(0,25), Uniform(25,600)], [.9,.1]), 0.001,600) # Uniform(0.001, 600.0) # 
factor_dist = Distributions.Categorical([0.5,0.5])*2-3 # Uniform(-10, 10) # Dirac(1) #  
bias_dist = Dirac(0) # Uniform(-10, 10) # 


results, filtered_results = test_datagen(
    seed=seed,
    n=n,
    log_weighting=true,
    marginal_output=true,
    samples_per_input=samples_per_input,
    normalize_results=normalize_results,
    metrics=metrics,
    dirac_std=dirac_std,
    variance_relative_epsilon=variance_relative_epsilon,
    std_magnitude_factor=std_magnitude_factor,
    variable_mean_dist=variable_mean_dist,
    variable_std_dist=variable_std_dist,
    factor_dist=factor_dist,
    bias_dist=bias_dist,
    implied_z_max_error=implied_z_max_error,
    uniform_quota=uniform_quota,
)

"""

seed = rand(1:1000000)
metrics = [:mae, :kldivergence,]
normalize_results = false

function objective(n, samples_per_input, variance_epsilon, std_magnitude_factor, var_mean_min, var_mean_max, var_std_min, var_std_max, var_factor_min, var_factor_max, var_bias_min, var_bias_max)
    
    # check if if max is always grater than min
    if var_mean_max < var_mean_min || var_std_max < var_std_min || var_factor_max < var_factor_min || var_bias_max < var_bias_min
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
        variable_mean_dist = Uniform(var_mean_min, var_mean_max),
        variable_std_dist = Uniform(var_std_min, var_std_max),
        factor_dist = Uniform(var_factor_min, var_factor_max),
        bias_dist = Uniform(var_bias_min, var_bias_max),
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
                                            inner=BOHB(dims=[Hyperopt.Continuous(), Hyperopt.Continuous(), Hyperopt.Continuous(), Hyperopt.Continuous(), Hyperopt.Continuous(), Hyperopt.Continuous(), Hyperopt.Continuous(), Hyperopt.Continuous(), Hyperopt.Continuous(), Hyperopt.Continuous()])),
                                            variance_epsilon=LinRange(0.01, 1, 100),
                                            std_magnitude_factor=LinRange(1, 5, 50),
                                            var_mean_min=LinRange(-100, 100, 200),
                                            var_mean_max=LinRange(-100, 100, 200),
                                            var_std_min=LinRange(-100, 100, 200),
                                            var_std_max=LinRange(-100, 100, 200),
                                            var_factor_min=LinRange(-100, 100, 200),
                                            var_factor_max=LinRange(-100, 100, 200),
                                            var_bias_min=LinRange(-100, 100, 200),
                                            var_bias_max=LinRange(-100, 100, 200)
                                            

    if state !== nothing
        variance_epsilon, std_magnitude_factor, var_mean_min, var_mean_max, var_std_min, var_std_max, var_factor_min, var_factor_max, var_bias_min, var_bias_max = state
    end

    res = Optim.optimize(x->objective(100, 10_000, x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10]), [variance_epsilon, std_magnitude_factor, var_mean_min, var_mean_max, var_std_min, var_std_max, var_factor_min, var_factor_max, var_bias_min, var_bias_max], NelderMead(), Optim.Options(f_calls_limit=round(Int, i)))
    Optim.minimum(res), Optim.minimizer(res)
end

# write results to file
open("SoHEstimation/approximate_message_passing/weighted_sum_factor/results.txt", "w") do io
println(io, "Best result: ", bohb)
println(io, "Seed:", seed)
end
"""

x = 1