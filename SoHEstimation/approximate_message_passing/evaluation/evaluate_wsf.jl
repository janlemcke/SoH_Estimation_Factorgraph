module EvalWSF

include("../../../lib/gaussian.jl")
using .GaussianDistribution

include("../weighted_sum_factor/generate_data_weighted_sum_factor.jl")
using .WeightedSumFactorGeneration: generate_input_weighted_sum_factor, generate_output_weighted_sum_factor, X, Y, Z, get_variable, a, b, c, to_mean_variance
using Distributions
using JLD2

include("../nn/mjl.jl")
using .NN: train_network, predict_sample, analyze_outlier

using StatsBase
using Plots
using DelimitedFiles

# Configs
seed = 123456789
n = 5_000
datapath = WeightedSumFactorGeneration.generate_dataset_weighted_sum_factor(
        n=                          n,
        patience=                   0.5,
        save_dataset=               true,
        log_weighting=              true,
        samples_per_input=          1_000_000,
        variance_relative_epsilon=  1e-10,
        dirac_std=                  0.1,
        std_magnitude_factor=       10, # 114.0 is the max in TTT analytical
        implied_z_max_error=        50, # 38.0 is the max in TTT analytical
        uniform_quota=              0.2,
        variable_mean_dist=         Uniform(-100, 100),
        variable_std_dist=          MixtureModel([Uniform(0,20), Uniform(20, 200)], [.9, .1]), #Truncated(MixtureModel([Uniform(0,25), Uniform(25,600)], [.97,.03]), 1.0,600),
        factor_dist=                Distributions.Categorical([0.5,0.5])*2-3,
        bias_dist=                  Dirac(0),
        algorithm=                  :adaptive_metropolis_hastings, 
        savepath=                   "SoHEstimation/approximate_message_passing/weighted_sum_factor/data",
        name_appendix=              "_TTTbased_1variableFixed_stdmin_1-0_strict",
        strict_convergence=false
)

inputs = JLD2.load(datapath, "samples")
inputs = Matrix(hcat(inputs...)')
@assert (n, 9) == size(inputs)

# TODO: WHY?
indexes = []
for i in 1:size(inputs, 1)
    if (inputs[i, 2] == 0 || inputs[i, 4] == 0 || inputs[i, 6] == 0)
        push!(indexes, i)
    end
end

println("Found ", length(indexes), " samples with precision 0. Removing them.")

inputs = inputs[setdiff(1:size(inputs, 1), indexes), :]

# NN Configs
fast_run = true
n_neurons = 32
factor = :wsf
use_polynomial = false
degree = 2
scaling = :zscore
scaling_output = :none
rm_outlier = true
n_layers = 2

# Data
use_log=false
transform_to_tau_rho=false
loss_function=:rmse

models = Dict{String, Any}()
labels = Dict{String, Any}()

for target in ["targets_X", "targets_Y", "targets_Z"]
    y = JLD2.load(datapath, target)
    y = hcat(y...)'

    y = y[setdiff(1:size(y, 1), indexes), :]

    if transform_to_tau_rho
        y = to_tau_rho(y)
    end

    labels[target] = y
end


if transform_to_tau_rho
    inputs = to_tau_rho(inputs) 
end

if rm_outlier
    combined_mask = analyze_outlier(inputs, labels["targets_X"], labels["targets_Y"], labels["targets_Z"])

    # Apply combined mask to X and y
    inputs = inputs[vec(combined_mask), :]
    
    for target in ["targets_X", "targets_Y", "targets_Z"]
        # p = plot()
        # scatter!(p, labels[target][:, 2], label="Befor removal data for $target second moment")
        # display(p)
        
        # p = plot()
        # scatter!(p, labels[target][:, 1], label="Before removal data for target $target first moment")
        # display(p)

        labels[target] = labels[target][vec(combined_mask), :]

        # p = plot()
        # scatter!(p, labels[target][:, 2], label="After removal data for target $target second moment")
        # display(p)

        # p = plot()
        # scatter!(p, labels[target][:, 1], label="After removal data for target $target first moment")
        # display(p)
    end
end


for target in ["targets_X", "targets_Y", "targets_Z"]
    for moment in [:both]

        if moment == :second
            labels[target] = labels[target][:, 2:2]
        end

        if moment == :first
            labels[target] = labels[target][:, 1:1]
        end
        
        for scaling in [:zscore]
            for activation_function in ["tanh_fast"]
                for output_layer_choice in ["softplus"]
                    #TODO: Bei wechsle zu mean & var muss der Residuallayer subtrahieren anstatt addieren
                    appendix = "eval_$(target)_$(n)_$(transform_to_tau_rho)_$(moment)_$(scaling)_$(scaling_output)_$(activation_function)_$(output_layer_choice)_y_scale_min"
                    mach, scaling_params, scaling, scaling_params_output, scaling_output = train_network(
                        inputs, labels[target];
                        target=target,
                        appendix=appendix,
                        seed=seed,
                        n_neurons=n_neurons,
                        n_layers=n_layers,
                        factor=factor,
                        use_polynomial=use_polynomial,
                        degree=degree,
                        scaling=scaling,
                        moment=moment,
                        transform_to_tau_rho=transform_to_tau_rho,
                        activation_function=activation_function,
                        output_layer_choice=output_layer_choice,
                        max_epochs=1_000,
                        loss_function=loss_function,
                        scaling_output=scaling_output,
                    )
                    models[target] = (mach, scaling_params, scaling, scaling_params_output, local_scaling_output)
                end
            end                
        end
    end
end


function approx_msg_x(input::AbstractVector)
    prediction = predict_sample(models["targets_X"], input)
    if !transform_to_tau_rho
        newMarginal = Gaussian1DFromMeanVariance(prediction[1], prediction[2])
    else
        newMarginal = Gaussian1D(prediction[1], prediction[2])
    end

    # assert new Marginal is finite
    @assert isfinite(GaussianDistribution.mean(newMarginal)) "Mean is infinite: $(newMarginal)"
    @assert isfinite(GaussianDistribution.variance(newMarginal)) "Variance is infinite: $(newMarginal)"
    @assert GaussianDistribution.variance(newMarginal) >= 0 "Variance is negative: $(newMarginal)"

    if !transform_to_tau_rho
        display(input)
        println("New marginal: ", newMarginal)
        println("Incoming message: ", Gaussian1DFromMeanVariance(get_variable(X, input)...))
        newMsg = newMarginal / Gaussian1DFromMeanVariance(get_variable(X, input)...)
    else
        newMsg = newMarginal / Gaussian1D(get_variable(X, input)...)
    end

    return newMsg
end

function approx_msg_y(input::AbstractVector)
    prediction = predict_sample(models["targets_Y"], input)
    if !transform_to_tau_rho
        newMarginal = Gaussian1DFromMeanVariance(prediction[1], prediction[2])
     else
        newMarginal = Gaussian1D(prediction[1], prediction[2])
     end

    # assert new Marginal is finite
    @assert isfinite(GaussianDistribution.mean(newMarginal))
    @assert isfinite(GaussianDistribution.variance(newMarginal))

    if !transform_to_tau_rho
        newMsg = newMarginal / Gaussian1DFromMeanVariance(get_variable(Y, input)...)
    else
        newMsg = newMarginal / Gaussian1D(get_variable(Y, input)...)
    end

    return newMsg
end

function approx_msg_z(input::AbstractVector)
    prediction = predict_sample(models["targets_Z"], input)
    if !transform_to_tau_rho
        newMarginal = Gaussian1DFromMeanVariance(prediction[1], prediction[2])
     else
        newMarginal = Gaussian1D(prediction[1], prediction[2])
     end

    # assert new Marginal is finite
    @assert isfinite(GaussianDistribution.mean(newMarginal)) "Marginal: $(newMarginal)"
    @assert isfinite(GaussianDistribution.variance(newMarginal)) "Marginal: $(newMarginal)"

    if !transform_to_tau_rho
        newMsg = newMarginal / Gaussian1DFromMeanVariance(get_variable(Z, input)...)
    else
        newMsg = newMarginal / Gaussian1D(get_variable(Z, input)...)
    end

    return newMsg
end



function evaluate(inputs::Matrix{Float32})
    rmse_x = []
    rmse_y = []
    rmse_z = []

    mae_x = []
    mae_y = []
    mae_z = []

    metrics = Matrix{Float32}(undef, 4*3*2, size(inputs, 1))
    
    for i in 1:size(inputs, 1)
        input = inputs[i, :]

        # Analytical solutions
        exact_x = calc_msg_x(input)
        exact_y = calc_msg_y(input)
        exact_z = calc_msg_z(input)

        # Approximate solutions
        approx_x = approx_msg_x(input)
        approx_y = approx_msg_y(input)
        approx_z = approx_msg_z(input)

        # Calculate errors
        metric = rmse(approx_x, exact_x)
        metrics[1, i] = metric[1]
        metrics[2, i] = metric[2]
        metrics[3, i] = metric[3]
        metrics[4, i] = metric[4]

        metric = rmse(approx_y, exact_y)
        metrics[5, i] = metric[1]
        metrics[6, i] = metric[2]
        metrics[7, i] = metric[3]
        metrics[8, i] = metric[4]

        metric = rmse(approx_z, exact_z)
        metrics[9, i] = metric[1]
        metrics[10, i] = metric[2]
        metrics[11, i] = metric[3]
        metrics[12, i] = metric[4]

        metric = mae(approx_x, exact_x)
        metrics[13, i] = metric[1]
        metrics[14, i] = metric[2]
        metrics[15, i] = metric[3]
        metrics[16, i] = metric[4]

        metric = mae(approx_y, exact_y)
        metrics[17, i] = metric[1]
        metrics[18, i] = metric[2]
        metrics[19, i] = metric[3]
        metrics[20, i] = metric[4]

        metric = mae(approx_z, exact_z)
        metrics[21, i] = metric[1]
        metrics[22, i] = metric[2]
        metrics[23, i] = metric[3]
        metrics[24, i] = metric[4]

    end

    println("Shape of metrics: ", size(metrics))

    # add last value to matrix for mean calculation
    metrics = hcat(metrics, StatsBase.mean(metrics, dims=2))

    # write matrix to txt file
    writedlm("metrics.txt", metrics)
end

function rmse(approx::Gaussian1D, exact::Gaussian1D)
    mean_diff = GaussianDistribution.mean(approx) - GaussianDistribution.mean(exact)
    var_exact = GaussianDistribution.variance(exact)
    var_approx = GaussianDistribution.variance(approx)

    if var_exact == Inf
        println("Exact variance is Inf:" , exact)
    end

    if var_approx == Inf
        println("Approx variance is Inf:" , approx)
    end
    variance_diff = var_approx - var_exact

    rho_diff = approx.rho - exact.rho
    tau_diff = approx.tau - exact.tau

    return sqrt(mean_diff^2), sqrt(variance_diff^2), sqrt(rho_diff^2), sqrt(tau_diff^2)
end

function mae(approx::Gaussian1D, exact::Gaussian1D)
    mean_diff = abs(GaussianDistribution.mean(approx) - GaussianDistribution.mean(exact))
    var_exact = GaussianDistribution.variance(exact)
    var_approx = GaussianDistribution.variance(approx)
    variance_diff = abs(var_approx - var_exact)

    if var_exact == Inf
        println("Exact variance is Inf:" , exact)
    end

    if var_approx == Inf
        println("Approx variance is Inf:" , approx)
    end

    rho_diff = abs(approx.rho - exact.rho)
    tau_diff = abs(approx.tau - exact.tau)

    return mean_diff, variance_diff, rho_diff, tau_diff
end

evaluate(inputs)

end