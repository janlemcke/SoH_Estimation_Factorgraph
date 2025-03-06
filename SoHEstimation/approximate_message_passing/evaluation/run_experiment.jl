include("../nn/mjl.jl")
using .NN: run_experiment, get_data, remove_outliers, rmse_loss
using ProgressMeter
using Random

"""
Possible parameters for get_model:
    n_neurons::Int,
    n_layers::Int,
    activation_function::String,
    batch_size::Int,
    output_layer_choice::String,
    epochs::Int=500,
    target::String,
    moment=:both,
    scale::Symbol=:none,
    scaling_params::Union{Tuple{Matrix{Float32}, Matrix{Float32}}, Nothing}=nothing,
    scale_output::Symbol=:none,
    scaling_params_output::Union{Tuple{Matrix{Float32}, Matrix{Float32}}, Nothing}=nothing,
    loss_function=:rmse,
    transform_to_tau_rho=false
"""


seed = 707542
Random.seed!(seed)
target = "targets_X"
datapath = "SoHEstimation/approximate_message_passing/data/dataset_weighted_sum_factor_10_000Experiment.jld2"
#datapath = "SoHEstimation/approximate_message_passing/data/dataset_em_factor_100Experiment.jld2"
nfolds = 5
max_epochs = 100
factor = :wsf

p = Progress(4*nfolds; desc="Calculating...")
for architecture in [:single, :both]
    for transform_to_tau_rho in [true, false]

        X, y =  get_data(datapath; target="targets_X", factor=factor, transform_to_tau_rho=transform_to_tau_rho)
        X, y, _, _ = remove_outliers(X , y)

        params = Dict(
            :n_neurons => 32,
            :n_layers => 2,
            :activation_function => "tanh_fast",
            :batch_size => 512,
            :output_layer_choice => "softplus",
            :target => target,
            :moment => nothing,
            :transform_to_tau_rho => transform_to_tau_rho,
            :scale => :zscore,
            :scale_output => architecture == :both ? :minmax : :none,
            :loss_function => rmse_loss,
            :factor => factor
        )

        if architecture == :both
            params[:moment] = :both
            params[:scale_output] = :minmax
        end

        result = run_experiment(X, y, params, nfolds, seed, max_epochs, architecture, p)

        rmse_avg_first_moment, rmse_avg_second_moment, mae_avg_first_moment, mae_avg_second_moment, mape_avg_first_moment, mape_avg_second_moment, huber_avg_first_moment, huber_avg_second_moment = result

        # write params and errors in txt file.^
        open("SoHEstimation/approximate_message_passing/evaluation/$(factor)_mean_var_vs_naturals.txt", "a") do f
            println(f, "Factor: ", factor)
            println(f, "Architecture: ", architecture)
            println(f, "Transform_to_tau_rho: ", transform_to_tau_rho)
            println(f, "nFolds: ", nfolds)
            println(f, "rmse_avg_first_moment: ", rmse_avg_first_moment)
            println(f, "rmse_avg_second_moment: ", rmse_avg_second_moment)
            println(f, "mae_avg_first_moment: ", mae_avg_first_moment)
            println(f, "mae_avg_second_moment: ", mae_avg_second_moment)
            println(f, "mape_avg_first_moment: ", mape_avg_first_moment)
            println(f, "mape_avg_second_moment: ", mape_avg_second_moment)
            println(f, "huber_avg_first_moment: ", huber_avg_first_moment)
            println(f, "huber_avg_second_moment: ", huber_avg_second_moment)
            println(f, "-----------------------------------")
        end

    end 
end