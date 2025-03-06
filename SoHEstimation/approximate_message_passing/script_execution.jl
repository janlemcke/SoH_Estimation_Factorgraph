include("./nn/mjl.jl")
using .NN: train_network, get_data, remove_outliers, tune_network
using Distributions

TRAIN = false
TUNE = true

if TRAIN
    seed = 123456789
    datapath = "SoHEstimation/approximate_message_passing/data/dataset_em_factor_100Experiment.jld2"
    #datapath = "SoHEstimation\\approximate_message_passing\\data\\dataset_weighted_sum_factor_5_000_TTTbased_1variableFixed_stdmin_1-0_JAN.jld2"
    #target = "targets_Y"
    fast_run = true
    tune = false
    n_neurons = 32
    factor = :gmf
    use_polynomial = false
    degree = 2
    scaling = :zscore
    rm_outlier = true
    use_residual = true
    # activation_function = "tanh_fast"
    # output_layer_choice = "softplus"

    for target in ["targets_X", "targets_Y"]
        for moment_to_learn in [:second, :first, :both]
            for scaling in [:zscore]
                for activation_function in ["tanh_fast", "relu"]
                    for output_layer_choice in ["relu", "softplus"]
                        # Bei wechsle zu mean & var muss der Residuallayer subtrahieren anstatt addieren
                        appendix = "_$(moment_to_learn)_$(scaling)_activation_$(activation_function)_outputlayer_$(output_layer_choice)_5k"
                        train_network(
                            datapath;
                            target=target,
                            appendix=appendix,
                            seed=seed,
                            fast_run=fast_run,
                            n_neurons=n_neurons,
                            factor=factor,
                            use_polynomial=use_polynomial,
                            degree=degree,
                            scaling=scaling,
                            rm_outlier=rm_outlier,
                            use_residual=use_residual,
                            moment=moment_to_learn,
                            activation_function=activation_function,
                            output_layer_choice=output_layer_choice,
                            use_log=false,
                            transform_to_tau_rho=false,
                            max_epochs=1_000
                        )
                    end
                end                
            end
        end
    end
end

if TUNE
    using Base.Threads

    println("Tuning weighted sum factor with nthreads: ", Threads.nthreads())

    datapath = "SoHEstimation/approximate_message_passing/data_masterarbeit/dataset_em_factor_5_000_1000000_Experiment_1.jld2"
    seed = 123456789
    factor = :emf
    transform_to_tau_rho = false

    dataset_map = Dict{String, Any}()
    for target in ["targets_X", "targets_Y", "targets_Z"]
        X, y = get_data(datapath; target=target, factor=factor, transform_to_tau_rho=transform_to_tau_rho)
        X, y, _, _ = remove_outliers(X, y)
        dataset_map[target] = (X, y)
    end

    #Threads.@threads for target in ["targets_X", "targets_Y", "targets_Z"]
    for target in ["targets_X", "targets_Y", "targets_Z"]
        X, y = dataset_map[target]
        tune_network(
            X, y;
            target=target,
            moment=:both,
            factor=factor,
            seed=seed,
            transform_to_tau_rho=transform_to_tau_rho,
        )
    end
end