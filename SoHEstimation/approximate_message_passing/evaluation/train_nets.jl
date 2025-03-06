include("../nn/mjl.jl")
using .NN: train_network, get_data, remove_outliers, rmse_loss
using ProgressMeter
using Random
using Format
using Base.Threads
using DelimitedFiles

seed = 707542
Random.seed!(seed)
max_epochs = 2_000

data_map = Dict{String, Any}()

ns = [10_000]

factors = [:emf]

combinations = [(n, factor) for n in ns, factor in factors]
n_combinations = length(combinations)

progressbar = Progress(n_combinations, desc="Loading data...")
for (n, factor) in combinations
    if factor == :wsf || factor == :emf
        targets = ["targets_X", "targets_Y", "targets_Z"]
    elseif factor == :gmf
        targets = ["targets_X", "targets_Y"]
    end
    for target in targets
        if factor == :wsf
            factor_path = "dataset_weighted_sum_factor_"
		elseif  factor == :gmf
            factor_path = "dataset_gaussian_mean_factor_"
        else
			factor_path = "dataset_em_factor_"
		end

        for i in 1:3
            nstring = replace(format(n, commas=true), "," => "_")
            filename = factor_path * "$(nstring)_1000000_Experiment_$(i).jld2"
            datapath = "SoHEstimation/approximate_message_passing/data_masterarbeit/" * filename

			# Check if the file exists
			if !isfile(datapath)
				error("File not found: $datapath")
			end

            X, y =  get_data(datapath; target=target, factor=factor, transform_to_tau_rho=false, verbose = 0)
			println("Size of X: ", size(X))
            X, y, _, _ = remove_outliers(X, y; verbose = 0)
			println("Size of X after removal: ", size(X))
            key = "$(factor)_$(n)_1000000_$(target)_$i"
            data_map[key] = (X, y)
        end
    end
    next!(progressbar)
end
finish!(progressbar)

params_dict = Dict(
	"gmf_X" => Dict(
		:n_neurons => 2024,
		:n_layers => 2,
		:activation_function => "relu",
		:batch_size => 512,
		:output_layer_choice => "relu",
		:target => "targets_X",
		:moment => :both,
		:transform_to_tau_rho => false,
		:scale => :minmax,
		:scale_output => :minmax,
		:loss_function => rmse_loss,
		:factor => "gmf",
	),
	"gmf_Y" => Dict(
		:n_neurons => 2024,
		:n_layers => 3,
		:activation_function => "relu",
		:batch_size => 512,
		:output_layer_choice => "softplus",
		:target => "targets_Y",
		:moment => :both,
		:transform_to_tau_rho => false,
		:scale => :minmax,
		:scale_output => :minmax,
		:loss_function => rmse_loss,
		:factor => "gmf",
	),
	"wsf_X" => Dict(
		:n_neurons => 512,
		:n_layers => 2,
		:activation_function => "tanh_fast",
		:batch_size => 512,
		:output_layer_choice => "relu",
		:target => "targets_X",
		:moment => :both,
		:transform_to_tau_rho => false,
		:scale => :zscore,
		:scale_output => :zscore,
		:loss_function => rmse_loss,
		:factor => "wsf",
	),
	"wsf_Y" => Dict(
		:n_neurons => 512,
		:n_layers => 2,
		:activation_function => "tanh_fast",
		:batch_size => 512,
		:output_layer_choice => "softplus",
		:target => "targets_Y",
		:moment => :both,
		:transform_to_tau_rho => false,
		:scale => :zscore,
		:scale_output => :zscore,
		:loss_function => rmse_loss,
		:factor => "wsf",
	),
	"wsf_Z" => Dict(
		:n_neurons => 512,
		:n_layers => 1,
		:activation_function => "tanh_fast",
		:batch_size => 512,
		:output_layer_choice => "softplus",
		:target => "targets_Z",
		:moment => :both,
		:transform_to_tau_rho => false,
		:scale => :zscore,
		:scale_output => :minmax,
		:loss_function => rmse_loss,
		:factor => "wsf",
	),
	"emf_X" => Dict(
		:n_neurons => 16,
		:n_layers => 1,
		:activation_function => "relu",
		:batch_size => 512,
		:output_layer_choice => "softplus",
		:target => "targets_X",
		:moment => :both,
		:transform_to_tau_rho => false,
		:scale => :zscore,
		:scale_output => :zscore,
		:loss_function => rmse_loss,
		:factor => "emf",
	),
	"emf_Y" => Dict(
		:n_neurons => 2024,
		:n_layers => 1,
		:activation_function => "tanh_fast",
		:batch_size => 512,
		:output_layer_choice => "relu",
		:target => "targets_Y",
		:moment => :both,
		:transform_to_tau_rho => false,
		:scale => :minmax,
		:scale_output => :zscore,
		:loss_function => rmse_loss,
		:factor => "emf",
	),
	"emf_Z" => Dict(
		:n_neurons => 512,
		:n_layers => 2,
		:activation_function => "tanh_fast",
		:batch_size => 512,
		:output_layer_choice => "softplus",
		:target => "targets_Z",
		:moment => :both,
		:transform_to_tau_rho => false,
		:scale => :zscore,
		:scale_output => :none,
		:loss_function => rmse_loss,
		:factor => "emf",
	),
)

progressbar = Progress(length(data_map), desc="Training networks...")
for (key, (X, y)) in data_map
    # update description of progressbar
    progressbar.desc = "Training network for $(key)"
	println("Key: ", key)

    factor, n, samples_per_input, target, variable, i = split(key, "_")
    target = target * "_$(variable)"

    params = deepcopy(params_dict[factor*"_$(variable)"])
    train_network(X, y, params, seed, max_epochs, :both, target; appendix="Experiment_$(i)", save=true)
    next!(progressbar)
end
finish!(progressbar)