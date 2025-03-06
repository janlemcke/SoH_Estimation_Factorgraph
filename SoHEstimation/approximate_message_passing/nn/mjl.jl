module NN
export train_network, load_model, predict_sample, experiment_data_size_vs_neurons, get_data, remove_outliers, analyze_outlier

using DataFrames: DataFrames

using JLD2
using MLJ
using MLJFlux: MLJFlux
using Flux
using CUDA
using StatsBase
using Plots
using TableTransforms
using Random
using IterTools
using ProgressMeter
using Dates
using DelimitedFiles
using Tables

include("./layer.jl")
using .Layer: ResidualMinimum

include("./custom_nn.jl")
using .CustomNN: MyNetworkBuilder

include("../../../lib/gaussian.jl")
using .GaussianDistribution: Gaussian1D, Gaussian1DFromMeanVariance

# include("./loss_functions.jl")
# using .LossFunctions: rmse_loss, huber_loss, mae_loss, mape_loss

include("../weighted_sum_factor/generate_data_weighted_sum_factor.jl")
include("../gaussian_mean_factor/generate_data_gaussian_mean_factor.jl")
include("../em_factor/generate_dataset_em_factor.jl")

MultitargetNeuralNetworkRegressor = MLJ.@load MultitargetNeuralNetworkRegressor pkg = MLJFlux verbosity = 0
Standardizer = MLJ.@load Standardizer pkg = MLJModels verbosity = 0
NeuralNetworkRegressor = MLJ.@load NeuralNetworkRegressor pkg = MLJFlux verbosity = 0


function rmse_loss(yhat::AbstractArray, y::AbstractArray)
	return sqrt(mean((y .- yhat) .^ 2))
end

function rmspe_loss(yhat::AbstractArray, y::AbstractArray)
	return sqrt(mean(((y .- yhat) ./ y) .^ 2))
end

function mae_loss(yhat::AbstractArray, y::AbstractArray)
	return mean(abs.(y .- yhat))
end

function mape_loss(yhat::AbstractArray, y::AbstractArray)
    mask = y .!= 0  # Create a mask where y is nonzero
    return mean(abs.((y[mask] .- yhat[mask]) ./ y[mask]))
end

function huber_loss(yhat::AbstractArray, y::AbstractArray)
	delta = 1.0
	diff = abs.(y .- yhat)
	return mean(ifelse.(diff .<= delta, 0.5 * diff .^ 2, delta * (diff .- 0.5 * delta)))
end


function load_model(model_path)
	# check if model ends with jld2 or jls
	if endswith(model_path, ".jld2")
		model_store = JLD2.load(model_path)
		return restore!(model_store["model"]), model_store["scaling_params"], model_store["scaling"], model_store["scaling_params_output"], model_store["scaling_output"]
	elseif endswith(model_path, ".jls")
		machine = MLJ.machine(model_path)
		#switch model path from .jls to jld2
		model_path = replace(model_path, ".jls" => "_scaling_params.jld2")
		model_store = JLD2.load(model_path)
		model_store["model"] = machine
		return model_store["model"], model_store["scaling_params"], model_store["scaling"], model_store["scaling_params_output"], model_store["scaling_output"]
	else
		error("Model path must end with .jld2 or .jls")
	end
end

function predict_sample(model_store, x::AbstractVector)
	model, scaling_params, scaling, scaling_params_output, scaling_output = model_store

	if length(x) == 9 #WSF
		X = DataFrames.DataFrame("mean_X" => x[1], "var_X" => x[2], "mean_Y" => x[3], "var_Y" => x[4], "mean_Z" => x[5], "var_Z" => x[6], "a" => x[7], "b" => x[8], "c" => x[9])
	elseif  length(x) == 5 #GMF
		X = DataFrames.DataFrame("mean_X" => x[1], "var_X" => x[2], "mean_Y" => x[3], "var_Y" => x[4], "beta" => x[5])
	elseif length(x) == 8 #EMF
		X = DataFrames.DataFrame("mean_X" => x[1], "var_X" => x[2], "mean_Y" => x[3], "var_Y" => x[4], "mean_Z" => x[5], "var_Z" => x[6], "q0" => x[7], "dt" => x[8])
	end
	X = Matrix{Float32}(X)
	#original_X = deepcopy(X)
	X = scale(X, scaling = scaling, scaling_params = scaling_params)
	prediction = MLJ.predict(model, table(X))

	if scaling_output != :none
		prediction = descale(prediction, scaling = scaling_output, scaling_params = scaling_params_output)
	end

	#if typeof(prediction) != Matrix{Float32}
	#		X = Tables.matrix(X)
	#end 

	prediction = Matrix{Float32}(prediction)

	#println("Prediction: ", prediction, " for input $original_X and scaled_input $X")
	return prediction
end

using Optimisers
function get_model(;
	n_neurons::Int,
	n_layers::Int,
	activation_function::String,
	batch_size::Int,
	output_layer_choice::String,
	epochs::Int = 500,
	target::String,
	moment = :both,
	scale::Symbol = :none,
	scaling_params::Union{Tuple{Matrix{Float32}, Matrix{Float32}}, Nothing} = nothing,
	scale_output::Symbol = :none,
	scaling_params_output::Union{Tuple{Matrix{Float32}, Matrix{Float32}}, Nothing} = nothing,
	loss_function = :rmse,
	transform_to_tau_rho = false,
)
	myBuilder = MyNetworkBuilder(n_neurons, n_layers, activation_function, output_layer_choice, target, moment, scale, scaling_params, scale_output, scaling_params_output, transform_to_tau_rho)
	optimiser = Optimisers.Adam()  # use the Adam optimiser with its default settings
	# Create the final model using MultitargetNeuralNetworkRegressor

	acceleration = CPU1()

	if CUDA.functional()
		acceleration = CUDALibs()
	end

	if moment == :both
		model = MultitargetNeuralNetworkRegressor(
			builder = myBuilder,
			batch_size = batch_size,
			epochs = epochs,
			acceleration = acceleration,
			loss = loss_function,
			optimiser = optimiser,
		)
	else
		model = NeuralNetworkRegressor(
			builder = myBuilder,
			batch_size = batch_size,
			epochs = epochs,
			acceleration = acceleration,
			loss = loss_function,
			optimiser = optimiser,
		)
	end

	return model, optimiser
end

# Function to compute a mask for outlier detection
function compute_outlier_mask(matrix, cols_to_check, threshold = 3)
	data_to_check = matrix[:, cols_to_check]
	column_means = mean(data_to_check, dims = 1)
	column_stds = std(data_to_check, dims = 1)

	# Avoid division by zero for columns with zero standard deviation
	column_stds[column_stds.==0] .= 1.0

	zscores = abs.((data_to_check .- column_means) ./ column_stds)

	# Assert that Z-scores are finite
	if any(isnan.(zscores))
		error("Z-scores contain NaN values. Skipping outlier removal.")
	end

	# Mask where no value exceeds the threshold in any column
	return all(zscores .< threshold, dims = 2)
end

function analyze_outlier(X, y_x, y_y, y_z)
	# Exclude the last column from X for outlier analysis
	x_mask = compute_outlier_mask(X, 1:(size(X, 2)-1))

	# Check all columns of y for outliers
	y_x_mask = compute_outlier_mask(y_x, 1:size(y_x, 2))
	y_y_mask = compute_outlier_mask(y_y, 1:size(y_y, 2))
	y_z_mask = compute_outlier_mask(y_z, 1:size(y_z, 2))

	# Combine masks using logical AND
	combined_mask = x_mask .& y_x_mask .& y_y_mask .& y_z_mask

	return combined_mask
end

function remove_outliers(X, y; verbose = 0)
	old_shape_X = size(X)
	old_shape_y = size(y)

	if verbose > 0
		println("Old shape: ", old_shape_X, " and ", old_shape_y)
	end

	x_mask = compute_outlier_mask(X, 1:size(X, 2))

	# Check all columns of y for outliers
	y_mask = compute_outlier_mask(y, 1:size(y, 2))

	# Combine masks using logical AND
	combined_mask = x_mask .& y_mask

	# Apply combined mask to X and y
	X = X[vec(combined_mask), :]
	y = y[vec(combined_mask), :]

	new_x_shape = size(X)
	new_y_shape = size(y)
	if verbose > 0
		println("Outliers removed. New shape of X: ", new_x_shape, " and y: ", new_y_shape)
	end

	removed_x_count = old_shape_X[1] - new_x_shape[1]
	removed_y_count = old_shape_y[1] - new_y_shape[1]

	return X, y, removed_x_count, removed_y_count

end

function learn_and_scale(X; scaling = :zscore)
	# Apply scaling based on the scaling parameter
	scaling_params = nothing
	if scaling == :minmax
		# Min-Max scaling
		min_vals = minimum(X, dims = 1)
		max_vals = maximum(X, dims = 1)
		range_vals = max_vals .- min_vals
		range_vals[range_vals.==0] .= 1.0  # Avoid division by zero for constant columns
		X = (X .- min_vals) ./ range_vals
		scaling_params = (min_vals, max_vals)
		@assert all(0 .<= X .<= 1) "Min-Max scaling failed: not all values are in [0, 1]"

	elseif scaling == :zscore
		# Z-score normalization
		means = mean(X, dims = 1)
		stds = std(X, dims = 1)
		stds[stds.==0] .= 1.0  # Avoid division by zero for constant columns
		X = (X .- means) ./ stds
		scaling_params = (means, stds)

	elseif scaling == :log
		X = log.(X .+ 1)
		scaling_params = nothing
	else
		error("Invalid scaling method. Use :minimal for Min-Max scaling or :zscore for Z-score normalization or :log for log scaling.")
	end


	return X, scaling_params
end

function scale(X; scaling, scaling_params)
	if scaling == :minmax
		# Min-Max scaling
		min_vals, max_vals = scaling_params
		range_vals = max_vals .- min_vals
		range_vals[range_vals.==0] .= 1.0  # Avoid division by zero for constant columns
		X = (X .- min_vals) ./ range_vals

	elseif scaling == :zscore
		# Z-score normalization
		means, stds = scaling_params
		stds[stds.==0] .= 1.0  # Avoid division by zero for constant columns
		X = (X .- means) ./ stds
	elseif scaling == :log
		X = log.(X .+ 1)
	else
		error("Invalid scaling method $(scaling). Use :minimal for Min-Max scaling or :zscore for Z-score normalization.")
	end

	return X
end

function descale(X; scaling, scaling_params)

	if typeof(X) == Matrix{Float32} && CUDA.functional()
		X = CUDA.CuArray(X)
	end

	if scaling == :minmax
		# Undo Min-Max scaling
		min_vals, max_vals = scaling_params
		if CUDA.functional()
			min_vals = CUDA.CuArray(min_vals)
			max_vals = CUDA.CuArray(max_vals)
		end
		range_vals = max_vals .- min_vals
		#range_vals[range_vals.==0] .= 1.0  # Avoid division by zero for constant columns
		X = X .* range_vals .+ min_vals

	elseif scaling == :zscore
		# Undo Z-score normalization
		means, stds = scaling_params
		if CUDA.functional()
			means = CUDA.CuArray(means)
			stds = CUDA.CuArray(stds)
		end
		#stds[stds.==0] .= 1.0  # Avoid division by zero for constant columns
		X = X .* stds .+ means

	elseif scaling == :log
		X = exp.(X) .- 1
	else
		error("Invalid scaling method $(scaling). Use :minimal for Min-Max scaling or :zscore for Z-score normalization or :log for log scaling.")
	end

	return X
end

function transform_polynomial(X; degree = 2)
	# Ensure degree is valid
	@assert degree >= 1 "Degree must be at least 1."

	# Include the original features in the transformed dataset
	X_transformed = X

	# Add higher-order polynomial terms
	if degree > 1
		polynomial_terms = hcat([X .^ d for d in 2:degree]...)  # Start from d=2 to avoid duplicating X
		X_transformed = hcat(X_transformed, polynomial_terms)
	end

	# Add reciprocal features
	reciprocal_terms = 1.0 ./ X
	# cap the reciprocal terms to avoid infinity
	reciprocal_terms[isinf.(reciprocal_terms)] .= 1.0

	X_transformed = hcat(X_transformed, reciprocal_terms)

	# Add interaction terms
	n_features = size(X, 2)
	interaction_terms = [X[:, i] .* X[:, j] for i in 1:n_features for j in (i+1):n_features]
	if !isempty(interaction_terms)
		interaction_terms_matrix = hcat(interaction_terms...)
		X_transformed = hcat(X_transformed, interaction_terms_matrix)
	end

	# convert to Float32
	X_transformed = Matrix{Float32}(X_transformed)
	return X_transformed
end



function get_data(datapath; target = "targets_X", factor = :gmf, transform_to_tau_rho = false, verbose = 0)
	# Check if models folder exists, if not create it
	if !isdir("SoHEstimation/approximate_message_passing/data")
		error("Data folder does not exist. Please generate data first.")
	end

	if verbose > 0
		println("[get_data] Factor: ", factor, " Transform to tau rho: ", transform_to_tau_rho)
	end

	# Load dataset
	# check whether "samples" can be load_model
	if !haskey(JLD2.load(datapath), "samples")
		if target == "targets_X"
			X = JLD2.load(datapath, "inputs_X")
		elseif target == "targets_Y"
			X = JLD2.load(datapath, "inputs_Y")
		elseif target == "targets_Z"
			X = JLD2.load(datapath, "inputs_Z")
		else
			error("Invalid target. Use 'targets_X', 'targets_Y', or 'targets_Z'.")
		end
	else
		X = JLD2.load(datapath, "samples")
	end

	# rho = precision, tau = precision mean,
	tau_X = []
	rho_X = []
	tau_Y = []
	rho_Y = []

	if factor == :gmf
		β2 = []
	elseif factor == :wsf
		tau_Z = []
		rho_Z = []
		a = []
		b = []
		c = []
	elseif factor == :emf
		tau_Z = []
		rho_Z = []
		q0 = []
		dt = []
	end

	for i in eachindex(X)
		push!(tau_X, X[i][1])
		push!(rho_X, X[i][2])
		push!(tau_Y, X[i][3])
		push!(rho_Y, X[i][4])
		if factor == :gmf
			push!(β2, X[i][5])
		elseif factor == :wsf
			push!(tau_Z, X[i][5])
			push!(rho_Z, X[i][6])
			push!(a, X[i][7])
			push!(b, X[i][8])
			push!(c, X[i][9])
		elseif factor == :emf
			push!(tau_Z, X[i][5])
			push!(rho_Z, X[i][6])
			push!(q0, X[i][7])
			push!(dt, X[i][8])
		end
	end

	if factor == :gmf
		X = DataFrames.DataFrame("tau_X" => tau_X, "rho_X" => rho_X, "tau_Y" => tau_Y, "rho_Y" => rho_Y, "β2" => β2)
	elseif factor == :wsf
		X = DataFrames.DataFrame("tau_X" => tau_X, "rho_X" => rho_X, "tau_Y" => tau_Y, "rho_Y" => rho_Y, "tau_Z" => tau_Z, "rho_Z" => rho_Z, "a" => a, "b" => b, "c" => c)
	elseif factor == :emf
		X = DataFrames.DataFrame("tau_X" => tau_X, "rho_X" => rho_X, "tau_Y" => tau_Y, "rho_Y" => rho_Y, "tau_Z" => tau_Z, "rho_Z" => rho_Z, "q0" => q0, "dt" => dt)
	end

	y = JLD2.load(datapath, target)

	taus = []
	rhos = []
	for i in eachindex(y)
		push!(taus, y[i][1])
		push!(rhos, y[i][2])
	end

	y = DataFrames.DataFrame("tau" => taus, "rho" => rhos)

	if verbose > 0
		println("Shape of X: ", size(X), " and shape of y: ", size(y))
	end

	X = Matrix{Float32}(X)
	y = Matrix{Float32}(y)

	# If we want to learn in mean / var not in precision mean / precision
	if transform_to_tau_rho
		if factor == :wsf
			X = WeightedSumFactorGeneration.to_tau_rho(X)
			y = WeightedSumFactorGeneration.to_tau_rho(y)
		elseif factor == :emf
			X = ElectricalModelFactorGeneration.to_tau_rho(X)
			y = ElectricalModelFactorGeneration.to_tau_rho(y)
		elseif factor == :gmf
			X = GaussianMeanFactorGeneration.to_tau_rho(X)
			y = GaussianMeanFactorGeneration.to_tau_rho(y)
		end
	else
		indexes = []
		for i in 1:size(X, 1)
			if (factor == :wsf || factor == :emf) && (X[i, 2] == Inf || X[i, 4] == Inf || X[i, 6] == Inf || y[i, 2] == Inf)
				push!(indexes, i)
			end

			if factor == :gmf && (X[i, 2] == Inf || X[i, 4] == Inf || y[i, 2] == Inf)
				push!(indexes, i)
			end
		end

		if verbose > 0
			println("Found ", length(indexes), " samples with variance == Inf (Uniform Distribution). Removing them.")
		end

		X = X[setdiff(1:size(X, 1), indexes), :]
		y = y[setdiff(1:size(y, 1), indexes), :]
	end

	X = Matrix{Float32}(X)
	y = Matrix{Float32}(y)

	# assert that X and y do not contain Inf values
	@assert !any(isinf.(X)) "X contains Inf values."
	@assert !any(isinf.(y)) "y contains Inf values."

	if verbose > 0
		println("Shape of X: ", size(X), " and shape of y: ", size(y))
	end
	return X, y

end


function run_experiment(X, y, params, nfolds, seed, max_epochs, architecture, target::String; progressbar = nothing)
	rng = Xoshiro(seed)
	key = pop!(params, :key)

	# get kfold split
	if nfolds > 1
		folds = get_folds(y, nfolds, rng)
	else
		folds = []
		train_indices, test_indices = MLJ.partition(1:size(y, 1), 0.8, rng = rng, shuffle = true)
		push!(folds, (train_indices, test_indices))
	end

	rmse_marginal_prediction_exact = zeros(nfolds, 4)  # mean, var, rho, tau
	rmse_message_prediction_exact = zeros(nfolds, 4)
	rmse_marginal_prediction_label = zeros(nfolds, 4)
	rmse_message_prediction_label = zeros(nfolds, 4)

	mae_marginal_prediction_exact = zeros(nfolds, 4)
	mae_message_prediction_exact = zeros(nfolds, 4)
	mae_marginal_prediction_label = zeros(nfolds, 4)
	mae_message_prediction_label = zeros(nfolds, 4)

	mape_marginal_prediction_exact = zeros(nfolds, 4)
	mape_message_prediction_exact = zeros(nfolds, 4)
	mape_marginal_prediction_label = zeros(nfolds, 4)
	mape_message_prediction_label = zeros(nfolds, 4)

	fold_rmse = []
	fold_mae = []
	fold_mape = []
	fold_huber = []

	factor = params[:factor]
	delete!(params, :factor)

	for fold_index in 1:nfolds
		train_indexes, test_indexes = folds[fold_index]
		X_train, y_train = X[train_indexes, :], y[train_indexes, :]
		X_test, y_test = X[test_indexes, :], y[test_indexes, :]
		X_test_evaluation = deepcopy(X_test)

		params[:scaling_params] = nothing
		params[:scaling_params_output] = nothing

		# Do scaling
		if params[:scale] != :none
			X_train, scaling_params = learn_and_scale(X_train, scaling = params[:scale])
			X_test_unscaled = deepcopy(X_test)
			X_test = scale(X_test, scaling = params[:scale], scaling_params = scaling_params)
			params[:scaling_params] = scaling_params
		end

		if params[:scale_output] != :none
			y_train, scaling_params_output = learn_and_scale(y_train, scaling = params[:scale_output])
			params[:scaling_params_output] = scaling_params_output
		end

		function update_lr(mach)
			# get number of epochs
			losses = report(mach).training_losses

			# if the last 10 losses are all the same or higher, reduce the learning rate
			if cooldown_counter == 0 && length(losses) > 10 && all(losses[end-10:end] .>= losses[end]) && opt.eta > 0.000001
				old_lr = mach.model.optimiser.eta
				new_lr = max(0.000001, 0.95 * old_lr)
				opt = Optimisers.ADAM(new_lr)
				mach.model.optimiser = opt
				cooldown_counter = 10  # set cooldown period
			end

			if cooldown_counter > 0
				cooldown_counter -= 1
			end

			updated += 1
		end

		controls = [Step(1),
			Patience(20),
			InvalidValue(),
			NumberLimit(max_epochs),
			Callback(update_lr),]

		if architecture == :both
			model, opt = get_model(; params...)
			updated = 0
			cooldown_counter = 0

			it_model = IteratedModel(
				model = model,
				resampling = nothing,
				controls = controls,
				measure = params[:loss_function],
				iteration_parameter = iteration_parameter(model),
			)

			mach = machine(it_model, table(X_train), y_train)

			fit!(mach, verbosity = 0)

			# Evaluate on validation set
			predictions = MLJ.predict(mach, table(X_test))
			# put predictions to CPU back
			predictions = Matrix{Float32}(predictions)

			# descale predictions
			if params[:scale_output] != :none
				predictions = descale(predictions, scaling = params[:scale_output], scaling_params = params[:scaling_params_output])
				# put predictions to CPU back
				predictions = Matrix{Float32}(predictions)
			end

		elseif architecture == :single
			params[:moment] = :first
			model, opt = get_model(; params...)
			updated = 0
			cooldown_counter = 0

			it_model = IteratedModel(
				model = model,
				resampling = nothing,
				controls = controls,
				measure = params[:loss_function],
				iteration_parameter = iteration_parameter(model),
			)

			mach = machine(it_model, table(X_train), y_train[:, 1:1])
			fit!(mach, verbosity = 0)
			predictions_first = MLJ.predict(mach, table(X_test))

			# Second Moment
			params[:moment] = :second
			model, opt = get_model(; params...)
			updated = 0
			cooldown_counter = 0

			it_model = IteratedModel(
				model = model,
				resampling = nothing,
				controls = controls,
				measure = params[:loss_function],
				iteration_parameter = iteration_parameter(model),
			)

			mach = machine(it_model, table(X_train), y_train[:, 2:2])
			fit!(mach, verbosity = 0)
			predictions_second = MLJ.predict(mach, table(X_test))

			# concat predictions
			predictions = hcat(predictions_first, predictions_second)

		else
			error("Invalid moment parameter. Use :both or :single instead of: ", architecture)
		end

		if params[:transform_to_tau_rho]
			println("Converting again...")
			if factor == "wsf"
				predictions = WeightedSumFactorGeneration.to_mean_variance(predictions)
				y_test = WeightedSumFactorGeneration.to_mean_variance(y_test)
			elseif factor == "em"
				predictions = ElectricalModelFactorGeneration.to_mean_variance(predictions)
				y_test = ElectricalModelFactorGeneration.to_mean_variance(y_test)
			elseif factor == "gmf"
				predictions = GaussianMeanFactorGeneration.to_mean_variance(predictions)
				y_test = GaussianMeanFactorGeneration.to_mean_variance(y_test)
			end
		end

		rmse_per_feature = sqrt.(mean((y_test .- predictions) .^ 2, dims = 1))
		mae_per_feature = mean(abs.(y_test .- predictions), dims = 1)
		mape_per_feature = mean(abs.((y_test .- predictions) ./ y_test), dims = 1)
		delta = 1.0
		huber_per_feature = mean(ifelse.(abs.(y_test .- predictions) .<= delta, 0.5 * (y_test .- predictions) .^ 2, delta * (abs.(y_test .- predictions) .- 0.5 * delta)), dims = 1)

		println("rmse_per_feature: ", rmse_per_feature)
		println("mae_per_feature: ", mae_per_feature)
		println("mape_per_feature: ", mape_per_feature)

		# Evaluate against analytical solution
		metrics = Matrix{Float32}(undef, 48, size(y_test, 1))

		for i in 1:size(X_test_evaluation, 1)
			input = X_test_evaluation[i, :]
			if factor == "wsf"
				if target == "targets_X"
					exact_marginal, exact_msg = WeightedSumFactorGeneration.calc_msg_x(input, params[:transform_to_tau_rho])
					if !params[:transform_to_tau_rho]
						predicted_marginal = Gaussian1DFromMeanVariance(predictions[i, 1], predictions[i, 2])
						msg_back = Gaussian1DFromMeanVariance(WeightedSumFactorGeneration.get_variable(WeightedSumFactorGeneration.X, input)...)
						predicted_msg = predicted_marginal / msg_back

						label_marginal = Gaussian1DFromMeanVariance(y_test[i, 1], y_test[i, 2])
						label_msg = label_marginal / msg_back
					else
						predicted_marginal = Gaussian1D(predictions[i, 1], predictions[i, 2])
						msg_back = Gaussian1D(GaussianMeanFactorGeneration.get_variable(GaussianMeanFactorGeneration.X, input)...)
						predicted_msg = predicted_marginal / msg_back

						label_marginal = Gaussian1D(y_test[i, 1], y_test[i, 2])
						label_msg = label_marginal / msg_back
					end
				elseif target == "targets_Y"
					exact_marginal, exact_msg = WeightedSumFactorGeneration.calc_msg_y(input, params[:transform_to_tau_rho])
					if !params[:transform_to_tau_rho]
						predicted_marginal = Gaussian1DFromMeanVariance(predictions[i, 1], predictions[i, 2])
						msg_back = Gaussian1DFromMeanVariance(WeightedSumFactorGeneration.get_variable(WeightedSumFactorGeneration.Y, input)...)
						predicted_msg = predicted_marginal / msg_back

						label_marginal = Gaussian1DFromMeanVariance(y_test[i, 1], y_test[i, 2])
						label_msg = label_marginal / msg_back
					else
						predicted_marginal = Gaussian1D(predictions[i, 1], predictions[i, 2])
						msg_back = Gaussian1D(GaussianMeanFactorGeneration.get_variable(GaussianMeanFactorGeneration.Y, input)...)
						predicted_msg = predicted_marginal / msg_back

						label_marginal = Gaussian1D(y_test[i, 1], y_test[i, 2])
						label_msg = label_marginal / msg_back
					end
				elseif target == "targets_Z"
					exact_marginal, exact_msg = WeightedSumFactorGeneration.calc_msg_z(input, params[:transform_to_tau_rho])
					if !params[:transform_to_tau_rho]
						predicted_marginal = Gaussian1DFromMeanVariance(predictions[i, 1], predictions[i, 2])
						msg_back = Gaussian1DFromMeanVariance(WeightedSumFactorGeneration.get_variable(WeightedSumFactorGeneration.Z, input)...)
						predicted_msg = predicted_marginal / msg_back
						if predicted_msg.rho == 0
							error("Predicted message rho is zero: ", predicted_msg, "predicted marginal: ", predicted_marginal, "msg_back: ", msg_back, "exact_msg: ", exact_msg, "exact_marginal: ", exact_marginal)
						end
						label_marginal = Gaussian1DFromMeanVariance(y_test[i, 1], y_test[i, 2])
						label_msg = label_marginal / msg_back
					else
						predicted_marginal = Gaussian1D(predictions[i, 1], predictions[i, 2])
						msg_back = Gaussian1D(GaussianMeanFactorGeneration.get_variable(GaussianMeanFactorGeneration.Z, input)...)
						predicted_msg = predicted_marginal / msg_back

						label_marginal = Gaussian1D(y_test[i, 1], y_test[i, 2])
						label_msg = label_marginal / msg_back
					end
				end
			elseif factor == "gmf"
				if target == "targets_X"
					exact_marginal, exact_msg = GaussianMeanFactorGeneration.calc_msg_x(input, params[:transform_to_tau_rho])
					if !params[:transform_to_tau_rho]
						predicted_marginal = Gaussian1DFromMeanVariance(predictions[i, 1], predictions[i, 2])
						msg_back = Gaussian1DFromMeanVariance(GaussianMeanFactorGeneration.get_variable(GaussianMeanFactorGeneration.X, input)...)
						predicted_msg = predicted_marginal / msg_back

						label_marginal = Gaussian1DFromMeanVariance(y_test[i, 1], y_test[i, 2])
						label_msg = label_marginal / msg_back
					else
						predicted_marginal = Gaussian1D(predictions[i, 1], predictions[i, 2])
						msg_back = Gaussian1D(GaussianMeanFactorGeneration.get_variable(GaussianMeanFactorGeneration.X, input)...)
						predicted_msg = predicted_marginal / msg_back

						label_marginal = Gaussian1D(y_test[i, 1], y_test[i, 2])
						label_msg = label_marginal / msg_back
					end
				elseif target == "targets_Y"
					exact_marginal, exact_msg = GaussianMeanFactorGeneration.calc_msg_y(input, params[:transform_to_tau_rho])
					if !params[:transform_to_tau_rho]
						predicted_marginal = Gaussian1DFromMeanVariance(predictions[i, 1], predictions[i, 2])
						msg_back = Gaussian1DFromMeanVariance(GaussianMeanFactorGeneration.get_variable(GaussianMeanFactorGeneration.Y, input)...)
						predicted_msg = predicted_marginal / msg_back

						label_marginal = Gaussian1DFromMeanVariance(y_test[i, 1], y_test[i, 2])
						label_msg = label_marginal / msg_back
					else
						predicted_marginal = Gaussian1D(predictions[i, 1], predictions[i, 2])
						msg_back = Gaussian1D(GaussianMeanFactorGeneration.get_variable(GaussianMeanFactorGeneration.Y, input)...)
						predicted_msg = predicted_marginal / msg_back

						label_marginal = Gaussian1D(y_test[i, 1], y_test[i, 2])
						label_msg = label_marginal / msg_back
					end
				end
			else
				error("Factor $factor not supported.")
			end

			metric = GaussianDistribution.squared_diff(predicted_marginal, Gaussian1D(exact_marginal.tau, exact_marginal.rho))
			metrics[1, i] = metric[1]
			metrics[2, i] = metric[2]
			metrics[3, i] = metric[3]
			metrics[4, i] = metric[4]

			metric = GaussianDistribution.squared_diff(predicted_msg, Gaussian1D(exact_msg.tau, exact_msg.rho))
			metrics[5, i] = metric[1]
			metrics[6, i] = metric[2]
			metrics[7, i] = metric[3]
			metrics[8, i] = metric[4]

			metric = GaussianDistribution.squared_diff(predicted_marginal, label_marginal)
			metrics[9, i] = metric[1]
			metrics[10, i] = metric[2]
			metrics[11, i] = metric[3]
			metrics[12, i] = metric[4]

			metric = GaussianDistribution.squared_diff(predicted_msg, label_msg)
			metrics[13, i] = metric[1]
			metrics[14, i] = metric[2]
			metrics[15, i] = metric[3]
			metrics[16, i] = metric[4]

			metric = GaussianDistribution.absolute_error(predicted_marginal, Gaussian1D(exact_marginal.tau, exact_marginal.rho))
			metrics[17, i] = metric[1]
			metrics[18, i] = metric[2]
			metrics[19, i] = metric[3]
			metrics[20, i] = metric[4]

			metric = GaussianDistribution.absolute_error(predicted_msg, Gaussian1D(exact_msg.tau, exact_msg.rho))
			metrics[21, i] = metric[1]
			metrics[22, i] = metric[2]
			metrics[23, i] = metric[3]
			metrics[24, i] = metric[4]

			metric = GaussianDistribution.absolute_error(predicted_marginal, label_marginal)
			metrics[25, i] = metric[1]
			metrics[26, i] = metric[2]
			metrics[27, i] = metric[3]
			metrics[28, i] = metric[4]

			metric = GaussianDistribution.absolute_error(predicted_msg, label_msg)
			metrics[29, i] = metric[1]
			metrics[30, i] = metric[2]
			metrics[31, i] = metric[3]
			metrics[32, i] = metric[4]

			metric = GaussianDistribution.absolute_percentage_error(predicted_marginal, Gaussian1D(exact_marginal.tau, exact_marginal.rho))
			metrics[33, i] = metric[1]
			metrics[34, i] = metric[2]
			metrics[35, i] = metric[3]
			metrics[36, i] = metric[4]

			metric = GaussianDistribution.absolute_percentage_error(predicted_msg, Gaussian1D(exact_msg.tau, exact_msg.rho))
			metrics[37, i] = metric[1]
			metrics[38, i] = metric[2]
			metrics[39, i] = metric[3]
			metrics[40, i] = metric[4]

			metric = GaussianDistribution.absolute_percentage_error(predicted_marginal, label_marginal)
			metrics[41, i] = metric[1]
			metrics[42, i] = metric[2]
			metrics[43, i] = metric[3]
			metrics[44, i] = metric[4]

			metric = GaussianDistribution.absolute_percentage_error(predicted_msg, label_msg)
			metrics[45, i] = metric[1]
			metrics[46, i] = metric[2]
			metrics[47, i] = metric[3]
			metrics[48, i] = metric[4]
		end

		rmse_marginal_prediction_exact[fold_index, :] = [
			sqrt(StatsBase.mean(metrics[1, :])),
			sqrt(StatsBase.mean(metrics[2, :])),
			sqrt(StatsBase.mean(metrics[3, :])),
			sqrt(StatsBase.mean(metrics[4, :])),
		]

		rmse_message_prediction_exact[fold_index, :] = [
			sqrt(StatsBase.mean(metrics[5, :])),
			sqrt(StatsBase.mean(metrics[6, :])),
			sqrt(StatsBase.mean(metrics[7, :])),
			sqrt(StatsBase.mean(metrics[8, :])),
		]

		rmse_marginal_prediction_label[fold_index, :] = [
			sqrt(StatsBase.mean(metrics[9, :])),
			sqrt(StatsBase.mean(metrics[10, :])),
			sqrt(StatsBase.mean(metrics[11, :])),
			sqrt(StatsBase.mean(metrics[12, :])),
		]

		rmse_message_prediction_label[fold_index, :] = [
			sqrt(StatsBase.mean(metrics[13, :])),
			sqrt(StatsBase.mean(metrics[14, :])),
			sqrt(StatsBase.mean(metrics[15, :])),
			sqrt(StatsBase.mean(metrics[16, :])),
		]

		mae_marginal_prediction_exact[fold_index, :] = [
			StatsBase.mean(metrics[17, :]),
			StatsBase.mean(metrics[18, :]),
			StatsBase.mean(metrics[19, :]),
			StatsBase.mean(metrics[20, :]),
		]

		mae_message_prediction_exact[fold_index, :] = [
			StatsBase.mean(metrics[21, :]),
			StatsBase.mean(metrics[22, :]),
			StatsBase.mean(metrics[23, :]),
			StatsBase.mean(metrics[24, :]),
		]

		mae_marginal_prediction_label[fold_index, :] = [
			StatsBase.mean(metrics[25, :]),
			StatsBase.mean(metrics[26, :]),
			StatsBase.mean(metrics[27, :]),
			StatsBase.mean(metrics[28, :]),
		]

		mae_message_prediction_label[fold_index, :] = [
			StatsBase.mean(metrics[29, :]),
			StatsBase.mean(metrics[30, :]),
			StatsBase.mean(metrics[31, :]),
			StatsBase.mean(metrics[32, :]),
		]

		mape_marginal_prediction_exact[fold_index, :] = [
			StatsBase.mean(metrics[33, :]),
			StatsBase.mean(metrics[34, :]),
			StatsBase.mean(metrics[35, :]),
			StatsBase.mean(metrics[36, :]),
		]

		mape_message_prediction_exact[fold_index, :] = [
			StatsBase.mean(metrics[37, :]),
			StatsBase.mean(metrics[38, :]),
			StatsBase.mean(metrics[39, :]),
			StatsBase.mean(metrics[40, :]),
		]

		mape_marginal_prediction_label[fold_index, :] = [
			StatsBase.mean(metrics[41, :]),
			StatsBase.mean(metrics[42, :]),
			StatsBase.mean(metrics[43, :]),
			StatsBase.mean(metrics[44, :]),
		]

		mape_message_prediction_label[fold_index, :] = [
			StatsBase.mean(metrics[45, :]),
			StatsBase.mean(metrics[46, :]),
			StatsBase.mean(metrics[47, :]),
			StatsBase.mean(metrics[48, :]),
		]

		push!(fold_rmse, rmse_per_feature)
		push!(fold_mae, mae_per_feature)
		push!(fold_mape, mape_per_feature)
		push!(fold_huber, huber_per_feature)

		if progressbar != nothing
			next!(progressbar)
		end

	end

	rmse_avg_first_moment = StatsBase.mean([fold_rmse[i][1] for i in 1:nfolds])
	rmse_avg_second_moment = StatsBase.mean([fold_rmse[i][2] for i in 1:nfolds])
	mae_avg_first_moment = StatsBase.mean([fold_mae[i][1] for i in 1:nfolds])
	mae_avg_second_moment = StatsBase.mean([fold_mae[i][2] for i in 1:nfolds])
	mape_avg_first_moment = StatsBase.mean([fold_mape[i][1] for i in 1:nfolds])
	mape_avg_second_moment = StatsBase.mean([fold_mape[i][2] for i in 1:nfolds])
	huber_avg_first_moment = StatsBase.mean([fold_huber[i][1] for i in 1:nfolds])
	huber_avg_second_moment = StatsBase.mean([fold_huber[i][2] for i in 1:nfolds])

	rmse_marginal_prediction_exact = StatsBase.mean(rmse_marginal_prediction_exact, dims = 1)
	rmse_message_prediction_exact = StatsBase.mean(rmse_message_prediction_exact, dims = 1)
	rmse_marginal_prediction_label = StatsBase.mean(rmse_marginal_prediction_label, dims = 1)
	rmse_message_prediction_label = StatsBase.mean(rmse_message_prediction_label, dims = 1)

	mae_marginal_prediction_exact = StatsBase.mean(mae_marginal_prediction_exact, dims = 1)
	mae_message_prediction_exact = StatsBase.mean(mae_message_prediction_exact, dims = 1)
	mae_marginal_prediction_label = StatsBase.mean(mae_marginal_prediction_label, dims = 1)
	mae_message_prediction_label = StatsBase.mean(mae_message_prediction_label, dims = 1)

	mape_marginal_prediction_exact = StatsBase.mean(mape_marginal_prediction_exact, dims = 1)
	mape_message_prediction_exact = StatsBase.mean(mape_message_prediction_exact, dims = 1)
	mape_marginal_prediction_label = StatsBase.mean(mape_marginal_prediction_label, dims = 1)
	mape_message_prediction_label = StatsBase.mean(mape_message_prediction_label, dims = 1)

	return Dict(
		:rmse_avg_first_moment=>rmse_avg_first_moment,
		:rmse_avg_second_moment=>rmse_avg_second_moment,
		:mae_avg_first_moment=>mae_avg_first_moment,
		:mae_avg_second_moment=>mae_avg_second_moment,
		:mape_avg_first_moment=>mape_avg_first_moment,
		:mape_avg_second_moment=>mape_avg_second_moment,
		:huber_avg_first_moment=>huber_avg_first_moment,
		:huber_avg_second_moment=>huber_avg_second_moment,
		:rmse_marginal_prediction_exact_mean=>rmse_marginal_prediction_exact[1],
		:rmse_marginal_prediction_exact_variance=>rmse_marginal_prediction_exact[2],
		:rmse_marginal_prediction_exact_rho=>rmse_marginal_prediction_exact[3],
		:rmse_marginal_prediction_exact_tau=>rmse_marginal_prediction_exact[4],
		:rmse_message_prediction_exact_mean=>rmse_message_prediction_exact[1],
		:rmse_message_prediction_exact_variance=>rmse_message_prediction_exact[2],
		:rmse_message_prediction_exact_rho=>rmse_message_prediction_exact[3],
		:rmse_message_prediction_exact_tau=>rmse_message_prediction_exact[4],
		:rmse_marginal_prediction_label_mean=>rmse_marginal_prediction_label[1],
		:rmse_marginal_prediction_label_variance=>rmse_marginal_prediction_label[2],
		:rmse_marginal_prediction_label_rho=>rmse_marginal_prediction_label[3],
		:rmse_marginal_prediction_label_tau=>rmse_marginal_prediction_label[4],
		:rmse_message_prediction_label_mean=>rmse_message_prediction_label[1],
		:rmse_message_prediction_label_variance=>rmse_message_prediction_label[2],
		:rmse_message_prediction_label_rho=>rmse_message_prediction_label[3],
		:rmse_message_prediction_label_tau=>rmse_message_prediction_label[4],
		:mae_marginal_prediction_exact_mean=>mae_marginal_prediction_exact[1],
		:mae_marginal_prediction_exact_variance=>mae_marginal_prediction_exact[2],
		:mae_marginal_prediction_exact_rho=>mae_marginal_prediction_exact[3],
		:mae_marginal_prediction_exact_tau=>mae_marginal_prediction_exact[4],
		:mae_message_prediction_exact_mean=>mae_message_prediction_exact[1],
		:mae_message_prediction_exact_variance=>mae_message_prediction_exact[2],
		:mae_message_prediction_exact_rho=>mae_message_prediction_exact[3],
		:mae_message_prediction_exact_tau=>mae_message_prediction_exact[4],
		:mae_marginal_prediction_label_mean=>mae_marginal_prediction_label[1],
		:mae_marginal_prediction_label_variance=>mae_marginal_prediction_label[2],
		:mae_marginal_prediction_label_rho=>mae_marginal_prediction_label[3],
		:mae_marginal_prediction_label_tau=>mae_marginal_prediction_label[4],
		:mae_message_prediction_label_mean=>mae_message_prediction_label[1],
		:mae_message_prediction_label_variance=>mae_message_prediction_label[2],
		:mae_message_prediction_label_rho=>mae_message_prediction_label[3],
		:mae_message_prediction_label_tau=>mae_message_prediction_label[4],
		:mape_marginal_prediction_exact_mean=>mape_marginal_prediction_exact[1],
		:mape_marginal_prediction_exact_variance=>mape_marginal_prediction_exact[2],
		:mape_marginal_prediction_exact_rho=>mape_marginal_prediction_exact[3],
		:mape_marginal_prediction_exact_tau=>mape_marginal_prediction_exact[4],
		:mape_message_prediction_exact_mean=>mape_message_prediction_exact[1],
		:mape_message_prediction_exact_variance=>mape_message_prediction_exact[2],
		:mape_message_prediction_exact_rho=>mape_message_prediction_exact[3],
		:mape_message_prediction_exact_tau=>mape_message_prediction_exact[4],
		:mape_marginal_prediction_label_mean=>mape_marginal_prediction_label[1],
		:mape_marginal_prediction_label_variance=>mape_marginal_prediction_label[2],
		:mape_marginal_prediction_label_rho=>mape_marginal_prediction_label[3],
		:mape_marginal_prediction_label_tau=>mape_marginal_prediction_label[4],
		:mape_message_prediction_label_mean=>mape_message_prediction_label[1],
		:mape_message_prediction_label_variance=>mape_message_prediction_label[2],
		:mape_message_prediction_label_rho=>mape_message_prediction_label[3],
		:mape_message_prediction_label_tau=>mape_message_prediction_label[4],
	)
end

function get_folds(X, n, rng)
	# Number of samples
	num_samples = size(X, 1)  # Number of rows in X

	# Check if n is valid
	if n > num_samples
		error("Number of folds n cannot exceed the number of samples.")
	end

	# Generate and shuffle indices
	indices = shuffle(rng, collect(1:num_samples))

	# Split indices into folds
	fold_sizes = fill(div(num_samples, n), n)
	remainder = mod(num_samples, n)
	for i in 1:remainder
		fold_sizes[i] += 1
	end

	# Create folds
	folds = Vector{Vector{Int}}(undef, n)
	start_idx = 1
	for i in 1:n
		end_idx = start_idx + fold_sizes[i] - 1
		folds[i] = indices[start_idx:end_idx]
		start_idx = end_idx + 1
	end

	# Create train-test splits
	train_test_splits = []
	for i in 1:n
		test_indices = folds[i]
		train_indices = vcat(folds[1:i-1]..., folds[i+1:end]...)
		push!(train_test_splits, (train_indices, test_indices))
	end

	return train_test_splits
end

function nested_cross_validation(X, y;
	outer_folds_amount = 5,
	inner_folds_amount = 3,
	factor = :wsf,
	target = "targets_X",
	seed = nothing,
	moment = :both,
	transform_to_tau_rho = false,
	max_epochs = 200,
	train_final_model=false,
)

	rng = Xoshiro(seed)

	path = "SoHEstimation/approximate_message_passing/tuning"
	datetime_as_string = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
	file_name = "ncv__$(factor)_$(target)_$(datetime_as_string).txt"

	# create folder path if not exist
	if !isdir(path)
		mkdir(path)
	end

	# get kfold split
	outer_folds = get_folds(y, outer_folds_amount, rng)

	# Define hyperparameter combinations
	hyperparameter_grid = product(
		[8, 16, 32, 64, 512, 1024, 2024],  # n_neurons
		[1, 2, 3],                   # n_layers
		["relu", "tanh_fast"],       # activation_function
		["softplus", "relu"],        # output_layer_choice
		[:minmax, :zscore],   # scaling input 
		[:minmax, :zscore],   # scaling output & both
		[512],                  # batch_size
		[rmse_loss],      # loss function
	)
	
	# Open the file for writing results
	open(path * "/" * file_name, "w") do io
		write(io, "Nested Cross-Validation Results\n")
		write(io, "Target: $target\n")
		write(io, "Factor: $factor\n")
		write(io, "Moment: $moment\n")
		write(io, "Seed: $seed\n")
		write(io, "Shape of X: $(size(X))\n")
		write(io, "Shape of y: $(size(y))\n")
		write(io, "------------------------------------------------------\n")
		write(io, "Hyperparameter Grid:\n")
		write(io, "N_neurons: [8, 16, 32, 64, 512, 1024, 2024]\n")
		write(io, "N_layers: [1, 2, 3]\n")
		write(io, "Activation Function: [relu, tanh_fast]\n")
		write(io, "Output Layer Choice: [softplus, relu]\n")
		write(io, "Scaling: [minmax, zscore]\n")
		write(io, "Output scaling: [minmax, zscore]\n")
		write(io, "Batch Size: [256, 512]\n")
		write(io, "Loss Function: [rmse_loss]\n")
		write(io, "Amount of combinations: $(length(hyperparameter_grid))\n")
		write(io, "------------------------------------------------------\n")
		write(io, "Outer folds: $outer_folds_amount, Inner folds: $inner_folds_amount\n")
		write(io, "------------------------------------------------------\n")
	end

	hyperparameter_grid = collect(hyperparameter_grid)

	total = length(hyperparameter_grid) * inner_folds_amount * outer_folds_amount

	# Initialize the progress bar
	p = Progress(total; desc = "Doing nested cross-validation for $(target) and $(total) combinations: ")

	# Prepare containers for results
	all_rmse = []
	best_hyperparams_overall = nothing
	best_rmse_overall = Inf

	for outer_fold_index in 1:outer_folds_amount

		train_indexes, test_indexes = outer_folds[outer_fold_index]
		X_train, y_train = X[train_indexes, :], y[train_indexes, :]
		X_test, y_test = X[test_indexes, :], y[test_indexes, :]

		# Inner cross-validation for hyperparameter tuning
		inner_folds = get_folds(y_train, inner_folds_amount, rng)
		best_hyperparams = nothing
		best_rmse = Inf

		for hp_params in hyperparameter_grid
			(n_neurons, n_layers, activation_function, output_layer_choice, scaling_type, scaling_type_output, batch_size, loss_function) = hp_params
			fold_rmses = []

			for inner_fold_index in 1:inner_folds_amount
				train_indexes_inner, val_indexes = inner_folds[inner_fold_index]
				X_train_inner = X_train[train_indexes_inner, :]
				y_train_inner = y_train[train_indexes_inner, :]
				X_val = X_train[val_indexes, :]
				y_val = y_train[val_indexes, :]

				scaling_params = nothing
				scaling_params_output = nothing
				if scaling_type != :none
					X_train_inner, scaling_params = learn_and_scale(X_train_inner, scaling = scaling_type)
					X_val = scale(X_val, scaling = scaling_type, scaling_params = scaling_params)
				end

				if scaling_type_output != :none
					y_train_inner, scaling_params_output = learn_and_scale(y_train_inner, scaling = scaling_type_output)
				end

				# Train model with current hyperparameters
				params = Dict(
					:n_neurons => n_neurons,
					:n_layers => n_layers,
					:activation_function => activation_function,
					:output_layer_choice => output_layer_choice,
					:batch_size => batch_size,
					:target => target,
					:epochs => max_epochs,
					:loss_function => loss_function,
					:transform_to_tau_rho => transform_to_tau_rho,
					:moment => moment,
					:scale => scaling_type,
					:scale_output => scaling_type_output,
					:scaling_params => scaling_params,
					:scaling_params_output => scaling_params_output,
				)
				model, _ = get_model(; params...)
				it_model = IteratedModel(
					model = model,
					resampling = nothing,
					controls = [Step(1),
						Patience(20),
						InvalidValue(),
						NumberLimit(max_epochs),
					],
					measure = loss_function,
					iteration_parameter = iteration_parameter(model),
				)

				mach = machine(it_model, table(X_train_inner), y_train_inner)

				fit!(mach, verbosity = 0)

				# Evaluate on validation set
				predictions = MLJ.predict(mach, table(X_val))

				if scaling_type_output != :none
					predictions = descale(predictions, scaling = scaling_type_output, scaling_params = scaling_params_output)
				end

				if typeof(y_val) == Matrix{Float32} && CUDA.functional()
					y_val = CuArray(y_val)
				end

				rmse_value = rmse_loss(predictions, y_val)

				push!(fold_rmses, rmse_value)
				next!(p)
			end

			# Average RMSE for current hyperparameters
			avg_rmse = mean(fold_rmses)
			if avg_rmse < best_rmse
				best_rmse = avg_rmse
				best_hyperparams = hp_params
			end
		end

		# Log best hyperparameters for this outer fold
		open(path * "/" * file_name, "a") do io
			write(io, "Outer fold $outer_fold_index:\n")
			write(io, "  Best hyperparameters: $best_hyperparams\n")
			write(io, "  Best inner RMSE: $best_rmse\n")
		end

		# Train on full outer training set with best hyperparameters
		(n_neurons, n_layers, activation_function, output_layer_choice, scaling_type, scaling_type_output, batch_size, loss_function) = best_hyperparams

		if scaling_type != :none
			X_train, scaling_params = learn_and_scale(X_train, scaling = scaling_type)
			X_test = scale(X_test, scaling = scaling_type, scaling_params = scaling_params)
		end

		if scaling_type_output != :none
			y_train, scaling_params_output = learn_and_scale(y_train, scaling = scaling_type_output)
		end

		params = Dict(
			:n_neurons => n_neurons,
			:n_layers => n_layers,
			:activation_function => activation_function,
			:output_layer_choice => output_layer_choice,
			:batch_size => batch_size,
			:target => target,
			:epochs => max_epochs,
			:loss_function => loss_function,
			:transform_to_tau_rho => transform_to_tau_rho,
			:moment => moment,
			:scale => scaling_type,
			:scale_output => scaling_type_output,
			:scaling_params => scaling_params != nothing ? scaling_params : nothing,
			:scaling_params_output => scaling_params_output != nothing ? scaling_params_output : nothing,
		)
		model, _ = get_model(; params...)

		it_model = IteratedModel(
			model = model,
			resampling = nothing,
			controls = [Step(1),
				Patience(20),
				InvalidValue(),
				NumberLimit(max_epochs*2),
			],
			measure = loss_function,
			iteration_parameter = iteration_parameter(model),
		)

		mach = machine(it_model, table(X_train), y_train)

		fit!(mach, verbosity = 0)

		# Evaluate on test set
		predictions = MLJ.predict(mach, table(X_test))

		if scaling_type_output != :none
			predictions = descale(predictions, scaling = scaling_type_output, scaling_params = scaling_params_output)
		end

		if typeof(y_test) == Matrix{Float32} && CUDA.functional()
			y_test = CuArray(y_test)
		end

		rmse_value = rmse_loss(predictions, y_test)
		push!(all_rmse, rmse_value)

		# Update the overall best model if necessary
		if rmse_value < best_rmse_overall
			best_rmse_overall = rmse_value
			best_hyperparams_overall = best_hyperparams
		end

		# Log outer fold test RMSE
		open(path * "/" * file_name, "a") do io
			write(io, "  Test RMSE: $rmse_value\n")
			write(io, "  Best RMSE Overall: $best_rmse_overall\n")
		end

		next!(p)
	end

	(n_neurons, n_layers, activation_function, output_layer_choice, scaling_type, scaling_type_output, batch_size, loss_function) = best_hyperparams_overall

	if train_final_model
		# Train final model on full dataset
		if scaling_type != :none
			X, scaling_params = learn_and_scale(X, scaling = scaling_type)
		end

		if scaling_type_output != :none
			y, scaling_params_output = learn_and_scale(y, scaling = scaling_type_output)
		end

		updated = 0
		cooldown_counter = 0

		params = Dict(
			:n_neurons => n_neurons,
			:n_layers => n_layers,
			:activation_function => activation_function,
			:output_layer_choice => output_layer_choice,
			:batch_size => batch_size,
			:target => target,
			:epochs => max_epochs*2,
			:loss_function => loss_function,
			:transform_to_tau_rho => transform_to_tau_rho,
			:moment => moment,
			:scale => scaling_type,
			:scale_output => scaling_type_output,
			:scaling_params => scaling_params != nothing ? scaling_params : nothing,
			:scaling_params_output => scaling_params_output != nothing ? scaling_params_output : nothing,
		)

		final_model, opt = get_model(; params...)

		function update_lr(mach)
			# get number of epochs
			losses = report(mach).training_losses

			# if the last 10 losses are all the same or higher, reduce the learning rate
			if cooldown_counter == 0 && length(losses) > 10 && all(losses[end-10:end] .>= losses[end]) && opt.eta > 0.000001
				old_lr = mach.model.optimiser.eta
				new_lr = max(0.000001, 0.95 * old_lr)
				opt = Optimisers.ADAM(new_lr)
				mach.model.optimiser = opt
				cooldown_counter = 10  # set cooldown period
			end

			if cooldown_counter > 0
				cooldown_counter -= 1
			end

			updated += 1
		end


		controls = [Step(1),
			Patience(20),
			InvalidValue(),
			NumberLimit(max_epochs*2),
			Callback(update_lr),
		]

		it_model = IteratedModel(
			model = final_model,
			resampling = nothing,
			controls = controls,
			measure = loss_function,
			iteration_parameter = iteration_parameter(final_model),
		)

		final_mach = machine(it_model, table(X), y)

		println("Fitting final model with best hyperparameters")
		fit!(final_mach, verbosity = 0)

		# Evaluate on test set
		predictions = MLJ.predict(final_mach, table(X))

		if scaling_type_output != :none
			predictions = descale(predictions, scaling = scaling_type_output, scaling_params = scaling_params_output)
		end

		if typeof(y) == Matrix{Float32} && CUDA.functional()
			y = CuArray(y)
		end

		rmse_value = rmse_loss(predictions, y)
		mae_value = mae_loss(predictions, y)
		mape_value = mape_loss(predictions, y)
		huber_value = huber_loss(predictions, y)

		println("RMSE: ", rmse_value, " MAE: ", mae_value, " MAPE: ", mape_value, " Huber: ", huber_value)

		if moment == :both
			rmse_per_feature = sqrt.(mean((y .- predictions) .^ 2, dims = 1))
			mae_per_feature = mean(abs.(y .- predictions), dims = 1)
			mape_per_feature = mean(abs.((y .- predictions) ./ y), dims = 1)
			delta = 1.0
			huber_per_feature = mean(ifelse.(abs.(y .- predictions) .<= delta, 0.5 * (y .- predictions) .^ 2, delta * (abs.(y .- predictions) .- 0.5 * delta)), dims = 1)
			println("[Per Feature] RMSE: ", rmse_per_feature, " MAE: ", mae_per_feature, " MAPE: ", mape_per_feature, " Huber: ", huber_per_feature)
		end

		MLJ.save("SoHEstimation/approximate_message_passing/models/tuned_$(factor)_$(target).jls", final_mach)
		JLD2.jldsave("SoHEstimation/approximate_message_passing/models/tuned_$(factor)_$(target)_scaling_params.jld2",
			scaling_type = scaling_type,
			scaling_type_output = scaling_type_output,
			scaling_params = scaling_params != nothing ? scaling_params : nothing,
			scaling_params_output = scaling_params_output != nothing ? scaling_params_output : nothing,
		)
	end

	# Log final summary
	open(path * "/" * file_name, "a") do io
		write(io, "\nNested Cross-Validation Final Results:\n")
		write(io, "  RMSE per Fold: $all_rmse\n")
		write(io, "  Best Hyperparameters Overall: $(best_hyperparams_overall)\n")
		write(io, "  Best RMSE Overall: $best_rmse_overall\n")
		write(io, "  Best Scale type: $scaling_type\n")
		write(io, "  Best Scale output type: $scaling_type_output\n")
		if train_final_model
			write(io, "  RMSE after training on all data: $rmse_value\n")
			write(io, "  MAE after training on all data: $mae_value\n")
			write(io, "  MAPE after training on all data: $mape_value\n")
			write(io, "  Huber after training on all data: $huber_value\n")
			if moment == :both
				write(io, "  RMSE per Feature: $rmse_per_feature\n")
				write(io, "  MAE per Feature: $mae_per_feature\n")
				write(io, "  MAPE per Feature: $mape_per_feature\n")
				write(io, "  Huber per Feature: $huber_per_feature\n")
			end
		end
	end
end


function train_network(X, y, params, seed, max_epochs, architecture, target::String; appendix="", save=true)
	
    factor = pop!(params, :factor)
    model_name = "nn_$(factor)_$(target)_$(appendix)"

	# check if models folder exists, if not create it
	if !isdir("SoHEstimation/approximate_message_passing/evaluation/models")
		mkdir("SoHEstimation/approximate_message_passing/evaluation/models")
	end

	# check if model already exists
	if isfile("SoHEstimation/approximate_message_passing/evaluation/models/$model_name.jld2") && save
		println("Model already exists. Skipping training.")
		return load_model("SoHEstimation/approximate_message_passing/evaluation/models/$model_name.jld2")
	end

	seed = isnothing(seed) ? rand(1:10^9) : seed

	params[:scaling_params] = nothing
	params[:scaling_params_output] = nothing

	# Do scaling
	if params[:scale] != :none
		X, scaling_params = learn_and_scale(X, scaling = params[:scale])
		params[:scaling_params] = scaling_params
	end

	original_y = deepcopy(y)
	if params[:scale_output] != :none
		y, scaling_params_output = learn_and_scale(y, scaling = params[:scale_output])
		params[:scaling_params_output] = scaling_params_output
	end

	updated = 0
    cooldown_counter = 0

	function update_lr(mach)
		# get number of epochs
		losses = report(mach).training_losses

		# if the last 10 losses are all the same or higher, reduce the learning rate
		if cooldown_counter == 0 && length(losses) > 10 && all(losses[end-10:end] .>= losses[end]) && opt.eta > 0.000001
			old_lr = mach.model.optimiser.eta
			new_lr = max(0.000001, 0.95 * old_lr)
			opt = Optimisers.ADAM(new_lr)
			mach.model.optimiser = opt
			cooldown_counter = 10  # set cooldown period
		end

		if cooldown_counter > 0
			cooldown_counter -= 1
		end

		updated += 1
	end

	best_training_loss = Inf

	function save_best_model(mach)
		loss = report(mach).training_losses[end]
		if loss < best_training_loss
			best_training_loss = loss
			MLJ.save("SoHEstimation/approximate_message_passing/evaluation/models/$model_name.jls", mach)
			JLD2.jldsave("SoHEstimation/approximate_message_passing/evaluation/models/$(model_name)_scaling_params.jld2",
				scaling_params = params[:scaling_params],
				scaling = params[:scale],
				scaling_params_output = params[:scaling_params_output],
				scaling_output = params[:scale_output],
			)
			println("New best model saved with loss: ", loss)
		end

		return false
	end


	controls = [
		Step(1),
		Patience(20),
		InvalidValue(),
		NumberLimit(max_epochs),
		Callback(update_lr),
	]

	if save
		push!(controls, WithMachineDo(save_best_model))
	end

    if architecture == :both
        model, opt = get_model(; params...)

        it_model = IteratedModel(
				model = model,
				resampling = nothing,
				controls = controls,
				measure = params[:loss_function],
				iteration_parameter = iteration_parameter(model),
			)


    	mach = machine(it_model, table(X), y)

	    fit!(mach, verbosity = 0)

		# load best model
		if save
			model, scaling_params, scaling, scaling_params_output, scaling_output = load_model("SoHEstimation/approximate_message_passing/evaluation/models/$model_name.jls")
		else
			model = mach
			scaling_params = params[:scaling_params]
			scaling = params[:scale]
			scaling_params_output = params[:scaling_params_output]
			scaling_output = params[:scale_output]
		end

		# Evaluate on test set
		predictions = MLJ.predict(model, table(X))

		if scaling_output != :none
			predictions = descale(predictions, scaling = scaling_output, scaling_params = scaling_params_output)
		end

		# put predictions to CPU back
		predictions = Matrix{Float32}(predictions)
		
		rmse_value = rmse_loss(predictions, original_y)
		mape_value = mape_loss(predictions, original_y)

		rmse_per_feature = sqrt.(mean((original_y .- predictions) .^ 2, dims = 1))
		mae_per_feature = mean(abs.(original_y .- predictions), dims = 1)
		mape_per_feature = mean(abs.((original_y .- predictions) ./ original_y), dims=1)
		println("Final for $factor RMSE: $rmse_value and MAPE = $mape_value")
		println("rmse_per_feature: ", rmse_per_feature)
		println("mae_per_feature: ", mae_per_feature)
		println("mape_per_feature: ", mape_per_feature)

	else
        error("Architecture $architecture not supported.")
    end
end

function tune_network(X::Matrix{Float32}, y::Matrix{Float32}; target = "targets_X", seed = nothing, factor = :wsf, moment = :both, transform_to_tau_rho = false)

	nested_cross_validation(X, y,
		outer_folds_amount = 5,
		inner_folds_amount = 3,
		factor = factor,
		target = target,
		seed = seed,
		moment = moment,
		transform_to_tau_rho = transform_to_tau_rho,
	)
end
end
