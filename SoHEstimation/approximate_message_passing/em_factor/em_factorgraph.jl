include("../../../lib/gaussian.jl")
include("../../../lib/distributionbag.jl")
include("../nn/mjl.jl")
include("../../../lib/factors.jl")


module SoHEstimation

export start_convergence, estimation_loss, add_noise, estimate_initial_state_y!

using ..Factors
using ..DistributionCollections
using ..GaussianDistribution
using ..NN

using DataFrames


# struct that saves the references to all factors and variables of one timestep (variables and factors connecting them to the previous timesteps), as well as the size of the time step
struct TimeStepGraph
	I::Int64
	SoC::Int64
	DSoC::Union{Int64, Nothing}
	SoH::Int64
	DQ::Int64

	F_I_data::Union{GaussianFactor, Nothing}
	F_SoC_data::Union{GaussianFactor, Nothing}
	F_DQ_data::Union{GaussianFactor, Nothing}

	F_prev_SoC_DSoC::Union{WeightedSumFactor, Nothing}
	F_prev_I_SoH::Union{ApproximateEMFactor, Nothing}
	F_prev_SoH_DQ::Union{WeightedSumFactor, Nothing}

	timestamp::Float64
end


function load_models(multiple=false, nn_index=0)
	if multiple
		nn_x = []
		nn_y = []
		nn_z = []
		for nn_index in 1:3
			push!(nn_x, load_model("SoHEstimation/approximate_message_passing/evaluation/models/nn_emf_targets_X_Experiment_$(nn_index).jls"))
			push!(nn_y, load_model("SoHEstimation/approximate_message_passing/evaluation/models/nn_emf_targets_Y_Experiment_$(nn_index).jls"))
			push!(nn_z, load_model("SoHEstimation/approximate_message_passing/evaluation/models/nn_emf_targets_Z_Experiment_$(nn_index).jls"))
		end
	else
		nn_x = load_model("SoHEstimation/approximate_message_passing/evaluation/models/nn_emf_targets_X_Experiment_$(nn_index).jls")
		nn_y = load_model("SoHEstimation/approximate_message_passing/evaluation/models/nn_emf_targets_Y_Experiment_$(nn_index).jls")
		nn_z = load_model("SoHEstimation/approximate_message_passing/evaluation/models/nn_emf_targets_Z_Experiment_$(nn_index).jls")
	end
    return nn_x, nn_y, nn_z
end

function addTimeStep!(
	db::DistributionBag,
	I::Int64,
	SoC::Int64,
	DSoC::Int64,
	SoH::Int64,
	DQ::Int64;
	datapoint::Union{Nothing, DataFrameRow{DataFrame, DataFrames.Index}} = nothing,
	prev_timestamp::Float64 = 0.0,
	nn_x::Any,
	nn_y::Any,
	nn_z::Any,
	multiple::Bool = false,
	sampling::Bool = false,
	delta_time::Float64 = 0.004166666666666667,
)
	new_I = add!(db)
	new_SoC = add!(db)
	new_DSoC = add!(db)
	new_SoH = add!(db)
	new_DQ = add!(db)

    # EM Factor from dsoc_new = - i_new/Soh 
	
	F_prev_I_SoH = ApproximateEMFactor(db, new_I, SoH, new_DSoC, 0.66, delta_time, nn_x, nn_y, nn_z, multiple, sampling)

	# Weighted sum factor from SoC_new= SoC_old + DS0C_new
	F_prev_SoC_DSoC = WeightedSumFactor(db, SoC, new_DSoC, new_SoC, 1.0, 1.0, 0.0)

	# Weighted sum factor from SoH_new = SoH_old - DQ_new
	F_prev_SoH_DQ = WeightedSumFactor(db, SoH, new_DQ, new_SoH, 1.0, 1.0, 0.0)
	

	if !isnothing(datapoint)
		F_I_data = GaussianFactor(db, new_I, Gaussian1DFromMeanVariance(datapoint.I, 0.01))
		F_SoC_data = GaussianFactor(db, new_SoC, Gaussian1DFromMeanVariance(datapoint.SoC, 0.1))
		F_DQ_data = GaussianFactor(db, new_DQ, Gaussian1DFromMeanVariance(datapoint.DQ, 0.001))

		return TimeStepGraph(new_I, new_SoC, new_DSoC, new_SoH, new_DQ, F_I_data, F_SoC_data, F_DQ_data, F_prev_SoC_DSoC, F_prev_I_SoH, F_prev_SoH_DQ, datapoint.timestamp)
	else
		return TimeStepGraph(new_I, new_SoC, new_DSoC, new_SoH, new_DQ, nothing, nothing, nothing, F_prev_SoC_DSoC, F_prev_I_SoH, F_prev_SoH_DQ, prev_timestamp + delta_time)
	end
end

function addFirstTimeStep!(
	db::DistributionBag,
	I_0_prior::Gaussian1D = Gaussian1DFromMeanVariance(0, 7.5^2),
	SoC_0_prior::Gaussian1D = Gaussian1DFromMeanVariance(0.15, 0.2^2),
	DSoC_0_prior::Gaussian1D = Gaussian1DFromMeanVariance(0, 0.4^2),
    SoH_0_prior::Gaussian1D = Gaussian1DFromMeanVariance(0.7, 0.2^2),
    DQ_0_prior::Gaussian1D = Gaussian1DFromMeanVariance(-0.08, 0.01^2);
	datapoint::Union{Nothing, DataFrameRow{DataFrame, DataFrames.Index}} = nothing,
)
	# add variables and factors for the 0th timestep (initial state)
	I_0 = add!(db)
	SoC_0 = add!(db)
	DSoC_0 = add!(db)
	SoH_0 = add!(db)
	DQ_0 = add!(db)

	# Check if we have data for the initial values of I, SoC and DSoC
	if isnothing(datapoint)
		F_I_data = GaussianFactor(db, I_0, I_0_prior)
		F_SoC_data = GaussianFactor(db, SoC_0, SoC_0_prior)
		F_DQ_data = GaussianFactor(db, DQ_0, DQ_0_prior)
	else
		F_I_data = GaussianFactor(db, I_0, Gaussian1DFromMeanVariance(datapoint.I, 0.0001))
		F_SoC_data = GaussianFactor(db, SoC_0, Gaussian1DFromMeanVariance(datapoint.SoC, 0.0001))
		F_DQ_data = GaussianFactor(db, DQ_0, Gaussian1DFromMeanVariance(datapoint.DQ, 0.0001))
	end

	# Both latent variables get a prior once outside the convergence (forward & backward pass) loop.
	prior_SoH_0 = GaussianFactor(db, SoH_0, SoH_0_prior)
	prior_DSoC_0 = GaussianFactor(db, DSoC_0, DSoC_0_prior)

	update_msg_to_x!(prior_SoH_0)
	update_msg_to_x!(prior_DSoC_0)

	return TimeStepGraph(I_0, SoC_0, DSoC_0, SoH_0, DQ_0, F_I_data, F_SoC_data, F_DQ_data, nothing, nothing, nothing, 0)
end

function pretty_print(db::DistributionBag, time_step_graphs::Vector{TimeStepGraph})
	for (i, e) in enumerate(time_step_graphs)
		println("Timestep ", i, ": Timestamp = ", e.timestamp, ", I = (", db[e.I], "), SoC = (", db[e.SoC], "), DSoC = (", db[e.DSoC], "), SoH = (", db[e.SoH], "), DQ = (", db[e.DQ], ")")
	end
end

# # helper function to perform the forward pass of the sum-product algorithm for the subgraph of one particular timestep (pass from previous timestep to current timestep)
function timeStepForwardPass!(time_step_graph::TimeStepGraph)
        delta = 0.0

        # Forward pass the data points
        if !isnothing(time_step_graph.F_I_data)
            delta = max(delta, update_msg_to_x!(time_step_graph.F_I_data))
        end

        if !isnothing(time_step_graph.F_SoC_data)
            delta = max(delta, update_msg_to_x!(time_step_graph.F_SoC_data))
        end

        if !isnothing(time_step_graph.F_DQ_data)
            delta = max(delta, update_msg_to_x!(time_step_graph.F_DQ_data))
        end

        
        # Forward pass other factors
		if !isnothing(time_step_graph.F_prev_I_SoH)
        	delta = max(delta, update_msg_to_z!(time_step_graph.F_prev_I_SoH))
		end

        if !isnothing(time_step_graph.F_prev_SoC_DSoC)
            delta = max(delta, update_msg_to_z!(time_step_graph.F_prev_SoC_DSoC))
        end

        if !isnothing(time_step_graph.F_prev_SoH_DQ)
            delta = max(delta, update_msg_to_z!(time_step_graph.F_prev_SoH_DQ))
        end

		

        return delta
end

# # helper function to perform the backward pass of the sum-product algorithm for the subgraph of one particular timestep (pass from current timestep to previous timestep)
function timeStepBackwardPass!(time_step_graph::TimeStepGraph)
    delta = 0.0

	# Update F_prev_SoC_DSoC backwards : WSF
	if !isnothing(time_step_graph.F_prev_SoC_DSoC)
		delta = max(delta, update_msg_to_x!(time_step_graph.F_prev_SoC_DSoC))
		delta = max(delta, update_msg_to_y!(time_step_graph.F_prev_SoC_DSoC))
	end

	# Update F_prev_I_SoH backwards : AEMF
	if !isnothing(time_step_graph.F_prev_I_SoH)
		delta = max(delta, update_msg_to_x!(time_step_graph.F_prev_I_SoH))
		delta = max(delta, update_msg_to_y!(time_step_graph.F_prev_I_SoH))
	end

	# Update F_prev_SoH_DQ backwards : WSF
	if !isnothing(time_step_graph.F_prev_SoH_DQ)
		delta = max(delta, update_msg_to_x!(time_step_graph.F_prev_SoH_DQ))
		delta = max(delta, update_msg_to_y!(time_step_graph.F_prev_SoH_DQ))
	end

    return delta
end



# Computes the TrueSkills for a two-player game
function start_convergence(
	data::DataFrame;
	max_iterations::Int64 = 2_000,
	time_step::Float64 = 0.004166666666666667,
	min_diff::Float64 = 1e-10,
	multiple::Bool = false,
	separate_loop::Bool = false,
	sampling::Bool = false,
	nn_index::Int = 1,
)
	nn_x, nn_y, nn_z = load_models(multiple, nn_index)

	db = DistributionBag(Gaussian1D(0, 0))

	time_step_graphs = Vector{TimeStepGraph}()
    
	max_timestamp = maximum(data.timestamp)
	n_timesteps = Int(ceil(max_timestamp / time_step))

	data_index = 1
	for t in 0:n_timesteps
		current_timestamp = t * time_step
		index = findfirst(ts -> abs(ts - current_timestamp) < 1e-5, data.timestamp)
		datapoint = isnothing(index) ? nothing : data[index, :]

		if isempty(time_step_graphs)
			push!(time_step_graphs, addFirstTimeStep!(db, datapoint = datapoint))
		else
			push!(
				time_step_graphs,
				addTimeStep!(
					db,
					time_step_graphs[end].I,
					time_step_graphs[end].SoC,
					time_step_graphs[end].DSoC,
					time_step_graphs[end].SoH,
					time_step_graphs[end].DQ,
					datapoint = datapoint,
					prev_timestamp = time_step_graphs[end].timestamp,
					nn_x=nn_x,
					nn_y=nn_y,
					nn_z=nn_z,
					multiple=multiple,
					sampling=sampling,
					delta_time=time_step,
				),
			)
		end
	end

	#pretty_print(db, time_step_graphs)

	iteration = 0
	patience = 0
	max_patience = 10
	best_forward_delta = Inf
	best_backward_delta = Inf

	previous_delta = Inf
	while iteration < max_iterations
		if separate_loop
		
			forward_delta = 0
			# do one complete forward pass through the graph
			#start_time = time()
			for time_step_graph in time_step_graphs
				forward_delta = max(forward_delta, timeStepForwardPass!(time_step_graph))
			end
			#println("Forward pass done in ", time() - start_time, " seconds.")
			#start_time = time()

			backwards_delta = 0
			# do one complete backward pass through the graph
			for time_step_graph in reverse(time_step_graphs[1:end])
				backwards_delta = max(backwards_delta, timeStepBackwardPass!(time_step_graph))
			end

			forward_delta = 0
			# only check forward pass delta for SoH
			for time_step_graph in time_step_graphs
				if !isnothing(time_step_graph.F_prev_SoH_DQ)
					forward_delta = max(forward_delta, update_msg_to_z!(time_step_graph.F_prev_SoH_DQ))
				end
			end
			println("Iteration: ", iteration)
			println("Forward delta overall: ", forward_delta)
			backward_delta = 0
			for time_step_graph in time_step_graphs
				if !isnothing(time_step_graph.F_prev_SoH_DQ)
					backward_delta = max(backward_delta, update_msg_to_x!(time_step_graph.F_prev_SoH_DQ))
					backward_delta = max(backward_delta, update_msg_to_y!(time_step_graph.F_prev_SoH_DQ))
				end
			end

			println("Backward delta overall: ", backward_delta)

			overall_delta = max(forward_delta, backward_delta)
			#println("Overall delta: ", overall_delta)

			if abs(previous_delta - overall_delta) < 1e-5 || previous_delta < overall_delta 
				println("Stopped due to convergence criteria: ", abs(previous_delta - overall_delta), " < 1e-5")
				break
			end
			previous_delta = overall_delta
			
		else

			# Forwards in time
			i = 1
			for time_step_graph in time_step_graphs
				previous_delta = 1e6
				local_delta = 0
				while abs(previous_delta - local_delta) > 1e-4
					previous_delta = local_delta
					local_delta = 0
					# Perform forward and backward passes
					local_delta = max(local_delta, timeStepForwardPass!(time_step_graph))
					local_delta = max(local_delta, timeStepBackwardPass!(time_step_graph))
					#println("Abs difference: ", abs(previous_delta - local_delta), " for timestep ", i)
					
				end
				#println("Forward local_delta: ", local_delta, " for timestep ", i)
				i+=1
			end
			
			i = 0
			# Backwards in time
			for time_step_graph in reverse(time_step_graphs[1:end])

				previous_delta = 1e6
				local_delta = 0
				while abs(previous_delta - local_delta) > 1e-4
					previous_delta = local_delta
					local_delta = 0
					# Perform forward and backward passes
					local_delta = max(local_delta, timeStepForwardPass!(time_step_graph))
					local_delta = max(local_delta, timeStepBackwardPass!(time_step_graph))
					#println("Abs difference: ", abs(previous_delta - local_delta), " for timestep ", i)
					
				end
				i += 1
				#println("Backward local_delta:", local_delta)
			end

			forward_delta = 0
			# only check forward pass delta for SoH
			for time_step_graph in time_step_graphs
				if !isnothing(time_step_graph.F_prev_SoH_DQ)
					forward_delta = max(forward_delta, update_msg_to_z!(time_step_graph.F_prev_SoH_DQ))
				end
			end
			println("Iteration: ", iteration)
			println("Forward delta overall: ", forward_delta)
			backward_delta = 0
			for time_step_graph in time_step_graphs
				if !isnothing(time_step_graph.F_prev_SoH_DQ)
					backward_delta = max(backward_delta, update_msg_to_x!(time_step_graph.F_prev_SoH_DQ))
					backward_delta = max(backward_delta, update_msg_to_y!(time_step_graph.F_prev_SoH_DQ))
				end
			end

			println("Backward delta overall: ", backward_delta)

			overall_delta = max(forward_delta, backward_delta)
			#println("Overall delta: ", overall_delta)

			if abs(previous_delta - overall_delta) < 1e-5  # If delta isn't changing much
				println("Stopped due to convergence criteria: ", abs(previous_delta - overall_delta), " < 1e-5")
				break
			end
			previous_delta = overall_delta
		end

		iteration += 1
	end
	

	soh_list = [db[time_step_graph.SoH] for time_step_graph in time_step_graphs]
	dsoc_list = [db[time_step_graph.DSoC] for time_step_graph in time_step_graphs]

	return soh_list, dsoc_list, iteration

end

# Add Gaussian noise to a DataFrame
function add_noise(data::DataFrame; beta_squared::Dict{Symbol, Float64})
	noisy_data = deepcopy(data)

	for col in names(noisy_data)
		if haskey(beta_squared, Symbol(col))
			beta = sqrt(beta_squared[Symbol(col)])
			for row in 1:nrow(noisy_data)
				error = beta * randn()
				noisy_data[row, col] += error
			end
		end
	end

	return noisy_data
end

end
