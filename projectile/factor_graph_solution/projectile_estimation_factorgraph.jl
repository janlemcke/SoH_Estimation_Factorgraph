include("../../lib/factors.jl")
# include("../../lib/gaussian.jl")
# include("../../lib/distributionbag.jl")

module ProjectileEstimation

export estimate_initial_state_x!, estimation_loss, add_noise, estimate_initial_state_y!

using DataStructures
using ..Factors
using ..DistributionCollections
using ..GaussianDistribution


# struct that saves the references to all factors and variables of one timestep (variables and factors connecting them to the previous timesteps), as well as the size of the time step
struct TimeStepGraph
    x::Int64
    v::Union{Int64, Nothing}
    F_prev_x_v::WeightedSumFactor
    F_prev_v::Union{ScalingFactor, Nothing}
    F_x_data::Union{GaussianFactor, Nothing}
    timestamp::Float64
end


TimeStepGraph(x::Int64, v::Int64, F_prev_x_v::WeightedSumFactor, F_prev_v::ScalingFactor) = TimeStepGraph(x, v, F_prev_x_v, F_prev_v, nothing, 0.0)
TimeStepGraph(x::Int64, v::Int64, F_prev_x_v::WeightedSumFactor, F_prev_v::ScalingFactor, F_x_data::GaussianFactor) = TimeStepGraph(x, v, F_prev_x_v, F_prev_v, F_x_data, 0.0)

function pretty_print(db::DistributionBag, x_0::Int, v_0::Int, graphs::Vector{TimeStepGraph})
    if isempty(graphs)
        println("No timesteps to display.")
        return
    end
    
    println("Timestep 0: x = (", db[x_0], "), v = (", db[v_0], ")")
    
    for (i, e) in enumerate(graphs)
        if !isnothing(e.v)
            println("Timestep ", i, ": x = (", db[e.x] , "), v = (", db[e.v], ")")
        else
            println("Timestep ", i, ": x = (", db[e.x] , ")")
        end
    end
end


# helper function to add the variables and factors for one additional timestep to the factor graph
function addTimeStep!(db::DistributionBag, prev_x::Int64, prev_v::Int64; datapoint::Union{Nothing, Pair{Float64, Float64}}=nothing, prev_timestamp::Float64=0.0, time_step::Float64=1.0, beta=2.0, factor_coordinate=(t -> t), bias_coordinate=(t -> 0.0), factor_velocity=(t -> 1.0), bias_velocity=(t -> 0.0))   
    x = add!(db)
    v = add!(db)
    delta_time = isnothing(datapoint) ? time_step : datapoint.first - prev_timestamp
    F_prev_x_v = WeightedSumFactor(db, prev_x, prev_v, x, 1.0, factor_coordinate(delta_time), bias_coordinate(delta_time))
    F_prev_v = ScalingFactor(db, prev_v, v, factor_velocity(delta_time), bias_velocity(delta_time))
    if !isnothing(datapoint)
        data = Gaussian1DFromMeanVariance(datapoint.second, beta^2)
        F_x_data = GaussianFactor(db, x, data)
        return TimeStepGraph(x, v, F_prev_x_v, F_prev_v, F_x_data, datapoint.first)
    else
        return TimeStepGraph(x, v, F_prev_x_v, F_prev_v, nothing, prev_timestamp + delta_time)
    end
end

# helper function to add the variables and factors for the final timestep to the factor graph
function addLastTimeStep!(db::DistributionBag, prev_x::Int64, prev_v::Int64; datapoint::Union{Nothing, Pair{Float64, Float64}}=nothing, prev_timestamp::Float64=0.0, time_step::Float64=1.0, beta=2.0, factor_coordinate=(t -> t), bias_coordinate=(t -> 0.0), factor_velocity=(t -> 1.0), bias_velocity=(t -> 0.0))   
    x = add!(db)
    delta_time = isnothing(datapoint) ? time_step : datapoint.first - prev_timestamp
    F_prev_x_v = WeightedSumFactor(db, prev_x, prev_v, x, 1.0, factor_coordinate(delta_time), bias_coordinate(delta_time))

    if !isnothing(datapoint)
        data = Gaussian1DFromMeanVariance(datapoint.second, beta^2)
        F_x_data = GaussianFactor(db, x, data)
        return TimeStepGraph(x, nothing, F_prev_x_v, nothing, F_x_data, datapoint.first)
    else
        return TimeStepGraph(x, nothing, F_prev_x_v, nothing, nothing, prev_timestamp + delta_time)
    end
end

# helper function to perform the forward pass of the sum-product algorithm for the subgraph of one particular timestep (pass from previous timestep to current timestep)
function timeStepForwardPass!(time_step_graph::TimeStepGraph)
        delta = 0.0

        # forward pass
        delta = max(delta, update_msg_to_z!(time_step_graph.F_prev_x_v))

        if !isnothing(time_step_graph.F_prev_v)
            delta = max(delta, update_msg_to_y!(time_step_graph.F_prev_v))
        end

        if !isnothing(time_step_graph.F_x_data)
            delta = max(delta, update_msg_to_x!(time_step_graph.F_x_data))
        end

        return delta
end

# helper function to perform the backward pass of the sum-product algorithm for the subgraph of one particular timestep (pass from current timestep to previous timestep)
function timeStepBackwardPass!(time_step_graph::TimeStepGraph)
    delta = 0.0


    delta = max(delta, update_msg_to_x!(time_step_graph.F_prev_v))
    delta = max(delta, update_msg_to_x!(time_step_graph.F_prev_x_v))
    delta = max(delta, update_msg_to_y!(time_step_graph.F_prev_x_v))

    return delta
end

# Computes the TrueSkills for a two-player game
function estimate_initial_state_x!(;x_0_prior::Gaussian1D=Gaussian1DFromMeanVariance(0.0, 200.0), v_0_prior::Gaussian1D=Gaussian1DFromMeanVariance(0.0, 200.0), beta::Float64=2.0, time_step::Float64=1.0, data::OrderedDict{Float64, Float64}=OrderedDict(2.0 => 11.0, 3.0 => 16.0), min_diff=1e-10, factor_coordinate=(t -> t), bias_coordinate=(t -> 0.0), factor_velocity=(t -> 1.0), bias_velocity=(t -> 0.0))
    sort!(data, by=first)
    
    db = DistributionBag(Gaussian1D(0, 0))

    # add variables and factors for the 0th timestep (initial state)
    x_0 = add!(db)
    v_0 = add!(db)

    # check if we have data for the 0th timestep, if so, add a factor for the data instead of posterior
    prior_x_0 = collect(data)[1].first > 0.0 ? GaussianFactor(db, x_0, x_0_prior) : GaussianFactor(db, x_0, Gaussian1DFromMeanVariance(collect(data)[1].second, beta^2))
    prior_v_0 = GaussianFactor(db, v_0, v_0_prior)

    update_msg_to_x!(prior_x_0)
    update_msg_to_x!(prior_v_0)
    

    time_step_graphs = Vector{TimeStepGraph}()


    # add timesteps to factor graph that we don't have data for
    # (check if there is a timestep between the 0th timestep and the first data point, if so, add timesteps for each of these steps to the factor graph)
    if collect(data)[1].first > time_step
        for _ in time_step:time_step:(collect(data)[1].first-time_step)
            if isempty(time_step_graphs)
                push!(time_step_graphs, addTimeStep!(db, x_0, v_0, beta=beta, time_step=time_step, prev_timestamp=0.0, factor_coordinate=factor_coordinate, bias_coordinate=bias_coordinate, factor_velocity=factor_velocity, bias_velocity=bias_velocity))
            else
                push!(time_step_graphs, addTimeStep!(db, time_step_graphs[end].x, time_step_graphs[end].v, prev_timestamp=time_step_graphs[end].timestamp, beta=beta, time_step=time_step,  factor_coordinate=factor_coordinate, bias_coordinate=bias_coordinate, factor_velocity=factor_velocity, bias_velocity=bias_velocity))
            end
        end
    end

    # add timesteps to factor graph that we do have data for 
    # (check if we already have a timestep for the first data point only add it to the factor graph if we don't have it yet. 
    # this should only be the case if we have a datapoint for the 0th timestep, i.e., timestamp = 0.0)
    first_timestep = collect(data)[1].first == 0.0 ? 2 : 1
    for (i, datapoint) in enumerate(collect(data)[first_timestep:end])
        
        if isempty(time_step_graphs)
            push!(time_step_graphs, addTimeStep!(db, x_0, v_0, datapoint=datapoint, beta=beta, time_step=time_step, prev_timestamp=0.0, factor_coordinate=factor_coordinate, bias_coordinate=bias_coordinate, factor_velocity=factor_velocity, bias_velocity=bias_velocity))
            continue
        end
        
        # if last data point, do net set something for v 
        if i == length(collect(data)[first_timestep:end])
            push!(time_step_graphs, addLastTimeStep!(db, time_step_graphs[end].x, time_step_graphs[end].v, prev_timestamp=time_step_graphs[end].timestamp, datapoint=datapoint, beta=beta, time_step=time_step,  factor_coordinate=factor_coordinate, bias_coordinate=bias_coordinate, factor_velocity=factor_velocity, bias_velocity=bias_velocity))    
            continue
        end

        push!(time_step_graphs, addTimeStep!(db, time_step_graphs[end].x, time_step_graphs[end].v, prev_timestamp=time_step_graphs[end].timestamp, datapoint=datapoint, beta=beta, time_step=time_step,  factor_coordinate=factor_coordinate, bias_coordinate=bias_coordinate, factor_velocity=factor_velocity, bias_velocity=bias_velocity))
    end


    delta = 2*min_diff
    iteration = 0
    while delta > min_diff
        delta = 0.0
        # do one complete forward pass through the graph
        for time_step_graph in time_step_graphs
            delta = max(delta, timeStepForwardPass!(time_step_graph))
        end

        # do one complete backward pass through the graph
        for time_step_graph in reverse(time_step_graphs[1:end-1])
            delta = max(delta, timeStepBackwardPass!(time_step_graph))
        end
        iteration += 1
    end
    println("iterations: ", iteration)

    pretty_print(db, x_0, v_0, time_step_graphs)
    return vcat([(db[x_0], db[v_0])], [(db[e.x], !isnothing(e.v) ? db[e.v] : nothing) for e in time_step_graphs])

end

estimate_initial_state_y!(;y_0_prior::Gaussian1D=Gaussian1DFromMeanVariance(0.0, 200.0), v_0_prior::Gaussian1D=Gaussian1DFromMeanVariance(0.0, 200.0), beta::Float64=2.0, time_step::Float64=1.0, data::OrderedDict{Float64, Float64}=OrderedDict(2.0 => 61.4, 3.0 => 76.9), min_diff=1e-10, factor_coordinate=(t -> t), bias_coordinate=(t -> 0.5*(-9.8)*t^2), factor_velocity=(t -> 1.0), bias_velocity=(t -> t * (-9.8))) = estimate_initial_state_x!(;x_0_prior=y_0_prior, v_0_prior=v_0_prior, beta=beta, time_step=time_step, data=data, min_diff=min_diff, factor_coordinate=factor_coordinate, bias_coordinate=bias_coordinate, factor_velocity=factor_velocity, bias_velocity=bias_velocity)

estimate_initial_state_y!(;y_0_prior::Gaussian1D=Gaussian1DFromMeanVariance(0.0, 200.0), v_0_prior::Gaussian1D=Gaussian1DFromMeanVariance(0.0, 200.0), beta::Float64=2.0, time_step::Float64=1.0, data::OrderedDict{Float64, Float64}=OrderedDict(2.0 => 61.4, 3.0 => 76.9), min_diff=1e-10, factor_coordinate=(t -> t), bias_coordinate=(t -> 0.5*(-9.8)*t^2), factor_velocity=(t -> 1.0), bias_velocity=(t -> t * (-9.8))) = estimate_initial_state_x!(;x_0_prior=y_0_prior, v_0_prior=v_0_prior, beta=beta, time_step=time_step, data=data, min_diff=min_diff, factor_coordinate=factor_coordinate, bias_coordinate=bias_coordinate, factor_velocity=factor_velocity, bias_velocity=bias_velocity)

function estimation_loss(data::OrderedDict{Float64, Float64}, estimated_data::Vector{Tuple{Gaussian1D, Any}})
    loss = 0.0

    for (i, (time, value)) in enumerate(data)
        loss += abs(mean(estimated_data[i][1]) - value)
    end

    velocity = diff([datapoint.second for datapoint in data])
    velocity = vcat(velocity, velocity[end])
    println(velocity)

    for (i, (time, value)) in enumerate(data)
        if !isnothing(estimated_data[i][2])
            loss += abs(mean(estimated_data[i][2]) - velocity[i])
        end

    end

    return loss / (length(data)*2)

end

function add_noise(data::OrderedDict{Float64, Float64}; beta_squared::Float64=0.1)
    beta = sqrt(beta_squared)
    # Create a new OrderedDict to store the updated values
    updated_data = OrderedDict{Float64, Float64}()
    
    for (time, value) in data
        # Generate a Gaussian, zero-mean measurement error with variance beta_squared
        error = beta * randn()
        # Add the error to the original value
        updated_value = value + error
        # Update the dictionary with the new value
        updated_data[time] = updated_value
    end
    
    return updated_data
end

end