module CustomNN
export MyNetworkBuilder

using MLJBase
using MLJFlux
using MLJ
using Flux
include("./layer.jl")
using .Layer



# Define the custom model struct
mutable struct MyNetworkBuilder  <: MLJFlux.Builder
    n_neurons::Int
    n_layers::Int
    activation_function::String
    output_layer_choice::String
    target::String
    moment::Symbol
    scale::Symbol
    scaling_params::Union{Tuple{Matrix{Float32}, Matrix{Float32}}, Nothing}
    scale_output::Symbol
    scaling_params_output::Union{Tuple{Matrix{Float32}, Matrix{Float32}}, Nothing}
    transform_to_tau_rho::Bool
end

function dimension(target::String, moment::Symbol, n_features::Int)
    # if both moments are learned, we have 9 features
    index = 0
    if moment == :both || moment == :second || moment == :first
        if target == "targets_X"
            index = 2
        elseif target == "targets_Y"
            index = 4
        elseif target == "targets_Z"
            index = 6
        end
    end
    return index
end

function MLJFlux.build(model::MyNetworkBuilder, rng, n_in, n_out)
    # Initialize variables
    output_layer = nothing

    if model.activation_function == "tanh_fast"
        init_func = Flux.glorot_normal()
        activation_function = tanh_fast
    elseif model.activation_function == "relu"
        init_func = Flux.kaiming_normal()
        activation_function = relu
    else
        error("Invalid activation function")
    end

    layers = []

    # Input layer
    push!(layers, Dense(n_in, model.n_neurons, activation_function, init=init_func))

    # Hidden layers
    hidden_layers = []
    for _ in 1:model.n_layers
        push!(hidden_layers, Dense(model.n_neurons, model.n_neurons, activation_function, init=init_func))
    end
    push!(layers, hidden_layers...)

    # Add a Dense layer before the output
    push!(layers, Dense(model.n_neurons, n_out, init=init_func))

    nn = Flux.Chain(layers...)

    if model.moment == :both
        nn = Flux.Chain(ResidualMinimum(model.scale, model.scaling_params, model.scale_output, model.scaling_params_output, model.moment, nn, model.transform_to_tau_rho, dimension(model.target, model.moment, n_in), 2,  model.output_layer_choice))
    elseif model.moment == :second
        nn =  Flux.Chain(ResidualMinimum(model.scale, model.scaling_params, model.scale_output, model.scaling_params_output, model.moment, nn, model.transform_to_tau_rho, dimension(model.target, model.moment, n_in), 1,  model.output_layer_choice))
    end
    return nn
end

end