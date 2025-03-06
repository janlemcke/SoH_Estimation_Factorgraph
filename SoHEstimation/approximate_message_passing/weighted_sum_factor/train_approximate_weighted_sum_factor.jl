using Flux, JLD2, Random, StatsBase, ProgressMeter, Format
using Flux: @functor, mse, withgradient, update!
using Base.Iterators: partition
#include("../../../lib/utils.jl")

# using .Utils
using Dates

struct ResidualMinimum
    model::Chain
    input_index::Int
    output_index::Int
end

ResidualMinimum(model, input_index) = ResidualMinimum(model, input_index, 2)

function (l::ResidualMinimum)(x::AbstractMatrix)
    output = l.model(x)
    result = copy(output)
    result[l.output_index,:] += x[l.input_index,:]
    return result
end

function (l::ResidualMinimum)(x::AbstractVector)
    output = l.model(x)
    result = copy(output)
    result[l.output_index] += x[l.input_index]
    return result
end

@functor ResidualMinimum

# Define the PositiveVariance struct and its behavior as an activation function.
struct PositiveVariance
    output_index::Int
end
function (l::PositiveVariance)(x::AbstractMatrix)
    if l.output_index == 0
        return vcat(x[1:1, :], Flux.Flux.softplus.(x[2:2, :]))
    elseif l.output_index == 1
        return x
    elseif l.output_index == 2
        return  Flux.Flux.softplus.(x[1:1, :])
    else
        error("Output index is out of range")
    end
end
function (l::PositiveVariance)(x::AbstractVector)
    if l.output_index == 0
        return [x[1], Flux.Flux.softplus(x[2])]
    elseif l.output_index == 1
        return x
    elseif l.output_index == 2
        return  Flux.Flux.softplus.(x[1:1, :])
    else
        error("Output index is out of range")
    end
    
end
@functor PositiveVariance

# Define the PositiveVariance struct and its behavior as an activation function.
struct ETransformation
    output_index::Int
end
function (l::ETransformation)(x::AbstractMatrix)
    if l.output_index == 0
        return vcat(x[1:1, :], exp.(x[2:2, :]))
    elseif l.output_index == 1
        return x
    elseif l.output_index == 2
        return  exp.(x[1:1, :])
    else
        error("Output index is out of range")
    end
end
function (l::ETransformation)(x::AbstractVector)
    if l.output_index == 0
        return [x[1], exp.(x[2])]
    elseif l.output_index == 1
        return x
    elseif l.output_index == 2
        return  exp.(x[1:1, :])
    else
        error("Output index is out of range")
    end
end
@functor ETransformation

# K-fold Cross-Validation Splitting
function k_fold_cross_validation(k, data)
    n = length(data)
    fold_size = div(n, k)
    folds = partition(1:n, fold_size)
    return [Set(fold) for fold in folds]
end

# Normalization helper functions
function compute_norms(data)
    norms = [Dict(:mean => StatsBase.mean([d[dim] for d in data]), :var => var([d[dim] for d in data])) for dim in 1:length(data[1])]
    return norms
end

function normalize_data!(data, norms)
    cols = length(data[1])
    for i in eachindex(data)
        for j in 1:cols 
            # Because in some datasets we might set an input to a constant value (e.g. c to 0), we check whether var is greater than zero
            data[i][j] = norms[j][:var] > 0 ? (data[i][j] - norms[j][:mean]) / sqrt(norms[j][:var]) : data[i][j] - norms[j][:mean]
        end
    end
    return data
end

# Function for evaluating the model
function eval_model(model, data_loader, lf, loss_choice)
    l = 0
    for (x, y) in data_loader
        l += lf(model, stack(x), stack(y), loss_choice)
    end
    return l / length(data_loader)
end

function loss_function(m, x, y, loss_choice)
    if loss_choice == "mse"
        return mse(m(x), y)
    elseif loss_choice == "kl"
        return kl_loss(y, m(x))
    end
end

# Inner loop for hyperparameter tuning using k_inner-fold cross-validation
function inner_cv_train(variable, train_samples, train_targets, k_inner, hyperparams_grid, loss_choice, output_index, patience=50, epochs=10000)
    best_val_score = Inf
    best_model = nothing
    current_model = nothing
    best_hyperparams = nothing

    # Split training data into k_inner folds for cross-validation
    inner_folds = k_fold_cross_validation(k_inner, train_samples)

    # Iterate over all possible combinations of hyperparameters
    for hyperparams in hyperparams_grid
        fold_val_scores = []

        for fold_idx in 1:k_inner
            println("Hyperparameters: $(hyperparams) - Starting inner fold $(fold_idx)...")
            # Create training and validation splits
            val_idx = inner_folds[fold_idx]
            train_idx = setdiff(1:length(train_samples), val_idx)
            val_idx = collect(val_idx)

            inner_train_samples = train_samples[train_idx]
            inner_train_targets = train_targets[train_idx]
            inner_val_samples = train_samples[val_idx]
            inner_val_targets = train_targets[val_idx]

            # Normalize the data
            norms = compute_norms(inner_train_samples)
            inner_train_samples = normalize_data!(inner_train_samples, norms)
            inner_val_samples = normalize_data!(inner_val_samples, norms)

            # Normalize the targets
            norms = compute_norms(inner_train_targets)
            inner_train_targets = normalize_data!(inner_train_targets, norms)
            inner_val_targets = normalize_data!(inner_val_targets, norms)

            # Create data loaders
            inner_train_loader = Flux.Data.DataLoader((inner_train_samples, inner_train_targets), batchsize=hyperparams[:batch_size], shuffle=true)
            inner_val_loader = Flux.Data.DataLoader((inner_val_samples, inner_val_targets), batchsize=hyperparams[:batch_size])

            # Build model based on hyperparameters
            current_model = ResidualMinimum(
                Chain(
                    Dense(size(inner_train_samples[1], 1) => hyperparams[:hidden_units], tanh_fast),
                    Dense(hyperparams[:hidden_units] => hyperparams[:hidden_units], tanh_fast),
                    Dense(hyperparams[:hidden_units] => size(inner_train_targets[1], 1)),
                    ETransformation(output_index)
                ),
                dimension(variable, 2),
                2,
            )

            optimizer = Flux.setup(ADAM(hyperparams[:learning_rate]), current_model)

            # Training with early stopping
            best_fold_val_score = Inf
            current_patience = 0

            for epoch in 1:div(epochs, 3)
                if current_patience >= patience
                    break
                end

                # Training loop
                for (x, y) in inner_train_loader
                    loss_value, grads = withgradient(current_model) do m
                        loss_function(m, stack(x), stack(y), loss_choice)
                    end
                    update!(optimizer, current_model, grads[1])
                end

                # Validation evaluation
                val_loss = eval_model(current_model, inner_val_loader, loss_function, loss_choice)
                if (best_fold_val_score - val_loss) > 1e-4
                    best_fold_val_score = val_loss
                    current_patience = 0
                    #println("INNER Epoch $(epoch) - Validation Loss: $(best_fold_val_score)")
                else
                    current_patience += 1
                end
            end

            println("Fold $(fold_idx) - Validation Loss: $(best_fold_val_score)")

            push!(fold_val_scores, best_fold_val_score)
        end

        # Average validation score for current hyperparameters
        avg_val_score = StatsBase.mean(fold_val_scores)
        if avg_val_score < best_val_score
            best_val_score = avg_val_score
            best_model = deepcopy(current_model)
            best_hyperparams = deepcopy(hyperparams)
            println("UPDATE! $(best_hyperparams) - Average Validation Loss: $(best_val_score)")
        end
    end

    return best_model, best_hyperparams, best_val_score
end

# Main training function with nested cross-validation
function train_approximate_weighted_sum_factor(; variable=X, k_outer=5, k_inner=3, hyperparams_grid, seed=nothing, datapath="data/dataset_gaussian_mean_factor_20_000.jld2", datakey="data", patience=50, epochs=10000, loss_choice="kl", output_index=0)
    seed = isnothing(seed) ? rand(1:10^9) : seed
    Random.seed!(seed)

    # Load dataset
    data = JLD2.load(datapath, datakey)

    #samples = [remove_variable(variable, d[1]) for d in data]
    samples = [d[1] for d in data]

    if output_index == 0
        targets = [get_variable(variable, d[2]) for d in data]
    else
        targets = [get_variable(variable, d[2])[output_index] for d in data]
        targets = [Float32[x] for x in targets]
    end

    println("Shape of samples: $(size(samples)) - Shape of targets: $(size(targets))")
    # Print first 5 inputs and outputs
    # original inputs are [msg_from_x_mean, msg_from_x_var, msg_from_y_mean, msg_from_y_var, msg_from_z_mean, msg_from_z_var, a, b, c]
    # original outputs = [GaussianDistribution.mean(new_msg_to_x), GaussianDistribution.variance(new_msg_to_x), GaussianDistribution.mean(new_msg_to_y), GaussianDistribution.variance(new_msg_to_y), GaussianDistribution.mean(new_msg_to_z), GaussianDistribution.variance(new_msg_to_z)]
    # sample input = [msg_from_y_mean, msg_from_y_var, msg_from_z_mean, msg_from_z_var, a, b, c]]
    # sample output = [mean_sampled_marginal_x, var_sampled_marginal_x]

    # Outer cross-validation
    outer_folds = k_fold_cross_validation(k_outer, samples)
    best_fold_score = Inf
    best_hyperparams = nothing

    for fold_idx in 1:k_outer
        println("\nStarting outer fold $(fold_idx)...")

        # Split data for outer fold: test set and training/validation set
        test_idx = outer_folds[fold_idx]
        train_val_idx = setdiff(1:length(samples), test_idx)

        train_val_samples = samples[train_val_idx]
        train_val_targets = targets[train_val_idx]
        # convert test_idx to array
        test_idx = collect(test_idx)
        test_samples = samples[test_idx]
        test_targets = targets[test_idx]

        # Perform hyperparameter tuning using inner cross-validation
        _, best_inner_hyperparams, _ = inner_cv_train(train_val_samples, train_val_targets, k_inner, hyperparams_grid, loss_choice, output_index, patience, epochs)
        
        # Normalize inputs
        norms = compute_norms(train_val_samples)
        train_val_samples = normalize_data!(train_val_samples, norms)
        test_samples = normalize_data!(test_samples, norms)

        # Normalize targets
        target_norms = compute_norms(train_val_targets)
        train_val_targets = normalize_data!(train_val_targets, target_norms)
        test_targets = normalize_data!(test_targets, target_norms)

        # Build model based on hyperparameters
        current_model = ResidualMinimum(
            Chain(
                Dense(size(train_val_samples[1], 1) => best_inner_hyperparams[:hidden_units], tanh_fast),
                Dense(best_inner_hyperparams[:hidden_units] => best_inner_hyperparams[:hidden_units], tanh_fast),
                Dense(best_inner_hyperparams[:hidden_units] => size(test_targets[1], 1)),
                ETransformation(output_index)
            ),
            dimension(variable, 2),
            2,
        )

        optimizer = Flux.setup(ADAM(best_inner_hyperparams[:learning_rate]), current_model)

        # Training with early stopping
        best_train_score = Inf
        current_patience = 0

        # Create data loaders
        train_loader = Flux.Data.DataLoader((train_val_samples, train_val_targets), batchsize=best_inner_hyperparams[:batch_size], shuffle=true)
        test_loader = Flux.Data.DataLoader((test_samples, test_targets), batchsize=best_inner_hyperparams[:batch_size])
        for epoch in 1:epochs
            if current_patience >= patience
                break
            end

            # Training loop
            for (x, y) in train_loader
                loss_value, grads = withgradient(current_model) do m
                    loss_function(m, stack(x), stack(y), loss_choice)
                end
                update!(optimizer, current_model, grads[1])
            end

            # Validation evaluation
            val_loss = eval_model(current_model, test_loader, loss_function, loss_choice)
            if (best_train_score - val_loss) > 1e-4
                best_train_score = val_loss
                current_patience = 0
                #println("Epoch $(epoch) - Best Validation Loss: $(best_train_score)")
            else
                current_patience += 1
            end
        end

        test_loss = eval_model(current_model, test_loader, loss_function, loss_choice)
        # test_loss is Nan in some instances, but why?

        println("Fold $(fold_idx) - Test Loss: $(test_loss) with Hyperparams: $(best_inner_hyperparams)")
        
        if test_loss < best_fold_score
            best_fold_score = test_loss
            best_hyperparams = deepcopy(best_inner_hyperparams)
            println("Fold $(fold_idx) - Best test_loss for entire fold : $(best_fold_score)")
        end
    end

    # Normalize the data
    norms = compute_norms(samples)
    samples = normalize_data!(samples, norms)

    # Normalize targets
    target_norms = compute_norms(targets)
    targets = normalize_data!(targets, target_norms)

    # Create data loaders
    final_train_loader = Flux.Data.DataLoader((samples, targets), batchsize=best_hyperparams[:batch_size], shuffle=true)

    final_model = ResidualMinimum(
        Chain(
            Dense(size(samples[1], 1) => best_hyperparams[:hidden_units], tanh_fast),
            Dense(best_hyperparams[:hidden_units] => best_hyperparams[:hidden_units], tanh_fast),
            Dense(best_hyperparams[:hidden_units] => size(targets[1], 1)),
            ETransformation(output_index)
        ),
        dimension(variable, 2),
        2,
    )

    # Define loss function and optimizer
    optimizer = Flux.setup(ADAM(best_hyperparams[:learning_rate]), final_model)

    # Training with early stopping
    current_patience = 0
    best_final_score = Inf
    lr_reductions = 1
    lr_reduction_max = 3

    while true
        if current_patience >= patience
            current_patience = 0

            new_lr = best_hyperparams[:learning_rate] * 0.01^lr_reductions
            Flux.adjust!(optimizer, new_lr)
            if lr_reductions == lr_reduction_max
                break
            end
            println("Learning rate updated new LR: ", new_lr)

            lr_reductions += 1
        end

        # Training loop
        for (x, y) in final_train_loader
            loss_value, grads = withgradient(final_model) do m
                loss_function(m, stack(x), stack(y), loss_choice)
            end
            update!(optimizer, final_model, grads[1])
        end

        # Validation evaluation
        val_loss = eval_model(final_model, final_train_loader, loss_function, loss_choice)
        if  (best_final_score - val_loss) > 1e-4
            println("Final Loss: $(best_final_score)")
            best_final_score = val_loss
            current_patience = 0
        else
            current_patience += 1
        end
    end


    return final_model, norms, best_hyperparams, best_final_score
end