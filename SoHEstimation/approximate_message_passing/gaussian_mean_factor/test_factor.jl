using .Utils
using .GaussianDistribution
using .DistributionCollections
using .Factors
using Plots

@enum Variable X Y

db = DistributionCollections.DistributionBag(GaussianDistribution.Gaussian1DFromMeanVariance(4, 3/2))
factorList = Vector{Factors.Factor}()
variableList = Vector{Int}()

function get_file_name(variable, loss_choice::String, output_index::Int)
    return "model_for_$(variable)_10_000_$(loss_choice)_$(output_index).jld2"
end

# helper function to add factors to a long list
function addFactor(f)
    push!(factorList, f)
    return (f)
end

# helper functions to add variables to a list (so they can be reset later on if needed)
function addVariable!(db::DistributionCollections.DistributionBag)
    index = DistributionCollections.add!(db)
    push!(variableList, index)
    return index
end

MEANS = collect(0:50)
VARIANCES = collect(1:50)
BETAS = collect(1:50)

function evalute_factor(beta::Float64, models::Vector{String})

    # Skill variables for the two players
    s1 = addVariable!(db)
    
    # Performance variables for the two players
    p1 = addVariable!(db)
    
    # Gaussian likelihood of performance for the two players
    likel_approx = addFactor(Factors.ApproximateGaussianMeanFactor(db, p1, s1, beta * beta, models))
    likel = addFactor(Factors.GaussianMeanFactor(db, p1, s1, beta * beta))
    
    update_x = update_msg_to_x!(likel)
    update_y = update_msg_to_y!(likel)
    println("True update x: ", update_x)
    println("True update y: ", update_y)

    update_x_approx = update_msg_to_x!(likel_approx)
    update_y_aprox = update_msg_to_y!(likel_approx)
    println("Approximate update x: ", update_x_approx)
    println("Approximate update y: ", update_y_aprox)

    diff_x = abs(update_x - update_x_approx)
    diff_y = abs(update_y - update_y_aprox)

    println("Difference in x: ", diff_x)
    println("Difference in y: ", diff_y)
    
end


types = [[X, 0, "mse"],[X, 1, "mse"], [X, 2, "mse"], [Y, 0, "mse"], [Y, 1, "mse"],
         [Y, 2, "mse"],[X, 0, "kl"], [Y, 0, "kl"]
]

models_kl = [get_file_name(X, "kl", 0), get_file_name(Y, "kl", 0)]
models_kl = [string(model) for model in models_kl]

models_mse = [get_file_name(X, "mse", 0), get_file_name(Y, "mse", 0)]
models_mse = [string(model) for model in models_mse]



evalute_factor(1.0, models_kl)
evalute_factor(1.0, models_mse)