module WeightedSumFactorGeneration
export generate_dataset_weighted_sum_factor, X, Y, Z, dimension, mean, variance, remove_variable, get_variable, Variable, to_tau_rho, to_mean_variance, tau, rho, a, b, c, metropolis_hastings, adaptive_metropolis_hastings, generate_output_weighted_sum_factor, generate_input_weighted_sum_factor, get_variable

include("../../../lib/gaussian.jl")
using .GaussianDistribution

using StatsBase
using Distributions
using JLD2
using ProgressMeter
using Format
using LogExpFunctions
using SpecialFunctions
using MCMCDiagnosticTools
using Statistics
using Humanize: digitsep

# using ..Utils
using Base.Threads
using Random

@enum Variable X Y Z

function get_params_dict(; kwargs...)
    return Dict(kwargs)
end

function dimension(v::Variable, moment::Int=1)
    @assert moment == 1 || moment == 2
    return 2 * Int(v) + moment
end

"""
Assumes location-scale parameters in input vector.
"""
function mean(v::Variable, x::AbstractVector)
    @assert length(x) == 9 || length(x) == 6
    return x[dimension(v, 1)]
end

"""
Assumes location-scale parameters in input vector.
"""
function variance(v::Variable, x::AbstractVector)
    @assert length(x) == 9 || length(x) == 6
    return x[dimension(v, 2)]
end

"""
Assumes location-scale parameters in input vector.
"""
function set_mean!(variable::Variable, x::AbstractVector, value::Number)
    @assert length(x) == 9 || length(x) == 6
    x[dimension(variable, 1)] = value
end

"""
Assumes location-scale parameters in input vector.
"""
function set_variance!(variable::Variable, x::AbstractVector, value::Number)
    @assert length(x) == 9 || length(x) == 6
    x[dimension(variable, 2)] = value
end

"""
Assumes location-scale parameters in input vector.
"""
function tau(v::Variable, x::AbstractVector)
    if variance(v, x) == Inf
        return 0
    end
    return mean(v, x) / variance(v, x)
end

"""
Assumes location-scale parameters in input vector.
"""
function rho(v::Variable, x::AbstractVector)
    if variance(v, x) == Inf
        return 0
    end
    return 1 / variance(v, x)
end

"""
Assumes location-scale parameters in input vector.
"""
function std(v::Variable, x::AbstractVector)
    return sqrt(variance(v, x))
end

function remove_variable(v::Variable, x::AbstractVector)
    @assert length(x) == 9 || length(x) == 6
    return vcat(x[1:dimension(v, 1)-1], x[dimension(v, 2)+1:end])
end

"""
Assumes location-scale parameters in input vector.
"""
function get_variable(v::Variable, x::AbstractVector)
    @assert length(x) == 9 || length(x) == 6
    return [x[dimension(v, 1)], x[dimension(v, 2)]]
end

function a(x::AbstractVector)
    @assert length(x) == 9
    return x[7]
end

function b(x::AbstractVector)
    @assert length(x) == 9
    return x[8]
end

function c(x::AbstractVector)
    @assert length(x) == 9
    return x[9]
end

"""
Assumes location-scale parameters in input vector.
Randomly chooses a variable and sets it to a uniform distribution, i.e., tau = 0, rho = 0 and consequently mean = Inf, variance = Inf.
"""
function set_uniform!(input::AbstractVector)
    v = sample([v for v in instances(Variable)])
    set_mean!(v, input, 0.0)
    set_variance!(v, input, Inf)
end

"""
Assumes location-scale parameters in input vector.
"""
function is_uniform(v::Variable, input::AbstractVector)::Bool
    if mean(v, input) == 0.0 && variance(v, input) == Inf
        return true
    end

    return false
end

"""
Assumes location-scale parameters in input vector.
"""
function has_uniform(input::AbstractVector)::Bool
    for v in [X, Y, Z]
        if is_uniform(v, input)
            return true
        end
    end

    return false
end

"""
Assumes location-scale parameters in input vector.
"""
function to_tau_rho(mean, variance)
    if variance == Inf
        return 0, 0
    end
    return mean / variance, 1 / variance
end

"""
Assumes location-scale parameters in input vector.
"""
function to_tau_rho(x::AbstractVector)
    @assert length(x) == 9 || length(x) == 6 || length(x) == 2
    if length(x) == 9
        return [tau(X, x), rho(X, x), tau(Y, x), rho(Y, x), tau(Z, x), rho(Z, x), a(x), b(x), c(x)]
    elseif length(x) == 2
        return [to_tau_rho(x[1], x[2])...]
    else
        return [tau(X, x), rho(X, x), tau(Y, x), rho(Y, x), tau(Z, x), rho(Z, x)]
    end
end

"""
Assumes location-scale parameters in input vector.
"""
function to_tau_rho(x::Matrix{T}) where T<:AbstractFloat
    @assert size(x, 2) == 9 || size(x, 2) == 6 || size(x, 2) == 2
    return hcat(map(to_tau_rho, eachrow(x))...)' |> Matrix{Float32}
end

"""
Assumes natural parameters in input vector.
"""
function to_mean_variance(tau, rho)
    if rho == 0
        return 0, Inf
    end
    return tau / rho, 1 / rho
end

"""
Assumes natural parameters in input vector.
"""
function to_mean_variance(x::AbstractVector)
    @assert length(x) == 9 || length(x) == 6 || length(x) == 2
    # tau = mean / variance
    # rho = 1 / variance
    if length(x) == 2
        return [to_mean_variance(x[1], x[2])...]
    end

    xyz = [
        to_mean_variance(x[1], x[2])...,
        to_mean_variance(x[3], x[4])...,
        to_mean_variance(x[5], x[6])...,
    ]

    if length(x) == 9
        return vcat(xyz, [a(x), b(x), c(x)])
    end

    return xyz
end

"""
Assumes natural parameters in input vector.
"""
function to_mean_variance(x::Matrix{T}) where T<:AbstractFloat
    @assert size(x, 2) == 9 || size(x, 2) == 6 || size(x, 2) == 2
    return hcat(map(to_mean_variance, eachrow(x))...)' |> Matrix{Float32}
end

function log_weighted_mean(x::Vector{Float64}, log_weights::Vector{Float64})
    # make sure all weights are negative. does not distort the result, but makes the calculation more stable
    adjusted_log_weights = any(log_weights .> 0) ? log_weights .- (maximum(log_weights) + 1) : log_weights

    # makes sure all values are positive. does not distort the result, but makes the calculation more stable
    shift = abs(minimum(x)) + 1

    return exp(LogExpFunctions.logsumexp(log.(x .+ shift) .+ adjusted_log_weights) - LogExpFunctions.logsumexp(adjusted_log_weights)) - shift
end

function log_weighted_var(x::Vector{Float64}, log_weights::Vector{Float64}; mean::Union{Float64,Nothing}=nothing)
    m = isnothing(mean) ? log_weighted_mean(x, log_weights) : mean
    return log_weighted_mean((x .- m) .^ 2, log_weights)
end

function log_weighted_mean_inspect(x::Vector{Float64}, log_weights::Vector{Float64})
    # make sure all weights are negative. does not distort the result, but makes the calculation more stable
    adjusted_log_weights = any(log_weights .> 0) ? log_weights .- (maximum(log_weights) + 1) : log_weights

    # makes sure all values are positive. does not distort the result, but makes the calculation more stable
    shift = abs(minimum(x)) + 1
    return (wx=LogExpFunctions.logsumexp(log.(x .+ shift) .+ adjusted_log_weights), w=LogExpFunctions.logsumexp(adjusted_log_weights))
end

function log_weighted_var_inspect(x::Vector{Float64}, log_weights::Vector{Float64}; mean::Union{Float64,Nothing}=nothing)
    m = isnothing(mean) ? log_weighted_mean(x, log_weights) : mean
    return log_weighted_mean_inspect((x .- m) .^ 2, log_weights)
end

function improved_log_weighted_mean(x::Vector{Float64}, log_weights::Vector{Float64})
    max_log_weight = maximum(log_weights)
    normalized_log_weights = log_weights .- max_log_weight
    log_numerator = LogExpFunctions.logsumexp(log.(x) .+ normalized_log_weights)
    log_denominator = LogExpFunctions.logsumexp(normalized_log_weights)
    return exp(log_numerator - log_denominator)
end

function improved_log_weighted_var(x::Vector{Float64}, log_weights::Vector{Float64}; mean::Union{Float64,Nothing}=nothing)
    m = isnothing(mean) ? improved_log_weighted_mean(x, log_weights) : mean
    return improved_log_weighted_mean((x .- m) .^ 2, log_weights)
end

# # debug code begin
# function log_weighted_mean(x::Vector{BigFloat}, log_weights::Vector{BigFloat})
#     # make sure all weights are negative. does not distort the result, but makes the calculation more stable
#     adjusted_log_weights = any(log_weights .> 0) ? log_weights .- (maximum(log_weights) + 1) : log_weights

#     # makes sure all values are positive. does not distort the result, but makes the calculation more stable
#     shift = abs(minimum(x)) + 1

#     return exp(LogExpFunctions.logsumexp(log.(x .+ shift) .+ adjusted_log_weights) - LogExpFunctions.logsumexp(adjusted_log_weights)) - shift
# end

# function log_weighted_var(x::Vector{BigFloat}, log_weights::Vector{BigFloat}; mean::Union{BigFloat,Nothing}=nothing)
#     m = isnothing(mean) ? log_weighted_mean(x, log_weights) : mean
#     return log_weighted_mean((x .- m) .^ 2, log_weights)
# end
# # debug code end

similar_maginitude(x::Float64, y::Float64; factor=3.0) = x * factor > y && y * factor > x

"""
Assumes location-scale parameters in input vector.
"""
function variance_magnitudes_match(input::Vector{Float64}; factor=3.0)
    x_term_variance = variance(X, input) * a(input)^2
    y_term_variance = variance(Y, input) * b(input)^2
    z_term_variance = variance(Z, input)

    return similar_maginitude(x_term_variance, y_term_variance; factor=factor) &&
           similar_maginitude(x_term_variance, z_term_variance; factor=factor) &&
           similar_maginitude(y_term_variance, z_term_variance; factor=factor)
end

"""
Assumes location-scale parameters in input vector.
"""
function std_magnitudes_match(input::Vector{Float64}; factor=3.0)
    x_term_std = std(X, input) * abs(a(input))
    y_term_std = std(Y, input) * abs(b(input))
    z_term_std = std(Z, input)

    return similar_maginitude(x_term_std, y_term_std; factor=factor) &&
           similar_maginitude(x_term_std, z_term_std; factor=factor) &&
           similar_maginitude(y_term_std, z_term_std; factor=factor)
end

"""
Assumes location-scale parameters in input vector.
Checks if one of the variances in input is larger than the other two by a certain factor.
If so, sets the variable with the large variance to a uniform distribution in a new input vector and resturns it.
"""
function large_variance_to_uniform(input::Vector{Float64}; factor=50.0)
    input_vector = deepcopy(input)
    for v in [X, Y, Z]
        if variance(v, input) > factor * sort([variance(X, input), variance(Y, input), variance(Z, input)])[2]
            set_variance!(v, input_vector, Inf)
            set_mean!(v, input_vector, 0.0)
            return input_vector
        end
    end
end

"""
given a fraction x, return the multiple of the standard deviation within which this fraction of the population lies
i.e. if x = .68, return 1, since 68% of the population lies within 1 standard deviation of the mean and so on
"""
sigmult(x) = quantile(Normal(), (1 + x) / 2)


"""
WIP: do not use until fixed
return the lower and upper bounds of the z variable, given the means and variances of x, y and z (implied mean only for z),
as well as the coefficients a, b and c
"""
function z_min_max(x_mean, x_var, y_mean, y_var, z_implied_mean, z_var, a, b, c; samples_per_input=1_000_000, M=1e300)::Tuple{Float64}
    # fraction = 200 / samples_per_input # guessing we need at least 100 samples to get a good estimate of the mean and variance
    # sigma_multiple = sigmult(1-fraction)
    # std_left_term = sqrt(x_var * a^2 + y_var * b^2)
    # std_right_term = sqrt(z_var)

    fraction = 200 / samples_per_input

    mean_term = (x_mean * a + y_mean * b + c)
    @assert mean_term == z_implied_mean

    var_term = sqrt(z_var + a^2 * x_var + b^2 * y_var) * sqrt(2) * erfinv(fraction)

    lb = mean_term - M + var_term
    ub = mean_term + M - var_term

    return lb, ub
end

"""
Assumes location-scale parameters in input vector.
Returns true if variance of output is larger than input variance by more than a certain factor.
"""
function exploding_variance(input::Vector{Float64}, output::Vector{Float64}; factor=10.0)
    for v in [X, Y, Z]
        if !similar_maginitude(variance(v, output), variance(v, input); factor=factor)
            return true
        end
    end

    return false
end

function eval(x::Float64, y::Float64, z::Float64, dirac_std::Float64=1e-1; a::Float64=1.0, b::Float64=-1.0, c::Float64=0.0, log_weighting=true)
    result = log_weighting ? logpdf(Normal(0.0, dirac_std), z - (x * a + y * b + c)) : pdf(Normal(0.0, dirac_std), z - (x * a + y * b + c))
    return result
end

function eval(x::Vector{Float64}, y::Vector{Float64}, z::Vector{Float64}, dirac_std::Float64=1e-1;
    a::Float64=1.0, b::Float64=-1.0, c::Float64=0.0, log_weighting=true)

    diff = z .- (x .* a .+ y .* b .+ c)  # Element-wise operation
    if log_weighting
        return logpdf.(Normal(0.0, dirac_std), diff)  # Broadcasted logpdf
    else
        return pdf.(Normal(0.0, dirac_std), diff)     # Broadcasted pdf
    end
end

# function relative_eval(x::Vector{Float64}, y::Vector{Float64}, z::Vector{Float64}, dirac_std::Float64=1e-1;
#     a::Float64=1.0, b::Float64=-1.0, c::Float64=0.0, log_weighting=true)

#     diff = z .- (x .* a .+ y .* b .+ c)  # Element-wise operation
#     normalized_diff = (diff .- StatsBase.mean(diff)) ./ StatsBase.std(diff)
#     if log_weighting
#         return logpdf.(Normal(0.0, dirac_std), normalized_diff)  # Broadcasted logpdf
#     else
#         return pdf.(Normal(0.0, dirac_std), normalized_diff)     # Broadcasted pdf
#     end
# end

function get_implied_variable(x_value, y_value, z_value, input, variable)
    if variable == Z
        return x_value, y_value, x_value * a(input) + y_value * b(input) + c(input)
    elseif variable == X
        return (z_value - y_value * b(input) - c(input)) / a(input), y_value, z_value
    elseif variable == Y
        return x_value, (z_value - x_value * a(input) - c(input)) / b(input), z_value
    end
    throw(ArgumentError("Invalid variable"))
end

function importance_sampling(input::Vector{Float64};
    samples_per_input::Int=1_000_000,
    dirac_std::Float64=1e-1,
    log_weighting=true,
)
    # Create normal distributions for sampling
    old_msg_from_x = Normal(mean(X, input), std(X, input))
    old_msg_from_y = Normal(mean(Y, input), std(Y, input))
    old_msg_from_z = Normal(mean(Z, input), std(Z, input))

    # Vectorized sampling of x, y, z
    samples_x = rand(old_msg_from_x, samples_per_input)
    samples_y = rand(old_msg_from_y, samples_per_input)
    samples_z = rand(old_msg_from_z, samples_per_input)

    # Vectorized weight computation
    weights = eval(
        samples_x, samples_y, samples_z, dirac_std;
        a=a(input), b=b(input), c=c(input), log_weighting=log_weighting
    )

    if log_weighting
        # Log-weighted mean and variance
        mean_sampled_marginal_x = log_weighted_mean(samples_x, weights)
        var_sampled_marginal_x = log_weighted_var(samples_x, weights; mean=mean_sampled_marginal_x)
        mean_sampled_marginal_y = log_weighted_mean(samples_y, weights)
        var_sampled_marginal_y = log_weighted_var(samples_y, weights; mean=mean_sampled_marginal_y)
        mean_sampled_marginal_z = log_weighted_mean(samples_z, weights)
        var_sampled_marginal_z = log_weighted_var(samples_z, weights; mean=mean_sampled_marginal_z)
    else
        # Weighted mean and variance
        mean_sampled_marginal_x = StatsBase.mean(samples_x, Weights(weights))
        var_sampled_marginal_x = StatsBase.var(samples_x, Weights(weights); mean=mean_sampled_marginal_x)
        mean_sampled_marginal_y = StatsBase.mean(samples_y, Weights(weights))
        var_sampled_marginal_y = StatsBase.var(samples_y, Weights(weights); mean=mean_sampled_marginal_y)
        mean_sampled_marginal_z = StatsBase.mean(samples_z, Weights(weights))
        var_sampled_marginal_z = StatsBase.var(samples_z, Weights(weights); mean=mean_sampled_marginal_z)
    end

    marginal_output = [mean_sampled_marginal_x, var_sampled_marginal_x, mean_sampled_marginal_y, var_sampled_marginal_y, mean_sampled_marginal_z, var_sampled_marginal_z]
    
    # specific to our application: clamp variances, so that they are not larger than input variances, 
    # since that will cause negative rho values when calculating the new messages to the variables
    for v in [X, Y, Z]
        marginal_output[dimension(v, 2)] = min(marginal_output[dimension(v, 2)], variance(v, input))
    end
    
    # # debug code begin
    # if !all(isfinite, marginal_output)
    #     println("________________________________________________________")
    #     println("NaN or Inf in updated marginals")
    #     println("Weights count: ", length(weights), " #nans:", sum(isnan, weights), " #infs:", sum(isinf, weights), " #zeros:", sum(weights .== 0))
    #     println("Input Rho/Tau: ", input)
    #     println("Input Mean/Var", to_mean_variance(input))
    #     println("Sampled Rhos/Taus: ", marginal_output)
    #     # println("Sampled Mean/Var", to_mean_variance(marginal_output))
    #     println("Sampled Mean/Var: ", [mean_sampled_marginal_x, var_sampled_marginal_x, mean_sampled_marginal_y, var_sampled_marginal_y, mean_sampled_marginal_z, var_sampled_marginal_z])

    #     println("analytical marginals Mean/Var: ", update_marginals(to_mean_variance(input)...; natural_parameters=false))
    #     println("analytical marginals Tau/Rho: ", update_marginals(to_mean_variance(input)...; natural_parameters=true))

    #     std_ax = std(X, input) * a(input)
    #     std_by = std(Y, input) * b(input)
    #     std_z = std(Z, input)
    #     std_axby = sqrt(variance(X, input) * a(input)^2 + variance(Y, input) * b(input)^2)
    #     std_axbyz = sqrt(variance(X, input) * a(input)^2 + variance(Y, input) * b(input)^2 + variance(Z, input))

    #     println("std_ax: ", std_ax)
    #     println("std_by: ", std_by)
    #     println("std_z: ", std_z)
    #     println("std_axby: ", std_axby)
    #     println("std_axbyz: ", std_axbyz)
    #     println("________________________________________________________")
    #     return (weights=weights, samples_x=samples_x, samples_y=samples_y, samples_z=samples_z)
    # end
    # # debug code end

    return marginal_output
end

# function relative_eval(x::Vector{Float64}, y::Vector{Float64}, z::Vector{Float64}, dirac_std::Float64=1e-1;
#     a::Float64=1.0, b::Float64=-1.0, c::Float64=0.0, log_weighting=true)

#     diff = z .- (x .* a .+ y .* b .+ c)  # Element-wise operation
#     normalized_diff = (diff .- StatsBase.mean(diff)) ./ StatsBase.std(diff)
#     if log_weighting
#         return logpdf.(Normal(0.0, dirac_std), normalized_diff)  # Broadcasted logpdf
#     else
#         return pdf.(Normal(0.0, dirac_std), normalized_diff)     # Broadcasted pdf
#     end
# end

# function adaptive_metropolis_hastings(input::Vector{Float64};
#     dirac_std::Float64=1e-1,
#     max_samples::Int=1_000_000,
#     burn_in::Float64=0.1,
#     variance_relative_epsilon=1e-10,
#     adaptation_interval::Int=100,
#     target_acceptance::Float64=0.234,
#     min_samples::Int=1000,
#     convergence_window::Int=200
# )
#     # Initialize
#     num_burn_in_samples = Int(max_samples * burn_in)
#     std_msgfrom_x, std_msgfrom_y, std_msgfrom_z = std(X, input), std(Y, input), std(Z, input)
    
#     # Define prior distributions
#     x_prior(x) = logpdf(Normal(mean(X, input), std_msgfrom_x), x)
#     y_prior(y) = logpdf(Normal(mean(Y, input), std_msgfrom_y), y)
#     z_prior(z) = logpdf(Normal(mean(Z, input), std_msgfrom_z), z)
#     likelihood(x, y, z) = eval(x, y, z, dirac_std; a=a(input), b=b(input), c=c(input), log_weighting=true)
#     multiply(x...) = +(x...)
#     divide(X...) = -(X...)
#     acceptance_rate() = log(rand())

#     # Initialize adaptive parameters
#     proposal_scales = [0.1, 0.1, 0.1]  # Initial scales for x, y, z
#     acceptance_history = Bool[]
    
#     # Initialize current state and samples
#     x_current = mean(X, input)
#     y_current = mean(Y, input)
#     z_current = mean(Z, input)
#     x_samples = Float64[]
#     y_samples = Float64[]
#     z_samples = Float64[]
    
#     # Function to adapt proposal scales
#     function adapt_proposals!(scales, history)
#         if length(history) >= adaptation_interval
#             recent_acceptance = StatsBase.mean(history[end-adaptation_interval+1:end])
#             for i in eachindex(scales)
#                 log_scale = log(scales[i])
#                 if recent_acceptance > target_acceptance
#                     log_scale += 0.1
#                 else
#                     log_scale -= 0.1
#                 end
#                 scales[i] = exp(log_scale)
#             end
#         end
#     end
    
#     # Function to check convergence
#     function check_convergence(x_samples, y_samples, z_samples, proposal_scales, acceptance_history)
#         if length(x_samples) < min_samples
#             return false
#         end
        
#         # Check recent samples
#         if length(x_samples) >= convergence_window
#             # Geweke diagnostic on x, y, z
#             for samples in (x_samples, y_samples, z_samples)
#                 recent = samples[end-convergence_window+1:end]
#                 first_20pct = recent[1:Int(0.2*convergence_window)]
#                 last_50pct = recent[Int(0.5*convergence_window):end]
                
#                 if abs(StatsBase.mean(first_20pct) -StatsBase. mean(last_50pct)) > 0.1
#                     return false
#                 end
#             end
            
#             # Check proposal scale stability
#             recent_scales = proposal_scales[end-convergence_window+1:end]
#             if any(StatsBase.std(scales) > 0.01 for scales in recent_scales)
#                 return false
#             end
            
#             # Check acceptance rate stability
#             recent_acceptance = StatsBase.mean(acceptance_history[end-convergence_window+1:end])
#             if abs(recent_acceptance - target_acceptance) > 0.05
#                 return false
#             end
            
#             return true
#         end
#         return false
#     end
    
#     accepted = 0
#     i = 0
    
#     while i < max_samples + num_burn_in_samples
#         i += 1
        
#         # Propose new values using adaptive scales
#         x_proposal = rand(Normal(x_current, std_msgfrom_x * proposal_scales[1]))
#         y_proposal = rand(Normal(y_current, std_msgfrom_y * proposal_scales[2]))
#         z_proposal = rand(Normal(z_current, std_msgfrom_z * proposal_scales[3]))
        
#         # Compute log probabilities
#         posterior_current = multiply(x_prior(x_current), y_prior(y_current), 
#                                   z_prior(z_current), likelihood(x_current, y_current, z_current))
#         posterior_proposal = multiply(x_prior(x_proposal), y_prior(y_proposal), 
#                                    z_prior(z_proposal), likelihood(x_proposal, y_proposal, z_proposal))
        
#         # Metropolis-Hastings acceptance step
#         accepted_step = acceptance_rate() < divide(posterior_proposal, posterior_current)
#         push!(acceptance_history, accepted_step)
        
#         if accepted_step
#             x_current, y_current, z_current = x_proposal, y_proposal, z_proposal
#             accepted += 1
#         end
        
#         if i > num_burn_in_samples
#             push!(x_samples, x_current)
#             push!(y_samples, y_current)
#             push!(z_samples, z_current)
#         end
        
#         # Adapt proposals periodically
#         if i % adaptation_interval == 0
#             adapt_proposals!(proposal_scales, acceptance_history)
#         end
        
#         # Check convergence periodically
#         if i % 100 == 0 && i > num_burn_in_samples
#             if check_convergence(x_samples, y_samples, z_samples, 
#                                [proposal_scales], acceptance_history)
#                 @info "Converged after $i iterations"
#                 break
#             end
#         end
#     end
        
#     # Calculate output statistics
#     output = [
#         StatsBase.mean(x_samples), StatsBase.var(x_samples),
#         StatsBase.mean(y_samples), StatsBase.var(y_samples),
#         StatsBase.mean(z_samples), StatsBase.var(z_samples)
#     ]
    
#     # Clamp variances
#     for v in [X, Y, Z]
#         output[dimension(v, 2)] = min(
#             output[dimension(v, 2)], 
#             variance(v, input) * (1 - variance_relative_epsilon)
#         )
#     end

#     @assert length(x_samples) < max_samples
    
#     return to_tau_rho(output)
# end

function adaptive_metropolis_hastings(input::Vector{Float64};
    dirac_std::Float64=1e-1,
    max_samples::Int=1_000_000,
    burn_in::Float64=0.5,
    variance_relative_epsilon=1e-10,
    adaptation_interval::Int=1_000,
    target_acceptance_ratio::Float64=0.23,
    min_samples::Int=100_000,
    n_next_iteration=x -> 2 * x,
    one_variable_fixed=false,
    # global_proposal_dist_scale=10.0,
    # local_proposal_dist_initial_weight=0.1,
    # global_proposal_dist_weight=0.99,
    # local_proposal_dist_weight=0.1,
)
    std_msgfrom_x, std_msgfrom_y, std_msgfrom_z = std(X, input), std(Y, input), std(Z, input)
    mean_msg_from_x, mean_msg_from_y, mean_msg_from_z = mean(X, input), mean(Y, input), mean(Z, input)
    
    x_prior(x) = logpdf(Normal(mean_msg_from_x, std_msgfrom_x), x)
    y_prior(y) = logpdf(Normal(mean_msg_from_y, std_msgfrom_y), y)
    z_prior(z) = logpdf(Normal(mean_msg_from_z, std_msgfrom_z), z)
    likelihood(x, y, z) = eval(x, y, z, dirac_std; a=a(input), b=b(input), c=c(input), log_weighting=true)
    multiply(x...) = +(x...)
    divide(X...) = -(X...)
    acceptance_rate() = log(rand())

    proposal_scale = 0.1
    acceptance_history = Bool[]

    # global_proposal_dist_x = Normal(mean_msg_from_x, std_msgfrom_x) * global_proposal_dist_scale
    # global_proposal_dist_y = Normal(mean_msg_from_y, std_msgfrom_y) * global_proposal_dist_scale
    # global_proposal_dist_z = Normal(mean_msg_from_z, std_msgfrom_z) * global_proposal_dist_scale
    
    x_current =  rand(Normal(mean_msg_from_x, std_msgfrom_x)) # mean_msg_from_x
    y_current = rand(Normal(mean_msg_from_y, std_msgfrom_y)) # mean_msg_from_y
    z_current = rand(Normal(mean_msg_from_z, std_msgfrom_z)) # mean_msg_from_z

    if one_variable_fixed
        x_current, y_current, z_current = get_implied_variable(x_current, y_current, z_current, input, rand(instances(Variable)))
    end

    x_samples = Float64[]
    y_samples = Float64[]
    z_samples = Float64[]
    
    
    # Function to adapt proposal scales
    function adapt_proposals(scale, history)
        if length(history) >= adaptation_interval
            recent_acceptance = StatsBase.mean(history[end-adaptation_interval+1:end])
            if recent_acceptance > target_acceptance_ratio
                return scale * 1.1
            elseif recent_acceptance < target_acceptance_ratio
                return scale * 0.9
            else
                return scale
            end
        end
    end
    
    # Function to check convergence
    function check_convergence(x_relevant_samples, y_relevant_samples, z_relevant_samples)
        if length(x_relevant_samples) < min_samples
            return false
        end
        
        function check_geweke(samples)
            zscore, pvalue = gewekediag(samples)
            return abs(zscore) < 1.96 && pvalue > 0.05
        end

        function check_ess_rhat(samples)
            ess_value, rhat_value = ess_rhat(samples)
            return ess_value > 100 && rhat_value < 1.1
        end

        # Geweke diagnostic on x, y, z
        for (i, samples) in enumerate((x_relevant_samples, y_relevant_samples, z_relevant_samples))
            if !check_geweke(samples) || bfmi(samples) > 0.3 || mcse(samples) > 0.1 || mcse(samples; kind=Statistics.std) > 0.1 || !check_ess_rhat(samples)
                return false, i
            end
        end
        
        return true
    end
    
    accepted = 0
    i = 0

    early_stop = false
    
    n_iterations = min_samples

    posterior_current = multiply(x_prior(x_current), y_prior(y_current), 
                                  z_prior(z_current), likelihood(x_current, y_current, z_current))

    while i < max_samples
        i += 1
        
        # Propose new values using adaptive scales
        x_proposal = rand(Normal(x_current, std_msgfrom_x * proposal_scale))
        y_proposal = rand(Normal(y_current, std_msgfrom_y * proposal_scale))
        z_proposal = rand(Normal(z_current, std_msgfrom_z * proposal_scale))

        if one_variable_fixed
            x_proposal, y_proposal, z_proposal = get_implied_variable(x_proposal, y_proposal, z_proposal, input, rand(instances(Variable)))
        end
        
        # Compute log probabilities
        posterior_proposal = multiply(x_prior(x_proposal), y_prior(y_proposal), 
                                   z_prior(z_proposal), likelihood(x_proposal, y_proposal, z_proposal))
        
        # Metropolis-Hastings acceptance step
        accepted_step = acceptance_rate() < divide(posterior_proposal, posterior_current)
        push!(acceptance_history, accepted_step)
        
        if accepted_step
            x_current, y_current, z_current = x_proposal, y_proposal, z_proposal
            posterior_current = posterior_proposal
            accepted += 1
        end
        
        push!(x_samples, x_current)
        push!(y_samples, y_current)
        push!(z_samples, z_current)
        
        # Adapt proposals periodically
        if i % adaptation_interval == 0
            proposal_scale = adapt_proposals(proposal_scale, acceptance_history)
        end
        
        # Check convergence periodically
        if i == n_iterations
            relevant_samples_x, relevant_samples_y, relevant_samples_z = x_samples[Int(end*burn_in):end], y_samples[Int(end*burn_in):end], z_samples[Int(end*burn_in):end]
            if check_convergence(relevant_samples_x, relevant_samples_y, relevant_samples_z)[1]
                #@info "Converged after $(digitsep(i, seperator= "_")) iterations, overall acceptance rate of $(StatsBase.mean(acceptance_history[Int(end*burn_in):end]))"
                x_samples, y_samples, z_samples = relevant_samples_x, relevant_samples_y, relevant_samples_z
                early_stop = true
                break
            else
                n_iterations = n_next_iteration(length(x_samples))
            end
        end
    end
        
    # Calculate output statistics
    output = [
        StatsBase.mean(x_samples), StatsBase.var(x_samples),
        StatsBase.mean(y_samples), StatsBase.var(y_samples),
        StatsBase.mean(z_samples), StatsBase.var(z_samples)
    ]
    
    # Clamp variances
    for v in [X, Y, Z]
        output[dimension(v, 2)] = min(
            output[dimension(v, 2)], 
            variance(v, input) * (1 - variance_relative_epsilon)
        )
    end

    converged = check_convergence(x_samples, y_samples, z_samples)
    if i >= max_samples && !converged[1]
        #@warn "Did not converge after $(digitsep(i, seperator= "_")) iterations, overall acceptance rate of $(StatsBase.mean(acceptance_history[Int(end*burn_in):end]))" # , failed on variable $(converged[2])"
    end
    
    return output, early_stop, converged[1]
end

function metropolis_hastings(input::Vector{Float64};
    dirac_std::Float64=1e-1,
    num_samples::Int=1_000_000,
    burn_in::Float64=0.1,
    log_weighting::Bool=true,
    variance_relative_epsilon=1e-10,
)
    num_burn_in_samples = Int(num_samples * burn_in)
    std_msgfrom_x, std_msgfrom_y, std_msgfrom_z = std(X, input), std(Y, input), std(Z, input)

    # Define prior distributions
    x_prior(x) = log_weighting ? logpdf(Normal(mean(X, input), std_msgfrom_x), x) : pdf(Normal(mean(X, input), std_msgfrom_x), x)
    y_prior(y) = log_weighting ? logpdf(Normal(mean(Y, input), std_msgfrom_y), y) : pdf(Normal(mean(Y, input), std_msgfrom_y), y)
    z_prior(z) = log_weighting ? logpdf(Normal(mean(Z, input), std_msgfrom_z), z) : pdf(Normal(mean(Z, input), std_msgfrom_z), z)
    likelihood(x, y, z) = eval(x, y, z, dirac_std; a=a(input), b=b(input), c=c(input), log_weighting=log_weighting)

    multiply(x...) = log_weighting ? +(x...) : *(x...)
    divide(X...) = log_weighting ? -(X...) : /(X...)
    acceptance_rate() = log_weighting ? log(rand()) : rand()


    # Initialize
    x_current = mean(X, input)
    y_current = mean(Y, input)
    z_current = mean(Z, input)

    x_samples = zeros(num_samples)
    y_samples = zeros(num_samples)
    z_samples = zeros(num_samples)
    accepted = 0

    for i in 1:(num_samples+num_burn_in_samples)
        # Propose new x and y
        x_proposal = rand(Normal(x_current, std_msgfrom_x * 0.1))
        y_proposal = rand(Normal(y_current, std_msgfrom_y * 0.1))
        # z_proposal = x_proposal * a + y_proposal * b + c + rand(Normal(0, 0.1))
        z_proposal = rand(Normal(z_current, std_msgfrom_z * 0.1))

        # Compute log probabilities
        posterior_current = multiply(x_prior(x_current), y_prior(y_current), z_prior(z_current), likelihood(x_current, y_current, z_current))
        posterior_proposal = multiply(x_prior(x_proposal), y_prior(y_proposal), z_prior(z_proposal), likelihood(x_proposal, y_proposal, z_proposal))

        # Metropolis-Hastings acceptance ratio
        if acceptance_rate() < divide(posterior_proposal, posterior_current)
            x_current, y_current, z_current = x_proposal, y_proposal, z_proposal
            accepted += 1
        end

        if i > num_burn_in_samples
            x_samples[i-num_burn_in_samples] = x_current
            y_samples[i-num_burn_in_samples] = y_current
            z_samples[i-num_burn_in_samples] = z_current
        end
    end

    acceptance_quota = accepted / (num_samples + num_burn_in_samples)
    # println("Acceptance quota: ", acceptance_quota)

    output = [StatsBase.mean(x_samples), StatsBase.var(x_samples), StatsBase.mean(y_samples), StatsBase.var(y_samples), StatsBase.mean(z_samples), StatsBase.var(z_samples)]
    
    # specific to our application: clamp variances, so that they are not larger than input variances, 
    # since that will cause negative rho values when calculating the new messages to the variables
    for v in [X, Y, Z]
        output[dimension(v, 2)] = min(output[dimension(v, 2)], variance(v, input)*(1-variance_relative_epsilon))
    end

    return output
end

function importance_sampling(input::Vector{Float64};
    samples_per_input::Int=1_000_000,
    dirac_std::Float64=1e-1,
    log_weighting=true,
)
    # Create normal distributions for sampling
    old_msg_from_x = Normal(mean(X, input), std(X, input))
    old_msg_from_y = Normal(mean(Y, input), std(Y, input))
    old_msg_from_z = Normal(mean(Z, input), std(Z, input))

    # Vectorized sampling of x, y, z
    samples_x = rand(old_msg_from_x, samples_per_input)
    samples_y = rand(old_msg_from_y, samples_per_input)
    samples_z = rand(old_msg_from_z, samples_per_input)

    # Vectorized weight computation
    weights = eval(
        samples_x, samples_y, samples_z, dirac_std;
        a=a(input), b=b(input), c=c(input), log_weighting=log_weighting
    )

    if log_weighting
        # Log-weighted mean and variance
        mean_sampled_marginal_x = log_weighted_mean(samples_x, weights)
        var_sampled_marginal_x = log_weighted_var(samples_x, weights; mean=mean_sampled_marginal_x)
        mean_sampled_marginal_y = log_weighted_mean(samples_y, weights)
        var_sampled_marginal_y = log_weighted_var(samples_y, weights; mean=mean_sampled_marginal_y)
        mean_sampled_marginal_z = log_weighted_mean(samples_z, weights)
        var_sampled_marginal_z = log_weighted_var(samples_z, weights; mean=mean_sampled_marginal_z)
    else
        # Weighted mean and variance
        mean_sampled_marginal_x = StatsBase.mean(samples_x, Weights(weights))
        var_sampled_marginal_x = StatsBase.var(samples_x, Weights(weights); mean=mean_sampled_marginal_x)
        mean_sampled_marginal_y = StatsBase.mean(samples_y, Weights(weights))
        var_sampled_marginal_y = StatsBase.var(samples_y, Weights(weights); mean=mean_sampled_marginal_y)
        mean_sampled_marginal_z = StatsBase.mean(samples_z, Weights(weights))
        var_sampled_marginal_z = StatsBase.var(samples_z, Weights(weights); mean=mean_sampled_marginal_z)
    end

    marginal_output = [mean_sampled_marginal_x, var_sampled_marginal_x, mean_sampled_marginal_y, var_sampled_marginal_y, mean_sampled_marginal_z, var_sampled_marginal_z]
    
    # specific to our application: clamp variances, so that they are not larger than input variances, 
    # since that will cause negative rho values when calculating the new messages to the variables
    for v in [X, Y, Z]
        marginal_output[dimension(v, 2)] = min(marginal_output[dimension(v, 2)], variance(v, input))
    end
    
    # # debug code begin
    # if !all(isfinite, marginal_output)
    #     println("________________________________________________________")
    #     println("NaN or Inf in updated marginals")
    #     println("Weights count: ", length(weights), " #nans:", sum(isnan, weights), " #infs:", sum(isinf, weights), " #zeros:", sum(weights .== 0))
    #     println("Input Rho/Tau: ", input)
    #     println("Input Mean/Var", to_mean_variance(input))
    #     println("Sampled Rhos/Taus: ", marginal_output)
    #     # println("Sampled Mean/Var", to_mean_variance(marginal_output))
    #     println("Sampled Mean/Var: ", [mean_sampled_marginal_x, var_sampled_marginal_x, mean_sampled_marginal_y, var_sampled_marginal_y, mean_sampled_marginal_z, var_sampled_marginal_z])

    #     println("analytical marginals Mean/Var: ", update_marginals(to_mean_variance(input)...; natural_parameters=false))
    #     println("analytical marginals Tau/Rho: ", update_marginals(to_mean_variance(input)...; natural_parameters=true))

    #     std_ax = std(X, input) * a(input)
    #     std_by = std(Y, input) * b(input)
    #     std_z = std(Z, input)
    #     std_axby = sqrt(variance(X, input) * a(input)^2 + variance(Y, input) * b(input)^2)
    #     std_axbyz = sqrt(variance(X, input) * a(input)^2 + variance(Y, input) * b(input)^2 + variance(Z, input))

    #     println("std_ax: ", std_ax)
    #     println("std_by: ", std_by)
    #     println("std_z: ", std_z)
    #     println("std_axby: ", std_axby)
    #     println("std_axbyz: ", std_axbyz)
    #     println("________________________________________________________")
    #     return (weights=weights, samples_x=samples_x, samples_y=samples_y, samples_z=samples_z)
    # end
    # # debug code end

    return marginal_output
end

function sample_uniform_update(input, samples_per_input)
    @assert has_uniform(input) "No uniform variable in input"

    if is_uniform(X, input)
        samples_y = rand(Normal(mean(Y, input), std(Y, input)), samples_per_input)
        samples_z = rand(Normal(mean(Z, input), std(Z, input)), samples_per_input)

        samples_x = (samples_z .- samples_y .* b(input) .- c(input)) ./ a(input)

        return [StatsBase.mean(samples_x), StatsBase.var(samples_x), mean(Y, input), variance(Y, input), mean(Z, input), variance(Z, input)]

    elseif is_uniform(Y, input)
        samples_x = rand(Normal(mean(X, input), std(X, input)), samples_per_input)
        samples_z = rand(Normal(mean(Z, input), std(Z, input)), samples_per_input)

        samples_y = (samples_z .- samples_x .* a(input) .- c(input)) ./ b(input)
        tau_y, rho_y = to_tau_rho(StatsBase.mean(samples_y), StatsBase.var(samples_y))

        return [mean(X, input), variance(X, input), StatsBase.mean(samples_y), StatsBase.var(samples_y), mean(Z, input), variance(Z, input)]

    elseif is_uniform(Z, input)
        samples_x = rand(Normal(mean(X, input), std(X, input)), samples_per_input)
        samples_y = rand(Normal(mean(Y, input), std(Y, input)), samples_per_input)

        samples_z = samples_x .* a(input) .+ samples_y .* b(input) .+ c(input)

        return [mean(X, input), variance(X, input), mean(Y, input), variance(Y, input), StatsBase.mean(samples_z), StatsBase.var(samples_z)]
    end

    return [StatsBase.mean(samples_x), StatsBase.var(samples_x), StatsBase.mean(samples_y), StatsBase.var(samples_y), StatsBase.mean(samples_z), StatsBase.var(samples_z)]
end

"""
generate a sample: old mean and var for msg_from_x, msg_from_y and msg_from_z, as well as a, b and c, so 9 Floats
"""
function generate_input_weighted_sum_factor(;
    variable_mean_dist::Distribution=Uniform(-100, 100),
    variable_std_dist::Distribution=Uniform(1e-3, 8.0),
    factor_dist::Distribution=Uniform(-100, 100),
    bias_dist::Distribution=Uniform(-100, 100),
    std_magnitude_factor=3.0,
    implied_z_max_error=15.0,
    uniform_quota=0.05,
)
    @assert isnothing(implied_z_max_error) || implied_z_max_error > 0 "implied_z_max_error must be positive or nothing"
    input = Vector{Float64}(undef, 9)
    trials = 0
    while trials < 5_000

        msg_from_x_std, msg_from_y_std, msg_from_z_std = shuffle(rand(variable_std_dist))
        msg_from_x_var, msg_from_y_var, msg_from_z_var = msg_from_x_std^2, msg_from_y_std^2, msg_from_z_std^2

        msg_from_x_mean = rand(variable_mean_dist)
        msg_from_y_mean = rand(variable_mean_dist)

        a, b = rand(factor_dist, 2)
        c = rand(bias_dist)

        z_implied_mean = (msg_from_x_mean) * a + (msg_from_y_mean) * b + c

        if !isnothing(implied_z_max_error)
            msg_from_z_mean_dist = Truncated(Normal(z_implied_mean, implied_z_max_error * 2/3), z_implied_mean - implied_z_max_error, z_implied_mean + implied_z_max_error)
            msg_from_z_mean = rand(msg_from_z_mean_dist)
        else
            msg_from_z_mean = rand(variable_mean_dist)
        end

        temp_input = [msg_from_x_mean, msg_from_x_var, msg_from_y_mean, msg_from_y_var, msg_from_z_mean, msg_from_z_var, a, b, c]

        if std_magnitudes_match(temp_input; factor=std_magnitude_factor) # floor(Int, log10(abs(msg_from_x_mean * a))) == floor(Int, log10(abs(msg_from_y_mean * b))) && floor(Int, log10(abs(a))) == floor(Int, log10(abs(b))) # 
            input = temp_input
            @assert all(isfinite, input)

            if rand() < uniform_quota
                set_uniform!(input)
            end

            return input
        end
        trials +=1
    end
    return error("Maximum trial count reached for input generation.")

end

"""
given a sample from generate_inputs_weighted_sum_factor() in natural parameters (tau/rho), calculate tau and rho for updated message to x, y and z, so 6 Floats
"""
function generate_output_weighted_sum_factor(input::Vector{Float64};
    samples_per_input::Int=1_000_000,
    dirac_std::Float64=1e-1,
    log_weighting=true,
    variance_relative_epsilon=1e-10,
    algorithm=:metropolis_hastings,
    set_large_variances_to_uniform=false,
    burn_in=0.5,
    debug=false,
    one_variable_fixed=true,
)

    @assert all(!isnan, [variance(v, input) for v in [X, Y, Z]]) "Inf in input variances" # Inf accepted for uniform variables
    @assert all(isfinite, [mean(v, input) for v in [X, Y, Z]]) "NaN or Inf in input means"

    input = set_large_variances_to_uniform ? large_variance_to_uniform(input) : input

    if has_uniform(input)
        marginal_output = sample_uniform_update(input, samples_per_input)
        converged = true
        early_stop = false
    else
        @assert all(>(0), [variance(v, input) for v in [X, Y, Z]]) "Values <= 0 in input variances"
        # TODO: Adapt this to input in location/scale parameters or assess if still necessary
        # @assert all(isfinite, [variance(v, input) for v in [X, Y, Z]]) "NaN or Inf in input variances. Cannot handle that yet."
        # @assert all(isfinite, [mean(v, input) for v in [X, Y, Z]]) "NaN or Inf in input means"

        if algorithm == :metropolis_hastings
            marginal_output = metropolis_hastings(input; dirac_std=dirac_std, num_samples=samples_per_input, log_weighting=log_weighting, variance_relative_epsilon=variance_relative_epsilon, burn_in=burn_in)
        elseif algorithm == :adaptive_metropolis_hastings
            marginal_output, early_stop, converged = adaptive_metropolis_hastings(input; dirac_std=dirac_std, max_samples=samples_per_input, burn_in=burn_in, variance_relative_epsilon=variance_relative_epsilon, one_variable_fixed=one_variable_fixed)
        elseif algorithm == :importance_sampling
            marginal_output = importance_sampling(input; samples_per_input=samples_per_input, dirac_std=dirac_std, log_weighting=log_weighting)
        else
            error("Unknown algorithm: $algorithm")
        end

        # if we clamp here, we will get means that are way off, because we need the original rho value to calculate the original mean. therefore, clamp the variance we get from sampling
    end

    @assert all(isfinite, marginal_output) "NaN or Inf in updated marginals for input $input and output $marginal_output"

    sampled_marginal_x = GaussianDistribution.Gaussian1D(tau(X, marginal_output), rho(X, marginal_output))
    sampled_marginal_y = GaussianDistribution.Gaussian1D(tau(Y, marginal_output), rho(Y, marginal_output))
    sampled_marginal_z = GaussianDistribution.Gaussian1D(tau(Z, marginal_output), rho(Z, marginal_output))

    # Precompute variances and means for Gaussian update
    new_msg_to_x = sampled_marginal_x / GaussianDistribution.Gaussian1D(tau(X, input), rho(X, input))
    new_msg_to_y = sampled_marginal_y / GaussianDistribution.Gaussian1D(tau(Y, input), rho(Y, input))
    new_msg_to_z = sampled_marginal_z / GaussianDistribution.Gaussian1D(tau(Z, input), rho(Z, input))

    if rho(X, marginal_output) == rho(X, input) || rho(Y, marginal_output) == rho(Y, input) || rho(Z, marginal_output) == rho(Z, input)
        error("No change in rho")
    end

    if isapprox(rho(X, marginal_output), rho(X, input); atol=1e-8) || isapprox(rho(Y, marginal_output), rho(Y, input); atol=1e-8) || isapprox(rho(Z, marginal_output), rho(Z, input); atol=1e-8)
         error("Too little change in rho")
    end
    
    # check if rho is zero
    if new_msg_to_x.rho == 0.0 || new_msg_to_y.rho == 0.0 || new_msg_to_z.rho == 0.0
        error("Zero rho in updated messages to variables")
    end

    if debug
        msg_to_output = [
            GaussianDistribution.mean(new_msg_to_x), GaussianDistribution.variance(new_msg_to_x), GaussianDistribution.mean(new_msg_to_y), GaussianDistribution.variance(new_msg_to_y), GaussianDistribution.mean(new_msg_to_z), GaussianDistribution.variance(new_msg_to_z)]
        @assert all(isfinite, msg_to_output) "NaN or Inf in updated messages to variables"
        return (marginal_output, msg_to_output), early_stop, converged
    else
        return marginal_output, early_stop, converged
    end
end

"""
generate a dataset of n samples, each sample consisting of 9 Floats for the inputs and 6 Floats for the outputs. save dataset as .jdl2 and return file path and variable key for loading
! make sure to avoid high variance in input distributions, low variances in input distributions, large differences between true means and input means, large dirac_std, low number of samples per input
! especially if several of these are combined. for example, combining the first three will quickly lead to the samples variances being larger than the input variances, 
! which will lead to negative rho values after division (impossible, will throw an error)
"""
function generate_dataset_weighted_sum_factor(;
    n::Int,
    variable_mean_dist::Distribution=Uniform(-100, 100),
    variable_std_dist::Distribution=Uniform(0.32, 6.0),
    factor_dist::Distribution=Uniform(-10, 10),
    bias_dist::Distribution=Uniform(-20, 20),
    samples_per_input::Int=1_000_000,
    dirac_std::Float64=1e-1,
    patience=0.1,
    log_weighting=true,
    save_dataset=true,
    variance_relative_epsilon=1e-10,
    std_magnitude_factor=3.0,
    algorithm=:adaptive_metropolis_hastings,
    implied_z_max_error=15.0,
    uniform_quota=0.1,
    set_large_variances_to_uniform=false,
    name_appendix="",
    savepath="",
    strict_convergence=false,
)
    nstring = replace(format(n, commas=true), "," => "_") * name_appendix
    savepath = joinpath(savepath, "dataset_weighted_sum_factor_" * nstring * ".jld2")

    if isfile(savepath) && save_dataset
        println("File already exists: $savepath. Skipping creation...")
       return savepath
    end

    if dirname(savepath) != "" && !isdir(dirname(savepath)) && save_dataset
        mkpath(dirname(savepath))
    end

    println("Number of threads: ", Threads.nthreads())

    if !strict_convergence
        dataset = Vector{Tuple{Vector{Float32}, Vector{Float32}}}(undef, n)
        attempts = Threads.Atomic{Int}(0)
        early_stop_counts = Threads.Atomic{Int}(0)
        converged_counts = Threads.Atomic{Int}(0)
        progress = ProgressMeter.Progress(n,  desc="Generating Dataset for $n samples and $samples_per_input using non-strict approach")

        Threads.@threads for i in 1:n
            while true
                try
                    inputs = generate_input_weighted_sum_factor(
                        variable_mean_dist=variable_mean_dist,
                        variable_std_dist=variable_std_dist,
                        factor_dist=factor_dist,
                        bias_dist=bias_dist,
                        std_magnitude_factor=std_magnitude_factor,
                        implied_z_max_error=implied_z_max_error,
                        uniform_quota=uniform_quota
                    )
                
                    outputs, early_stop, converged = generate_output_weighted_sum_factor(inputs;
                        samples_per_input=samples_per_input,
                        dirac_std=dirac_std,
                        log_weighting=log_weighting,
                        variance_relative_epsilon=variance_relative_epsilon,
                        algorithm=algorithm,
                        set_large_variances_to_uniform=set_large_variances_to_uniform,
                    )
                    dataset[i] = (Float32.(inputs), Float32.(outputs))
                    if early_stop
                        Threads.atomic_add!(early_stop_counts, 1)
                    end
                    if converged
                        Threads.atomic_add!(converged_counts, 1)
                    end
                    break
                catch e
                    #println("Error during sampling: $e. Retrying...")
                end
            end
            ProgressMeter.next!(progress)
        end

        early_stop_counts = early_stop_counts[]
        converged_counts = converged_counts[]
        println("Early stops: $early_stop_counts / $n --> $(early_stop_counts / n)%")
        println("Converged: $converged_counts / $n --> $(converged_counts / n)%")
    else
        # Thread-safe dataset container and counters
        dataset = Vector{Tuple{Vector{Float32}, Vector{Float32}}}(undef, n)
        accepted = Threads.Atomic{Int}(0)
        early_stop_counts = Threads.Atomic{Int}(0)
        tries = Threads.Atomic{Int}(0)

        # Thread-safe progress bar
        progress = Progress(n, desc="Generating Dataset for $n samples using strict approach")
        datalock = ReentrantLock()

        # Threaded processing
        while accepted[] < n
            @threads for _ in 1:Threads.nthreads()
                # Generate input
                inputs = generate_input_weighted_sum_factor(
                    variable_mean_dist=variable_mean_dist,
                    variable_std_dist=variable_std_dist,
                    factor_dist=factor_dist,
                    bias_dist=bias_dist,
                    std_magnitude_factor=std_magnitude_factor,
                    implied_z_max_error=implied_z_max_error,
                    uniform_quota=uniform_quota
                )

                # Generate output
                outputs, early_stop, converged = generate_output_weighted_sum_factor(inputs;
                    samples_per_input=samples_per_input,
                    dirac_std=dirac_std,
                    log_weighting=log_weighting,
                    variance_relative_epsilon=variance_relative_epsilon,
                    algorithm=algorithm,
                    set_large_variances_to_uniform=set_large_variances_to_uniform,
                )

                if converged
                    lock(datalock)
                    # Atomically reserve a slot in the dataset
                    Threads.atomic_add!(accepted, 1)
                    dataset[accepted[]] = (Float32.(inputs), Float32.(outputs))
                    if early_stop
                        Threads.atomic_add!(early_stop_counts, 1)
                    end
                    ProgressMeter.next!(progress)
                    unlock(datalock)
                end

                # Increment the total number of tries
                Threads.atomic_add!(tries, 1)
            end
        end

        early_stop_counts = early_stop_counts[]
        tries = tries[]
        converged_counts = n
        println("Early stops: $early_stop_counts / $n --> $(early_stop_counts / n)%")
        println("Converged after $tries tries for n=$n")

    end

    samples = [d[1] for d in dataset]
    targets_X = [get_variable(X, d[2]) for d in dataset]
    targets_Y = [get_variable(Y, d[2]) for d in dataset]
    targets_Z = [get_variable(Z, d[2]) for d in dataset]

    if save_dataset
        parameters = get_params_dict(; n, variable_mean_dist, variable_std_dist, factor_dist, bias_dist, samples_per_input, dirac_std, patience, log_weighting, save_dataset, std_magnitude_factor, variance_relative_epsilon, algorithm, implied_z_max_error, uniform_quota, set_large_variances_to_uniform, name_appendix)
        jldsave(savepath, samples=samples, targets_X=targets_X, targets_Y=targets_Y, targets_Z=targets_Z, parameters=parameters, early_stop_counts=early_stop_counts, converged_counts=converged_counts)
        return savepath
    else
        return dataset
    end
end

function generate_dataset_weighted_sum_factor_TTT_logged(;
    n=1_000,
    path="SoHEstimation/approximate_message_passing/data/log_msgfrom_WSF_TTT_3.jld2",
    samples_per_input::Int=1_000_000,
    dirac_std::Float64=1e-1,
    patience=0.1,
    log_weighting=true,
    variance_relative_epsilon=1e-10,
    algorithm=:adaptive_metropolis_hastings,
    set_large_variances_to_uniform=false,
    save_dataset=true,
    savepath="SoHEstimation/approximate_message_passing/weighted_sum_factor/data/",
    name_appendix="",
    strict_convergence=false,
)

    nstring = replace(format(n, commas=true), "," => "_") 

    savepath = joinpath(savepath, "dataset_weighted_sum_factor_" * nstring * name_appendix * ".jld2")

    if isfile(savepath) && save_dataset
        error("File already exists: $savepath")
    end

    if dirname(savepath) != "" && !isdir(dirname(savepath)) && save_dataset
        mkpath(dirname(savepath))
    end
    
    println("Number of threads: ", Threads.nthreads())
    println("Loading file... (This may take a while)")

    log = load(path)

    msgfromx_rhos = log["msgfromx_rhos"]
    msgfromy_rhos = log["msgfromy_rhos"]
    msgfromz_rhos = log["msgfromz_rhos"]

    msgfromx_taus = log["msgfromx_taus"]
    msgfromy_taus = log["msgfromy_taus"]
    msgfromz_taus = log["msgfromz_taus"]

    update_log = log["update_log"]

    outputs_x = []
    outputs_y = []
    outputs_z = []

    inputs_x = []
    inputs_y = []
    inputs_z = []

    attempts = 0

    indices_x = sample(findall(s -> s == :x, update_log), n, replace=false)
    indices_y = sample(findall(s -> s == :y, update_log), n, replace=false)
    indices_z = sample(findall(s -> s == :z, update_log), n, replace=false)

    for variable in instances(Variable)
        if variable == X
            indices = indices_x
            inputs = inputs_x
            outputs = outputs_x
        elseif variable == Y
            indices = indices_y
            inputs = inputs_y
            outputs = outputs_y
        elseif variable == Z
            indices = indices_z
            inputs = inputs_z
            outputs = outputs_z
        end

        attempts = Threads.Atomic{Int}(0)
        early_stop_counts = Threads.Atomic{Int}(0)
        converged_counts = Threads.Atomic{Int}(0)

        if !strict_convergence
            progress = ProgressMeter.Progress(n,  desc="Generating Dataset for $n samples using non-strict approach")
            Threads.@threads for i in indices
                input = [
                    msgfromx_taus[i], 
                    msgfromx_rhos[i], 
                    msgfromy_taus[i], 
                    msgfromy_rhos[i], 
                    msgfromz_taus[i], 
                    msgfromz_rhos[i], 
                    1.0, -1.0, 0.0
                ]

                input = to_mean_variance(input)

                output, early_stop, converged = generate_output_weighted_sum_factor(input;
                    samples_per_input=samples_per_input,
                    dirac_std=dirac_std,
                    log_weighting=log_weighting,
                    variance_relative_epsilon=variance_relative_epsilon,
                    algorithm=algorithm,
                    set_large_variances_to_uniform=set_large_variances_to_uniform,
                )
                if early_stop
                    Threads.atomic_add!(early_stop_counts, 1)
                end
                if converged
                    Threads.atomic_add!(converged_counts, 1)
                end

                push!(inputs, input)
                push!(outputs, get_variable(variable, output))
                ProgressMeter.next!(progress)
            end

            early_stop_counts = early_stop_counts[]
            converged_counts = converged_counts[]
            println("Early stops for variable $variable: $early_stop_counts / $n --> $(early_stop_counts / n)%")
            println("Converged for variable $variable: $converged_counts / $n --> $(converged_counts / n)%")
        else
            # Thread-safe dataset container and counters
            early_stop_counts = Threads.Atomic{Int}(0)
            tries = Threads.Atomic{Int}(0)

            # Thread-safe progress bar
            progress = Progress(n, desc="Generating Dataset for $n samples using strict approach")
            Threads.@threads for i in indices
                input = [msgfromx_taus[i], msgfromx_rhos[i], msgfromy_taus[i], msgfromy_rhos[i], msgfromz_taus[i], msgfromz_rhos[i], 1.0, -1.0, 0.0]
                input = to_mean_variance(input)
                converged = false
                while !converged
                    output, early_stop, converged = generate_output_weighted_sum_factor(input;
                        samples_per_input=samples_per_input,
                        dirac_std=dirac_std,
                        log_weighting=log_weighting,
                        variance_relative_epsilon=variance_relative_epsilon,
                        algorithm=algorithm,
                        set_large_variances_to_uniform=set_large_variances_to_uniform,
                    )
                    if early_stop
                        Threads.atomic_add!(early_stop_counts, 1)
                    end
                    if !converged
                        Threads.atomic_add!(tries, 1)
                    end
                end
                Threads.atomic_add!(converged_counts, 1)
                push!(inputs, input)
                push!(outputs, get_variable(variable, output))
                ProgressMeter.next!(progress)
            end

            early_stop_counts = early_stop_counts[]
            tries = tries[]
            converged_counts = n
            println("Early stops: $early_stop_counts / $n --> $(early_stop_counts / n)%")
            println("Converged after $tries tries for n=$n")
        end
    end

    if save_dataset
        parameters = get_params_dict(; samples_per_input, dirac_std, patience, log_weighting, save_dataset, variance_relative_epsilon, algorithm, set_large_variances_to_uniform, name_appendix)
        jldsave(savepath, inputs_X=inputs_x, inputs_Y=inputs_y, inputs_Z=inputs_z, targets_X=outputs_x, targets_Y=outputs_y, targets_Z=outputs_z, parameters=parameters)
        return savepath
    else
        return (inputs_x, outputs_x), (inputs_y, outputs_y), (inputs_z, outputs_z), savepath, get_params_dict(; samples_per_input, dirac_std, patience, log_weighting, save_dataset, variance_relative_epsilon, algorithm, set_large_variances_to_uniform, name_appendix)
    end

end

function calc_msg_x(input::AbstractVector, transform_to_tau_rho)
    if !transform_to_tau_rho
        msgBackY = Gaussian1DFromMeanVariance(get_variable(Y, input)...)
        msgBackZ = Gaussian1DFromMeanVariance(get_variable(Z, input)...)
        msgBackX = Gaussian1DFromMeanVariance(get_variable(X, input)...)
    else
        msgBackY = Gaussian1D(get_variable(Y, input)...) #f.db[f.y] / f.db[f.msg_to_y]
        msgBackZ = Gaussian1D(get_variable(Z, input)...) #f.db[f.z] / f.db[f.msg_to_z]
        msgBackX = Gaussian1D(get_variable(X, input)...) #f.db[f.x] / f.db[f.msg_to_x]
    end
    a_value = a(input)
    b_value = b(input)
    c_value = c(input)
    
    newMsgX = Gaussian1DFromMeanVariance(
        GaussianDistribution.mean(msgBackZ) / a_value - b_value / a_value * GaussianDistribution.mean(msgBackY) - c_value / a_value,
        GaussianDistribution.variance(msgBackZ) / (a_value * a_value) + b_value * b_value / (a_value * a_value) * GaussianDistribution.variance(msgBackY),
    )
    
    newMarginal = msgBackX * newMsgX
    return newMarginal, newMsgX
end
    
function calc_msg_y(input::AbstractVector, transform_to_tau_rho)
    if !transform_to_tau_rho
        msgBackY = Gaussian1DFromMeanVariance(get_variable(Y, input)...)
        msgBackZ = Gaussian1DFromMeanVariance(get_variable(Z, input)...)
        msgBackX = Gaussian1DFromMeanVariance(get_variable(X, input)...)
    else
        msgBackY = Gaussian1D(get_variable(Y, input)...) #f.db[f.y] / f.db[f.msg_to_y]
        msgBackZ = Gaussian1D(get_variable(Z, input)...) #f.db[f.z] / f.db[f.msg_to_z]
        msgBackX = Gaussian1D(get_variable(X, input)...) #f.db[f.x] / f.db[f.msg_to_x]
    end

    a_value = a(input)
    b_value = b(input)
    c_value = c(input)
    
    newMsgY = Gaussian1DFromMeanVariance(
        GaussianDistribution.mean(msgBackZ) / b_value - a_value / b_value * GaussianDistribution.mean(msgBackX) - c_value  / b_value,
        GaussianDistribution.variance(msgBackZ) / (b_value * b_value) + a_value * a_value / (b_value * b_value) * GaussianDistribution.variance(msgBackX),
    )
    
    newMarginal = msgBackY * newMsgY
    return newMarginal, newMsgY
end
    
function calc_msg_z(input::AbstractVector, transform_to_tau_rho)
    if !transform_to_tau_rho
        msgBackY = Gaussian1DFromMeanVariance(get_variable(Y, input)...)
        msgBackZ = Gaussian1DFromMeanVariance(get_variable(Z, input)...)
        msgBackX = Gaussian1DFromMeanVariance(get_variable(X, input)...)
    else
        msgBackY = Gaussian1D(get_variable(Y, input)...) #f.db[f.y] / f.db[f.msg_to_y]
        msgBackZ = Gaussian1D(get_variable(Z, input)...) #f.db[f.z] / f.db[f.msg_to_z]
        msgBackX = Gaussian1D(get_variable(X, input)...) #f.db[f.x] / f.db[f.msg_to_x]
    end
    
    a_value = a(input)
    b_value = b(input)
    c_value = c(input)
    
    newMsgZ = Gaussian1DFromMeanVariance(
        a_value * GaussianDistribution.mean(msgBackX) + b_value *  GaussianDistribution.mean(msgBackY) + c_value,
        a_value * a_value * GaussianDistribution.variance(msgBackX) + b_value * b_value * GaussianDistribution.variance(msgBackY),
    )
    
    newMarginal = msgBackZ * newMsgZ
    return newMarginal, newMsgZ
end


end
