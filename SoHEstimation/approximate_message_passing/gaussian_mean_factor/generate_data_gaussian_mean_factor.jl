module GaussianMeanFactorGeneration
export generate_dataset_gaussian_mean_factor, X, Y, dimension, mean, variance, remove_variable, get_variable, generate_output_gaussian_mean_factor


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
using Base.Threads
using Random


function get_params_dict(; kwargs...)
    return Dict(kwargs)
end

@enum Variable X Y

function dimension(v::Variable, moment::Int=1)
    @assert moment == 1 || moment == 2
    return 2 * Int(v) + moment
end

"""
Assumes location-scale parameters in input vector.
"""
function mean(v::Variable, x::AbstractVector)
    @assert length(x) == 5 || length(x) == 4
    return x[dimension(v, 1)]
end

"""
Assumes location-scale parameters in input vector.
"""
function variance(v::Variable, x::AbstractVector)
    @assert length(x) == 5 || length(x) == 4
    return x[dimension(v, 2)]
end

"""
Assumes location-scale parameters in input vector.
"""
function set_mean!(variable::Variable, x::AbstractVector, value::Number)
    @assert length(x) == 5 || length(x) == 4
    x[dimension(variable, 1)] = value
end

"""
Assumes location-scale parameters in input vector.
"""
function set_variance!(variable::Variable, x::AbstractVector, value::Number)
    @assert length(x) == 5 || length(x) == 4
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
    @assert length(x) == 5 || length(x) == 4
    return vcat(x[1:dimension(v, 1)-1], x[dimension(v, 2)+1:end])
end

"""
Assumes location-scale parameters in input vector.
"""
function get_variable(v::Variable, x::AbstractVector)
    @assert length(x) == 5 || length(x) == 4
    return [x[dimension(v, 1)], x[dimension(v, 2)]]
end

function β2(x::AbstractVector)::Float64
    @assert length(x) == 5
    return x[5]
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
    for v in [X, Y]
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
        return 0.0, 0.0
    end
    return mean / variance, 1 / variance
end

"""
Assumes location-scale parameters in input vector.
"""
function to_tau_rho(x::AbstractVector)
    @assert length(x) == 5 || length(x) == 4 || length(x) == 2
    if length(x) == 5
        return [tau(X, x), rho(X, x), tau(Y, x), rho(Y, x), β2(x)]
    elseif length(x) == 2
        return [to_tau_rho(x...)...]
    else
        return [tau(X, x), rho(X, x), tau(Y, x), rho(Y, x)]
    end
end

"""
Assumes location-scale parameters in input vector.
"""
function to_tau_rho(x::Matrix{T}) where T<:AbstractFloat
    @assert size(x, 2) == 5 || size(x, 2) == 4 || size(x, 2) == 2
    return hcat(map(to_tau_rho, eachrow(x))...)' |> Matrix{Float32}
end

"""
Assumes natural parameters in input vector.
"""
function to_mean_variance(tau, rho)
    if rho == 0.0
        return 0.0, Inf
    end
    return tau / rho, 1 / rho
end

"""
Assumes natural parameters in input vector.
"""
function to_mean_variance(x::AbstractVector)
    @assert length(x) == 5 || length(x) == 4 || length(x) == 2
    # tau = mean / variance
    # rho = 1 / variance
    if length(x) == 2
        return [to_mean_variance(x...)...]
    end

    xy = [
        to_mean_variance(x[1:2]...)...,
        to_mean_variance(x[4:3]...)...,
    ]

    if length(x) == 5
        return vcat(xy, [β2(x)])
    end

    return xy
end

"""
Assumes natural parameters in input vector.
"""
function to_mean_variance(x::Matrix{T}) where T<:AbstractFloat
    @assert size(x, 2) == 5 || size(x, 2) == 4 || size(x, 2) == 2
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

similar_maginitude(x::Float64, y::Float64; factor=3.0) = x * factor > y && y * factor > x

"""
Assumes location-scale parameters in input vector.
"""
function variance_magnitudes_match(input::Vector{Float64}; factor=3.0)
    x_term_variance = variance(X, input)
    y_term_variance = variance(Y, input)

    return similar_maginitude(x_term_variance, y_term_variance; factor=factor)
end

"""
Assumes location-scale parameters in input vector.
"""
function std_magnitudes_match(input::Vector{Float64}; factor=3.0)
    x_term_std = std(X, input)
    y_term_std = std(Y, input)

    return similar_maginitude(x_term_std, y_term_std; factor=factor)
end

"""
Assumes location-scale parameters in input vector.
Checks if one of the variances in input is larger than the other two by a certain factor.
If so, sets the variable with the large variance to a uniform distribution in a new input vector and resturns it.
"""
function large_variance_to_uniform(input::Vector{Float64}; factor=50.0)
    input_vector = deepcopy(input)
    for v in [X, Y]
        if variance(v, input) > factor * sort([variance(X, input), variance(Y, input)])[2]
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
Assumes location-scale parameters in input vector.
Returns true if variance of output is larger than input variance by more than a certain factor.
"""
function exploding_variance(input::Vector{Float64}, output::Vector{Float64}; factor=10.0)
    for v in [X, Y]
        if !similar_maginitude(variance(v, output), variance(v, input); factor=factor)
            return true
        end
    end

    return false
end

function eval(x::Float64, y::Float64, β2::Float64, log_weighting::Bool=true)
    result = log_weighting ? logpdf(Normal(x, sqrt(β2)), y) : pdf(Normal(x, sqrt(β2)), y)
    return result
end

function eval(x::Vector{Float64}, y::Vector{Float64}, β2::Float64, log_weighting::Bool=true)
    # Ensure inputs are broadcasted if they are vectors
    if log_weighting
        return logpdf.(Normal.(x, sqrt.(β2)), y)
    else
        return pdf.(Normal.(x, sqrt.(β2)), y)
    end
end


"""
Assumes location-scale parameters in input vector.
"""
function sample_uniform_update(input, samples_per_input)
    @assert has_uniform(input) "No uniform variable in input"

    if is_uniform(X, input)
        samples_y = rand(Normal(mean(Y, input), std(Y, input)), samples_per_input)
        samples_x = deepcopy(samples_y)

        return [StatsBase.mean(samples_x), StatsBase.var(samples_x), mean(Y, input), variance(Y, input)]

    elseif is_uniform(Y, input)
        samples_x = rand(Normal(mean(X, input), std(X, input)), samples_per_input)
        samples_y = deepcopy(samples_x)

        return [mean(X, input), variance(X, input), StatsBase.mean(samples_y), StatsBase.var(samples_y)]

    end

    return [StatsBase.mean(samples_x), StatsBase.var(samples_x), StatsBase.mean(samples_y), StatsBase.var(samples_y)]
end


"""
generate a sample: old mean and var for msg_from_x, msg_from_y and msg_from_z, as well as a, b and c, so 9 Floats
"""
function generate_input_gaussian_mean_factor(;
    variable_mean_dist::Distribution=Uniform(-100, 100),
    variable_std_dist::Distribution=Uniform(1e-3, 8.0),
    factor_dist::Distribution=Uniform(-2, 2),
    std_magnitude_factor=3.0,
    uniform_quota=0.05,
)
    input = Vector{Float64}(undef, 5)
    trials = 0
    while trials < 500
        msg_from_x_mean = rand(variable_mean_dist)
        msg_from_y_mean = rand(variable_mean_dist)
        msg_from_x_std, msg_from_y_std = shuffle(rand(variable_std_dist))
        msg_from_x_var, msg_from_y_var = msg_from_x_std^2, msg_from_y_std^2
        β2 = rand(factor_dist)


        temp_input = [msg_from_x_mean, msg_from_x_var, msg_from_y_mean, msg_from_y_var, β2]

        if std_magnitudes_match(temp_input; factor=std_magnitude_factor)
            input = temp_input
            @assert all(isfinite, input)

            if rand() < uniform_quota
                set_uniform!(input)
            end
            return input
        end
        trials += 1
    end
    return error("Maximum trial count reached for input generation.")
end

function adaptive_metropolis_hastings(input::Vector{Float64};
    max_samples::Int=1_000_000,
    burn_in::Float64=0.5,
    variance_relative_epsilon=5e-2,
    adaptation_interval::Int=1_000,
    target_acceptance_ratio::Float64=0.23,
    min_samples::Int=100_000,
    n_next_iteration=x -> 2 * x,
)
    std_msgfrom_x, std_msgfrom_y = std(X, input), std(Y, input)
    mean_msg_from_x, mean_msg_from_y = mean(X, input), mean(Y, input)
    
    x_prior(x) = logpdf(Normal(mean_msg_from_x, std_msgfrom_x), x)
    y_prior(y) = logpdf(Normal(mean_msg_from_y, std_msgfrom_y), y)
    likelihood(x, y) = eval(x, y, β2(input), true)
    multiply(x...) = +(x...)
    divide(X...) = -(X...)
    acceptance_rate() = log(rand())

    proposal_scale = 0.1
    acceptance_history = Bool[]
    
    x_current =  rand(Normal(mean_msg_from_x, std_msgfrom_x)) # mean_msg_from_x 
    y_current = rand(Normal(mean_msg_from_y, std_msgfrom_y)) # mean_msg_from_y

    x_samples = Float64[]
    y_samples = Float64[]
    
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
    function check_convergence(x_relevant_samples, y_relevant_samples)
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
        for (i, samples) in enumerate((x_relevant_samples, y_relevant_samples))
            if !check_geweke(samples) || bfmi(samples) > 0.3 || mcse(samples) > 0.1 || mcse(samples; kind=Statistics.std) > 0.1 || !check_ess_rhat(samples)
                return false, i
            end
        end
        
        return true
    end
    
    accepted = 0
    i = 0
    
    n_iterations = min_samples
    posterior_current = multiply(x_prior(x_current), y_prior(y_current), likelihood(x_current, y_current))

    early_stop = false

    while i < max_samples
        i += 1
        
        # Propose new values using adaptive scales
        x_proposal = rand(Normal(x_current, std_msgfrom_x * proposal_scale))
        y_proposal = rand(Normal(y_current, std_msgfrom_y * proposal_scale))
        
        # Compute log probabilities
        posterior_proposal = multiply(x_prior(x_proposal), y_prior(y_proposal), likelihood(x_proposal, y_proposal))
        
        # Metropolis-Hastings acceptance step
        accepted_step = acceptance_rate() < divide(posterior_proposal, posterior_current)
        push!(acceptance_history, accepted_step)
        
        if accepted_step
            x_current, y_current = x_proposal, y_proposal
            posterior_current = posterior_proposal
            accepted += 1
        end
        
        push!(x_samples, x_current)
        push!(y_samples, y_current)
        
        # Adapt proposals periodically
        if i % adaptation_interval == 0
            proposal_scale = adapt_proposals(proposal_scale, acceptance_history)
        end
        
        # Check convergence periodically
        if i == n_iterations
            relevant_samples_x, relevant_samples_y = x_samples[Int(end*burn_in):end], y_samples[Int(end*burn_in):end]
            if check_convergence(relevant_samples_x, relevant_samples_y)[1]
                #@info "Converged after $(digitsep(i, seperator= "_")) iterations, overall acceptance rate of $(StatsBase.mean(acceptance_history[Int(end*burn_in):end]))"
                x_samples, y_samples = relevant_samples_x, relevant_samples_y
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
        StatsBase.mean(y_samples), StatsBase.var(y_samples)
    ]
    
    # Clamp variances
    for v in [X, Y]
        output[dimension(v, 2)] = min(
            output[dimension(v, 2)], 
            variance(v, input) * (1 - variance_relative_epsilon)
        )
    end

    converged = check_convergence(x_samples, y_samples)
    if i >= max_samples && !converged[1]
        #@warn "Did not converge after $(digitsep(i, seperator= "_")) iterations, overall acceptance rate of $(StatsBase.mean(acceptance_history[Int(end*burn_in):end]))" # , failed on variable $(converged[2])"
    end
    
    return output, early_stop, converged[1]
end


"""
given a sample from generate_inputs_weighted_sum_factor(), calculate mean and var for updated message to x and y so 4 Floats
"""
function generate_output_gaussian_mean_factor(input::Vector{Float64};
    samples_per_input::Int=1_000_000,
    variance_relative_epsilon=5e-2,
    log_weighting=true,
    burn_in=0.5,
    debug=false,
)
    @assert all(!isnan, [variance(v, input) for v in [X, Y]]) "NaN in input variances" # Inf allowed for uniform input variables
    @assert all(isfinite, [mean(v, input) for v in [X, Y]]) "NaN or Inf in input means"

    if has_uniform(input)
        marginal_output = sample_uniform_update(input, samples_per_input)
        converged = true
        early_stop = false
    else
        @assert all(>(0), [variance(v, input) for v in [X, Y]]) "Values <= 0 in input rhos"
        # TODO: Adapt this to input in location/scale parameters or assess if still necessary
        # @assert all(isfinite, [variance(v, input) for v in [X, Y]]) "NaN or Inf in input variances. Cannot handle that yet."
        # @assert all(isfinite, [mean(v, input) for v in [X, Y]]) "NaN or Inf in input means"

        marginal_output, early_stop, converged = adaptive_metropolis_hastings(input; max_samples=samples_per_input, burn_in=burn_in, variance_relative_epsilon=variance_relative_epsilon)
    end

    @assert all(isfinite, marginal_output) "NaN or Inf in updated marginals"

    sampled_marginal_x = GaussianDistribution.Gaussian1D(tau(X, marginal_output), rho(X, marginal_output))
    sampled_marginal_y = GaussianDistribution.Gaussian1D(tau(Y, marginal_output), rho(Y, marginal_output))

    # Precompute variances and means for Gaussian update
    new_msg_to_x = sampled_marginal_x / GaussianDistribution.Gaussian1D(tau(X, input), rho(X, input))
    new_msg_to_y = sampled_marginal_y / GaussianDistribution.Gaussian1D(tau(Y, input), rho(Y, input))

    if rho(X, marginal_output) == rho(X, input) || rho(Y, marginal_output) == rho(Y, input)
        error("No change in rho")
    end

    if isapprox(rho(X, marginal_output), rho(X, input); atol=1e-8) || isapprox(rho(Y, marginal_output), rho(Y, input); atol=1e-8)
        error("Too little change in rho")
    end
    
    # check if rho is zero
    if new_msg_to_x.rho == 0.0 || new_msg_to_y.rho == 0.0
        error("Zero rho in updated messages to variables")
    end

    msg_to_output = [GaussianDistribution.mean(new_msg_to_x), GaussianDistribution.variance(new_msg_to_x), GaussianDistribution.mean(new_msg_to_y), GaussianDistribution.variance(new_msg_to_y)]

    @assert all(isfinite, msg_to_output) "NaN or Inf in updated messages to variables"

    if debug
        return (marginal_output, msg_to_output), early_stop, converged
    else
        return marginal_output, early_stop, converged
    end
end

"""
generate a dataset of n samples, each sample consisting of 5 Floats for the inputs and 4 Floats for the outputs. save dataset as .jdl2 and return file path and variable key for loading
! make sure to avoid high variance in input distributions, low variances in input distributions, large differences between true means and input means, large dirac_std, low number of samples per input
! especially if several of these are combined. for example, combining the first three will quickly lead to the samples variances being larger than the input variances, 
! which will lead to negative rho values after division (impossible, will throw an error)
"""
function generate_dataset_gaussian_mean_factor(;
    n::Int,
    variable_mean_dist::Distribution=Uniform(-100, 100),
    variable_std_dist::Distribution=Uniform(1e-1, 3.5),
    factor_dist::Distribution=Uniform(0, 10),
    samples_per_input::Int=1_000_000,
    patience=0.1,
    log_weighting=true,
    save_dataset=true,
    std_magnitude_factor=3.0,
    name_appendix="",
    savepath="",
    strict_convergence=false,
    variance_relative_epsilon=5e-2,
)
    nstring = replace(format(n, commas=true), "," => "_") * name_appendix
    savepath = savepath * "dataset_gaussian_mean_factor_" * nstring * ".jld2"

    if isfile(savepath) && save_dataset
        error("File already exists: $savepath")
    end

    if dirname(savepath) != "" && !isdir(dirname(savepath)) && save_dataset
        mkpath(dirname(savepath))
    end

    if !strict_convergence
        dataset = Vector{Tuple{Vector{Float32},Vector{Float32}}}(undef, n)
        early_stop_counts = Threads.Atomic{Int}(0)
        converged_counter = Threads.Atomic{Int}(0)
        println("Number of threads: ", Threads.nthreads())

        # Use a thread-safe progress bar
        progress = ProgressMeter.Progress(n)

        Threads.@threads for i in 1:n
            while true
                try
                    inputs = generate_input_gaussian_mean_factor(
                        variable_mean_dist=variable_mean_dist,
                        variable_std_dist=variable_std_dist,
                        factor_dist=factor_dist,
                        std_magnitude_factor=std_magnitude_factor
                    )
                    outputs, early_stop, converged = generate_output_gaussian_mean_factor(inputs; samples_per_input, log_weighting,variance_relative_epsilon=variance_relative_epsilon)
                
                    dataset[i] = (Float32.(inputs), Float32.(outputs))
                    if early_stop
                        Threads.atomic_add!(early_stop_counts, 1)
                    end
                    if converged
                        Threads.atomic_add!(converged_counter, 1)
                    end
                    break
                catch e
                   
                end
            end
            ProgressMeter.next!(progress)
        end

        early_stop_counts = early_stop_counts[]
        converged_counter = converged_counter[]
        println("Early stops: $early_stop_counts / $n --> $(early_stop_counts / n)%")
        println("Converged: $converged_counter / $n --> $(converged_counter / n)%")
        
    else
        # Thread-safe dataset container and counters
        dataset = Vector{Tuple{Vector{Float32}, Vector{Float32}}}(undef, n)
        accepted = Threads.Atomic{Int}(0)
        early_stop_counts = Threads.Atomic{Int}(0)
        tries = Threads.Atomic{Int}(0)
        converged_counter = Threads.Atomic{Int}(0)

        # Thread-safe progress bar
        progress = Progress(n, desc="Generating Dataset for $n samples using strict approach")
        datalock = ReentrantLock()

        while accepted[] < n
            @threads for _ in 1:Threads.nthreads()
                try
                    inputs = generate_input_gaussian_mean_factor(
                        variable_mean_dist=variable_mean_dist,
                        variable_std_dist=variable_std_dist,
                        factor_dist=factor_dist,
                        std_magnitude_factor=std_magnitude_factor
                    )
                    outputs, early_stop, converged = generate_output_gaussian_mean_factor(inputs; samples_per_input, log_weighting)
                    if converged
                        lock(datalock)
                        if accepted[] < n  # Ensure we do not exceed n
                            Threads.atomic_add!(accepted, 1)
                            Threads.atomic_add!(converged_counter, 1)
                            
                            dataset[accepted[]] = (Float32.(inputs), Float32.(outputs))
        
                            if early_stop
                                Threads.atomic_add!(early_stop_counts, 1)
                            end
                            ProgressMeter.next!(progress)
                        end
                        unlock(datalock)
                    end
                    # Increment the total number of tries
                    Threads.atomic_add!(tries, 1)
                catch e
                    println("Error during sampling: $e. Retrying...")
                end
            end
        end
        early_stop_counts = early_stop_counts[]
        converged_counter = converged_counter[]
        tries = tries[]
        println("Early stops: $early_stop_counts / $n --> $(early_stop_counts / n)%")
        println("Converged in $converged_counter / $n")
        println("Converged after $tries tries for n=$n")

    end

    samples = [d[1] for d in dataset]
    targets_X = [get_variable(X, d[2]) for d in dataset]
    targets_Y = [get_variable(Y, d[2]) for d in dataset]

    if save_dataset
        parameters = get_params_dict(; n, variable_mean_dist, variable_std_dist, factor_dist, samples_per_input, patience, log_weighting, std_magnitude_factor, name_appendix)
        jldsave(savepath, samples=samples, targets_X=targets_X, targets_Y=targets_Y, parameters=parameters, early_stop_counts=early_stop_counts, converged_counter=converged_counter)
        return savepath
    else
        return dataset
    end
end

    function calc_msg_x(input::AbstractVector, transform_to_tau_rho)
        if !transform_to_tau_rho
            msgBackX = Gaussian1DFromMeanVariance(get_variable(X, input)...)
            msgBackY = Gaussian1DFromMeanVariance(get_variable(Y, input)...)
        else
            msgBackY = Gaussian1D(get_variable(Y, input)...) # msgBack = f.db[f.y] / f.db[f.msg_to_y]
            msgBackX = Gaussian1D(get_variable(X, input)...) # msgBack = f.db[f.x] / f.db[f.msg_to_x]
        end
    
        beta_squared = β2(input)
    
        c = 1 / (1 + beta_squared * msgBackY.rho)
        
        newMsg = Gaussian1D(msgBackY.tau * c, msgBackY.rho * c)
        newMarginal = newMsg * msgBackX

    return newMarginal, newMsg
  end

  function calc_msg_y(input::AbstractVector, transform_to_tau_rho)

    if !transform_to_tau_rho
        msgBackX = Gaussian1DFromMeanVariance(get_variable(X, input)...)
        msgBackY = Gaussian1DFromMeanVariance(get_variable(Y, input)...)
    else
        msgBackX = Gaussian1D(get_variable(X, input)...) # msgBack = f.db[f.x] / f.db[f.msg_to_x]
        msgBackY = Gaussian1D(get_variable(Y, input)...) # msgBack = f.db[f.y] / f.db[f.msg_to_y]
    end

    beta_squared = β2(input)

    c = 1 / (1 + beta_squared * msgBackX.rho)

    newMsg = Gaussian1D(msgBackX.tau * c, msgBackX.rho * c)
    newMarginal = newMsg * msgBackY

    return newMarginal, newMsg
    
  end
end