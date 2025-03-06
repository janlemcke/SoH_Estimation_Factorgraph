module ElectricalModelFactorGeneration
export generate_dataset_em_factor, I, DSOC, SOH, dimension, mean, variance, remove_variable, get_variable, generate_output_em_factor

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

@enum Variable I SOH DSOC

function get_params_dict(; kwargs...)
	return Dict(kwargs)
end

function dimension(v::Variable, moment::Int = 1)
	@assert moment == 1 || moment == 2
	return 2 * Int(v) + moment
end

"""
Assumes location-scale parameters in input vector.
"""
function mean(v::Variable, x::AbstractVector)
	@assert length(x) == 8 || length(x) == 6
	return x[dimension(v, 1)]
end

"""
Assumes location-scale in input vector.
"""
function variance(v::Variable, x::AbstractVector)
	@assert length(x) == 8 || length(x) == 6
	return x[dimension(v, 2)]
end

"""
Assumes location-scale in input vector.
"""
function set_mean!(variable::Variable, x::AbstractVector, value::Number)
	@assert length(x) == 8 || length(x) == 6
	x[dimension(variable, 1)] = value
end

"""
Assumes location-scale in input vector.tor.
"""
function set_variance!(variable::Variable, x::AbstractVector, value::Number)
	@assert length(x) == 8 || length(x) == 6
	x[dimension(variable, 2)] = value
end

"""
Assumes location-scale in input vector.
"""
function tau(v::Variable, x::AbstractVector)
	if variance(v, x) == Inf
		return 0
	end
	return mean(v, x) / variance(v, x)
end


"""
Assumes location-scale in input vector.
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
	@assert length(x) == 8 || length(x) == 6
	return vcat(x[1:dimension(v, 1)-1], x[dimension(v, 2)+1:end])
end

"""
Assumes natural parameters in input vector.
"""
function get_variable(v::Variable, x::AbstractVector)
	@assert length(x) == 8 || length(x) == 6
	return [x[dimension(v, 1)], x[dimension(v, 2)]]
end

function q0(x::AbstractVector)::Float64
	@assert length(x) == 8
	return x[7]
end

function dt(x::AbstractVector)::Float64
	@assert length(x) == 8
	return x[8]
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
	for v in [I, SOH, DSOC]
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
	@assert length(x) == 8 || length(x) == 6 || length(x) == 2
	if length(x) == 8
		return [tau(I, x), rho(I, x), tau(SOH, x), rho(SOH, x), tau(DSOC, x), rho(DSOC, x), qo(x), dt(x)]
	elseif length(x) == 2
		return [to_tau_rho(x...)...]
	else
		return [tau(I, x), rho(I, x), tau(SOH, x), rho(SOH, x), tau(DSOC, x), rho(DSOC, x)]
	end
end

"""
Assumes location-scale parameters in input vector.
"""
function to_tau_rho(x::Matrix{T}) where T <: AbstractFloat
	@assert size(x, 2) == 8 || size(x, 2) == 6 || size(x, 2) == 2
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
function to_mean_variance(x::Matrix{T}) where T <: AbstractFloat
	@assert size(x, 2) == 8 || size(x, 2) == 6 || size(x, 2) == 2
	return hcat(map(to_mean_variance, eachrow(x))...)' |> Matrix{Float32}
end


function log_weighted_mean(x::Vector{Float64}, log_weights::Vector{Float64})
	# make sure all weights are negative. does not distort the result, but makes the calculation more stable
	adjusted_log_weights = any(log_weights .> 0) ? log_weights .- (maximum(log_weights) + 1) : log_weights

	# makes sure all values are positive. does not distort the result, but makes the calculation more stable
	shift = abs(minimum(x)) + 1

	return exp(LogExpFunctions.logsumexp(log.(x .+ shift) .+ adjusted_log_weights) - LogExpFunctions.logsumexp(adjusted_log_weights)) - shift
end


similar_maginitude(x::Float64, y::Float64; factor = 3.0) = x * factor > y && y * factor > x


"""
Assumes natural parameters in input vector.

function std_magnitudes_match(input::Vector{Float64}; factor = 3.0)
	I_term_std = std(I, input) * abs(dt(input))
	SOH_term_std = std(SOH, input) * abs(q0(input))
	DOSC_term_std = std(DSOC, input)

	return similar_maginitude(I_term_std, SOH_term_std; factor = factor) &&
		   similar_maginitude(I_term_std, DOSC_term_std; factor = factor) &&
		   similar_maginitude(SOH_term_std, DOSC_term_std; factor = factor)
end
"""

function std_magnitudes_match(input::Vector{Float64}; factor = 3.0)
    I_std = std(I, input)
    SOH_std = std(SOH, input)
	#SOH_std_log = log(std(SOH, input) + 1e-3)  # Log-transform
    q0_value = q0(input)  # Since q0 is fixed, get its value once

    # Compute standard deviation contributions
    I_term_std = (I_std / q0_value) * abs(dt(input) / mean(SOH, input))
    SOH_term_std = (SOH_std / q0_value) * abs(mean(I, input) * dt(input) / mean(SOH, input)^2) #exp(log(mean(SOH, input) + 1e-3))) #mean(SOH, input)^2)

    total_term_std = I_term_std + SOH_term_std
    DOSC_term_std = std(DSOC, input)

	#println("DSOC_term_std: ", DOSC_term_std, " total_term_std: ", total_term_std, " I_term_std: ", I_term_std, " SOH_term_std: ", SOH_term_std)

    return similar_maginitude(total_term_std, DOSC_term_std; factor = factor)
end


"""
Assumes natural parameters in input vector.
Checks if one of the variances in input is larger than the other two by a certain factor.
If so, sets the variable with the large variance to a uniform distribution in a new input vector and resturns it.
"""
function large_variance_to_uniform(input::Vector{Float64}; factor = 50.0)
	input_vector = deepcopy(input)
	for v in [I, SOH, DSOC]
		if variance(v, input) > factor * sort([variance(I, input), variance(SOH, input), variance(DSOC, input)])[2]
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
Assumes natural parameters in input vector.
Returns true if variance of output is larger than input variance by more than a certain factor.
"""
function exploding_variance(input::Vector{Float64}, output::Vector{Float64}; factor = 10.0)
	for v in [I, SOH, DSOC]
		if !similar_maginitude(variance(v, output), variance(v, input); factor = factor)
			return true
		end
	end

	return false
end

function eval(i::Float64, soh::Float64, dsoc::Float64, dirac_std::Float64 = 1e-1; q0::Float64 = 66.0, dt::Float64 = 1.0, log_weighting = true)
	result = log_weighting ? logpdf(Normal(0.0, dirac_std), dsoc - ((-i * dt) / (q0 * soh))) : pdf(Normal(0.0, dirac_std), dsoc - ((-i * dt) / (q0 * soh)))
	return result
end

function eval(i::Vector{Float64}, soh::Vector{Float64}, dsoc::Vector{Float64}, dirac_std::Float64 = 1e-1;
	q0::Float64 = 66.0, dt::Float64 = 1.0, log_weighting = true)

	diff = dsoc .- ((-i .* dt) ./ (q0 .* soh)) # Element-wise operation
	if log_weighting
		return logpdf.(Normal(0.0, dirac_std), diff)  # Broadcasted logpdf
	else
		return pdf.(Normal(0.0, dirac_std), diff)     # Broadcasted pdf
	end
end


function sample_uniform_update(input, samples_per_input, variance_relative_epsilon)
	@assert has_uniform(input) "No uniform variable in input"

	if is_uniform(I, input)
		samples_soh = rand(Truncated(Normal(mean(SOH, input), std(SOH, input)), 0, 1), samples_per_input)
		samples_dosc = rand(Truncated(Normal(mean(DSOC, input), std(DSOC, input)), -1, 1), samples_per_input)

		samples_i = -(samples_dosc .* q0(input) .* samples_soh ./ dt(input))
		mean_i, var_i = StatsBase.mean(samples_i), StatsBase.var(samples_i)

		output = [mean_i, var_i, mean(SOH, input), variance(SOH, input), mean(DSOC, input), variance(DSOC, input)]

	elseif is_uniform(SOH, input)
		samples_i = rand(Normal(mean(I, input), std(I, input)), samples_per_input)
		samples_dosc = rand(Truncated(Normal(mean(DSOC, input), std(DSOC, input)), -1, 1), samples_per_input)

		samples_soh = -(samples_i .* dt(input) ./ (q0(input) .* samples_dosc))
		mean_soh, var_soh = StatsBase.mean(samples_soh), StatsBase.var(samples_soh)

		output = [mean(I, input), variance(I, input), mean_soh, var_soh, mean(DSOC, input), variance(DSOC, input)]

	elseif is_uniform(DSOC, input)
		samples_i = rand(Normal(mean(I, input), std(I, input)), samples_per_input)
		samples_soh = rand(Truncated(Normal(mean(SOH, input), std(SOH, input)), 0, 1), samples_per_input)

		samples_dsoc = -(samples_i .* dt(input) ./ (q0(input) .* samples_soh))
		mean_dsoc, var_dsoc = StatsBase.mean(samples_dsoc), StatsBase.var(samples_dsoc)

		output = [mean(I, input), variance(I, input), mean(SOH, input), variance(SOH, input), mean_dsoc, var_dsoc]
	end

	if abs(mean(I, input)) > 15
		error("Current I is out of range.")
	end

	if mean(SOH, input) > 1 || mean(SOH, input) < 0
		error("SoH is not in range.")
	end

	if abs(mean(DSOC, input)) > 1
		error("DSOC is out of range.")
	end

	# Clamp variances
	for v in [I, SOH, DSOC]
		output[dimension(v, 2)] = min(
			output[dimension(v, 2)],
			variance(v, input) * (1 - variance_relative_epsilon),
		)
	end

	return output
end


function adaptive_metropolis_hastings(input::Vector{Float64};
	dirac_std::Float64 = 1e-1,
	max_samples::Int = 1_000_000,
	burn_in::Float64 = 0.5,
	variance_relative_epsilon = 1e-10,
	adaptation_interval::Int = 1_000,
	target_acceptance_ratio::Float64 = 0.23,
	min_samples::Int = 100_000,
	n_next_iteration = x -> 2 * x,
)
	std_msgfrom_i, std_msgfrom_soh, std_msgfrom_dsoc = std(I, input), std(SOH, input), std(DSOC, input)
	mean_msg_from_i, mean_msg_from_soh, mean_msg_from_dsoc = mean(I, input), mean(SOH, input), mean(DSOC, input)

	i_prior(i) = logpdf(Normal(mean_msg_from_i, std_msgfrom_i), i)
	soh_prior(soh) = logpdf(Normal(mean_msg_from_soh, std_msgfrom_soh), soh)
	dsoc_prior(dsoc) = logpdf(Normal(mean_msg_from_dsoc, std_msgfrom_dsoc), dsoc)
	likelihood(i, soh, dsoc) = eval(i, soh, dsoc, dirac_std; q0 = q0(input), dt = dt(input), log_weighting = true)
	multiply(x...) = +(x...)
	divide(X...) = -(X...)
	acceptance_rate() = log(rand())

	proposal_scale = 0.1
	acceptance_history = Bool[]

	i_current = rand(Normal(mean_msg_from_i, std_msgfrom_i)) # mean_msg_from_x
	soh_current = rand(Truncated(Normal(mean_msg_from_soh, std_msgfrom_soh), 0, 1)) # mean_msg_from_y
	dsoc_current = rand(Truncated(Normal(mean_msg_from_dsoc, std_msgfrom_dsoc), -1, 1)) # mean_msg_from_z

	i_samples = Float64[]
	soh_samples = Float64[]
	dsoc_samples = Float64[]

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
			if !check_geweke(samples) || bfmi(samples) > 0.3 || mcse(samples) > 0.1 || mcse(samples; kind = Statistics.std) > 0.1 || !check_ess_rhat(samples)
				return false, i
			end
		end

		return true
	end

	accepted = 0
	i = 0
	early_stop = false

	n_iterations = min_samples
	posterior_current = multiply(i_prior(i_current), soh_prior(soh_current),
		dsoc_prior(dsoc_current), likelihood(i_current, soh_current, dsoc_current))

	while i < max_samples
		i += 1

		# Propose new values using adaptive scales
		i_proposal = rand(Normal(i_current, std_msgfrom_i * proposal_scale))
		soh_proposal = rand(Truncated(Normal(soh_current, std_msgfrom_soh * proposal_scale), 0, 1))
		dsoc_proposal = rand(Truncated(Normal(dsoc_current, std_msgfrom_dsoc * proposal_scale), -1, 1))

		# Compute log probabilities
		posterior_proposal = multiply(i_prior(i_proposal), soh_prior(soh_proposal),
			dsoc_prior(dsoc_proposal), likelihood(i_proposal, soh_proposal, dsoc_proposal))

		# Metropolis-Hastings acceptance step
		accepted_step = acceptance_rate() < divide(posterior_proposal, posterior_current)
		push!(acceptance_history, accepted_step)

		if accepted_step
			i_current, soh_current, dsoc_current = i_proposal, soh_proposal, dsoc_proposal
			posterior_current = posterior_proposal
			accepted += 1
		end

		if soh_current < 0 || soh_current > 1
			error("SoH has wrong value $soh_current")
		end

		if abs(dsoc_current) > 1
			error("DSoC has wrong value $dsoc_current")
		end

		push!(i_samples, i_current)
		push!(soh_samples, soh_current)
		push!(dsoc_samples, dsoc_current)

		# Adapt proposals periodically
		if i % adaptation_interval == 0
			proposal_scale = adapt_proposals(proposal_scale, acceptance_history)
		end

		# Check convergence periodically
		if i == n_iterations
			relevant_samples_i, relevant_samples_soh, relevant_samples_dsoc = i_samples[Int(end * burn_in):end], soh_samples[Int(end * burn_in):end], dsoc_samples[Int(end * burn_in):end]
			if check_convergence(relevant_samples_i, relevant_samples_soh, relevant_samples_dsoc)[1]
				#@info "Converged after $(digitsep(i, seperator= "_")) iterations, overall acceptance rate of $(StatsBase.mean(acceptance_history[Int(end*burn_in):end]))"
				i_samples, soh_samples, dsoc_samples = relevant_samples_i, relevant_samples_soh, relevant_samples_dsoc
				early_stop = true
				break
			else
				n_iterations = n_next_iteration(length(i_samples))
			end
		end
	end

	output = [
		StatsBase.mean(i_samples), StatsBase.var(i_samples),
		StatsBase.mean(soh_samples), StatsBase.var(soh_samples),
		StatsBase.mean(dsoc_samples), StatsBase.var(dsoc_samples),
	]

	# Clamp variances
	for v in [I, SOH, DSOC]
		output[dimension(v, 2)] = min(
			output[dimension(v, 2)],
			variance(v, input) * (1 - variance_relative_epsilon),
		)
	end

	converged = check_convergence(i_samples, soh_samples, dsoc_samples)
	if i >= max_samples && !converged[1]
		#@warn "Did not converge after $(digitsep(i, seperator= "_")) iterations, overall acceptance rate of $(StatsBase.mean(acceptance_history[Int(end*burn_in):end]))" # , failed on variable $(converged[2])"
	end

	return output, early_stop, converged[1]
end

"""
generate a sample: old mean and var for msg_from_x, msg_from_y and msg_from_z, as well as a, b and c, so 9 Floats
"""
function generate_input_em_factor(;
	I_mean_dist::Distribution = Uniform(-100, 100),
	I_std_dist::Distribution = Uniform(1e-3, 8.0),
	SOH_mean_dist::Distribution = Uniform(0.1, 1.0),
	SOH_std_dist::Distribution = Uniform(1e-3, 0.5),
	DSOC_mean_dist::Distribution = Uniform(0.1, 1.0),
	DSOC_std_dist::Distribution = Uniform(1e-3, 0.5),
	qo_dist::Distribution = Uniform(30, 120.0),
	dt_dist::Distribution = Uniform(0.1, 10.0),
	std_magnitude_factor = 3.0,
	implied_z_max_error = 15.0,
	uniform_quota = 0.05,
)
	@assert isnothing(implied_z_max_error) || implied_z_max_error > 0 "implied_z_max_error must be positive or nothing"
	input = Vector{Float64}(undef, 8)
	trials = 0
	while trials < 500
		
		msg_from_i_mean = rand(I_mean_dist)
		msg_from_i_var = rand(I_std_dist)^2
		msg_from_soh_mean = rand(SOH_mean_dist)
		msg_from_soh_var = rand(SOH_std_dist)^2
		msg_from_dsoc_var = rand(DSOC_std_dist)^2
		q0 = rand(qo_dist)
		dt = rand(dt_dist)

		dsoc_implied_mean = -(msg_from_i_mean * dt) / (q0 * msg_from_soh_mean)

		#I_std = sqrt(msg_from_i_var)
		#SOH_std = sqrt(msg_from_soh_var)

		# Compute standard deviation contributions
		#I_term_std = (I_std / q0) * abs(dt / msg_from_soh_mean)
		#SOH_term_std = (SOH_std / q0) * abs(msg_from_i_mean * dt / msg_from_soh_mean^2)

		#total_term_std = I_term_std + SOH_term_std
		#msg_from_dsoc_var = total_term_std^2

		if (abs(dsoc_implied_mean) > 1)
			continue
		end

		if !isnothing(implied_z_max_error)
			lower_bound = max(dsoc_implied_mean - implied_z_max_error, -1)
			upper_bound = min(dsoc_implied_mean + implied_z_max_error, 1)
			msg_from_dsoc_mean_dist = Truncated(Normal(dsoc_implied_mean, implied_z_max_error * 2 / 3), lower_bound, upper_bound)
			msg_from_dsoc_mean = rand(msg_from_dsoc_mean_dist)
		else
			msg_from_dsoc_mean = rand(DSOC_mean_dist)
		end

		temp_input = [msg_from_i_mean, msg_from_i_var, msg_from_soh_mean, msg_from_soh_var, msg_from_dsoc_mean, msg_from_dsoc_var, q0, dt]

		#if std_magnitudes_match(temp_input; factor = std_magnitude_factor)
		#	input = temp_input

		#	@assert all(isfinite, input)

		# 	if rand() < uniform_quota
		# 		set_uniform!(input)
		# 	end

		# 	return input
		#end
		input = temp_input

		@assert all(isfinite, input)

		if rand() < uniform_quota
			set_uniform!(input)
		end

		return input
		trials += 1
	end
	return error("Maximum trial count reached for input generation.")
end


"""
given a sample from generate_inputs_weighted_sum_factor(), calculate mean and var for updated message to x and y so 4 Floats
"""
function generate_output_em_factor(input::Vector{Float64};
	samples_per_input::Int = 1_000_000,
	dirac_std::Float64 = 1e-1,
	log_weighting = true,
	variance_relative_epsilon = 1e-10,
	set_large_variances_to_uniform = false,
	burn_in = 0.5,
	debug = false,
)
	@assert all(!isnan, [variance(v, input) for v in [I, SOH, DSOC]]) "NaN or Inf in input rhos : $input"
	@assert all(isfinite, [mean(v, input) for v in [I, SOH, DSOC]]) "NaN or Inf in input taus: $input"

	input = set_large_variances_to_uniform ? large_variance_to_uniform(input) : input

	if has_uniform(input)
		marginal_output = sample_uniform_update(input, samples_per_input, variance_relative_epsilon)
		converged = true
		early_stop = false

		if is_uniform(I, input) && rho(I, marginal_output) == rho(I, input)
			error("No change in rho")
		elseif is_uniform(SOH, input) && rho(SOH, marginal_output) == rho(SOH, input)
			error("No change in rho")
		elseif is_uniform(DSOC, input) && rho(DSOC, marginal_output) == rho(DSOC, input)
			error("No change in rho")
		end
	else
		@assert all(>(0), [variance(v, input) for v in [I, SOH, DSOC]]) "Values <= 0 in input rhos"
		#@assert all(isfinite, [variance(v, input) for v in [I, SOH, DSOC]]) "NaN or Inf in input variances. Cannot handle that yet."
		#@assert all(isfinite, [mean(v, input) for v in [I, SOH, DSOC]]) "NaN or Inf in input means"

		marginal_output, early_stop, converged = adaptive_metropolis_hastings(input; dirac_std = dirac_std, max_samples = samples_per_input, burn_in = burn_in, variance_relative_epsilon = variance_relative_epsilon)

		if rho(I, marginal_output) == rho(I, input) || rho(SOH, marginal_output) == rho(SOH, input) || rho(DSOC, marginal_output) == rho(DSOC, input)
			error("No change in rho")
		end
	end

	@assert all(isfinite, marginal_output) "NaN or Inf in updated marginals"

	if isapprox(rho(I, marginal_output), rho(I, input); atol = 1e-8) || isapprox(rho(SOH, marginal_output), rho(SOH, input); atol = 1e-8) || isapprox(rho(DSOC, marginal_output), rho(DSOC, input); atol = 1e-8)
		error("Too little change in rho")
	end

	if mean(SOH, marginal_output) < 0
		error("SoH must be > 0 ")
	end

	return marginal_output, early_stop, converged
end

"""
generate a dataset of n samples, each sample consisting of 5 Floats for the inputs and 4 Floats for the outputs. save dataset as .jdl2 and return file path and variable key for loading
! make sure to avoid high variance in input distributions, low variances in input distributions, large differences between true means and input means, large dirac_std, low number of samples per input
! especially if several of these are combined. for example, combining the first three will quickly lead to the samples variances being larger than the input variances, 
! which will lead to negative rho values after division (impossible, will throw an error)
"""
function generate_dataset_em_factor(;
	n::Int,
	I_mean_dist::Distribution = Uniform(-100, 100),
	I_std_dist::Distribution = Uniform(1e-3, 8.0),
	SOH_mean_dist::Distribution = Uniform(0.1, 1.0),
	SOH_std_dist::Distribution = Uniform(1e-3, 0.5),
	DSOC_mean_dist::Distribution = Uniform(0.1, 1.0),
	DSOC_std_dist::Distribution = Uniform(1e-3, 0.5),
	qo_dist::Distribution = Uniform(30, 120.0),
	dt_dist::Distribution = Uniform(0.1, 10.0),
	samples_per_input::Int = 1_000_000,
	patience = 0.1,
	dirac_std::Float64 = 1e-1,
	log_weighting = true,
	save_dataset = true,
	variance_relative_epsilon = 1e-10,
	std_magnitude_factor = 3.0,
	uniform_quota = 0.1,
	set_large_variances_to_uniform = false,
	name_appendix = "",
	implied_z_max_error = 10,
	savepath = "",
	strict_convergence = false,
)
	nstring = replace(format(n, commas = true), "," => "_") * name_appendix
	savepath = joinpath(savepath, "dataset_em_factor_" * nstring * ".jld2")

	if isfile(savepath) && save_dataset
		error("File already exists: $savepath")
	end

	if dirname(savepath) != "" && !isdir(dirname(savepath)) && save_dataset
		mkpath(dirname(savepath))
	end

	println("Number of threads: ", Threads.nthreads())

	dataset = Vector{Tuple{Vector{Float32}, Vector{Float32}}}(undef, n)
	start_time = time()

	if !strict_convergence
		early_stop_counts = Threads.Atomic{Int}(0)
		converged_counts = Threads.Atomic{Int}(0)
		error_counts = Threads.Atomic{Int}(0)
		progress = ProgressMeter.Progress(n, desc = "Generating Dataset for $n samples using non-strict approach")

		Threads.@threads for i in 1:n
		#for i in 1:n
			while true
				try
					inputs = generate_input_em_factor(
							I_mean_dist = I_mean_dist,
							I_std_dist = I_std_dist,
							SOH_mean_dist = SOH_mean_dist,
							SOH_std_dist = SOH_std_dist,
							DSOC_mean_dist = DSOC_mean_dist,
							DSOC_std_dist = DSOC_std_dist,
							qo_dist = qo_dist,
							dt_dist = dt_dist,
							std_magnitude_factor = std_magnitude_factor,
							uniform_quota = uniform_quota,
							implied_z_max_error = implied_z_max_error,
					)
					outputs, early_stop, converged = generate_output_em_factor(inputs;
						samples_per_input = samples_per_input,
						dirac_std = dirac_std,
						log_weighting = log_weighting,
						variance_relative_epsilon = variance_relative_epsilon,
						set_large_variances_to_uniform = set_large_variances_to_uniform,
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
					println(e)
					Threads.atomic_add!(error_counts, 1)
				end
			end
			ProgressMeter.next!(progress)
		end

		early_stop_counts = early_stop_counts[]
		converged_counts = converged_counts[]
		error_counts = error_counts[]
		println("Early stops: $early_stop_counts / $n --> $(early_stop_counts / n)%")
		println("Converged: $converged_counts / $n --> $(converged_counts / n)%")
		println("Errors: $error_counts")

	else
		# Thread-safe dataset container and counters
		accepted = Threads.Atomic{Int}(0)
		early_stop_counts = Threads.Atomic{Int}(0)
		tries = Threads.Atomic{Int}(0)

		# Thread-safe progress bar
		progress = Progress(n, desc = "Generating Dataset for $n samples using strict approach")
		datalock = ReentrantLock()

		while accepted[] < n
			@threads for _ in 1:Threads.nthreads()
				inputs = generate_input_em_factor(
					I_mean_dist = I_mean_dist,
					I_std_dist = I_std_dist,
					SOH_mean_dist = SOH_mean_dist,
					SOH_std_dist = SOH_std_dist,
					qo_dist = qo_dist,
					dt_dist = dt_dist,
					std_magnitude_factor = std_magnitude_factor,
					uniform_quota = uniform_quota,
					implied_z_max_error = implied_z_max_error,
				)
				outputs, early_stop, converged = generate_output_em_factor(inputs;
					samples_per_input = samples_per_input,
					dirac_std = dirac_std,
					log_weighting = log_weighting,
					variance_relative_epsilon = variance_relative_epsilon,
					set_large_variances_to_uniform = set_large_variances_to_uniform,
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
	targets_X = [get_variable(I, d[2]) for d in dataset]
	targets_Y = [get_variable(SOH, d[2]) for d in dataset]
	targets_Z = [get_variable(DSOC, d[2]) for d in dataset]
	time_in_seconds = time() - start_time

	if save_dataset
		parameters = get_params_dict(;
			n,
			I_mean_dist,
			I_std_dist,
			SOH_mean_dist,
			SOH_std_dist,
			qo_dist,
			dt_dist,
			samples_per_input,
			patience,
			log_weighting,
			variance_relative_epsilon,
			std_magnitude_factor,
			uniform_quota,
			set_large_variances_to_uniform,
			name_appendix,
		)
		jldsave(savepath, samples = samples, targets_X = targets_X, targets_Y = targets_Y, targets_Z = targets_Z, parameters = parameters, early_stop_counts = early_stop_counts, converged_counts = converged_counts, time_in_seconds=time_in_seconds)
		return savepath
	else
		return dataset
	end
end
end
