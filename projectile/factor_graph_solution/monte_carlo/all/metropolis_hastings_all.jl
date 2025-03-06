module MetropolisHastings

include("../projectile.jl")
using Distributions
using ..GaussianDistribution: Gaussian1DFromMeanVariance

function sampleProjectileState(stateprior::ProjectileStatePrior=ProjectileStatePrior())
    state = ProjectileState(
        stateprior.time,
        rand(stateprior.x),
        rand(stateprior.y),
        rand(stateprior.velocity_x),
        rand(stateprior.velocity_y)
    )
    return state
end

function log_prior(state, prior::ProjectileStatePrior)
    lp = 0.0
    lp += logpdf(prior.x, state.x)
    lp += logpdf(prior.y, state.y)
    lp += logpdf(prior.velocity_x, state.velocity_x)
    lp += logpdf(prior.velocity_y, state.velocity_y)
    return lp
end

# calculate the likelihood of a proposed state given samples from the the successor_states
function likelihood(measured_data, proposed_state::ProjectileState, time_step, use_backwards=false)
    simulated_states = get_discrete_states(proposed_state, time_step)
    if use_backwards
        simulated_states_backwards = get_discrete_states_backwards(proposed_state, time_step)
    end

    # Check if there are enough simulated states
    if length(simulated_states) == 0
        return -Inf
    end

    ll_value = 0.0

    # Create a dictionary for quick lookup of simulated states by time
    simulated_dict = Dict(state.time => state for state in simulated_states)

    if use_backwards
        simulated_dict_backwards = Dict(state.time => state for state in simulated_states_backwards)
        simulated_dict = merge(simulated_dict, simulated_dict_backwards)
    end


    for measured in measured_data
        if haskey(simulated_dict, measured.time)
            simulated = simulated_dict[measured.time]

            ll_value += logpdf(Normal(measured.x, 1), simulated.x)
            ll_value += logpdf(Normal(measured.y, 1), simulated.y)
            ll_value += logpdf(Normal(measured.velocity_x, 1), simulated.velocity_x)
            ll_value += logpdf(Normal(measured.velocity_y, 1), simulated.velocity_y)
        end
    end
    
    return ll_value
end


function metropolis_hastings(measured_data, num_samples, time_step::Float64, burn_in=0.1)
    burn_in_samples = round(Int, num_samples * burn_in)
    total_samples = num_samples + burn_in_samples

    # get last time step of measured_data
    last_time = measured_data[end].time
    current_timestep = last_time - time_step

    sampled_states = []
    push!(sampled_states, 
        ProjectileStatePrior(last_time,
            Normal(measured_data[end].x, .1),
            Normal(measured_data[end].y, .1),
            Normal(measured_data[end].velocity_x, .1),
            Normal(measured_data[end].velocity_y, .1)
        )
    )
    sampled_states_index = 1

    while current_timestep >= 0

        # check if there exists a measured data point at the current time step
        if current_timestep in [state.time for state in measured_data]
            # get index of measured data point
            index = findfirst(x -> x.time == current_timestep, measured_data)
            push!(sampled_states, 
                ProjectileStatePrior(current_timestep,
                    Normal(measured_data[index].x, .1),
                    Normal(measured_data[index].y, .1),
                    Normal(measured_data[index].velocity_x, .1),
                    Normal(measured_data[index].velocity_y, .1)
                )
            )
            current_timestep -= time_step
            sampled_states_index += 1
            continue
        end

        successor_posterior = sampled_states[sampled_states_index]
        successor_states = [sampleProjectileState(successor_posterior) for _ in 1:100]
        successor_state = ProjectileState(current_timestep + time_step, 
                                            mean([state.x for state in successor_states]),
                                            mean([state.y for state in successor_states]),
                                            mean([state.velocity_x for state in successor_states]),
                                            mean([state.velocity_y for state in successor_states])
                                            )

        current_prior = ProjectileStatePrior(
            current_timestep,
            Normal(successor_state.x, 10),
            Normal(successor_state.y, 10),
            Normal(successor_state.velocity_x, 10),
            Normal(successor_state.velocity_y, 10)
        )

        current_state = sampleProjectileState(current_prior)
        current_likelihood = likelihood(measured_data, current_state, time_step)
        current_log_prior = log_prior(current_state, current_prior)

        current_samples = Vector{ProjectileState}(undef, total_samples)

        for i in 1:total_samples
            proposed_state = sampleProjectileState(current_prior)
            proposed_likelihood = likelihood(measured_data, proposed_state, time_step)
            proposed_log_prior = log_prior(proposed_state, current_prior)
    
            acceptance_ratio = min(1, exp((proposed_likelihood + proposed_log_prior) - (current_likelihood + current_log_prior)))
    
            if rand() < acceptance_ratio
                current_state = proposed_state
                current_likelihood = proposed_likelihood
                current_log_prior = proposed_log_prior
            end
    
            current_samples[i] = current_state
        end

        # Create a new posterior state based on the current samples
        current_posterior = ProjectileStatePrior(
            current_timestep,
            Normal(mean([sample.x for sample in current_samples[burn_in_samples:end]]), std([sample.x for sample in current_samples[burn_in_samples:end]])),
            Normal(mean([sample.y for sample in current_samples[burn_in_samples:end]]), std([sample.y for sample in current_samples[burn_in_samples:end]])),
            Normal(mean([sample.velocity_x for sample in current_samples[burn_in_samples:end]]), std([sample.velocity_x for sample in current_samples[burn_in_samples:end]])),
            Normal(mean([sample.velocity_y for sample in current_samples[burn_in_samples:end]]), std([sample.velocity_y for sample in current_samples[burn_in_samples:end]]))
        )

        push!(sampled_states, current_posterior)

        current_timestep -= time_step
        sampled_states_index += 1
    end

    # retun sampled_states in reverse order
    return reverse(sampled_states)
end



function calc_error(true_states, simulated_states)
    error = 0.0
    for (true_state, simulated_state) in zip(true_states, simulated_states)
        error += state_distance(true_state, simulated_state)
    end

    return error / length(true_states)
end

function get_continuous_plot_map(start_state::ProjectileState)
    function plotmap(x)
        time = (x - start_state.x) / start_state.velocity_x
        return start_state.y + start_state.velocity_y * time - 4.9 * time^2
    end
end

function plot_trajectory!(s::ProjectileState; 
    linecolor="blue", 
    linealpha::Float64=1.0, 
    x_step=TIME_STEP, 
    label="", 
    x_max=100.0,
    x_min=0.0
)
    x = x_min:x_step:x_max
    y = map(get_continuous_plot_map(s), x)
    plot!(
        x,
        y,
        ylims=(0, Inf),
        legend=true,
        linecolor=linecolor,
        linewidth=3,
        linealpha=linealpha,
        label=label)
end

function plot_point_estimation(point::Pair{Normal{Float64}, Normal{Float64}})
    # convert normal to 1D Gaussian
    first_point = Gaussian1DFromMeanVariance(point.first.μ, point.first.σ)
    second_point = Gaussian1DFromMeanVariance(point.second.μ, point.second.σ)

    μ = [first_point.tau/first_point.rho, second_point.tau/second_point.rho]
    Σ = [1.0/first_point.rho 0; 0 1.0/second_point.rho]

    dist = MvNormal(μ, Σ)

    x = μ[1]-Σ[1,1]*2:0.1:μ[1]+Σ[1,1]*2
    y = μ[2]-Σ[2,2]*2:0.1:μ[2]+Σ[2,2]*2
    f(x, y) = pdf(dist, [x, y])

    contour!(x, y, f, fill=true, color=reverse(palette(:acton)[1:end-32]), clim=(1e-3,Inf))

end


const NUM_DATA_POINTS = 2
const ERROR_DISTR = Normal(0, 1)
const TIME_STEP = 1.0
const MAX_NUM_SAMPLES = 100000

const true_starting_state = ProjectileState(0, 12.5, 20, 10.0, 50.0)
const true_data = get_true_data(true_starting_state, TIME_STEP)
const true_data_points = sample(true_data[2:end], NUM_DATA_POINTS; replace=false)
measured_data = [ProjectileState(state.time, state.x + rand(ERROR_DISTR), state.y + rand(ERROR_DISTR), state.velocity_x + rand(ERROR_DISTR), state.velocity_y + rand(ERROR_DISTR)) for state in true_data_points]
sort!(measured_data, by=x->x.time)
println("Observed data points: ", measured_data)
println("True data points: ", true_data)

metropolis_hastings_posteriors = metropolis_hastings(measured_data, MAX_NUM_SAMPLES, TIME_STEP)


# simulated_states = [sampleProjectileState(posterior) for posterior in metropolis_hastings_posteriors]
# sample 50 states from the last posterior and then take the mean to create the simulated states
simulated_states = [ProjectileState(posterior.time, 
                                    mean([rand(posterior.x) for _ in 1:100]),
                                    mean([rand(posterior.y) for _ in 1:100]),
                                    mean([rand(posterior.velocity_x) for _ in 1:100]),
                                    mean([rand(posterior.velocity_y) for _ in 1:100])
                                    ) for posterior in metropolis_hastings_posteriors]

# calc_error
println("Simulated states: ", simulated_states)
error = calc_error(true_data, simulated_states)
println("Error: ", error)

p = plot()

"""
println("Length of learned data: ", length(metropolis_hastings_posteriors))
for posterior in metropolis_hastings_posteriors
    plot_point_estimation(Pair(posterior.x, posterior.y))
end
"""


plot_trajectory!(true_data[1], linecolor="grey", label="true trajectory", x_max=last(true_data).x, x_min=first(true_data).x)

scatter!(
    [s.x for s in measured_data],
    [s.y for s in measured_data],
    markercolor="green",
    label="measured data",
    markershape=:xcross
)

scatter!(
    [s.x for s in true_data],
    [s.y for s in true_data],
    markercolor="red",
    label="true data",
    markershape=:xcross
)

scatter!(
    [s.x for s in simulated_states],
    [s.y for s in simulated_states],
    markercolor="orange",
    label="learned data",
    markershape=:xcross
)


display(p)
end