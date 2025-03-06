
import Distributions
include("projectile.jl")


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

function likelihood(measured_data, proposed_state::ProjectileState, time_step)
    simulated_states = get_discrete_states(proposed_state, time_step)

    # check if there are enough simulated states
    if length(simulated_states) == 0
        return -Inf
    end

    ll_value =  0
    # get only the states from measured_data and simulated_states that are within the same time frame
    for i in eachindex(measured_data)
        if i > length(simulated_states)
            break
        end

        if measured_data[i].time != simulated_states[i].time
            continue
        end

        ll_value += logpdf(Normal(measured_data[i].x, 1), simulated_states[i].x)
        ll_value += logpdf(Normal(measured_data[i].y, 1), simulated_states[i].y)
        ll_value += logpdf(Normal(measured_data[i].velocity_x, 1), simulated_states[i].velocity_x)
        ll_value += logpdf(Normal(measured_data[i].velocity_y, 1), simulated_states[i].velocity_y)
    end
    
    return ll_value
end

function propose_new_state(current_state, state_prior::ProjectileStatePrior)
    new_x = rand(Normal(current_state.x, state_prior.x.σ))
    new_y = rand(Normal(current_state.y, state_prior.y.σ))
    new_velocity_x = rand(Normal(current_state.velocity_x, state_prior.velocity_x.σ))
    new_velocity_y = rand(Normal(current_state.velocity_y, state_prior.velocity_y.σ))

    return ProjectileState(0.0, new_x, new_y, new_velocity_x, new_velocity_y)
end

function metropolis_hastings(initial_state, measured_data , num_samples, time_step, prior::ProjectileStatePrior, burn_in=0.1)
    burn_in_samples = round(Int, num_samples * burn_in)
    total_samples = num_samples + burn_in_samples

    samples = Vector{ProjectileState}(undef, total_samples)
    current_state = initial_state
    current_likelihood = likelihood(measured_data, current_state, time_step)
    current_log_prior = log_prior(current_state, prior)

    for i in 1:total_samples
        proposed_state = propose_new_state(current_state, prior)
        proposed_likelihood = likelihood(measured_data, proposed_state, time_step)
        proposed_log_prior = log_prior(proposed_state, prior)

        acceptance_ratio = min(1, exp((proposed_likelihood + proposed_log_prior) - (current_likelihood + current_log_prior)))


        if rand() < acceptance_ratio
            current_state = proposed_state
            current_likelihood = proposed_likelihood
            current_log_prior = proposed_log_prior
        end

        samples[i] = current_state
    end

    return samples[burn_in_samples+1:end]
end



function calc_error(true_states, simulated_states)
    error = 0.0
    for (true_state, simulated_state) in zip(true_states, simulated_states)
        error += state_distance(true_state, simulated_state)
    end

    return error / length(true_states)
end
