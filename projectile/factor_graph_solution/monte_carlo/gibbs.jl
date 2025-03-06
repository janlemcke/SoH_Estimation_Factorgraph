using Plots, Random, Distributions, LinearAlgebra
import .Projectile

# Gibbs sampling function
function gibbs_sampling(initial_state, measured_data, num_iterations, time_step, prior::ProjectileStatePrior, burn_in_ratio=0.1)
    burn_in = round(Int, num_iterations * burn_in_ratio)
    total_iterations = num_iterations + burn_in

    x_samples = Vector{Float64}(undef, total_iterations + 1)
    y_samples = Vector{Float64}(undef, total_iterations + 1)
    vx_samples = Vector{Float64}(undef, total_iterations + 1)
    vy_samples = Vector{Float64}(undef, total_iterations + 1)

    x_samples[1] = initial_state.x
    y_samples[1] = initial_state.y
    vx_samples[1] = initial_state.velocity_x
    vy_samples[1] = initial_state.velocity_y

    current_state = initial_state
    current_likelihood = likelihood(measured_data, current_state, time_step)
    current_log_prior = log_prior(current_state, prior)

    for i in 1:total_iterations
        # Sample x given y, vx, vy
        x_proposal = rand(Normal(current_state.x, prior.x.σ))
        new_state = ProjectileState(current_state.time, x_proposal, current_state.y, current_state.velocity_x, current_state.velocity_y)

        proposed_likelihood = likelihood(measured_data, new_state, time_step)
        proposed_log_prior = log_prior(new_state, prior)

        acceptance_ratio = min(1, exp((proposed_likelihood + proposed_log_prior) - (current_likelihood + current_log_prior)))
        if rand() < acceptance_ratio
            current_state = new_state
            current_likelihood = proposed_likelihood
            current_log_prior = proposed_log_prior
        end
        x_samples[i + 1] = current_state.x

        # Sample y given x, vx, vy
        y_proposal = rand(Normal(current_state.y, prior.y.σ))
        new_state = ProjectileState(current_state.time, current_state.x, y_proposal, current_state.velocity_x, current_state.velocity_y)

        proposed_likelihood = likelihood(measured_data, new_state, time_step)
        proposed_log_prior = log_prior(new_state, prior)

        acceptance_ratio = min(1, exp((proposed_likelihood + proposed_log_prior) - (current_likelihood + current_log_prior)))
        if rand() < acceptance_ratio
            current_state = new_state
            current_likelihood = proposed_likelihood
            current_log_prior = proposed_log_prior
        end
        y_samples[i + 1] = current_state.y

        # Sample vx given x, y, vy
        vx_proposal = rand(Normal(current_state.velocity_x, prior.velocity_x.σ))
        new_state = ProjectileState(current_state.time, current_state.x, current_state.y, vx_proposal, current_state.velocity_y)
        
        proposed_likelihood = likelihood(measured_data, new_state, time_step)
        proposed_log_prior = log_prior(new_state, prior)

        acceptance_ratio = min(1, exp((proposed_likelihood + proposed_log_prior) - (current_likelihood + current_log_prior)))
        if rand() < acceptance_ratio
            current_state = new_state
            current_likelihood = proposed_likelihood
            current_log_prior = proposed_log_prior
        end
        vx_samples[i + 1] = current_state.velocity_x

        # Sample vy given x, y, vx
        vy_proposal = rand(Normal(current_state.velocity_y, prior.velocity_y.σ))
        new_state = ProjectileState(current_state.time, current_state.x, current_state.y, current_state.velocity_x, vy_proposal)

        proposed_likelihood = likelihood(measured_data, new_state, time_step)
        proposed_log_prior = log_prior(new_state, prior)

        acceptance_ratio = min(1, exp((proposed_likelihood + proposed_log_prior) - (current_likelihood + current_log_prior)))
        if rand() < acceptance_ratio
            current_state = new_state
            current_likelihood = proposed_likelihood
            current_log_prior = proposed_log_prior
        end
        vy_samples[i + 1] = current_state.velocity_y
    end
    
    states = [ProjectileState(0, x_samples[i], y_samples[i], vx_samples[i], vy_samples[i]) for i in burn_in+1:total_iterations]
    return states
end

