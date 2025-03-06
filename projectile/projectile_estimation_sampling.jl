using Plots, Random, Distributions, LinearAlgebra

# Random.seed!(123)

const TIME_STEP = 1.0
const NUM_DATA_POINTS = 3
const ERROR_PRECISION = 0.50
const ERROR_DISTR = Normal(0.0, 1 / ERROR_PRECISION)
const LIKELIHOOD_STD = 10.0
const NUM_SAMPLES = 5000

# so far this script is assuming perfect choice of prior and measurement precision

struct ProjectileStatePrior
    time::Float64
    x::Normal{Float64}
    y::Normal{Float64}
    velocity_x::Normal{Float64}
    velocity_y::Normal{Float64}
end

ProjectileStatePrior() = ProjectileStatePrior(
    0.0,
    Normal(0.0, 2.0),
    Normal(0.0, 0.01),
    Normal(10.0, 1.0),
    Normal(50.0, 1.0)
)

ProjectileStatePriorWide() = ProjectileStatePrior(
    0.0,
    Normal(0.0, 5.0),
    Normal(0.0, 5.0),
    Normal(10.0, 10.0),
    Normal(50.0, 10.0)
)

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

struct ProjectileState
    time::Float64
    x::Float64
    y::Float64
    velocity_x::Float64
    velocity_y::Float64
end

function state_distance(s1::ProjectileState, s2::ProjectileState)
    return abs(s1.x - s2.x) + abs(s1.y - s2.y) + abs(s1.velocity_x - s2.velocity_x) + abs(s1.velocity_y - s2.velocity_y)
end

function simulate_projectile_step(state::ProjectileState, time_step::Float64=0.1)
    new_x = state.x + state.velocity_x * time_step
    new_y = state.y + state.velocity_y * time_step - 4.9 * time_step^2
    new_velocity_x = state.velocity_x
    new_velocity_y = state.velocity_y - 9.8 * time_step
    new_time = state.time + time_step
    return ProjectileState(new_time, new_x, new_y, new_velocity_x, new_velocity_y)
end

function get_discrete_states(start_state, time_step, stop_time)
    time_values = start_state.time:time_step:stop_time
    return [simulate_projectile_step(start_state, time) for time in time_values]
end

function get_discrete_states(start_state, time_step)
    # from start point to when y < 0
    states = []
    state = start_state

    while state.y <= 0.0
        push!(states, state)
        state = simulate_projectile_step(state, time_step)
        # println("state.y = ", state.y)
    end

    while state.y > 0.0
        push!(states, state)
        state = simulate_projectile_step(state, time_step)
        # println("state.y = ", state.y)
    end
    return states
end

function get_discrete_plot(start_state, time_step, stop_time)
    states = get_discrete_states(start_state, time_step, stop_time)
    x_values = [state.x for state in states]
    y_values = [state.y for state in states]
    return x_values, y_values
end

function get_continuous_plot_map(start_state::ProjectileState)
    function plotmap(x)
        time = (x - start_state.x) / start_state.velocity_x
        return start_state.y + start_state.velocity_y * time - 4.9 * time^2
    end
end

function get_log_likelihood(initial_state, datapoints)
    log_likelihood_x = 0.0
    log_likelihood_y = 0.0

    # new_x = state.x + state.velocity_x * time_step
    # new_y = state.y + state.velocity_y * time_step - 4.9 * time_step^2

    for point in datapoints
        log_likelihood_x += log(pdf(Normal(initial_state.x + initial_state.velocity_x * point.time, LIKELIHOOD_STD), point.x))
        log_likelihood_y += log(pdf(Normal(initial_state.y + initial_state.velocity_y * point.time - 4.9 * point.time^2, LIKELIHOOD_STD), point.y))
    end
    return log_likelihood_x, log_likelihood_y
end

function get_log_prior(initial_state, initial_state_prior)
    log_prior_x = log(pdf(initial_state_prior.x, initial_state.x)) + log(pdf(initial_state_prior.velocity_x, initial_state.velocity_x))
    log_prior_y = log(pdf(initial_state_prior.y, initial_state.y)) + log(pdf(initial_state_prior.velocity_y, initial_state.velocity_y))
    return log_prior_x, log_prior_y
end

function get_log_posterior_unnormalized(initial_state, datapoints, initial_state_prior)
    log_prior_x, log_prior_y = get_log_prior(initial_state, initial_state_prior)
    log_likelihood_x, log_likelihood_y = get_log_likelihood(initial_state, datapoints)
    return log_prior_x + log_likelihood_x, log_prior_y + log_likelihood_y
end

function get_combined_log_posterior_unnormalized(initial_state, datapoints, initial_state_prior)
    log_posterior_x, log_posterior_y = get_log_posterior_unnormalized(initial_state, datapoints, initial_state_prior)
    return log_posterior_x + log_posterior_y
end

function Base.:*(dist1::Normal, dist2::Normal)
    τ1 = mean(dist1) / var(dist1)
    ρ1 = 1 / var(dist1)
    τ2 = mean(dist2) / var(dist2)
    ρ2 = 1 / var(dist2)

    τ = τ1 + τ2
    ρ = ρ1 + ρ2

    μ = τ / ρ
    σ = √(1.0 / ρ)

    return Normal(μ, σ)
end

function posterior_dist_unnormalized(datapoints, initial_state_prior)
    dist_x = initial_state_prior.x * initial_state_prior.velocity_x
    dist_y = initial_state_prior.y * initial_state_prior.velocity_y

    for point in datapoints
        dist_x *= Normal(initial_state.x + initial_state.velocity_x * point.time, LIKELIHOOD_STD)
    end
end

p = plot()

function get_true_data()
    true_data = []
    while length(true_data) < NUM_DATA_POINTS
        true_start = sampleProjectileState()
        true_data = get_discrete_states(true_start, TIME_STEP)
        # println("true_data  = ", true_data)
    end
    return true_data
end

# println("got here")
true_data = get_true_data()

true_data_points = sample(true_data, NUM_DATA_POINTS; replace=false)
measured_data = [ProjectileState(state.time, state.x + rand(ERROR_DISTR), state.y + rand(ERROR_DISTR), state.velocity_x + rand(ERROR_DISTR), state.velocity_y + rand(ERROR_DISTR)) for state in true_data_points]

sample_states = [sampleProjectileState(ProjectileStatePriorWide()) for _ in 1:NUM_SAMPLES]
samples_losses = [state_distance(s, true_data[1]) for s in sample_states]
samples_log_posteriors = [get_combined_log_posterior_unnormalized(s, measured_data, ProjectileStatePrior()) for s in sample_states]
samples_posteriors = [exp(i) for i in samples_log_posteriors]
samples_posteriors_normalized = samples_posteriors / norm(samples_posteriors)
order = sortperm(samples_posteriors_normalized)
samples_posteriors_normalized = samples_posteriors_normalized[order]
sample_states = sample_states[order]
println(first(samples_posteriors_normalized))
println(last(samples_posteriors_normalized))
max_loss = maximum(samples_losses)

function plot_trajectory!(s::ProjectileState; 
    linecolor="blue", 
    linealpha::Float64=1.0, 
    x_step=TIME_STEP, 
    label="", 
    x_max=100.0
)
    x = -10.0:x_step:x_max
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

gradient = cgrad(:devon100)
for i in 1:NUM_SAMPLES
    # plot_trajectory!(sample_states[i], linecolor=gradient[trunc(Int, ((samples_losses[i]/max_loss))*98+1)], linealpha=(1-(samples_losses[i]/max_loss)))
    # , linealpha=(1-(samples_posteriors_normalized[i]))
    plot_trajectory!(
        sample_states[i], 
        # linecolor=gradient[trunc(Int, ((1 - samples_posteriors_normalized[i])) * 70 + 1)], 
        linealpha=clamp(samples_posteriors_normalized[i], 0.01, 1.0), 
        x_max=last(true_data).x
    )
end

plot_trajectory!(true_data[1], linecolor="red", label="true trajectory", x_max=last(true_data).x)

scatter!(
    [s.x for s in measured_data],
    [s.y for s in measured_data],
    markercolor="green",
    label="measured data"
)

scatter!(
    [s.x for s in true_data_points],
    [s.y for s in true_data_points],
    markercolor="red",
    label="true data"
)

p2 = histogram(samples_posteriors_normalized, bins=0.0:0.01:1.0, ylims=(0, 100))
display(p2)
display(p)