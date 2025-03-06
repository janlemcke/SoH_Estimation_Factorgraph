using Plots, Random, Distributions, LinearAlgebra


struct ProjectileState
    time::Float64
    x::Float64
    y::Float64
    velocity_x::Float64
    velocity_y::Float64
end

struct ProjectileStatePrior
    time::Float64
    x::Normal{Float64}
    y::Normal{Float64}
    velocity_x::Normal{Float64}
    velocity_y::Normal{Float64}
end

ProjectileStatePrior() = ProjectileStatePrior(
    0.0,
    Normal(12.5, 1),
    Normal(20, 1),
    Normal(10.0, 1),
    Normal(50.0, 1)
)

# pretty print state with floats rounded to 2 decimal places
function Base.show(io::IO, state::ProjectileState)
    println(io, "ProjectileState(time=$(round(state.time, digits=2)), x=$(round(state.x, digits=2)), y=$(round(state.y, digits=2)), velocity_x=$(round(state.velocity_x, digits=2)), velocity_y=$(round(state.velocity_y, digits=2)))")
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

function simulate_projectile_step_backwards(state::ProjectileState, time_step::Float64=0.1)
    new_x = state.x - state.velocity_x * time_step
    new_y = state.y - state.velocity_y * time_step - 4.9 * time_step^2
    new_velocity_x = state.velocity_x
    new_velocity_y = state.velocity_y + 9.8 * time_step
    new_time = state.time - time_step
    return ProjectileState(new_time, new_x, new_y, new_velocity_x, new_velocity_y)
end

function get_discrete_states(start_state, time_step, stop_time)
    time_values = start_state.time:time_step:stop_time
    return [simulate_projectile_step(start_state, time) for time in time_values]
end

function get_discrete_states_backwards(start_state::ProjectileState, time_step::Float64)
    # from start point to when y < 0
    states = []
    state = start_state

    while state.time >= 0.0
        state = simulate_projectile_step_backwards(state, time_step)
        push!(states, state)
        # println("state.y = ", state.y)
    end
    return states
end

function get_discrete_states(start_state::ProjectileState, time_step)
    # from start point to when y < 0
    states = []
    state = start_state

    while state.y >= 0.0
        state = simulate_projectile_step(state, time_step)
        push!(states, state)
        # println("state.y = ", state.y)
    end
    return states
end

function get_true_data(true_starting_state::ProjectileState, time_step::Float64=1.0)
    return get_discrete_states(true_starting_state, time_step)
end