include("projectile_estimation_factorgraph.jl")
using Plots, Random, Distributions, LinearAlgebra, DataStructures, ColorSchemes
using ..ProjectileEstimation, ..GaussianDistribution

d = Gaussian1D(0, 0)

# Random.seed!(123)

const TIME_STEP = 1.0
const NUM_DATA_POINTS = 3
const ERROR_PRECISION = 0.50
const ERROR_DISTR = Gaussian1DFromMeanVariance(0.0, 1 / ERROR_PRECISION)
const LIKELIHOOD_STD = 10.0
const NUM_SAMPLES = 5000

get_noise() = rand(Normal(ERROR_DISTR.tau/ERROR_DISTR.rho, 1.0/ERROR_DISTR.rho))

struct ProjectileState
    time::Float64
    x::Float64
    y::Float64
    velocity_x::Float64
    velocity_y::Float64
end

struct ProjectileStatePrior
    time::Float64
    x::Gaussian1D
    y::Gaussian1D
    velocity_x::Gaussian1D
    velocity_y::Gaussian1D
end

ProjectileStatePrior() = ProjectileStatePrior(
    0.0,
    Gaussian1DFromMeanVariance(0.0, 2.0^2),
    Gaussian1DFromMeanVariance(0.0, 0.01^2),
    Gaussian1DFromMeanVariance(10.0, 1.0^2),
    Gaussian1DFromMeanVariance(50.0, 1.0^2)
)

ProjectileStatePriorWide() = ProjectileStatePrior(
    0.0,
    Gaussian1DFromMeanVariance(0.0, 5.0^2),
    Gaussian1DFromMeanVariance(0.0, 5.0^2),
    Gaussian1DFromMeanVariance(10.0, 10.0^2),
    Gaussian1DFromMeanVariance(50.0, 10.0^2)
)

function sampleProjectileState(stateprior::ProjectileStatePrior=ProjectileStatePrior())
    state = ProjectileState(
        stateprior.time,
        rand(Normal(stateprior.x.tau/stateprior.x.rho, 1.0/stateprior.x.rho)),
        rand(Normal(stateprior.y.tau/stateprior.y.rho, 1.0/stateprior.y.rho)),
        rand(Normal(stateprior.velocity_x.tau/stateprior.velocity_x.rho, 1.0/stateprior.velocity_x.rho)),
        rand(Normal(stateprior.velocity_y.tau/stateprior.velocity_y.rho, 1.0/stateprior.velocity_y.rho))
    )
    return state
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
    states = []
    state = start_state

    while state.y <= 0.0
        push!(states, state)
        state = simulate_projectile_step(state, time_step)
    end

    while state.y > 0.0
        push!(states, state)
        state = simulate_projectile_step(state, time_step)
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

p = plot()

function get_true_data(time_step=TIME_STEP)
    true_data = []
    while length(true_data) < NUM_DATA_POINTS
        true_start = sampleProjectileState()
        while true_start.y < 0.0 || true_start.x < 0.0 || true_start.velocity_x < 0.0 || true_start.velocity_y < 0.0
            true_start = sampleProjectileState()
        end
        true_data = get_discrete_states(true_start, time_step)
    end
    return true_data
end

true_data = get_true_data()

true_data_points = true_data[length(true_data) - NUM_DATA_POINTS + 1:end]
measured_data = [ProjectileState(state.time, state.x + get_noise(), state.y + get_noise(), state.velocity_x + get_noise(), state.velocity_y + get_noise()) for state in true_data_points]

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

function plot_point_estimation(point::Pair{Gaussian1D, Gaussian1D})
    μ = [point.first.tau/point.first.rho, point.second.tau/point.second.rho]
    Σ = [1.0/point.first.rho 0; 0 1.0/point.second.rho]

    dist = MvNormal(μ, Σ)

    x = μ[1]-Σ[1,1]*2:0.1:μ[1]+Σ[1,1]*2
    y = μ[2]-Σ[2,2]*2:0.1:μ[2]+Σ[2,2]*2
    f(x, y) = pdf(dist, [x, y])

    contour!(x, y, f, fill=true, color=reverse(palette(:acton)[1:end-32]), clim=(1e-3,Inf))

end

estimations_x = estimate_initial_state_x!(data=OrderedDict([(datapoint.time, datapoint.x) for datapoint in measured_data]))
estimations_y = estimate_initial_state_y!(data=OrderedDict([(datapoint.time, datapoint.y) for datapoint in measured_data]))

for (x, y) in collect(zip(estimations_x, estimations_y))
    plot_point_estimation(Pair(x[1], y[1]))
end

plot_trajectory!(true_data[1], linecolor="red", label="true trajectory", x_max=last(true_data).x)

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

println("true_data:")
for s in true_data
    println(s)
end

display(p)