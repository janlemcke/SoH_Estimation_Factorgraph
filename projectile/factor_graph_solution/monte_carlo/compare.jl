using Plots, Random, Distributions, LinearAlgebra, Dates

include("projectile.jl")
include("metropolis_hastings.jl")
include("gibbs.jl")
include("plots.jl")
include("metrics.jl")

const NUM_DATA_POINTS = 1
const ERROR_DISTR = Normal(0, 1)
const TIME_STEP = 1.0
const MAX_NUM_SAMPLES = 100
const SAMPLE_STEP = 1

const true_starting_state = ProjectileState(0, 12.5, 20, 10.0, 50.0)
const true_data = get_true_data(true_starting_state, TIME_STEP)
const true_data_points = sample(true_data[2:end], NUM_DATA_POINTS; replace=false)
const measured_data = [ProjectileState(state.time, state.x + rand(ERROR_DISTR), state.y + rand(ERROR_DISTR), state.velocity_x + rand(ERROR_DISTR), state.velocity_y + rand(ERROR_DISTR)) for state in true_data_points]
println("Observed data points: ", measured_data)

# Assume an initial guess for the initial state
const initial_state_guess = sampleProjectileState()
println("Initial guess: ", initial_state_guess)

# Run Metropolis-Hastings and Gibbs Sampling with increasing number of samples
function evaluate_sample_size_vs_error_runtime()
    num_samples_list = 1:SAMPLE_STEP:MAX_NUM_SAMPLES
    errors_MH = Float64[]
    errors_gibbs = Float64[]
    times_MH = Float64[]
    times_gibbs = Float64[]
    for num_samples in num_samples_list
        #println("Running with ", num_samples, " samples...")
        
        # Measure runtime for Metropolis-Hastings
        start_time = time()
        samples_MH = metropolis_hastings(initial_state_guess, measured_data, num_samples, TIME_STEP, ProjectileStatePrior())
        end_time = time()
        push!(times_MH, end_time - start_time)
        posterior_state_MH = ProjectileState(0, mean([sample.x for sample in samples_MH]), mean([sample.y for sample in samples_MH]), mean([sample.velocity_x for sample in samples_MH]), mean([sample.velocity_y for sample in samples_MH]))
        states_MH = get_discrete_states(posterior_state_MH, TIME_STEP)
        error_MH = calc_error(true_data, states_MH)
        push!(errors_MH, error_MH)
        
        # Measure runtime for Gibbs Sampling
        start_time = time()
        samples_gibbs = gibbs_sampling(initial_state_guess, measured_data, num_samples, TIME_STEP, ProjectileStatePrior())
        posterior_state_gibbs = ProjectileState(0, mean([sample.x for sample in samples_gibbs]), mean([sample.y for sample in samples_gibbs]), mean([sample.velocity_x for sample in samples_gibbs]), mean([sample.velocity_y for sample in samples_gibbs]))
        end_time = time()
        push!(times_gibbs, end_time - start_time)
        
    
        states_gibbs = get_discrete_states(posterior_state_gibbs, TIME_STEP)
        error_gibbs = calc_error(true_data, states_gibbs)
        push!(errors_gibbs, error_gibbs)
    end

    return num_samples_list, errors_MH, errors_gibbs, times_MH, times_gibbs
end


num_samples_list, errors_MH, errors_gibbs, times_MH, times_gibbs = evaluate_sample_size_vs_error_runtime()


display(autocorrelation_plot(metropolis_hastings(initial_state_guess, measured_data, MAX_NUM_SAMPLES, TIME_STEP, ProjectileStatePrior()), :x))
display(autocorrelation_plot(gibbs_sampling(initial_state_guess, measured_data, MAX_NUM_SAMPLES, TIME_STEP, ProjectileStatePrior()), :x))


# Plot the number of samples with the error
p = plot(num_samples_list, errors_MH, label="MH Error", xlabel="Number of Samples", ylabel="Error", title="Error vs Number of Samples", legend=:topright)
plot!(num_samples_list, errors_gibbs, label="Gibbs Error")
display(p)

# Plot the number of samples with the runtime
p = plot(num_samples_list, times_MH, label="MH Runtime (ms)", xlabel="Number of Samples", ylabel="Runtime (ms)", title="Runtime vs Number of Samples", legend=:topright)
plot!(num_samples_list, times_gibbs, label="Gibbs Runtime (ms)")
display(p)

println("[MH] Final Error: ", last(errors_MH))
println("[GIBBS] Final Error: ", last(errors_gibbs))
println("[MH] Final Runtime: ", last(times_MH), " ms")
println("[GIBBS] Final Runtime: ", last(times_gibbs), " ms")


function evaluate_chains()
    NUM_OF_CHAINS = 20
    TIME_STEP = 1.0

    initial_guesses = []

    # add a lot of random noise to the initial guesses
    for _ in 1: NUM_OF_CHAINS
        push!(initial_guesses, ProjectileState(0, rand(Normal(12.5, 15)), rand(Normal(20, 15)), rand(Normal(10, 15)), rand(Normal(50, 15))))
    end

    println(initial_guesses)

    
    chains_MH = [metropolis_hastings(initial_guess, measured_data, MAX_NUM_SAMPLES, TIME_STEP, ProjectileStatePrior()) for initial_guess in initial_guesses]
    chains_gibbs = [gibbs_sampling(initial_guess, measured_data, MAX_NUM_SAMPLES, TIME_STEP, ProjectileStatePrior()) for initial_guess in initial_guesses]

    epsr_MH = gelman_rubin(chains_MH)
    epsr_gibbs = gelman_rubin(chains_gibbs)

    println("Estimated potential scale reduction (EPSR) for Metropolis-Hastings: ", epsr_MH)
    println("Estimated potential scale reduction (EPSR) for Gibbs Sampling: ", epsr_gibbs)

    
end

evaluate_chains()