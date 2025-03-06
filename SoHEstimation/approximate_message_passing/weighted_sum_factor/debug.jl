include("../../../lib/gaussian.jl")
include("../../../lib/utils.jl")
include("generate_data_weighted_sum_factor.jl")
include("plot_nn_results.jl")

# meankldiv(analytical_output, sampled_output; marginal_output=true) = StatsBase.mean([kldivergence(Normal(mean(v, analytical_output), std(v, analytical_output)), Normal(mean(v, sampled_output), std(v, sampled_output))) for v in instances(Variable) if marginal_output || !any(==(0), [tau(v, analytical_output), rho(v, analytical_output)])])

kldivs(analytical_output, sampled_output) = [kldivergence(Normal(tau(v, analytical_output), sqrt(rho(v, analytical_output))), Normal(tau(v, sampled_output), sqrt(rho(v, sampled_output)))) for v in instances(Variable)]

function mh_debug(input_mean_var::Vector{Float64};
    dirac_std::Float64=1e-1,
    num_samples::Int=1_000_000,
    burn_in::Union{Int,Nothing}=nothing,
    log_weighting::Bool=true,
    seed=rand(1:1000000),
)
    input_tau_rho = to_tau_rho(input_mean_var)
    Random.seed!(seed)
    burn_in = isnothing(burn_in) ? Int(num_samples * 0.1) : burn_in
    std_msgfrom_x, std_msgfrom_y, std_msgfrom_z = std(X, input_tau_rho), std(Y, input_tau_rho), std(Z, input_tau_rho)

    # Define prior distributions
    x_prior(x) = log_weighting ? logpdf(Normal(mean(X, input_tau_rho), std_msgfrom_x), x) : pdf(Normal(mean(X, input_tau_rho), std_msgfrom_x), x)
    y_prior(y) = log_weighting ? logpdf(Normal(mean(Y, input_tau_rho), std_msgfrom_y), y) : pdf(Normal(mean(Y, input_tau_rho), std_msgfrom_y), y)
    z_prior(z) = log_weighting ? logpdf(Normal(mean(Z, input_tau_rho), std_msgfrom_z), z) : pdf(Normal(mean(Z, input_tau_rho), std_msgfrom_z), z)
    likelihood(x, y, z) = eval(x, y, z, dirac_std; a=a(input_tau_rho), b=b(input_tau_rho), c=c(input_tau_rho), log_weighting=log_weighting)

    multiply(x...) = log_weighting ? +(x...) : *(x...)
    divide(X...) = log_weighting ? -(X...) : /(X...)
    acceptance_rate() = log_weighting ? log(rand()) : rand()


    # Initialize
    x_current = mean(X, input_tau_rho)
    y_current = mean(Y, input_tau_rho)
    z_current = mean(Z, input_tau_rho)

    x_samples = zeros(num_samples)
    y_samples = zeros(num_samples)
    z_samples = zeros(num_samples)
    accepted = 0

    for i in 1:(num_samples+burn_in)
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

        if i > burn_in
            x_samples[i-burn_in] = x_current
            y_samples[i-burn_in] = y_current
            z_samples[i-burn_in] = z_current
        end
    end

    acceptance_quota = accepted / (num_samples + burn_in)
    # println("Acceptance quota: ", acceptance_quota)

    output = [StatsBase.mean(x_samples), StatsBase.var(x_samples), StatsBase.mean(y_samples), StatsBase.var(y_samples), StatsBase.mean(z_samples), StatsBase.var(z_samples)]
    analytical = update_marginals(input_mean_var...; natural_parameters=false)
    return (kldivs=kldivs(analytical, output), seed=seed, sampled_output=output, analytical_output=analytical, input=input_mean_var, x_samples=x_samples, y_samples=y_samples, z_samples=z_samples)
end

r = mh_debug([32,1, 27,10, 1,100, 1,-10,0.0])