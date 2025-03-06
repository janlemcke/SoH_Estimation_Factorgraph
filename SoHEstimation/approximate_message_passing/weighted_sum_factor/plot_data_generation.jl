include("../../../lib/gaussian.jl")
include("../../../lib/distributionbag.jl")
include("../../../lib/factors.jl")
include("../../../lib/utils.jl")
include("generate_data_weighted_sum_factor.jl")

using Plots
using Distributions
using .WeightedSumFactorGeneration: to_tau_rho, to_mean_variance, tau, rho, get_variable, dimension, generate_output_weighted_sum_factor, X, Y, Z

# using JuliaInterpreter
# union!(JuliaInterpreter.compiled_modules, setdiff(Base.loaded_modules_array(), [Main, WeightedSumFactorGeneration]))

plotlyjs()

# function verbose_generate(x,y)
#     in = to_tau_rho([0.0, x, 0.0, y, 0.0, 10.0, a, b, c])
#     @assert all(isfinite, in)
#     out = generate_output_weighted_sum_factor(in; debug=true)
#     outmarg = out[1]
#     outmsg = out[2]
#     @assert all(isfinite, vcat(outmarg, outmsg))
#     outmv = to_mean_variance(outmsg[3:4])
#     @assert all(isfinite, outmv) "outmv: $outmv, outmarg: $outmarg, outmsg: $outmsg"
#     return outmv[2]
# end

xs = [1.0,20,40,60,80,100, 200, 400, 600]
ys = [1.0,20,40,60,80,100, 200, 400, 600]

a, b, c = 1.0, -1.0, 0.0
update_variable = Y
update_moment = 2

function get_input(x,y; variable=Z, moment=2)
    if variable == Z
        return moment == 1 ? [x, 10.0, y, 10.0, 2.0, 10.0, a, b, c] : [0.0, x, 0.0, y, 0.0, 10.0, a, b, c]
    elseif variable == X
        return moment == 1 ? [2.0, 10.0, x, 10.0, y, 10.0, a, b, c] : [0.0, 10.0, 0.0, x, 0.0, y, a, b, c]
    elseif variable == Y
        return moment == 1 ? [x, 10.0, 2.0, 10.0, y, 10.0, a, b, c] : [0.0, x, 0.0, 10.0, 0.0, y, a, b, c]
    end
end

function true_function(x,y; variable=Z, moment=2)
    if variable == Z
        result = moment == 1 ? a * x + b * y + c : a^2 * x + b^2 * y + c
    elseif variable == X
        result = moment == 1 ? y / a - b / a * x - c  / a : y / (a * a) + b * b / (a * a) * x
    elseif variable == Y
        result = moment == 1 ? y / b - a / b * x - c  / b : y / (b * b) + a * a / (b * b) * x
    end
    return result
end

generate(x,y; samples_per_input=10_000_000, update_variable=Z, moment=2, marginal=false) = to_mean_variance(generate_output_weighted_sum_factor(to_tau_rho(get_input(x, y; variable=update_variable, moment=moment)); debug=true, samples_per_input=samples_per_input, burn_in=0.5, algorithm=:adaptive_metropolis_hastings, one_variable_fixed=true)[marginal ? 1 : 2][dimension(update_variable,1):dimension(update_variable,2)])[moment]


p1 = surface(xs, ys, (x,y) -> true_function(x,y; variable=update_variable, moment=update_moment), label="true", c=cgrad([:green,:green]),colorbar=false, alpha=0.7)
points = vec([(x,y,generate(x,y; samples_per_input=1_000_000, update_variable=update_variable, moment=update_moment)) for x in xs, y in ys])

scatter!(points, label="data", zlim=(0,2000))

display(p1)

p2 = surface(xs, ys, (x,y) -> true_function(x,y; variable=update_variable, moment=update_moment), label="true", c=cgrad([:green,:green]),colorbar=false, alpha=0.7)
scatter!(points, label="data")

display(p2)