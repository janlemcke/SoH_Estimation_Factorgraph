include("./generate_dataset_em_factor.jl")
using .ElectricalModelFactorGeneration: generate_output_em_factor

input = [0, 0.01, 0.7, 0.04000000000000001, 0.0, Inf, 0.66, 0.004166666666666667]
global trial = 0

while true
    try
        output = generate_output_em_factor(input, variance_relative_epsilon=1e-2)
        println(output)
        println("Trial: ", trial)
        break
    catch e
        println("Error: ", e)
        global trial += 1
    end
end