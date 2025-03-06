include("battery_pack.jl")

using Plots

"""
Plot the simulation of the battery pack for the given number of iterations, either using the C or the Julia implementation of pow.
"""
function plot_simulator(use_c_pow=true, iterations=10_000_000)
    battery = BatteryPack()
    state = BatteryPackState()

    current = 0.05

    temperature = Float64[]
    voltage = Float64[]
    current_over_time = Float64[]
    soc = Float64[]
    soh_c = Float64[]
    soh_r = Float64[]

    for i in 1:iterations
        state, soc_limiting = simulate_time_step(battery, state,  15.0 / 3600.0, current, 20.0 + 273.15, use_c_pow=use_c_pow)
        # push!(temperature, state.cells.temperature[1])
        # push!(voltage, state.cells.voltage_terminal[1])
        # push!(current_over_time, state.cells.current[1])
        # push!(soc, state.cells.soc[1])
        # push!(soh_c, state.cells.soh_capacity[1])
        # push!(soh_r, state.cells.soh_resistance[1])

        if soc_limiting
            current = -current
        end

        if i % 100 == 0
            push!(temperature, state.cells.temperature[1])
            push!(voltage, state.cells.voltage_terminal[1])
            push!(current_over_time, state.cells.current[1])
            push!(soc, state.cells.soc[1])
            push!(soh_c, state.cells.soh_capacity[1])
            push!(soh_r, state.cells.soh_resistance[1])
            if i % 10000 == 0
                println("Step: ", i)
            end
        end
    end
    pow_label = use_c_pow ? "_cpow" : "_jpow"
    # plot(temperature .- 273.15, label="Temperature")
    # plot!(voltage, label="Voltage")
    # plot!(current_over_time, label="Current")
    plot!(soc, label="SOC"*pow_label)
    plot!(soh_c, label="SOH_C"*pow_label)
    # plot!(soh_r, label="SOH_R")
end

plot_simulator()
plot_simulator(false)