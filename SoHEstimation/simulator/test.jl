include("battery_pack.jl")

using Printf

function test_simulator()
    battery = BatteryPack()
    state = BatteryPackState()

    current = 10.0

    for i in 0:9999
        state, soc_limiting = simulate_time_step(battery, state,  15.0 / 3600.0, current, 20.0 + 273.15, iteration=i)
        @printf "Temperature[%i] =  %.10f Voltage[%i] = %.6f Current[%i] =  %.5f SOC[%i] =  %.8f SOH_C[%i] =  %.8f SOH_R[%i] =  %.8f \n"  i state.cells.temperature[1] i state.cells.voltage_terminal[1] i state.cells.current[1] i state.cells.soc[1] i state.cells.soh_capacity[1] i state.cells.soh_resistance[1]

        # println("Cell Currents[%i] : ", state.cells.current)

        # print bitstrings (for debugging differences between the C++ and Julia implementations)
        if i == 9999
            @printf "Temperature[%i] =  %s Voltage[%i] = %s Current[%i] =  %s SOC[%i] =  %s SOH_C[%i] =  %s SOH_R[%i] =  %s \n"  i bitstring(state.cells.temperature[1]) i bitstring(state.cells.voltage_terminal[1]) i bitstring(state.cells.current[1]) i bitstring(state.cells.soc[1]) i bitstring(state.cells.soh_capacity[1]) i bitstring(state.cells.soh_resistance[1])
            show([bitstring(state.cells.temperature[1]), bitstring(state.cells.voltage_terminal[1]), bitstring(state.cells.current[1]), bitstring(state.cells.soc[1]), bitstring(state.cells.soh_capacity[1]), bitstring(state.cells.soh_resistance[1])])
        end

        if soc_limiting
            println("Flipping current ", i, "[SOH = ", state.cells.soh_capacity[1], "]")
            current = -current
        end
    end
end

"""
    Compare two arrays of bitstrings and return true if they all match, false otherwise. 
    Needed for debugging differences between the C++ and Julia implementations.
"""
function bitcompare(a, b)
    not_matching = []
    for (index, (ai, bi)) in enumerate(zip(a,b))
        if ai != bitstring(bi)
            push!(not_matching, index)
        end
    end
    if isempty(not_matching) 
        return true
    else 
        println("Not matching at indices: ", not_matching)
        return false
    end
end

test_simulator()