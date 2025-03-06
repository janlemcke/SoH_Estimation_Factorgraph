include("battery_constants.jl")
include("battery_cell.jl")
include("soc_ocv_mapping.jl")
include("soh_cr_mapping.jl")
include("current_limits.jl")
include("internal_resistance.jl")
using DataFrames
using CSV

struct BatteryPack
    no_cells::Int64                         # Number of battery cells that form a battery pack
    constants::Constants                    # Constants for the simulation
    internal_resistance::InternalResistance # Mapping for internal resistance from temperature and state-of-charge
    current_limits::CurrentLimits           # Mapping for input & output current limits from temperature and state-of-charge
    sococv_mapping::SOCOCVMapping           # Mapping from state-of-charge to open circuit voltage
    sohcr_mapping::SOHCRMapping             # Mapping from capacity state-of-health to resistance state-of-health
    battery_cell_capacity::Float64          # Usable capacity of a battery cell in Ah
end

Base.broadcastable(b::BatteryPack) = Ref(b)

BatteryPack(;
    constants::Constants=Constants(),
    internal_resistance::InternalResistance=InternalResistance(),
    current_limits::CurrentLimits=CurrentLimits(),
    sococv_mapping::SOCOCVMapping=SOCOCVMapping(),
    sohcr_mapping::SOHCRMapping=SOHCRMapping(),
    no_cells::Int64=14,
    battery_cell_capacity::Float64=66.0
) = BatteryPack(
    no_cells,
    constants,
    internal_resistance,
    current_limits,
    sococv_mapping,
    sohcr_mapping,
    battery_cell_capacity,
)

struct BatteryPackState
    cells::BatteryCellsState              # Reference to the current state of all battery cells of the battery pack
    temperature_housing::Float64            # In-housing temperature (in K)
end

BatteryPackState(; battery_cells_state=BatteryCellsState(), initial_housing_temp::Float64=20.0 + 273.15) = BatteryPackState(battery_cells_state, initial_housing_temp)

# # Sets the initial state of health (capacity-wise) for all cells
# function set_initial_soh(battery_pack::BatteryPackState, initial_soh::Float64)
#     @assert (initial_soh > 0.0) && (initial_soh <= 1.0)
#     for c in battery_pack.curr
#         c.soh_capacity = initial_soh
#     end
# end

# # Sets the initial state of health (resistance-wise) for all cells
# function set_initial_sohr(battery_pack::BatteryPackState, initial_sohr::Float64)
#     @assert initial_sohr >= 0.0
#     for c in battery_pack.curr
#         c.soh_resistance = initial_sohr
#     end
# end

# # Sets the initial state of health (capacity-wise) for all cells (with varying SOH per cell)
# function set_initial_soh(battery_pack::BatteryPackState, initial_soh::Vector{Float64})
#     @assert length(initial_soh) == length(battery_pack.curr)
#     for i in eachindex(initial_soh)
#         @assert (initial_soh[i] > 0.0) && (initial_soh[i] <= 1.0)
#         battery_pack.battery_pack_state.cells[i].set_soh_capacity(initial_soh[i], battery_pack.sohcr_mapping)
#     end
# end

# function set_initial_sohr(battery_pack::BatteryPackState, initial_sohr::Vector{Float64})
#     @assert length(initial_sohr) == length(battery_pack.curr)
#     for i in eachindex(initial_sohr)
#         @assert initial_sohr[i] > 0.0
#         battery_pack.battery_pack_state.cells[i].soh_resistance = initial_sohr[i]
#     end
# end

# # Sets the initial cell temperature
# function set_initial_cell_temperature(battery_pack::BatteryPackState, initial_cell_temperature::Float64)
#     for c in battery_pack.curr
#         c.temperature = initial_cell_temperature
#     end
# end

# function set_initial_cell_temperature(battery_pack::BatteryPackState, initial_cell_temperature::Vector{Float64})
#     @assert length(initial_cell_temperature) == length(battery_pack.curr)
#     for i in eachindex(initial_cell_temperature)
#         battery_pack.battery_pack_state.cells.temperature[i] = initial_cell_temperature[i]
#     end
# end

# function set_soc(battery_pack::BatteryPackState, soc::Vector{Float64})
#     @assert length(soc) == length(battery_pack.curr)
#     for i in eachindex(soc)
#         @assert (soc[i] >= 0.0) && (soc[i] <= 1.0)
#         battery_pack.battery_pack_state.cells[i].set_soc(soc[i], battery_pack.sococv_mapping)
#     end
# end

# Limits the current based on any battery cell reaching their SOC limit
function socLimit(battery_pack::BatteryPack, battery_pack_state::BatteryPackState, current::Float64)
    # Discharging
    if current > 0.0
        for i in 1:battery_pack.no_cells
            if battery_pack_state.cells.soc[i] <= battery_pack.constants.soc_min
                return 0.0
            end
        end
    # Charging
    elseif current < 0.0
        for i in 1:battery_pack.no_cells
            if battery_pack_state.cells.soc[i] >= battery_pack.constants.soc_max
                return 0.0
            end
        end
    end
    return current
end

function parallel_balancing!(next::BatteryCellsState, current::Float64)
    next.current .= current
end

function passive_balancing!(next, current, battery_pack_state, battery_pack)
    if current != 0.0
        return
    end

    if any(>=(battery_pack.constants.soc_balancing), battery_pack_state.cells.soc)
        min_soc = minimum(battery_pack_state.cells.soc)
        for i in 1:battery_pack.no_cells
            if battery_pack_state.cells.soc[i] - min_soc >= battery_pack.constants.delta_soc_balancing
                r0 = get_analytic_resistance(battery_pack_state.cells.temperature[i], battery_pack_state.cells.soc[i], battery_pack.constants)
                voltage_oc = get_ocv(battery_pack_state.cells.soc[i], battery_pack.sococv_mapping)
                next.current[i] = (voltage_oc + battery_pack_state.cells.voltage_capacitor[i] + battery_pack.constants.r_balancing * next.current[i]) /
                                  (r0 * battery_pack_state.cells.soh_resistance[i] + battery_pack.constants.r_balancing)
            end
        end
    end
end

function apply_current_limits!(next::BatteryCellsState, battery_pack_state::BatteryPackState, battery_pack::BatteryPack)
    limitCounter = 0

    for i in 1:battery_pack.no_cells
        limits = get_current_limits(battery_pack_state.cells.temperature[i], battery_pack_state.cells.soc[i], battery_pack.current_limits)
        current = next.current[i]
        # Check that we are not overcharging and (possibly) limit charging current
        if (current < 0.0) && (-current > limits[1])
            next.current[i] = -limits[1]
            limitCounter += 1
        end

        # Check that we are not overdischarging and (possibly) limit discharging current
        if (current > 0.0) && (current > limits[2])
            next.current[i] = limits[2]
            limitCounter += 1
        end
    end

    limitCounter
end

function electrical_model_forward!(delta_time::Float64, next::BatteryCellsState, battery_pack::BatteryPack, battery_pack_state::BatteryPackState; verbose=true)

    df = DataFrame(
        "I" => battery_pack_state.cells.current[1],
        "SoC" => battery_pack_state.cells.soc[1],
        "SoH" => battery_pack_state.cells.soh_capacity[1],
        "timestamp" => Float64(0),
        "DSoC" => Float64(0),  # Placeholder for now
        "DQ" => Float64(0),
    )


    for i in 1:battery_pack.no_cells
        # Retrieve the internal resistance of the battery cell
        r0 = get_analytic_resistance(battery_pack_state.cells.temperature[i], battery_pack_state.cells.soc[i], battery_pack.constants)
        next.internal_res[i] = r0

        # Update state-of-charge of the cell # formula 2.2 in simons' thesis
        delta_soc = (-next.current[i] / (battery_pack_state.cells.soh_capacity[i] * battery_pack.battery_cell_capacity)) * delta_time # Formula (23) - Cordoba (2015)
        next.soc[i] = battery_pack_state.cells.soc[i] + delta_soc

        if !(0.0 <= next.soc[i] <= 1.0)
            if verbose
                println("Warning: clamping soc value ", next.soc[i])
            end
            next.soc[i] = clamp(next.soc[i], 0.0, 1.0)
        end

        # Update voltages
        #   1. Determine open circuit voltage as a function of the state of charge, simon thesis formula 2.5
        voltage_oc = get_ocv(next.soc[i], battery_pack.sococv_mapping) # look up table with interpolation
        #   2. Compute voltage across capacitor; Formula (15) - Andersson (2017)
        exp_term = exp((-delta_time * 3600.0) / (battery_pack.constants.r1 * battery_pack.constants.c1))
        next.voltage_capacitor[i] = (battery_pack_state.cells.voltage_capacitor[i] * exp_term) + (battery_pack.constants.r1 * (1 - exp_term) * next.current[i])
        #   3. Compute terminal voltage; Formula (2) in Cordoba (2015)
        next.voltage_terminal[i] = voltage_oc - battery_pack_state.cells.soh_resistance[i] * r0 * next.current[i] - next.voltage_capacitor[i]

        if i == 1
            df[1, :DSoC] = delta_soc  # Directly update the first row
        end
    end

    return df
end

function thermal_model_forward!(delta_time::Float64, air_temp::Float64, next::BatteryCellsState, battery_pack::BatteryPack, battery_pack_state::BatteryPackState)::Float64
    qku_total = 0.0

    for i in 1:battery_pack.no_cells
        # Conduction to the left
        qcc_left = (i == 1) ? 0.0 : ((battery_pack_state.cells.temperature[i] - battery_pack_state.cells.temperature[i-1]) / battery_pack.constants.rcc)
        # Conduction to the right
        qcc_right = (i == (battery_pack.no_cells)) ? 0.0 : ((battery_pack_state.cells.temperature[i] - battery_pack_state.cells.temperature[i+1]) / battery_pack.constants.rcc)

        # Convection
        qku = (battery_pack_state.cells.temperature[i] - battery_pack_state.temperature_housing) / (((i == 1) || (i == battery_pack.no_cells)) ? battery_pack.constants.rku_side : battery_pack.constants.rku_middle)

        # Total heat dissipated by the battery cell
        q_all = qcc_left + qcc_right + qku
        # Heat generated by each battery cell (Formula 4) - A. Cordoba et al. (2015)
        q_g = next.current[i] * (get_ocv(battery_pack_state.cells.soc[i], battery_pack.sococv_mapping) - battery_pack_state.cells.voltage_terminal[i])

        # Temperature of each battery cell (Formula 23) - A. Cordoba et al. (2015)
        next.temperature[i] = battery_pack_state.cells.temperature[i] + (q_g - q_all) / (battery_pack.constants.battery_cell_mass * battery_pack.constants.cp_battery_cell) * delta_time * 3600.0
        # Total heat dissipated by convection by all battery cells
        qku_total += qku
    end

    # Temperature in the housing Area
    return (qku_total * battery_pack.constants.thickness_housing) / (battery_pack.constants.lambda_housing * battery_pack.constants.surface_area_housing) + air_temp

end

function aging_model_forward!(delta_time::Float64, next::BatteryCellsState, battery_pack::BatteryPack, battery_pack_state::BatteryPackState; use_c_pow=true, df=nothing)
    for i in 1:battery_pack.no_cells
        if next.current[i] == 0.0
            # Calendric aging
            exp_term = exp(-battery_pack.constants.ea_calendric / (battery_pack.constants.rg * battery_pack_state.cells.temperature[i]))
            next.calendric_aging[i] = battery_pack_state.cells.calendric_aging[i] + delta_time
            capacity_loss = battery_pack.constants.ac_calendric * exp_term * (sqrt(next.calendric_aging[i] / 24.0) - sqrt(battery_pack_state.cells.calendric_aging[i] / 24.0))
            if i == 1 && df != nothing
                df[1, :DQ] = capacity_loss / 100.0
            end
            next.soh_capacity[i] = battery_pack_state.cells.soh_capacity[i] - capacity_loss / 100.0
            next.soh_resistance[i] = battery_pack_state.cells.soh_resistance[i]
            next.throughput[i] = battery_pack_state.cells.throughput[i]
        else
            # Cyclic aging
            exp_term = exp(-battery_pack.constants.ea_cyclic / (battery_pack.constants.rg * battery_pack_state.cells.temperature[i]))
            delta_throughput = delta_time * abs(next.current[i])
            next.throughput[i] = battery_pack_state.cells.throughput[i] + delta_throughput
            if use_c_pow
                capacity_loss = battery_pack.constants.ac_cyclic * exp_term * ((@ccall pow((10000.0 + next.throughput[i])::Float64, battery_pack.constants.z::Float64)::Float64) - (@ccall pow((10000.0 + battery_pack_state.cells.throughput[i])::Float64, battery_pack.constants.z::Float64)::Float64))
            else
                capacity_loss = battery_pack.constants.ac_cyclic * exp_term * (((10000.0 + next.throughput[i])^battery_pack.constants.z) - ((10000.0 + battery_pack_state.cells.throughput[i])^battery_pack.constants.z))
            end
            resistance_growth = battery_pack.constants.ar * exp(-battery_pack.constants.ea_resistance / (battery_pack.constants.rg * battery_pack_state.cells.temperature[i])) * delta_throughput

            if i == 1 && df != nothing
                df[1, :DQ] = capacity_loss / 100.0
            end
            next.soh_capacity[i] = battery_pack_state.cells.soh_capacity[i] - capacity_loss / 100.0
            next.soh_resistance[i] = battery_pack_state.cells.soh_resistance[i] + resistance_growth / 100.0
            next.calendric_aging[i] = battery_pack_state.cells.calendric_aging[i]
        end
    end
end

function update_energy!(delta_time::Float64, next::BatteryCellsState, battery_pack::BatteryPack, battery_pack_state::BatteryPackState)
    for i in 1:battery_pack.no_cells
        energy_timestep = abs(next.voltage_terminal[i] * next.current[i] * delta_time)
        next.energy_charged[i] = battery_pack_state.cells.energy_charged[i] + energy_timestep * (next.current[i] < 0)
        next.energy_discharged[i] = battery_pack_state.cells.energy_discharged[i] + energy_timestep * (next.current[i] > 0)
    end
end

"""
Simulate one time step of the battery pack. Set use_c_pow to false to use the Julia implementation of the pow function. Might lead to slightly different results form the C++ implementation.
"""
function simulate_time_step(battery_pack::BatteryPack, battery_pack_state::BatteryPackState, delta_time::Float64, current::Float64, air_temp::Float64; use_c_pow=true, verbose=true, filepath="SoHEstimation/simulator/data/battery_pack.csv", iteration=0)::Tuple{BatteryPackState,Bool}
    @assert delta_time > 0.0

    next = BatteryCellsState(; no_cells=battery_pack.no_cells)

    actual_current = socLimit(battery_pack, battery_pack_state, current)

    temperature_housing = battery_pack_state.temperature_housing

    if current != 0.0
        # Compute the next state for all battery cells
        parallel_balancing!(next, actual_current)
        passive_balancing!(next, actual_current, battery_pack_state, battery_pack)
        apply_current_limits!(next, battery_pack_state, battery_pack)

        df = electrical_model_forward!(delta_time, next, battery_pack, battery_pack_state; verbose=verbose)
        temperature_housing = thermal_model_forward!(delta_time, air_temp, next, battery_pack, battery_pack_state)
    end

    aging_model_forward!(delta_time, next, battery_pack, battery_pack_state, use_c_pow=use_c_pow, df=df)

    df[1, :timestamp] = delta_time * iteration # Directly update the first row

    if isfile(filepath)
        CSV.write(filepath, df; append=true)
    else
        CSV.write(filepath, df)
    end

    update_energy!(delta_time, next, battery_pack, battery_pack_state)

    soc_limiting = !iszero(current) && iszero(actual_current)

    return BatteryPackState(next, temperature_housing), soc_limiting
end
