struct BatteryCellsState
    voltage_terminal::Vector{Float64}               # Terminal cell voltage (in V), this is cell voltage in the dataset
    voltage_capacitor::Vector{Float64}              # Voltage across capacitor (in V)
    current::Vector{Float64}                        # Battery cell current (in A)
    temperature::Vector{Float64}                    # Battery cell temperature (in K)
    soc::Vector{Float64}                            # State of charge (in percent)
    internal_res::Vector{Float64}                   # Internal resistance
    soh_capacity::Vector{Float64}                   # State of health of battery cell (based on capacity fade)
    soh_resistance::Vector{Float64}                 # State of health of battery cell (based on resistance growth)

    calendric_aging::Vector{Float64}                # Accumulated time of calendric aging (in h)
    throughput::Vector{Float64}                     # Throughput (in Ah)
    energy_charged::Vector{Float64}                 # Total charged energy (in Wh)
    energy_discharged::Vector{Float64}              # Total dis-charged energy (in Wh)
end

BatteryCellsState(; initial_temp::Float64=20.0 + 273.15, no_cells=14, soh_capacity=0.7, soh_resistance=2.15, soc=0.15) = BatteryCellsState(
    repeat([3.681], no_cells),
    repeat([0.0], no_cells),
    repeat([0.0], no_cells),
    repeat([initial_temp], no_cells),
    repeat([soc], no_cells),
    repeat([0.0], no_cells),
    repeat([soh_capacity], no_cells),
    repeat([soh_resistance], no_cells),
    repeat([2000.0], no_cells),
    repeat([0.0], no_cells),
    repeat([0.0], no_cells),
    repeat([0.0], no_cells)
)

BatteryCellsState(;
    no_cells::Int=14,
    temperature::Vector{Float64}=repeat([20.0 + 273.15], 14),
    soc::Vector{Float64}=repeat([0.15], 14),
    soh_capacity::Vector{Float64}=repeat([0.7], 14),
    soh_resistance::Vector{Float64}=repeat([2.15], 14)
) = BatteryCellsState(
    repeat([3.681], no_cells),
    repeat([0.0], no_cells),
    repeat([0.0], no_cells),
    temperature,
    soc,
    repeat([0.0], no_cells),
    soh_capacity,
    soh_resistance,
    repeat([2000.0], no_cells),
    repeat([0.0], no_cells),
    repeat([0.0], no_cells),
    repeat([0.0], no_cells)
)