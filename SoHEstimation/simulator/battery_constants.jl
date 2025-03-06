struct Constants
    soc_min::Float64                # Minimal state of charge below of which a discharging current is cut to zero
    soc_max::Float64                # Maximal state of charge above of which a charging current is cut to zero

    soc_balancing::Float64          # state of charge of a battery cell where balancing starts
    delta_soc_balancing::Float64    # State of charge difference where balancing is activated
    r_balancing::Float64            # Balancing resistance in Ohm

    r1::Float64                     # Resistance of the RC module for a battery cell
    c1::Float64                     # Capacitance of the RC module for a battery cell

    rg::Float64                     # Universal gas constant (in J/K*mol)
    z::Float64                      # Dimensionless constant coefficient for aging
    ac_cyclic::Float64              # Severity factor for cyclic aging
    ea_cyclic::Float64              # Battery cell activation energy for the capacity fade process due to cyclic aging (in J/mol)
    ac_calendric::Float64           # Severity factor for calendric aging
    ea_calendric::Float64           # Battery cell activation energy for the capacity fade process due to calendric aging (in J/mol)
    ar::Float64                     # Resistance severity factor
    ea_resistance::Float64          # Battery cell activation energy for the resistance growth (in J/mol)

    battery_cell_mass::Float64      # Mass of a battery cell in kg
    battery_cell_width::Float64     # Width of a battery cell in m
    battery_cell_height::Float64    # Height of a battery cell in m
    battery_cell_depth::Float64     # Depth of a battery cell in m
    cp_battery_cell::Float64        # Specific heat capacity at constant pressure in J/kg*K
    h::Float64                      # Heat transfer coefficient for non-moving air in W/m2K
    surface_area_housing::Float64   # Surface housing in m^2
    thickness_housing::Float64      # Wall thickness housing in m
    lambda_housing::Float64         # Thermal conductivity housing material PA6 Nylon in W/mK
    lambda_air::Float64             # Thermal conductivity air gap between battery cells in W/m
    thickness_airgap::Float64       # Thickness airgap between battery cells in m
    ir_a::Float64                   # Parameter A of analytic R0 function
    ir_b::Float64                   # Parameter B of analytic R0 function
    ir_c::Float64                   # Parameter C of analytic R0 function
    ir_d::Float64                   # Parameter D of analytic R0 function
    ir_e::Float64                   # Parameter E of analytic R0 function
    rku_middle::Float64          # Surface convection resistance for battery cells in the middle
    rku_side::Float64            # Surface convection resistance for battery cells on the sides
    rcc::Float64                 # Conduction thermal resistance between two battery cells
end

function Constants(;
    soc_min::Float64=0.15,
    soc_max::Float64=0.95,
    soc_balancing::Float64=0.5,
    delta_soc_balancing::Float64=0.01,
    r_balancing::Float64=47.0,
    r1::Float64=0.000785637,
    c1::Float64=173043.7151,
    rg::Float64=8.314,
    z::Float64=0.48,
    ac_cyclic::Float64=137.0 + 420.0,
    ea_cyclic::Float64=22406.0,
    ac_calendric::Float64=14876.0,
    ea_calendric::Float64=24500.0,
    ar::Float64=320530.0 + 3.6342e3 * exp(0.9179 * 4.0),
    ea_resistance::Float64=51800.0,
    battery_cell_mass::Float64=0.9 * (3.85 / 2.0),
    battery_cell_width::Float64=0.2232,
    battery_cell_height::Float64=0.303,
    battery_cell_depth::Float64=0.0076,
    cp_battery_cell::Float64=800.0,
    h::Float64=3.0,
    surface_area_housing::Float64=0.834554,
    thickness_housing::Float64=0.0025,
    lambda_housing::Float64=0.38,
    lambda_air::Float64=0.0262,
    thickness_airgap::Float64=0.001,
    ir_a::Float64=0.0,
    ir_b::Float64=0.0,
    ir_c::Float64=0.0,
    ir_d::Float64=0.0,
    ir_e::Float64=0.0,
)

    # Heat transfer surface area for battery cells in the middle
    area_middle::Float64 = 2.0 * battery_cell_depth * (battery_cell_height + battery_cell_width)
    # Heat conduction area
    area_conduction::Float64 = battery_cell_height * battery_cell_width
    # Heat transfer surface area for battery cells on the sides
    area_side::Float64 = area_middle + area_conduction

    rku_middle = 1.0 / (h * area_middle)
    rku_side = 1.0 / (h * area_side)
    rcc = thickness_airgap / (area_conduction * lambda_air)

    return Constants(
        soc_min,
        soc_max,
        soc_balancing,
        delta_soc_balancing,
        r_balancing,
        r1,
        c1,
        rg,
        z,
        ac_cyclic,
        ea_cyclic,
        ac_calendric,
        ea_calendric,
        ar,
        ea_resistance,
        battery_cell_mass,
        battery_cell_width,
        battery_cell_height,
        battery_cell_depth,
        cp_battery_cell,
        h,
        surface_area_housing,
        thickness_housing,
        lambda_housing,
        lambda_air,
        thickness_airgap,
        ir_a,
        ir_b,
        ir_c,
        ir_d,
        ir_e,
        rku_middle,
        rku_side,
        rcc
)
end

# Recomputes rku_middle, rku_side and rcc from the geometry of the battery cell as well as conductivity
function recompute_convection_and_conduction(constants::Constants)
    # Heat transfer surface area for battery cells in the middle
    area_middle::Float64 = 2.0 * constants.battery_cell_depth * (constants.battery_cell_height + constants.battery_cell_width)
    # Heat conduction area
    area_conduction::Float64 = constants.battery_cell_height * constants.battery_cell_width
    # Heat transfer surface area for battery cells on the sides
    area_side::Float64 = area_middle + area_conduction

    constants.rku_middle = 1.0 / (constants.h * area_middle)
    constants.rku_side = 1.0 / (constants.h * area_side)
    constants.rcc = constants.thickness_airgap / (area_conduction * constants.lambda_air)
end