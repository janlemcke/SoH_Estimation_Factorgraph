struct InternalResistance
    temperatures::Vector{Float64}   # Discrete temperatures (in Kelvin)
    soc::Vector{Float64}            # Discrete state-of-charge
    resistance::Matrix{Float64}     # Internal resistance (in Ohm)
end

function InternalResistance()
    temperatures = [-25., -20., -10., 0., 10., 25., 35., 45., 60.] # Discrete temperatures (in Celsius)

    # Shift the temperature index to Kelvin so we do not run into a units problem when using these tables
    temperatures = [t + 273.15 for t in temperatures]

    soc = [.0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]

    resistance = [
        21.60  25.82  20.30  16.71  13.64  13.41  13.03  12.47  12.66  12.14  12.07  12.24;
        18.40  16.43  15.46  11.13  11.03  10.82  10.83  9.88   10.43  10.13  9.94   9.80;
        10.48  8.96   7.47   6.65   6.73   6.60   6.68   6.24   6.63   6.35   6.30   6.31;
        7.08   5.61   4.70   3.86   3.88   3.77   3.90   3.65   3.97   3.76   3.79   3.91;
        4.85   3.58   2.95   2.41   2.43   2.35   2.38   2.28   2.49   2.33   2.35   2.49;
        3.74   2.04   1.74   1.41   1.41   1.37   1.38   1.38   1.46   1.37   1.36   1.48;
        2.56   1.65   1.39   1.15   1.13   1.11   1.09   1.12   1.17   1.11   1.09   1.18;
        2.14   1.42   1.20   0.98   0.96   0.94   0.92   0.95   0.99   0.94   0.85   0.97;
        1.69   1.06   0.90   0.76   0.74   0.73   0.70   0.74   0.73   0.73   0.69   0.73
    ] # Internal resistance (in mOhm)

    # Convert from mOhm to Ohm and round to achieve parity with Python code
    resistance = round.((resistance ./ 1000.0) .* 1000000) ./ 1000000

    return InternalResistance(temperatures, soc, resistance)
end


function get_resistance(temperature::Float64, soc::Float64, ir::InternalResistance)
    xGrid = ir.temperatures
    yGrid = ir.soc
    fValues = ir.resistance
    x = temperature
    y = soc
    xIndex = 1
    while xIndex < length(xGrid) && x > xGrid[xIndex + 1]
        xIndex += 1
    end
    yIndex = 1
    while yIndex < length(yGrid) && y > yGrid[yIndex + 1]
        yIndex += 1
    end
    x1 = xGrid[xIndex]
    x2 = xGrid[xIndex + 1]
    y1 = yGrid[yIndex]
    y2 = yGrid[yIndex + 1]
    f11 = fValues[xIndex, yIndex]
    f12 = fValues[xIndex, yIndex + 1]
    f21 = fValues[xIndex + 1, yIndex]
    f22 = fValues[xIndex + 1, yIndex + 1]
    f = ((f11 * (x2 - x) * (y2 - y)) + (f21 * (x - x1) * (y2 - y)) + (f12 * (x2 - x) * (y - y1)) + (f22 * (x - x1) * (y - y1))) /
        ((x2 - x1) * (y2 - y1))
    return f
end

function get_analytic_resistance(temperature::Float64, soc::Float64, C::Constants)
    cst_start = 0.2
    temp_abs = (temperature - 273.15) + 10
    temp_dep = C.ir_a * exp(-(C.ir_b * temp_abs)) + C.ir_c
    if soc < cst_start
        return temp_dep + C.ir_d * (soc - cst_start)^2 * (C.ir_e + (70 - temp_abs) / 70)^2
    else
        return temp_dep
    end
end