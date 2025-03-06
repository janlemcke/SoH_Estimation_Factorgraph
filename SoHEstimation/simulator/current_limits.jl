struct CurrentLimits
    temperatures::Vector{Float64}
    soc::Vector{Float64}
    input_limits::Matrix{Float64}
    output_limits::Matrix{Float64}
end

function CurrentLimits()
    temperatures = [10., 15., 20., 25., 30., 35., 40., 41., 42., 43., 44., 45., 46.]
    temperatures = [t + 273.15 for t in temperatures]

    soc = [.1, .2, .3, .4, .5, .6, .7, .8, .9]

    input_limits = [
        25. 19. 17. 15. 14. 13. 12. 11. 6.;
        25. 25. 21. 19. 17. 17. 15. 13. 7.;
        25. 25. 25. 22. 20. 20. 18. 15. 8.;
        25. 25. 25. 25. 24. 23. 20. 17. 9.;
        25. 25. 25. 25. 25. 25. 24. 18. 10.;
        25. 25. 25. 25. 25. 25. 25. 20. 12.;
        19. 19. 19. 19. 19. 19. 19. 15. 9.;
        16. 16. 16. 16. 16. 16. 16. 13. 7.;
        12. 12. 12. 12. 12. 12. 12. 10. 6.;
        9. 9. 9. 9. 9. 9. 9. 8. 4.;
        6. 6. 6. 6. 6. 6. 6. 5. 3.;
        3. 3. 3. 3. 3. 3. 3. 2. 1.;
        0. 0. 0. 0. 0. 0. 0. 0. 0.;
    ]

    output_limits = [
        50. 50. 50. 50. 50. 50. 50. 50. 50.;
        50. 50. 50. 50. 50. 50. 50. 50. 50.;
        50. 50. 50. 50. 50. 50. 50. 50. 50.;
        50. 50. 50. 50. 50. 50. 50. 50. 50.;
        50. 50. 50. 50. 50. 50. 50. 50. 50.;
        50. 50. 50. 50. 50. 50. 50. 50. 50.;
        38. 38. 38. 38. 38. 38. 38. 38. 38.;
        31. 31. 31. 31. 31. 31. 31. 31. 31.;
        25. 25. 25. 25. 25. 25. 25. 25. 25.;
        19. 19. 19. 19. 19. 19. 19. 19. 19.;
        13. 13. 13. 13. 13. 13. 13. 13. 13.;
        6. 6. 6. 6. 6. 6. 6. 6. 6.;
        0. 0. 0. 0. 0. 0. 0. 0. 0.
    ]

    return CurrentLimits(temperatures, soc, input_limits, output_limits)
end

# Computes the nearest current limits for a temperature/SOC pair
function get_current_limits(temperature::Float64, soc::Float64, current_limits:: CurrentLimits)
    # Determine the closest temperature in the table
    min_temp_idx = findmin(abs.(current_limits.temperatures .- temperature))[2]

    # Determine the closest state-of-charge in the table
    min_soc_idx = findmin(abs.(current_limits.soc .- soc))[2]

    return (current_limits.input_limits[min_temp_idx, min_soc_idx], current_limits.output_limits[min_temp_idx, min_soc_idx])
end
