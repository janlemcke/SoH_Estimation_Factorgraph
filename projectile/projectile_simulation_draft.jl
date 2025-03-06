# first draft of simultaion, the one used in projectile_esiimation_sampling.jl is more recent

using Plots
velocity_y = 7
velocity_x = 7

delta_x(time::Float64, velocity::Float64=100.0, start::Float64=0.0) = velocity * time

delta_y(time::Float64, velocity::Float64=100.0, start::Float64=0.0) = velocity * time + 1/2 * (-9.8) * time^2
    
get_start_velocity_y(time::Float64, delta::Float64) = delta/time - 1/2 * (-9.8) * time

get_start_velocity_x(time::Float64, delta::Float64) = delta/time

function simulate_projectile(velocity_x::Float64=10.0, velocity_y::Float64=velocity_x; start_x::Float64=0.0, start_y::Float64=0.0, time_step::Float64=0.1)
    x = [start_x]
    y = [start_y]
    t = time_step

    while delta_y(t, velocity_y, start_y) > 0 
        append!(x, delta_x(t, velocity_x, start_x))
        append!(y, delta_y(t, velocity_y, start_y))
        t += time_step
    end
    return x, y
end

time_step = 0.0001
x,y = simulate_projectile(10.0, time_step=time_step)
scatter(x,y)
scatter!([last(x)], [last(y)])

starting_velocity = get_start_velocity_x(convert(Float64, length(x))*time_step, last(x)), get_start_velocity_y(convert(Float64, length(y))*time_step, last(y))

println("starting velocity $starting_velocity")

