struct SOHCRMapping
    soh_c::Vector{Float64}
    sohr::Vector{Float64}
end

function SOHCRMapping()
    soh_c = [0.6, 0.7, 0.78, 0.84, 0.9, 0.94, 1.0]
    soh_r = [0.6, 0.7, 0.78, 0.84, 0.9, 0.94, 1.0]

    return SOHCRMapping(soh_c, soh_r)
end

# Computes the interpolated value for state-of-health resistance for a given state-of-health capacity
function getSOHR(soh_c::Float64, mapping::SOHCRMapping)
    if soh_c <= mapping.soh_c[1]
        return mapping.soh_r[1]
    end
    
    if soh_c >= mapping.soh_c[end]
        return mapping.soh_r[end]
    end
    
    i = 1
    while soh_c > mapping.soh_c[i]
        i += 1
    end
    
    if soh_c == mapping.soh_c[i]
        return mapping.soh_r[i]
    end
    
    return mapping.soh_r[i-1] + (soh_c - mapping.soh_c[i-1]) / (mapping.soh_c[i] - mapping.soh_c[i-1]) * (mapping.soh_r[i] - mapping.soh_r[i-1])
end