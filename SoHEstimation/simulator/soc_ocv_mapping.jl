include("splines.jl")

struct SOCOCVMapping
    soc::Vector{Float64}
    ocv::Vector{Float64}
    splines::Vector{SplineSet}
end

function SOCOCVMapping()

    mapping = [
        0.0    3.0;
        0.05   3.345195448;
        0.0956 3.524044168;
        0.1412 3.668789347;
        0.1868 3.756903848;
        0.2324 3.792111367;
        0.278  3.811224443;
        0.3236 3.829726734;
        0.3692 3.854033465;
        0.4148 3.883274149;
        0.4604 3.911271536;
        0.506  3.935296102;
        0.5516 3.955656351;
        0.5972 3.971966243;
        0.6428 3.986217049;
        0.6884 3.998405745;
        0.734  4.009954179;
        0.7796 4.020956794;
        0.8252 4.032438866;
        0.8708 4.045250608;
        0.9164 4.059940448;
        0.962  4.077624842;
        1.0    4.12
    ]

    return SOCOCVMapping(
        mapping[:, 1],
        mapping[:, 2],
        spline(mapping[:, 1], mapping[:, 2])
    )
end


# Computes the interpolated value for OCV for a given soc in [0,1]
function get_ocv(soc::Float64, mapping::SOCOCVMapping)
    i = 1
    while soc > mapping.soc[i]
        i += 1
    end
    if soc == mapping.soc[i]
        return mapping.ocv[i]
    end
    s = mapping.splines[i-1]
    return s.a + s.b * (soc - s.x) + s.c * (soc - s.x)^2 + s.d * (soc - s.x)^3
end