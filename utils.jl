# integrations, sums and derivatives utils


∑(v::Vector, i::Int64=1, j::Int64=length(v)) = sum(v[n] for n in i:j)
∑(v::Vector, w::Vector, i::Int64=1, j::Int64=length(v), k::Int64=1, l::Int64=length(v)) = sum(v[n] * w[m] for n in i:j for m in k:l)
# d is the nearest neighbor parameter (we use PBC)
∑(v::Vector, w::Vector, i::Int64=1, j::Int64=length(v); d::Int64=1) = sum(v[n] * w[mod1(n+m, j)] for n in i:j for m in 1:d)

∫D(ϕ::Vector, dx) = dx * ∑(ϕ)

# to implement the Hamiltonian we need the discrete derivatives 

function ∂(v::Vector, dx; bc="pbc")
    n = length(v)
    if bc == "pbc"
        return [(v[mod1(i+1, n)] - v[i]) / dx for i in 1:n]
    elseif bc == "obc"
        return [(v[i+1] - v[i]) / dx for i in 1:n-1]
    end
end
