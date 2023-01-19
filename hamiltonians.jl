# Hamiltonians and Distributions

# Sine-gordon H = 1/2 ∫ |∂ϕ|^2 -(α^2 / γ^2 cos(γ ϕ) -1)

h(ϕ::Vector, dx; α=1., γ=1.) = abs.(∂(ϕ, dx)).^2  .- (α^2 / γ^2 * cos.(γ * ϕ) .- 1)
H(ϕ::Vector, dx; α=1., γ=1.) = 1/2. * ∫D(h(ϕ, dx; α, γ), dx)

