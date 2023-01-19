# random fields

using Distributions, Random

init_field(L; dx=0.05) = collect(0:dx:L-dx) |> x->1/ sqrt(2π * L^2/36) .* exp.(-(x .- L/2).^2 / (L^2/36))
random_field(L; dx=0.05) = init_field(L; dx=dx) .+ rand(Normal(0,0.001), Int(round(L/dx)))
random_integer(n::Int64) = rand(DiscreteUniform(1, n))
