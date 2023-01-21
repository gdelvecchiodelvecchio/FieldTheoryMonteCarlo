# finite temperature quantum montecarlo quantum sine-gordon model

# the partition function in this case is Z = ∫Dx(τ) exp[- ∫^β_0 dτ 1/2 (∂_τ ϕ)^2 + 1/2 (∂_x ϕ)^2  + V(x(τ))]

# the field will be a Matrix instead of a Vector

using Distributions, Plots, Random, LinearAlgebra

# Space-time grid for 2d simulation. This will be a matrix of coordinates.
# Since to compute any observable I have to average over the euclidean time direction, for performance,
# it is better to take the row index as time and the column as space

grid(n::Int64=100, m::Int64=100; dτ=0.05, dx=dτ) = [(i*dτ, j*dx) for i in 0:n-1, j in 0:m-1]

#integrations

∑∑(v::Matrix, i::Int64=1, n::Int64=size(v)[1], j::Int64=1, m::Int64=size(v)[2]) = sum(v[s, r] for s in i:n, r in j:m)

# double integral

∫D(v::Matrix, dτ=0.05, dx=dτ) = dx * dτ * ∑∑(v)

# to implement the Hamiltonian we need the discrete derivatives 

function ∂(v::Vector, dx=0.05; bc="pbc")
    n = length(v)
    if bc == "pbc"
        return [(v[mod1(i+1, n)] - v[i]) / dx for i in 1:n]
    elseif bc == "obc"
        return [(v[i+1] - v[i]) / dx for i in 1:n-1]
    end
end

function ∂(v::Matrix, dx=0.05; dir="x", bc="pbc")
    if dir == "x"
        if bc == "pbc"
            return mapslices(x->∂(x, dx; bc="pbc"), v; dims=1)
        elseif bc == "obc"
            return mapslices(x->∂(x, dx; bc="obc"), v, dims=1)
        end
    elseif dir == "τ"
        if bc == "pbc"
            return mapslices(x->∂(x, dx; bc="pbc"), v; dims=2)
        elseif bc == "obc"
            return mapslices(x->∂(x, dx; bc="obc"), v, dims=2)
        end
    end
end
    

# Sine-gordon with Euclidean time direction

h2d(ϕ::Matrix, dτ=0.05, dx=dτ; α=1., γ=1.) = 1/2. * abs.(∂(ϕ, dτ, dir="τ")).^2 .+ 1/2. * abs.(∂(ϕ, dx, dir="x")).^2  .- (α^2 / γ^2 * cos.(γ * ϕ) .- 1)
H2d(ϕ::Matrix, dτ=0.05, dx=dτ; α=1., γ=1.) = ∫D(h2d(ϕ, dτ, dx; α, γ), dτ, dx)

#not normalized Boltzmann distribution
P(ϕ::Matrix, dτ=0.05, dx=dτ; α=1., γ=1., β=1) = exp(- β * H(ϕ, dτ, dx; α, γ))

gaussian2d(τ, x; n::Int64=100, m::Int64=100, dτ=0.05, dx=0.05) = 1/ sqrt(2π * ((dx*n)^2 + (dτ*m)^2)/36) * exp(-((x - (dx*n)/2)^2 + (τ - (dx*n)/2)^2) / ((dx*n)^2 /36 + (dτ*m)^2 /36))


init_field(n::Int64=100, m::Int64=100; dτ=0.05, dx=dτ) = grid(n, m; dτ, dx) .|> x -> gaussian2d(x...; n=n, m=m, dτ=dτ, dx=dx)
random_field(n::Int64=100, m::Int64=100; dτ=0.05, dx=dτ) = init_field(n, m; dτ=dτ, dx=dx) .+ rand(Normal(0,0.001), n, m)

function cumulant(ϕ::Matrix, i::Int64; n=size(ϕ)[2], m=2) 
    ϕ_ave = reshape(mapslices(mean, ϕ; dims=2), (n,))
    return mean([(ϕ_ave[mod1(i + s, n)] - ϕ[s]).^m for s in 1:n])
end

function sg_montecarlo_equilibrate_forever(dτ=0.05, dx=0.05, σ0=random_field(100, 100; dτ=0.05, dx=dτ); α=1., γ=1., β=1., observables=true)
    n, m = size(σ0)
    ϕ = σ0 # distribution of spins to update
    if observables
        # I want to monitor second and fourth cumulant
        C2 = zeros(m)
        C2_loc = zeros(n, 10^2)
        C2_loc[:, 1] = [cumulant(ϕ, s) for s in 1:m]
        C4 = zeros(m)
        C4_loc = zeros(m, 10^2)
        C4_loc[:, 1] = [cumulant(ϕ, s; m=4) for s in 1:m]
        #C2 = [cumulant(π)] 
        #en = zeros(n, nsteps) 
    end
    acc_rate = 1
    acc_count = 1
    tot_count = 1
    loc_step = 1
    loc_av_step = 1
    variance = 0.01
    gr(show = true) # in IJulia this would be: gr(show = :ijulia)
    display(scatter([1], [mean(C2[s, loc_step]/(s * dx) for s in 1:m)], legend=:none, layout=2))
    display(scatter!([1], [mean(C4[s, loc_step]/(s * dx) for s in 1:m)], legend=:none, subplot=2))
    @inbounds @fastmath while tot_count <= 10^6
        if tot_count % 10^2 == 0
            loc_av_step = 1
            loc_step += 1
            C2 = hcat(C2, reshape(mapslices(mean, C2_loc, dims=[2]), m))
            display(scatter!([loc_step], [mean(C2[s, loc_step]/(s * dx) for s in 1:m)], subplot=1))
            C4 = hcat(C4, reshape(mapslices(mean, C4_loc, dims=[2]), n))
            display(scatter!([loc_step], [mean(C4[s, loc_step]/(s * dx) for s in 1:m)], subplot=2))
        end
        i, j = (rand(DiscreteUniform(1, n)), rand(DiscreteUniform(1, m))) # for multiple updates use rand(DiscreteUniform(1, n), frac)
        ϕnew = copy(ϕ)
        if acc_rate > 0.55
            variance = variance*2
        elseif acc_rate < 0.45
            variance = variance/2
        end
        ϕnew[i, j] = ϕnew[i, j] + rand(Normal(0, variance)) #for multiple updates use .+
        ΔE = H2d(ϕnew, dτ, dx; α, γ) - H2d(ϕ, dτ, dx; α, γ)
        #a = ΔE < 0 ?  P(πnew, dx; α, γ, β) / P(π, dx; α, γ, β) : 1 #min(P(πnew, dx; α, γ, β) / P(π, dx; α, γ, β), 1) 
        #a = min(P(πnew, dx; α, γ, β) / P(π, dx; α, γ, β), 1) 
        a = min(exp(- ΔE * β), 1)
        if rand() < a
            tot_count += 1
            #print("Accept \n")
            acc_count += 1
            ϕ = ϕnew
        else
            tot_count += 1
            #print("Reject \n")
        end
        loc_av_step += 1
        C2_loc[:,loc_av_step-1] = [cumulant(ϕ, s) for s in 1:m]
        C4_loc[:,loc_av_step-1] = [cumulant(ϕ, s; m=4) for s in 1:m]
        acc_rate = acc_count/tot_count
        print("Acceptance rate= $(acc_rate)  \t nsteps=$tot_count \r")
        #sleep(1)
    end
    print("Acceptance rate= $(acc_rate) \n")
    if observables
        return ϕ, C2, C4#, en
    else
        return ϕ
    end
end


sample = sg_montecarlo_equilibrate_forever(0.05, 0.05;  observables=true)
