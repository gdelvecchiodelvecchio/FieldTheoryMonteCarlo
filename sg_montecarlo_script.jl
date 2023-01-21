
using QuadGK, LinearAlgebra, Distributions, Random, Plots

# configuration here is a string of +1 or -1, the spins σ

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

# H = 1/2 ∫ |∂ϕ|^2 -(α^2 / γ^2 cos(γ ϕ) -1)

h(ϕ::Vector, dx=0.05; α=1., γ=1.) = 1/2. * abs.(∂(ϕ, dx)).^2  .- (α^2 / γ^2 * cos.(γ * ϕ) .- 1)
H(ϕ::Vector, dx=0.05; α=1., γ=1.) = ∫D(h(ϕ, dx; α, γ), dx)

#not normalized Boltzmann distribution
P(ϕ::Vector, dx=0.05; α=1., γ=1., β=1) = exp(- β * H(ϕ, dx; α, γ))

init_field(L; dx=0.05) = collect(0:dx:L-dx) |> x->1/ sqrt(2π * L^2/36) .* exp.(-(x .- L/2).^2 / (L^2/36))
random_field(L; dx=0.05) = init_field(L; dx=dx) .+ rand(Normal(0,0.001), Int(round(L/dx)))
random_integer(n::Int64) = rand(DiscreteUniform(1, n))
cumulant(ϕ::Vector, i::Int64; n=length(ϕ), m=2) = mean([(ϕ[mod1(i + s, n)] - ϕ[s]).^m for s in 1:n])


function sg_montecarlo_equilibrate_forever(dx=0.05, σ0=rand(100) .* dx#=random_field(100; dx)=#; tol=10e-2, α=1., γ=1., β=1., observables=true)
    n = length(σ0)
    ϕ = σ0 # distribution of spins to update
    if observables
        # I want to monitor second and fourth cumulant
        C2 = zeros(n)
        C2_loc = zeros(n, 10^2)
        C2_loc[:, 1] = [cumulant(ϕ, s) for s in 1:n]
        C4 = zeros(n)
        C4_loc = zeros(n, 10^2)
        C4_loc[:, 1] = [cumulant(ϕ, s; m=4) for s in 1:n]
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
    display(scatter([1], [0 *sum(C2_loc[s, 1]/(s * dx) for s in 1:n)], legend=:none, layout=2))
    display(scatter!([1], [0 *sum(C4_loc[s, 1]/(s * dx) for s in 1:n)], legend=:none, subplot=2))
    @inbounds @fastmath while tot_count <= 10^6
        if tot_count % 10^2 == 0
            loc_av_step = 1
            loc_step += 1
            C2 = hcat(C2, reshape(mapslices(mean, C2_loc, dims=[2]), n))
            display(scatter!([loc_step], [mean(C2[s, loc_step]/(s * dx) for s in 1:n)], subplot=1))
            C4 = hcat(C4, reshape(mapslices(mean, C4_loc, dims=[2]), n))
            display(scatter!([loc_step], [mean(C4[s, loc_step]/(s * dx) for s in 1:n)], subplot=2))
        end
        j = rand(DiscreteUniform(1, n)) # for multiple updates use rand(DiscreteUniform(1, n), frac)
        ϕnew = copy(ϕ)
        if acc_rate > 0.55
            variance = variance*2
        elseif acc_rate < 0.45
            variance = variance/2
        end
        ϕnew[j] = ϕnew[j] + rand(Normal(0, variance)) #for multiple updates use .+
        ΔE = H(ϕnew, dx; α, γ) - H(ϕ, dx; α, γ)
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
        C2_loc[:,loc_av_step-1] = [cumulant(ϕ, s) for s in 1:n]
        C4_loc[:,loc_av_step-1] = [cumulant(ϕ, s; m=4) for s in 1:n]
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


sample = sg_montecarlo_equilibrate_forever(0.05;  observables=true)

#hard to equilibrate and decrease the acceptance rate let me try local but multiple updates

function sg_montecarlo_equilibrate_forever_multiupdates(dx=0.05, σ0=rand(100) .* dx#=random_field(100; dx)=#; tol=10e-2, α=1., γ=1., β=1., observables=true)
    n = length(σ0)
    ϕ = σ0 # distribution of spins to update
    if observables
        C2 = zeros(n)
        C2_loc = zeros(n, 10^2)
        C2_loc[:, 1] = [cumulant(ϕ, s) for s in 1:n]
        #C2 = [cumulant(π)] 
        #en = zeros(n, nsteps) 
    end
    acc_rate = 1
    acc_count = 1
    tot_count = 1
    nups = 2
    loc_step = 1
    loc_av_step = 1
    variance = 0.01
    d = Int(nups/2)
    gr(show = true) 
    display(scatter([1], [0 *sum(C2_loc[s, 1]/(s * dx) for s in 1:n)], legend=:none))
    while tot_count <= 10^6
        if tot_count % 10^2 == 0
            loc_av_step = 1
            loc_step += 1
            C2 = hcat(C2, reshape(mapslices(mean, C2_loc, dims=[2]), n))
            display(scatter!([loc_step], [mean(C2[s, loc_step]/(s * dx) for s in 1:n)]))
        end
        j = rand(DiscreteUniform(1+nups, n-nups)) # for multiple updates use rand(DiscreteUniform(1, n), frac)
        ϕnew = copy(ϕ)
        if acc_rate > 0.55
            variance = variance*2
        elseif acc_rate < 0.45
            variance = variance/2
        end
        ϕnew[j-d:j+d-1] = ϕnew[j] .+ rand(Normal(0, variance), nups) #for multiple updates use .+
        ΔE = H(ϕnew, dx; α, γ) - H(ϕ, dx; α, γ)
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
        C2_loc[:,loc_av_step-1] = [cumulant(ϕ, s) for s in 1:n]
        acc_rate = acc_count/tot_count
        print("Acceptance rate= $(acc_rate)  \t nsteps=$tot_count \r")
        #sleep(1)
    end
    print("Acceptance rate= $(acc_rate) \n")
    if observables
        return ϕ, C2#, en
    else
        return ϕ
    end
end


sample, mag = sg_montecarlo_equilibrate_forever_multiupdates(0.5, random_field(100); β=1., observables=true)

