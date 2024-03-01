using Flux, CSV, CUDA, Plots, Statistics, ProgressMeter, JLD2

const dir = @__DIR__

include(dir * "/../src/RlgcNn.jl")
include("evaluate.jl")

using .RlgcNn

# Prepare training data
file = CSV.File(dir * "/../data/HFSS_2D_SS_RLGC.csv")

x = vcat(file["width_mm"]', file["freq_ghz"]')
y = vcat(file["r_ohm"]', file["l_uH"]', file["g_uS"]', file["c_pf"]')

model = SsRlgcModel() |> gpu

loader = Flux.DataLoader((x, y) |> gpu, batchsize=10149, shuffle=true)
optim = Flux.setup(Flux.Adam(0.01), model)

losses = []
gamma_error = []
z0_error = []

@showprogress for epoch in 1:100_000
    for (x, y) in loader
        loss, grads = Flux.withgradient(model) do m
            Flux.msle(m.model(x), y)
        end
        Flux.update!(optim, model, grads[1])
        push!(losses, loss)
        push!(gamma_error, gamma_mean_percent_error(model, x, y))
        push!(z0_error, z0_mean_percent_error(model, x, y))
    end
end

plot(z0_error; ylims=(0, 20), label="Z₀ %-Error")
plot!(gamma_error, label="Γ %-Error")
last(z0_error)

# Turn off learning so we can use it in other AD contexts
testmode!(model)

state = Flux.state(cpu(model))
jldsave(dir * "/../data/ss_model.jld2"; state)