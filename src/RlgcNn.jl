module RlgcNn

using Flux, JLD2, StaticArrays

struct SsRlgcModel{T}
    model::T
end

Flux.@functor SsRlgcModel

SsRlgcModel(; N=16) = SsRlgcModel(Chain(Dense(2 => N, tanh), BatchNorm(N), Dense(N => 4, softplus)))

const PretrainedSsRlgcModel = SsRlgcModel()

state = JLD2.load("$(@__DIR__)/../data/ss_model.jld2", "state")
Flux.loadmodel!(PretrainedSsRlgcModel, state)

# Un-scaled functor
function (m::SsRlgcModel)(width::Real, frequency::Real)
    m.model(@SMatrix [width*1e3; frequency/1e9;;]) .* @SVector [1, 1e-6, 1e-6, 1e-12]
end

export SsRlgcModel, PretrainedSsRlgcModel

end