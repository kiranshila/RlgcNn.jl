module RlgcNn

using Flux, JLD2, StaticArrays

struct SsRlgcModel
    model
end

Flux.@functor SsRlgcModel

SsRlgcModel(; N=16) = SsRlgcModel(Chain(Dense(2 => N, tanh), BatchNorm(N), Dense(N => 4, softplus)))

const PretrainedSsRlgcModel = SsRlgcModel()

state = JLD2.load("data/ss_model.jld2", "state")
Flux.loadmodel!(PretrainedSsRlgcModel, state)

export SsRlgcModel, PretrainedSsRlgcModel

# Un-scaled functor
function (m::SsRlgcModel)(width::Real, frequency::Real)
    m.model(@SMatrix [width*1e3; frequency/1e9;;])
end

end