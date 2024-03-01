# Given a model and training data, compare results in real(z0) and real(gamma)

using CircuitNetworks, KernelAbstractions, Tullio

import CircuitNetworks.Z₀
import CircuitNetworks.γ

@. percent_error(y, ŷ) = 100 * abs(y - ŷ) / y

function z0(model::SsRlgcModel, x)
    ŷ = model.model(x) .* gpu([1, 1e-6, 1e-6, 1e-12])
    @tullio z[i] := Z₀(ŷ[1, i], ŷ[2, i], ŷ[3, i], ŷ[4, i], x[2, i] * 1e-9)
end

z0(x, y) = @tullio z[i] := Z₀(y[1, i], y[2, i] * 1e-6, y[3, i] * 1e-6, y[4, i] * 1e-12, x[2, i] * 1e-9)

function gamma(model::SsRlgcModel, x)
    ŷ = model.model(x) .* gpu([1, 1e-6, 1e-6, 1e-12])
    @tullio gamma[i] := γ(ŷ[1, i], ŷ[2, i], ŷ[3, i], ŷ[4, i], x[2, i] * 1e-9)
end

gamma(x, y) = @tullio gamma[i] := γ(y[1, i], y[2, i] * 1e-6, y[3, i] * 1e-6, y[4, i] * 1e-12, x[2, i] * 1e-9)

function z0_mean_percent_error(model, x, y)
    model_z0 = z0(model, x) .|> real
    simul_z0 = z0(x, y) .|> real
    mean(percent_error(simul_z0, model_z0))
end

function gamma_mean_percent_error(model, x, y)
    model_gamma = gamma(model, x) .|> real
    simul_gamma = gamma(x, y) .|> real
    mean(percent_error(simul_gamma, model_gamma))
end


# fs = range(0.1, 5, 51)
# ws = range(0.2, 20.0, step=0.1)
# rs = reshape(file["r_ohm"], (length(fs), length(ws)))
# ls = reshape(file["l_uH"], (length(fs), length(ws))) ./ 1e6
# gs = reshape(file["g_uS"], (length(fs), length(ws))) ./ 1e6
# cs = reshape(file["c_pf"], (length(fs), length(ws))) ./ 1e12