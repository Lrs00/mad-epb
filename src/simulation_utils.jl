# Reusable simulation helpers and evaluation metrics.

using Distributions
using MultipleTesting
using Random
using Statistics

# ============================================================
# Error generators used in the simulation study
# ============================================================

"""Variance-one Uniform errors."""
function rand_scaled_uniform_errors(rng::AbstractRNG, n::Int)
    return rand(rng, Uniform(-sqrt(3), sqrt(3)), n)
end

"""Contaminated Gaussian errors: (1-eps)N(0,1) + eps N(0,tau^2)."""
function rand_mixture_normal_errors(
    rng::AbstractRNG,
    n::Int;
    eps::Real,
    tau::Real
)
    z = Vector{Float64}(undef, n)

    @inbounds for j in 1:n
        z[j] = rand(rng) < eps ?
            rand(rng, Normal(0.0, tau)) :
            rand(rng, Normal(0.0, 1.0))
    end

    return z
end

"""Contaminated Laplace errors with baseline variance one."""
function rand_mixture_laplace_errors(
    rng::AbstractRNG,
    n::Int;
    eps::Real,
    tau::Real
)
    b_main = 1 / sqrt(2)
    b_out  = tau / sqrt(2)
    z = Vector{Float64}(undef, n)

    @inbounds for j in 1:n
        z[j] = rand(rng) < eps ?
            rand(rng, Laplace(0.0, b_out)) :
            rand(rng, Laplace(0.0, b_main))
    end

    return z
end

# ============================================================
# Multiple-testing and performance metrics
# ============================================================

"""Apply Benjamini-Hochberg and return adjusted p-values and rejections."""
function bh_rejections(pvalues; alpha::Real = 0.05)
    adjusted = adjust(pvalues, BenjaminiHochberg())
    reject = adjusted .<= alpha

    return (
        adjusted_pvalues = adjusted,
        reject = reject
    )
end

"""TP, FP, power, and replicate-level FDP."""
function rejection_metrics(reject, is_DE)
    TP = sum(reject .& is_DE)
    FP = sum(reject .& .!is_DE)

    power = TP / max(sum(is_DE), 1)
    fdp = FP / max(sum(reject), 1)

    return (
        TP = TP,
        FP = FP,
        Power = power,
        FDP = fdp
    )
end

"""Top-k true positives, precision, and recall."""
function topk_metrics(pvalues, is_DE; k::Int = 100)
    k_eff = min(k, length(pvalues))
    idx = partialsortperm(pvalues, 1:k_eff)

    TP = sum(is_DE[idx])
    precision = TP / k_eff
    recall = TP / max(sum(is_DE), 1)

    return (
        TP = TP,
        Precision = precision,
        Recall = recall
    )
end

"""Average precision using p-values as ranking scores (smaller is better)."""
function average_precision(pvalues, is_DE)
    ord = sortperm(pvalues)
    y = is_DE[ord]
    n_pos = sum(y)

    n_pos == 0 && return 0.0

    cum_tp = cumsum(y)
    precision_at_rank = cum_tp ./ (1:length(y))

    return sum(precision_at_rank[y]) / n_pos
end

"""Apply BH and compute rejection/ranking metrics for one replicate."""
function evaluate_pvalues(
    pvalues,
    is_DE;
    alpha::Real = 0.05,
    topk::Int = 100
)
    bh = bh_rejections(pvalues; alpha = alpha)
    rejection = rejection_metrics(bh.reject, is_DE)
    ranking = topk_metrics(pvalues, is_DE; k = topk)

    return (
        adjusted_pvalues = bh.adjusted_pvalues,
        reject = bh.reject,
        TP = rejection.TP,
        FP = rejection.FP,
        Power = rejection.Power,
        FDP = rejection.FDP,
        TP_topk = ranking.TP,
        Precision_topk = ranking.Precision,
        Recall_topk = ranking.Recall,
        AP = average_precision(pvalues, is_DE)
    )
end

"""Store the true DE indicators for each (n, replicate) key."""
function extract_is_DE_store(datasets)
    return Dict(
        key => copy(value.is_DE)
        for (key, value) in datasets
    )
end
