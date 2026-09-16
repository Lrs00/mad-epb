# Non-empirical-Bayes benchmark tests.

using Distributions
using FastGaussQuadrature
using LinearAlgebra
using Statistics

if !isdefined(@__MODULE__, :MADDistribution)
    include("mad_distribution.jl")
end

"""Mean absolute deviation about the sample mean."""
mad_about_mean(x) = mean(abs.(x .- mean(x)))

"""
    build_mad_null_quadrature(n; K=400)

Precompute the exact Gaussian-null distribution needed for the
MAD-normalized statistic

    H = sqrt(n) * mean(Y) / D,

where D is the mean absolute deviation about the sample mean.
If W = D/σ, then under H0, H = Z/W with Z ~ N(0,1)
independent of W.
"""
function build_mad_null_quadrature(n::Int; K::Int = 400)
    x, qweights = gausslegendre(K)

    # (-1,1) -> (0,1)
    u = (x .+ 1) ./ 2
    qweights ./= 2

    # (0,1) -> (0,Inf), w = u/(1-u)
    w_nodes = u ./ (1 .- u)
    jac = 1 ./ (1 .- u).^2

    mad_density = [
        pdf(MADDistribution(1.0, n), w)
        for w in w_nodes
    ]

    weights = qweights .* jac .* mad_density
    weights ./= sum(weights)

    return (
        w = w_nodes,
        weights = weights
    )
end

"""
    mad_H_pvalue_fast(h_obs, quad)

Exact two-sided Gaussian-null p-value for H = sqrt(n)*mean(Y)/D.
"""
function mad_H_pvalue_fast(h_obs::Real, quad)
    x = abs(h_obs)

    tails =
        2 .* ccdf.(
            Normal(),
            x .* quad.w
        )

    p = dot(quad.weights, tails)
    return clamp(p, 0.0, 1.0)
end

"""
    mad_test_pvalues(Y; quad=nothing, K=400)

Compute non-EB MAD-normalized p-values for all genes. `Y[g]` is the
vector of observations for gene g. A common sample size is assumed.
"""
function mad_test_pvalues(Y; quad = nothing, K::Int = 400)
    n = length(first(Y))

    if any(length(y) != n for y in Y)
        error("mad_test_pvalues currently assumes a common sample size.")
    end

    quad === nothing &&
        (quad = build_mad_null_quadrature(n; K = K))

    beta_hat = [mean(y) for y in Y]
    mad_obs  = [mad_about_mean(y) for y in Y]
    H = sqrt(n) .* beta_hat ./ mad_obs

    pvalues = [
        mad_H_pvalue_fast(h, quad)
        for h in H
    ]

    return (
        pvalues = pvalues,
        statistic = H,
        beta_hat = beta_hat,
        mad = mad_obs,
        quadrature = quad
    )
end

"""
    ttest_pvalues(Y)

Ordinary two-sided one-sample t-test p-values for all genes.
"""
function ttest_pvalues(Y)
    n = length(first(Y))

    if any(length(y) != n for y in Y)
        error("ttest_pvalues currently assumes a common sample size.")
    end

    beta_hat = [mean(y) for y in Y]
    s_hat = [std(y, corrected = true) for y in Y]
    t_stats = sqrt(n) .* beta_hat ./ s_hat

    tdist = TDist(n - 1)
    pvalues = 2 .* ccdf.(Ref(tdist), abs.(t_stats))

    return (
        pvalues = pvalues,
        statistic = t_stats,
        beta_hat = beta_hat,
        scale = s_hat
    )
end
