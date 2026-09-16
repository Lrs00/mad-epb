# Empirical partially Bayes procedures based on mean absolute deviation.
# Includes both parametric and nonparametric across-gene scale models.

using Distributions
using Empirikos
using FastGaussQuadrature
using Hypatia
using LinearAlgebra
using Optim
using Statistics
using StatsFuns: logsumexp

if !isdefined(@__MODULE__, :MADDistribution)
    include("mad_distribution.jl")
end

mad_mean(x) = mean(abs.(x .- mean(x)))

# ============================================================
# Parametric MAD EPB
# ============================================================

struct InverseScaledChiSquare{T,S} <: ContinuousUnivariateDistribution
    σ²::T
    ν::S
end

function Distributions.InverseGamma(d::InverseScaledChiSquare)
    InverseGamma(d.ν / 2, d.ν * d.σ² / 2)
end

Distributions.pdf(d::InverseScaledChiSquare, x::Real) =
    pdf(InverseGamma(d), x)

Distributions.logpdf(d::InverseScaledChiSquare, x::Real) =
    logpdf(InverseGamma(d), x)

prior_sigma2(σ0², ν0) = InverseScaledChiSquare(σ0², ν0)

"""Gauss-Legendre nodes and weights on (0,1)."""
function gausslegendre01(K::Int)
    x, w = gausslegendre(K)
    u = (x .+ 1) ./ 2
    w = w ./ 2
    return u, w
end

"""Marginal log-density of one observed MAD under the parametric scale prior."""
function marginal_mad_logdensity_gl(
    D::Float64,
    n::Int,
    σ0²::Float64,
    ν0::Float64,
    u_nodes::Vector{Float64},
    u_weights::Vector{Float64}
)
    prior = prior_sigma2(σ0², ν0)
    logvals = similar(u_nodes)

    @inbounds for j in eachindex(u_nodes)
        u = clamp(u_nodes[j], 1e-12, 1 - 1e-12)
        s2 = u / (1 - u)
        jac = 1 / (1 - u)^2

        logvals[j] =
            logpdf(MADDistribution(sqrt(s2), n), D) +
            logpdf(prior, s2) +
            log(jac) +
            log(u_weights[j])
    end

    return logsumexp(logvals)
end

"""Negative marginal log-likelihood across genes."""
function negloglik_mad_parametric_gl(
    par::Vector{Float64},
    mad_obs::Vector{Float64},
    n::Int,
    u_nodes::Vector{Float64},
    u_weights::Vector{Float64}
)
    η, ξ = par
    σ0² = exp(η)
    ν0  = exp(ξ)

    total = 0.0
    @inbounds for D in mad_obs
        total -= marginal_mad_logdensity_gl(
            D, n, σ0², ν0, u_nodes, u_weights
        )
    end

    return total
end

"""
    fit_mad_parametric_prior_gl(mad_obs, n; K_quad=64, max_iter=80)

Estimate the scaled-inverse-chi-square prior hyperparameters by
marginal maximum likelihood.
"""
function fit_mad_parametric_prior_gl(
    mad_obs::Vector{Float64},
    n::Int;
    K_quad::Int = 64,
    init_logσ0² = log(max(median(mad_obs)^2, 1e-6)),
    init_logν0  = log(10.0),
    max_iter::Int = 80
)
    u_nodes, u_weights = gausslegendre01(K_quad)

    obj(par) = negloglik_mad_parametric_gl(
        par, mad_obs, n, u_nodes, u_weights
    )

    res = optimize(
        obj,
        [init_logσ0², init_logν0],
        NelderMead(),
        Optim.Options(
            iterations = max_iter,
            show_trace = false
        )
    )

    ηhat, ξhat = Optim.minimizer(res)

    return (
        σ0² = exp(ηhat),
        ν0 = exp(ξhat),
        optim = res,
        u_nodes = u_nodes,
        u_weights = u_weights
    )
end

"""Posterior quadrature nodes and weights for σ² given one observed MAD."""
function posterior_sigma2_weights_mad_gl(
    D::Float64,
    n::Int,
    σ0²::Float64,
    ν0::Float64,
    u_nodes::Vector{Float64},
    u_weights::Vector{Float64}
)
    prior = prior_sigma2(σ0², ν0)

    logpost = similar(u_nodes)
    s2_nodes = similar(u_nodes)

    @inbounds for j in eachindex(u_nodes)
        u = clamp(u_nodes[j], 1e-12, 1 - 1e-12)
        s2 = u / (1 - u)
        s2_nodes[j] = s2
        jac = 1 / (1 - u)^2

        logpost[j] =
            logpdf(MADDistribution(sqrt(s2), n), D) +
            logpdf(prior, s2) +
            log(jac) +
            log(u_weights[j])
    end

    m = maximum(logpost)
    w = exp.(logpost .- m)
    w ./= sum(w)

    return s2_nodes, w
end

"""
    mad_parametric_pvalue_gl(beta_tilde, D, n, σ0², ν0, u_nodes, u_weights)

Posterior-averaged two-sided Gaussian null tail probability, where
`beta_tilde = sqrt(n) * beta_hat`.
"""
function mad_parametric_pvalue_gl(
    β_tilde::Float64,
    D::Float64,
    n::Int,
    σ0²::Float64,
    ν0::Float64,
    u_nodes::Vector{Float64},
    u_weights::Vector{Float64}
)
    s2_nodes, w = posterior_sigma2_weights_mad_gl(
        D, n, σ0², ν0, u_nodes, u_weights
    )

    σ_nodes = sqrt.(s2_nodes)
    tails = 2 .* ccdf.(Normal.(0, σ_nodes), abs(β_tilde))

    return clamp(dot(tails, w), 0.0, 1.0)
end

"""Convenience wrapper for the parametric MAD EPB procedure."""
function mad_parametric_pvalues(
    Y;
    K_quad::Int = 100,
    max_iter::Int = 80
)
    n = length(first(Y))

    if any(length(y) != n for y in Y)
        error("mad_parametric_pvalues currently assumes a common sample size.")
    end

    β_hat = [mean(y) for y in Y]
    mad_obs = [mad_mean(y) for y in Y]
    β_tilde = sqrt(n) .* β_hat

    fit = fit_mad_parametric_prior_gl(
        Float64.(mad_obs),
        n;
        K_quad = K_quad,
        max_iter = max_iter
    )

    pvalues = [
        mad_parametric_pvalue_gl(
            Float64(β_tilde[g]),
            Float64(mad_obs[g]),
            n,
            fit.σ0²,
            fit.ν0,
            fit.u_nodes,
            fit.u_weights
        )
        for g in eachindex(Y)
    ]

    return (
        pvalues = pvalues,
        beta_hat = β_hat,
        beta_tilde = β_tilde,
        mad = mad_obs,
        fit = fit
    )
end

# ============================================================
# Nonparametric MAD EPB
# ============================================================

struct MADSample{T,S} <: Empirikos.EBayesSample{T}
    Z::T
    n::S
end

Empirikos.likelihood_distribution(Z::MADSample, σ) =
    MADDistribution(σ, Z.n)

Empirikos.components(
    convexclass::Empirikos.DiscretePriorClass,
    Z::MADSample
) = Empirikos.likelihood_distribution.(
    Ref(Z),
    Empirikos.support(convexclass)
)

"""Posterior-averaged p-value for a fitted discrete scale prior."""
function mad_nonparametric_pvalue(
    β_tilde::Real,
    Z::MADSample,
    prior::Empirikos.DiscreteNonParametric
)
    post = Empirikos.posterior(Z, prior)
    σs = Empirikos.support(post)
    πs = Empirikos.probs(post)

    tails = 2 .* ccdf.(Normal.(0, σs), abs(β_tilde))
    return clamp(dot(tails, πs), 0.0, 1.0)
end

"""Fit the nonparametric mixing distribution for σ using the MAD likelihood."""
function fit_mad_npmle(
    mad_obs::AbstractVector,
    n::Int;
    n_grid::Int = 200
)
    Zs_mad = [MADSample(mad_obs[i], n) for i in eachindex(mad_obs)]

    σ_grid_raw = sort(unique(mad_obs))
    σ_grid = unique(
        quantile(
            σ_grid_raw,
            range(0, 1; length = min(n_grid, length(σ_grid_raw)))
        )
    )

    prior_class = Empirikos.DiscretePriorClass(σ_grid)
    npmle = Empirikos.NPMLE(prior_class, Hypatia.Optimizer)
    fitres = Empirikos.fit(npmle, Zs_mad)
    prior_hat = Empirikos.clean(fitres.prior)

    return (
        prior = prior_hat,
        fit = fitres,
        samples = Zs_mad,
        grid = σ_grid
    )
end

"""Convenience wrapper for the nonparametric MAD EPB procedure."""
function mad_nonparametric_pvalues(Y; n_grid::Int = 200)
    n = length(first(Y))

    if any(length(y) != n for y in Y)
        error("mad_nonparametric_pvalues currently assumes a common sample size.")
    end

    β_hat = [mean(y) for y in Y]
    β_tilde = sqrt(n) .* β_hat
    mad_obs = [mad_mean(y) for y in Y]

    fit = fit_mad_npmle(mad_obs, n; n_grid = n_grid)

    pvalues = [
        mad_nonparametric_pvalue(
            β_tilde[i],
            fit.samples[i],
            fit.prior
        )
        for i in eachindex(Y)
    ]

    return (
        pvalues = pvalues,
        beta_hat = β_hat,
        beta_tilde = β_tilde,
        mad = mad_obs,
        fit = fit
    )
end
