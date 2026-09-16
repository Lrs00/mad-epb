# Exact finite-sample distribution of the mean absolute deviation
# about the sample mean under Gaussian sampling.

using Distributions
using StatsFuns: logsumexp
using ApproxFun

import Distributions: pdf, logpdf, insupport, minimum, maximum

struct MADDistribution{T<:Real,S<:Integer} <: ContinuousUnivariateDistribution
    σ::T
    n::S
end

"""
    G_recursive(max_r; a=0.0, b=100)

Recursive functions used in the exact Gaussian finite-sample density
of the mean absolute deviation about the sample mean.
"""
function G_recursive(max_r::Int; a=0.0, b=100)
    d = Interval(a, b)
    G = Vector{Fun}(undef, max_r + 1)
    G[1] = Fun(x -> 1.0, d)

    for r in 1:max_r
        integrand = Fun(x -> exp(-(x^2) / (2r * (r + 1))), d) * G[r]
        G[r + 1] = cumsum(integrand)
    end

    return G
end

# Cache because the recursion depends only on n.
const _Gcache = Dict{Int,Vector{Fun}}()

_getG(n::Int) = get!(_Gcache, n) do
    G_recursive(n)
end

"""
    _pdf_sigma1(n, m)

Exact density of D = mean(abs.(Y .- mean(Y))) at m when σ = 1.
"""
function _pdf_sigma1(n::Int, m::Real)
    m < 0 && return zero(float(m))

    z   = n * m / 2
    cst = n^(3 / 2) / (2^((n + 1) / 2) * π^((n - 1) / 2))
    G   = _getG(n)

    s = 0.0
    @inbounds for k in 1:(n - 1)
        s += binomial(n, k) *
             exp(-(m^2 * n^3) / (8k * (n - k))) *
             G[k](z) * G[n - k](z)
    end

    return cst * s
end

insupport(::MADDistribution, x::Real) = x >= 0
minimum(::MADDistribution) = 0.0
maximum(::MADDistribution) = Inf

# Scale-family representation:
# f_D(d | σ, n) = (1/σ) f_D(d/σ | 1, n).
pdf(d::MADDistribution, m::Real) =
    m < 0 ? 0.0 : (1 / d.σ) * _pdf_sigma1(d.n, m / d.σ)

function logpdf(d::MADDistribution, m::Real)
    m < 0 && return -Inf

    σ, n = d.σ, d.n
    z = n * m / (2σ)

    logC =
        (3 / 2) * log(n) -
        ((n + 1) / 2) * log(2) -
        ((n - 1) / 2) * log(π)

    G = _getG(n)
    logs = Float64[]

    @inbounds for k in 1:(n - 1)
        val1 = G[k](z)
        val2 = G[n - k](z)

        if val1 > 0 && val2 > 0
            push!(
                logs,
                log(binomial(n, k)) -
                (m^2 * n^3) / (8σ^2 * k * (n - k)) +
                log(val1) + log(val2)
            )
        end
    end

    return -log(σ) + logC +
           (isempty(logs) ? -Inf : logsumexp(logs))
end
