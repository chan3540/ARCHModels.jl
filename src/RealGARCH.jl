


"""
    RealGARCH{d, o, p, q, T<:AbstractFloat} <: UnivariateVolatilitySpec{T}
"""
struct RealGARCH{d, o, p, q, T<:AbstractFloat} <: UnivariateXVolatilitySpec{T}
    coefs::Vector{T}
    function RealGARCH{d, o, p, q, T}(coefs::Vector{T}) where {d,o,p,q,T}
        length(coefs) == nparams(RealGARCH{d, o, p, q})  || throw(NumParamError(nparams(RealGARCH{d, o, p, q}), length(coefs)))
        new{d, o, p, q, T}(coefs)
    end
end

"""
    RealGARCH{d,o,p,q}(coefs) -> UnivariateVolatilitySpec

Construct an RealizedGARCH specification with the given parameters.

# Example:
```jldoctest
julia> RealGARCH{1,1,1,1}([-0.1 , -0.1 , 0.1 , 0.8 , 0.1,  -0.1,  -0.1 , 0.1]) 
RealGARCH{1,1,1,1} specification.

─────────────────────────────────────────────────────────
                ω₁    τ₁   τ₂   β    γ     ξ₁   δ₁   δ₂ 
─────────────────────────────────────────────────────────
Parameters:   -0.1  -0.1  0.1  0.8  0.1  -0.1  -0.1  0.1
─────────────────────────────────────────────────────────
```
ω₁ = coefs[1]  
τ₁ = coefs[2]  
τ₂ = coefs[3] 
β = coefs[4] 
γ = coefs[5] 
ξ₁ = coefs[6] 
δ₁₁ = coefs[7] 
δ₁₂ = coefs[8] 
"""

RealGARCH{d,o,p,q}(coefs::Vector{T}) where {d, o, p, q, T}  = RealGARCH{d, o, p, q, T}(coefs)

@inline nparams(::Type{<:RealGARCH{d,o,p,q}}) where {d,o,p,q} = d+o+p+q+5

@inline presample(::Type{<:RealGARCH{d,o,p,q}}) where {d,o,p,q} = max(o,p,q)

Base.@propagate_inbounds @inline function update!(
    ht::AbstractVector{T}, lht::AbstractVector{T}, zt::AbstractVector{T}, ut::AbstractVector{T}, ::Type{<:RealGARCH{d,o,p,q}}, garchcoefs::AbstractVector{T1},t
    ) where {d,o,p,q,T,T1}
    mlht = garchcoefs[(t-1)%d+1]
    @muladd begin
        for i = 1:o
            mlht = mlht + garchcoefs[i+d]*zt[end-i+1]
        end
        for i = 1:p
            mlht = mlht + garchcoefs[i+d+o]*(zt[end-i+1]^2-1)
        end
        for i = 1:q
            mlht = mlht + garchcoefs[i+d+o+p]*lht[end-i+1]
        end
        mlht = mlht + garchcoefs[d+o+p+q+1]*ut[end]
        push!(lht, mlht)
        push!(ht, exp(mlht))
    end
    return nothing
end

Base.@propagate_inbounds @inline function ut_update!(
    ht::AbstractVector{T}, lht::AbstractVector{T}, zt::AbstractVector{T}, ut::AbstractVector{T}, data_X::Vector{T1}, ::Type{<:RealGARCH{d,o,p,q}}, garchcoefs::AbstractVector{T2}, t
    ) where {d,o,p,q,T,T1<:AbstractFloat,T2}
    @muladd begin
        mut = log(data_X[t]) 
        mut = mut - garchcoefs[d+o+p+q+2]
        mut = mut - garchcoefs[d+o+p+q+3]*lht[end]
        mut = mut - garchcoefs[d+o+p+q+4]*zt[end] 
        mut = mut - garchcoefs[d+o+p+q+5]*(zt[end]^2 - 1)
    end
    push!(ut, mut)
    return nothing
end

@inline function uncond(::Type{<:RealGARCH{d, o, p, q}}, coefs::Vector{T}) where {d, o, p, q, T}
    h0 = T(exp(coefs[1]/(1-sum(coefs[d+o+p+1:d+o+p+q]))))
end


function startingvals(spec::Type{<:RealGARCH{d, o, p, q}}, data::Array{T}) where {d, o, p, q, T}
    x0 = zeros(T, d+o+p+q+5)
    x0[d+o+1:d+o+p] .= 0.3/q
    x0[d+o+p+1:d+o+p+q] .= 0.7/q
    x0[1] = log.(mean(data))
    return x0
end


function constraints(::Type{<:RealGARCH{d, o, p, q}}, ::Type{T}) where {d, o, p, q, T}
    lower = zeros(T, d+o+p+q+5)
    upper = zeros(T, d+o+p+q+5)
    lower .=  T(-Inf)
    upper .= T(Inf)
    lower[1] = T(-Inf)
    return lower, upper
end

function coefnames(::Type{<:RealGARCH{d, o, p, q}}) where {d, o, p, q}
    names = Array{String, 1}(undef, d+o+p+q+5)
    names[1:d] .= (i -> "ω"*subscript(i)).([1:d...]) 
    names[d+1:d+o] .= (i -> "τ"*subscript(i)*subscript(1)).([1:o...])
    names[d+o+1:d+o+p] .= (i -> "τ"*subscript(i)*subscript(2)).([1:p...])
    names[d+o+p+1:d+o+p+q] .= (i -> "β"*subscript(i)).([1:q...])
    names[d+o+p+q+1] = "γ"
    names[d+o+p+q+2] = "ξ"
    names[d+o+p+q+3] = "ϕ"
    names[d+o+p+q+4] = "δ"*subscript(1)
    names[d+o+p+q+5] = "δ"*subscript(2)
    return names
end

