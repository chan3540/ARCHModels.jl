


"""
    pRealGARCH{S, p, q₁, q₂, T<:AbstractFloat} <: UnivariateVolatilitySpec{T}
"""
struct pRealGARCH{S, p, q₁, q₂, T<:AbstractFloat} <: UnivariateVolatilitySpec{T}
    coefs::Vector{T}
    function pRealGARCH{S, p, q₁, q₂, T}(coefs::Vector{T}) where {S, p, q₁, q₂, T}
        length(coefs) == nparams(pRealGARCH{S, p, q₁, q₂})  || throw(NumParamError(nparams(pRealGARCH{S, p, q₁, q₂}), length(coefs)))
        new{S, p, q₁, q₂, T}(coefs)
    end
end

"""
    pRealGARCH{S, p, q₁, q₂}(coefs) -> UnivariateVolatilitySpec

Construct an RealGARCH specification with the given parameters.

# Example:
```jldoctest
julia> pRealGARCH{1,1,1,1}([-0.1 , -0.1 , 0.1 , 0.8 , 0.1,  -0.1,  0.1 , -0.1]) 
RealGARCH{1,1,1,1} specification.

─────────────────────────────────────────────────────────
                ω₁    τ₁   τ₂   β    γ     ξ₁   δ₁   δ₂ 
─────────────────────────────────────────────────────────
Parameters:   -0.1  -0.1  0.1  0.8  0.1  -0.1  0.1  -0.1
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

pRealGARCH{S, p, q₁, q₂}(coefs::Vector{T}) where {S, p, q₁, q₂, T}  = pRealGARCH{S, p, q₁, q₂, T}(coefs)

@inline nparams(::Type{<:pRealGARCH{S, p, q₁, q₂}}) where {S, p, q₁, q₂} = S + p + q₁ + q₂+4

@inline presample(::Type{<:pRealGARCH{S, p, q₁, q₂}}) where {S, p, q₁, q₂} = max(p, q₁, q₂)

const RealGARCH = pRealGARCH{1}

RealGARCH{p, q}(coefs::Vector{T}) where {p, q, T}  = RealGARCH{p, q, q, T}(coefs)



Base.@propagate_inbounds @inline function update!(
    ht::AbstractVector{T}, lht::AbstractVector{T}, zt::AbstractVector{T}, ut::AbstractVector{T}, spec::Type{<:pRealGARCH{S, p, q₁, q₂}}, garchcoefs::AbstractVector{T1},t
    ) where {S, p, q₁, q₂, T, T1}
    mlht = garchcoefs[(t-1)%S+1]
    @muladd begin
        for i = 1:p
            mlht = mlht + garchcoefs[i+S]*lht[end-i+1]
        end
        for i = 1:q₁
            mlht = mlht + garchcoefs[i+S+p]*zt[end-i+1]
        end
        for i = 1:q₂
            mlht = mlht + garchcoefs[i+S+p+q₁]*(zt[end-i+1]^2-1)
        end
        mlht = mlht + garchcoefs[S+p+q₁+q₂+1]*ut[end]
        push!(lht, mlht)
        push!(ht, exp(mlht))
    end
    return nothing
end

Base.@propagate_inbounds @inline function ut_update!(
    ht::AbstractVector{T}, lht::AbstractVector{T}, zt::AbstractVector{T}, ut::AbstractVector{T}, data_X::Vector{T1}, ::Type{<:pRealGARCH{S, p, q₁, q₂}}, garchcoefs::AbstractVector{T2}, t
    ) where {S, p, q₁, q₂, T, T1<:AbstractFloat,T2}
    @muladd begin
        mut = log(data_X[t]) 
        mut = mut - garchcoefs[S+p+q₁+q₂+2]
        mut = mut - lht[end]#garchcoefs[d+o+p+q+3]*lht[end]
        mut = mut - garchcoefs[S+p+q₁+q₂+3]*zt[end] 
        mut = mut - garchcoefs[S+p+q₁+q₂+4]*(zt[end]^2 - 1)
    end
    push!(ut, mut)
    return nothing
end

@inline function uncond(::Type{<:pRealGARCH{S, p, q₁, q₂}}, coefs::Vector{T}) where {S, p, q₁, q₂, T}
    h0 = T(exp(mean(coefs[1:S])/(1-sum(coefs[S+1:S+p]))))
end


function startingvals(spec::Type{<:pRealGARCH{S, p, q₁, q₂}}, data::Array{T}) where {S, p, q₁, q₂, T}
    x0 = zeros(T, S+p+q₁+q₂+4)
    x0[S+1:S+p] .= 0.8/p
    x0[1] = log.(mean(data.^2))
    return x0
end


function constraints(::Type{<:pRealGARCH{S, p, q₁, q₂}}, ::Type{T}) where {S, p, q₁, q₂, T}
    lower = zeros(T, S+p+q₁+q₂+4)
    upper = zeros(T, S+p+q₁+q₂+4)
    lower .=  T(-Inf)
    upper .= T(Inf)
    return lower, upper
end

function coefnames(::Type{<:pRealGARCH{S, p, q₁, q₂}}) where {S, p, q₁, q₂}
    names = Array{String, 1}(undef, S+p+q₁+q₂+4)
    names[1:S] .= (i -> "ω"*subscript(i)).([1:S...]) 
    names[S+1:S+p] .= (i -> "β"*subscript(i)).([1:p...])
    names[S+p+1:S+p+q₁] .= (i -> "τ"*subscript(i)*subscript(1)).([1:q₁...])
    names[S+p+q₁+1:S+p+q₁+q₂] .= (i -> "τ"*subscript(i)*subscript(2)).([1:q₂...])
    names[S+p+q₁+q₂+1] = "γ"
    names[S+p+q₁+q₂+2] = "ξ"
    names[S+p+q₁+q₂+3] = "δ"*subscript(1)
    names[S+p+q₁+q₂+4] = "δ"*subscript(2)
    return names
end

