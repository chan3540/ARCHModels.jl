
struct dRealGARCH{S, p, q₁, q₂, T<:AbstractFloat} <: UnivariateVolatilitySpec{T}
    coefs::Vector{T}
    function dRealGARCH{S, p, q₁, q₂, T}(coefs::Vector{T}) where {S, p, q₁, q₂, T}
        length(coefs) == nparams(dRealGARCH{S, p, q₁, q₂})  || throw(NumParamError(nparams(dRealGARCH{S, p, q₁, q₂}), length(coefs)))
        new{S, p, q₁, q₂, T}(coefs)
    end
end


dRealGARCH{S, p, q₁, q₂}(coefs::Vector{T}) where {S, p, q₁, q₂, T}  = dRealGARCH{S, p, q₁, q₂, T}(coefs)

@inline nparams(::Type{<:dRealGARCH{S, p, q₁, q₂}}) where {S, p, q₁, q₂} = S + p + q₁ + q₂ + 4

@inline presample(::Type{<:dRealGARCH{S, p, q₁, q₂}}) where {S, p, q₁, q₂} = max(p, q₁, q₂)


dRealGARCH{p, q}(coefs::Vector{T}) where {p, q, T}  = dRealGARCH{1, p, q, q, T}(coefs)


@muladd @inline function damper(z)
    return z/sqrt(1+z*z*0.1)
end

Base.@propagate_inbounds @inline function update!(
    ht::AbstractVector{T}, lht::AbstractVector{T}, zt::AbstractVector{T}, ut::AbstractVector{T}, spec::Type{<:dRealGARCH{S, p, q₁, q₂}}, garchcoefs::AbstractVector{T1},t
    ) where {S, p, q₁, q₂, T, T1}
    mlht = garchcoefs[(t-1)%S+1]
    @muladd begin
        for i = 1:p
            mlht = mlht + garchcoefs[i+S]*lht[end-i+1]
        end
        for i = 1:q₁
            mlht = mlht + garchcoefs[i+S+p]*damper(zt[end-i+1])
        end
        for i = 1:q₂
            mlht = mlht + garchcoefs[i+S+p+q₁]*(damper(zt[end-i+1])^2-1)
        end
        mlht = mlht + garchcoefs[S+p+q₁+q₂+1]*damper(ut[end])
        push!(lht, mlht)
        push!(ht, exp(mlht))
    end
    return nothing
end

Base.@propagate_inbounds @inline function ut_update!(
    ht::AbstractVector{T}, lht::AbstractVector{T}, zt::AbstractVector{T}, ut::AbstractVector{T}, data_X::Vector{T1}, ::Type{<:dRealGARCH{S, p, q₁, q₂}}, garchcoefs::AbstractVector{T2}, t
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

@inline function uncond(::Type{<:dRealGARCH{S, p, q₁, q₂}}, coefs::Vector{T}) where {S, p, q₁, q₂, T}
    h0 = T(exp(mean(coefs[1:S])/(1-sum(coefs[S+1:S+p]))))
end


function startingvals(spec::Type{<:dRealGARCH{S, p, q₁, q₂}}, data::Array{T}) where {S, p, q₁, q₂, T}
    x0 = zeros(T, S+p+q₁+q₂+4)
    x0[S+1:S+p] .= 0.8/p
    x0[1] = log.(mean(data.^2))
    return x0
end


function constraints(::Type{<:dRealGARCH{S, p, q₁, q₂}}, ::Type{T}) where {S, p, q₁, q₂, T}
    lower = zeros(T, S+p+q₁+q₂+4)
    upper = zeros(T, S+p+q₁+q₂+4)
    lower .=  T(-Inf)
    upper .= T(Inf)
    return lower, upper
end

function coefnames(::Type{<:dRealGARCH{S, p, q₁, q₂}}) where {S, p, q₁, q₂}
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

