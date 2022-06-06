"""
    UnivariateXVolatilitySpec{T} <: UnivariateVolatilitySpec{T} end

Abstract supertype that univariate volatility specifications inherit from.
"""

abstract type UnivariateXVolatilitySpec{T} <: UnivariateVolatilitySpec{T} end

"""
    StandardizedDistribution{T} <: Distributions.Distribution{Univariate, Continuous}

Abstract supertype that standardized distributions inherit from.
"""

"""
    UnivariateARCHXModel{T<:AbstractFloat,
              		    VS<:UnivariateVolatilitySpec
              			} <: ARCHXModel
"""
mutable struct UnivariateARCHXModel{T<:AbstractFloat,
                 				   VS<:UnivariateXVolatilitySpec
                 				   } <: ARCHXModel
    spec::VS
    data::Vector{T}
	data_X::Vector{T}
	fitted::Bool
    function UnivariateARCHXModel{T, VS}(spec, data, data_X, fitted) where {T, VS}
		@assert length(data) == length(data_X)
        new(spec, data, data_X, fitted)
    end
end


"""
    UnivariateARCHXModel(spec::UnivariateVolatilitySpec, data::Vector, data_X::Vector ; fitted=false)

Create a UnivariateARCHXModel.

# Example:
```jldoctest
julia> UnivariateARCHXModel(RealGARCH{1, 1}([1., .9, .05]), randn(10))

RealizedGARCH{1, 1, 1, 1} model with Gaussian errors, T=10.


─────────────────────────────────────────
                             ω   β₁    α₁
─────────────────────────────────────────
Volatility parameters:     1.0  0.9  0.05
─────────────────────────────────────────
```
"""
function UnivariateARCHXModel(spec::VS,
          		 			 data::Vector{T},
							 data_X::Vector{T};
							 fitted::Bool=false
          					 ) where {T<:AbstractFloat,
                    			 	  VS<:UnivariateVolatilitySpec
                   			 		  }
    UnivariateARCHXModel{T, VS}(spec, data, data_X, fitted)
end

loglikelihood(am::UnivariateARCHXModel) = loglik(typeof(am.spec), am.data, am.data_X,
                                      am.spec.coefs)

dof(am::UnivariateARCHXModel) = nparams(typeof(am.spec)) 
coef(am::UnivariateARCHXModel) = am.spec.coefs
coefnames(am::UnivariateARCHXModel) = coefnames(typeof(am.spec))


# documented in general
#function simulate(spec::UnivariateVolatilitySpec{T2}, nobs; warmup=100, dist::StandardizedDistribution{T2}=StdNormal{T2}(),
#                  meanspec::MeanSpec{T2}=NoIntercept{T2}(),
#				  rng=GLOBAL_RNG
#                  ) where {T2<:AbstractFloat}
#    data = zeros(T2, nobs)
#	data_X = zeros(T2, nobs)
#    _simulate!(data, data_X, spec; warmup=warmup, dist=dist, meanspec=meanspec, rng=rng)
#    UnivariateARCHXModel(spec, data, data_X; dist=dist, meanspec=meanspec, fitted=false)
#end
## σᵤₜ??
#function _simulate!(data::Vector{T2}, data_X::Vector{T2},spec::UnivariateVolatilitySpec{T2};
#                  warmup=100,
#                  dist::StandardizedDistribution{T2}=StdNormal{T2}(),
#                  meanspec::MeanSpec{T2}=NoIntercept{T2}(),
#				  rng=GLOBAL_RNG
#                  ) where {T2<:AbstractFloat}
#	@assert warmup>=0
#	append!(data, zeros(T2, warmup))
#	append!(data_X, zeros(T2, warmup))
#    T = length(data)
#	r1 = presample(typeof(spec))
#	K = numrm(typeof(spec))
#	r2 = presample(meanspec)
#	r = max(r1, r2)
#	r = max(r, 1) # make sure this works for, e.g., ARCH{0}; CircularBuffer requires at least a length of 1
#    ht = CircularBuffer{T2}(r)
#    lht = CircularBuffer{T2}(r)
#    zt = CircularBuffer{T2}(r)
#	at = CircularBuffer{T2}(r)
#	ut = CircularBuffer{T2}(K)
#    @inbounds begin
#        h0 = uncond(typeof(spec), spec.coefs)
#		m0 = uncond(meanspec)
#        h0 > 0 || error("Model is nonstationary.")
#        for t = 1:T
#			if t>r2
#				themean = mean(at, ht, lht, data, meanspec, meanspec.coefs, t)
#			else
#				themean = m0
#			end
#			if t>r1
#                update!(ht, lht, zt, at, ut, typeof(spec), spec.coefs)
#            else
#				push!(ht, h0)
#                push!(lht, log(h0))
#            end
#			push!(zt, rand(rng, dist))
#			push!(ut, rand(rng, dist))
#			push!(at, sqrt(ht[end])*zt[end])
#			data[t] = themean + at[end]
#        end
#    end
#    deleteat!(data, 1:warmup)
#end


"""
    volatilities(am::UnivariateARCHModel)
Return the conditional volatilities.
"""
function volatilities(am::UnivariateARCHXModel{T, VS}) where {T, VS}
	ht = Vector{T}(undef, 0)
	lht = Vector{T}(undef, 0)
	zt = Vector{T}(undef, 0)
	at = Vector{T}(undef, 0)
	ut = Vector{T}(undef, 0)
	loglik!(ht, lht, zt, ut, VS, am.data, am.data_X, am.spec.coefs)
	return sqrt.(ht)
end

#"""
#    predict(am::UnivariateARCHXModel, what=:volatility, horizon=1; level=0.01)
#Form a `horizon`-step ahead prediction from `am`. `what` controls which object is predicted.
#The choices are `:volatility` (the default), `:variance`, `:return`, and `:VaR`. The VaR
#level can be controlled with the keyword argument `level`.
#
#Not all prediction targets / volatility specifications support multi-step predictions.
#"""
function predict(am::UnivariateARCHXModel{T1, VS}; horizon=1) where {T1, VS}
	ht = volatilities(am).^2
	lht = log.(ht)
	zt = residuals(am)
    ut = T1[0]
    T = length(am.data)
	for current_horizon in 1:horizon
        update!(ht, lht, zt, ut, VS, am.spec.coefs,T+current_horizon)
		push!(zt, 0.)
	end
    return ht
end




"""
    means(am::UnivariateARCHModel)
Return the conditional means of the model.
"""
#function means(am::UnivariateARCHModel)
#	return am.data-residuals(am; standardized=false)
#end

"""
    residuals(am::UnivariateARCHModel; standardized=true)
Return the residuals of the model. Pass `standardized=false` for the non-devolatized residuals.
"""
function residuals(am::UnivariateARCHXModel{T, VS}; standardized=true) where {T, VS, SD}
	ht = Vector{T}(undef, 0)
	lht = Vector{T}(undef, 0)
	zt = Vector{T}(undef, 0)
	ut = Vector{T}(undef, 0)
	loglik!(ht, lht, zt, ut, VS, am.data, am.data_X, am.spec.coefs)
	return standardized ? zt : sqrt(ht).*zt
end

"""
    VaRs(am::UnivariateARCHModel, level=0.01)
Return the in-sample Value at Risk implied by `am`.
"""
#function VaRs(am::UnivariateARCHXModel, level=0.01)
#    return -means(am) .- volatilities(am) .* quantile(am.dist, level)
#end

#this works on CircularBuffers. The idea is that ht/lht/zt need to be allocated
#inside of this function, when the type that Optim it with is known (because
#it calls it with dual numbers for autodiff to work). It works with arrays, too,
#but grows them by length(data); hence it should be called with an empty one-
#dimensional array of the right type.

@inline function loglik!(ht::AbstractVector{T1}, lht::AbstractVector{T1},
    zt::AbstractVector{T1}, ut::AbstractVector{T1}, vs::Type{VS},
    data::Vector{T2}, data_X::Vector{T2}, coefs::AbstractVector{T3}
    ) where {VS<:UnivariateVolatilitySpec, T1,T2<:AbstractFloat,T3}
    garchcoefs = coefs
    lowergarch, uppergarch = constraints(VS, T3)
    all(lowergarch.<garchcoefs.<uppergarch) || return T3(-Inf)
    T = length(data)
    r = presample(VS)
    T > r || error("Sample too small.")
    @inbounds begin
        h0 = var(data) 
        LL = zero(T3)
        σu2 = zero(T3)
        for t = 1:T
            if t > r
                update!(ht, lht, zt, ut, VS, garchcoefs,t)
            else
                push!(ht, h0)
                push!(lht, log(h0))
            end
            ht[end] < 0 && return T3(NaN)
            push!(zt, data[t]/sqrt(ht[end]))
            ut_update!(ht, lht, zt, ut, data_X, VS, garchcoefs, t)
            LL += - lht[end] - zt[end]^2 
            σu2 += ut[end]^2
        end
    end
    if σu2>0
	   σu2 = σu2/T
       LL -= log(σu2)*T
	end
	LL = LL/2
end

function loglik(spec::Type{VS}, 
                   data::Vector{<:AbstractFloat}, data_X::Vector{<:AbstractFloat}, coefs::AbstractVector{T2}
                   ) where {VS<:UnivariateVolatilitySpec,T2}
	r = max(presample(VS), 1) # make sure this works for, e.g., ARCH{0}; CircularBuffer requires at least a length of 1
    ht = CircularBuffer{T2}(r)
    lht = CircularBuffer{T2}(r)
    zt = CircularBuffer{T2}(r)
	ut = CircularBuffer{T2}(1)
    loglik!(ht, lht, zt, ut, spec, data, data_X, coefs)
end

function logliks(spec, data, data_X, coefs::AbstractVector{T}) where {T}
    garchcoefs = coefs
    ht = T[]
    lht = T[]
    zt = T[]
	ut = T[]
    loglik!(ht, lht, zt, ut, spec, data, data_X, coefs)
	σᵤ² = mean(ut.^2)
    LLs = (-lht .- zt.^2 .- log(σᵤ²) .- (ut.^2)./σᵤ²)./2
end


function informationmatrix(am::UnivariateARCHXModel; expected::Bool=false)
	expected && error("expected informationmatrix is not implemented for UnivariateARCHModel. Use expected=false.")
	g = x -> sum(logliks(typeof(am.spec), am.data, am.data_X, x))
	H = ForwardDiff.hessian(g, am.spec.coefs)
	J = -H
end

function scores(am::UnivariateARCHXModel)
	f = x -> logliks(typeof(am.spec), am.data, am.data_X, x)
	S = ForwardDiff.jacobian(f, am.spec.coefs)
end


function _fit!(coefs::Vector{T}, ::Type{VS},
              data::Vector{T}, data_X::Vector{T}; algorithm=NelderMead(), autodiff=:forward, kwargs...
              ) where {VS<:UnivariateVolatilitySpec, T<:AbstractFloat
                       }
    obj = x -> -loglik(VS, data, data_X, x)
	option = Optim.Options(g_tol = 1e-4, iterations=50000, show_trace=true, show_every=500)
	res = optimize(obj, coefs,algorithm,option; autodiff=:forward)
    coefs .= Optim.minimizer(res)
    return nothing
end

"""
    fit(VS::Type{<:UnivariateVolatilitySpec}, data; dist=StdNormal, meanspec=Intercept,
        algorithm=BFGS(), autodiff=:forward, kwargs...)

Fit the ARCH model specified by `VS` to `data`. `data` can be a vector or a
GLM.LinearModel (or GLM.TableRegressionModel).

# Keyword arguments:
- `dist=StdNormal`: the error distribution.
- `meanspec=Intercept`: the mean specification, either as a type or instance of that type.
- `algorithm=BFGS(), autodiff=:forward, kwargs...`: passed on to the optimizer.

# Example: EGARCH{1, 1, 1} model without intercept, Student's t errors.
```jldoctest
julia> fit(EGARCH{1, 1, 1}, BG96; meanspec=NoIntercept, dist=StdT)

EGARCH{1, 1, 1} model with Student's t errors, T=1974.


Volatility parameters:
──────────────────────────────────────────────
      Estimate  Std.Error    z value  Pr(>|z|)
──────────────────────────────────────────────
ω   -0.0162014  0.0186806  -0.867286    0.3858
γ₁  -0.0378454  0.018024   -2.09972     0.0358
β₁   0.977687   0.012558   77.8538      <1e-99
α₁   0.255804   0.0625497   4.08961     <1e-04
──────────────────────────────────────────────

Distribution parameters:
─────────────────────────────────────────
   Estimate  Std.Error  z value  Pr(>|z|)
─────────────────────────────────────────
ν   4.12423    0.40059  10.2954    <1e-24
─────────────────────────────────────────
```
"""
function fit(::Type{VS}, data::Vector{T}, data_X::Vector{T}; algorithm=NelderMead(),
             autodiff=:forward, kwargs...
             ) where {VS<:UnivariateVolatilitySpec, T<:AbstractFloat
                      }
	#can't use dispatch for this b/c meanspec is a kwarg
    coefs = startingvals(VS, data)
	_fit!(coefs, VS, data, data_X ; algorithm=algorithm, autodiff=autodiff, kwargs...)
	return UnivariateARCHXModel(VS(coefs), data, data_X ; fitted=true)
end


function fit!(am::UnivariateARCHXModel; algorithm=NelderMead(), autodiff=:forward, kwargs...)
    am.spec.coefs .= startingvals(typeof(am.spec), am.data)
	_fit!(am.spec.coefs, typeof(am.spec),
         am.data, am.data_X; algorithm=algorithm,
         autodiff=autodiff, kwargs...
         )
	am.fitted=true
    am
end

function fit(am::UnivariateARCHXModel; algorithm=NelderMead(), autodiff=:forward, kwargs...)
    am2=deepcopy(am)
    fit!(am2; algorithm=algorithm, autodiff=autodiff, kwargs...)
    return am2
end

#function fit(vs::Type{VS}, lm::TableRegressionModel{<:LinearModel}; kwargs...) where VS<:UnivariateVolatilitySpec
#	fit(vs, response(lm.model); meanspec=Regression(modelmatrix(lm.model); coefnames=coefnames(lm)), kwargs...)
#end

#function fit(vs::Type{VS}, lm::LinearModel; kwargs...) where VS<:UnivariateVolatilitySpec#
#	fit(vs, response(lm); meanspec=Regression(modelmatrix(lm)), kwargs...)
#end

"""
    selectmodel(::Type{VS}, data; kwargs...) -> UnivariateARCHModel

Fit the volatility specification `VS` with varying lag lengths and return that which
minimizes the [BIC](https://en.wikipedia.org/wiki/Bayesian_information_criterion).

# Keyword arguments:
- `dist=StdNormal`: the error distribution.
- `meanspec=Intercept`: the mean specification, either as a type or instance of that type.
- `minlags=1`: minimum lag length to try in each parameter of `VS`.
- `maxlags=3`: maximum lag length to try in each parameter of `VS`.
- `criterion=bic`: function that takes a `UnivariateARCHModel` and returns the criterion to minimize.
- `show_trace=false`: print `criterion` to screen for each estimated model.
- `algorithm=BFGS(), autodiff=:forward, kwargs...`: passed on to the optimizer.

# Example
```jldoctest
julia> selectmodel(EGARCH, BG96)

EGARCH{1, 1, 2} model with Gaussian errors, T=1974.

Mean equation parameters:
───────────────────────────────────────────────
      Estimate   Std.Error    z value  Pr(>|z|)
───────────────────────────────────────────────
μ  -0.00900018  0.00943948  -0.953461    0.3404
───────────────────────────────────────────────

Volatility parameters:
──────────────────────────────────────────────
      Estimate  Std.Error    z value  Pr(>|z|)
──────────────────────────────────────────────
ω   -0.0544398  0.0592073  -0.919478    0.3578
γ₁  -0.0243368  0.0270414  -0.899985    0.3681
β₁   0.960301   0.0388183  24.7384      <1e-99
α₁   0.405788   0.067466    6.0147      <1e-08
α₂  -0.207357   0.114161   -1.81636     0.0693
──────────────────────────────────────────────
```
"""
#function selectmodel(::Type{VS}, data::Vector{T};
#                     dist::Type{SD}=StdNormal{T}, meanspec::Union{MS, Type{MS}}=Intercept{T},
#                     maxlags::Integer=3, minlags::Integer=1, criterion=bic, show_trace=false, algorithm=BFGS(),
#                     autodiff=:forward, kwargs...
#                     ) where {VS<:UnivariateVolatilitySpec, T<:AbstractFloat,
#                              SD<:StandardizedDistribution, MS<:MeanSpec
#                              }
#	@assert maxlags >= minlags >= 0
#
#	#threading sometimes segfaults in tests locally. possibly https://github.com/JuliaLang/julia/issues/29934
#	mylock=Threads.ReentrantLock()
#    ndims = max(my_unwrap_unionall(VS)-1, 0) # e.g., two (p and q) for GARCH{p, q, T}
#	ndims2 = max(my_unwrap_unionall(MS)-1, 0 )# e.g., two (p and q) for ARMA{p, q, T}
#    res = Array{UnivariateSubsetARCHModel, ndims+ndims2}(undef, ntuple(i->maxlags - minlags + 1, ndims+ndims2))
#    Threads.@threads for ind in collect(CartesianIndices(size(res)))
#		tup = (ind.I[1:ndims] .+ minlags .-1)
#		MSi = (ndims2==0 ? deepcopy(meanspec) : meanspec{ind.I[ndims+1:end] .+ minlags .- 1...})
#		res[ind] = fitsubset(VS, data, maxlags, tup; dist=dist, meanspec=MSi,
#                       algorithm=algorithm, autodiff=autodiff, kwargs...)
#        if show_trace
#            lock(mylock)
#			VSi = VS{tup...}
#            Core.print(modname(VSi))
#			ndims2>0 && Core.print("-", modname(MSi))
#			Core.println(" model has ",
#                              uppercase(split("$criterion", ".")[end]), " ",
#                              criterion(res[ind]), "."
#                              )
#            unlock(mylock)
#        end
#    end
#    crits = criterion.(res)
#    _, ind = findmin(crits)
#	return fit(VS{res[ind].subset...}, data; dist=dist, meanspec=res[ind].meanspec, algorithm=algorithm, autodiff=autodiff, kwargs...)
#end

function coeftable(am::UnivariateARCHXModel)
    cc = coef(am)
    se = stderror(am)
    zz = cc ./ se
    CoefTable(hcat(cc, se, zz, 2.0 * normccdf.(abs.(zz))),
              ["Estimate", "Std.Error", "z value", "Pr(>|z|)"],
              coefnames(am), 4)
end

function show(io::IO, am::UnivariateARCHXModel)
	if isfitted(am)
		cc = coef(am)
	    se = stderror(am)
#	    ccg, ccd, ccm = splitcoefs(cc, typeof(am.spec),
#	                               typeof(am.dist), am.meanspec
#	                               )
#	    seg, sed, sem = splitcoefs(se, typeof(am.spec),
#	                               typeof(am.dist), am.meanspec
#	                               )
	    zz = cc ./ se
#		zzg = ccg ./ seg
#	    zzd = ccd ./ sed
#	    zzm = ccm ./ sem
	    println(io, "\n", modname(typeof(am.spec)), " model.",
	            "T=", nobs(am), ".\n")

#	    length(sem) > 0 && println(io, "Mean equation parameters:", "\n",
#	                               CoefTable(hcat(ccm, sem, zzm, 2.0 * normccdf.(abs.(zzm))),
#	                                         ["Estimate", "Std.Error", "z value", "Pr(>|z|)"],
#	                                         coefnames(am.meanspec), 4
#	                                         )
#	                              )
	    println(io, "\nVolatility parameters:", "\n",
	            CoefTable(hcat(cc, se, zz, 2.0 * normccdf.(abs.(zz))),
	                      ["Estimate", "Std.Error", "z value", "Pr(>|z|)"],
	                      coefnames(typeof(am.spec)), 4
	                      )
	            )
#	    length(sed) > 0 && println(io, "\nDistribution parameters:", "\n",
#	                               CoefTable(hcat(ccd, sed, zzd, 2.0 * normccdf.(abs.(zzd))),
#	                                         ["Estimate", "Std.Error", "z value", "Pr(>|z|)"],
#	                                         coefnames(typeof(am.dist)), 4
#	                                         )
#	                              )

   else
	   println(io, "\n", typeof(am.spec), " model.",
			   " T=", nobs(am), ".\n\n")
#	   length(am.meanspec.coefs) > 0 && println(io, CoefTable(am.meanspec.coefs, coefnames(am.meanspec), ["Mean equation parameters:"]))
	   println(io, CoefTable(am.spec.coefs, coefnames(typeof(am.spec)), ["Volatility parameters:   "]))
#	   length(am.dist.coefs) > 0 && println(io, CoefTable(am.dist.coefs, coefnames(typeof(am.dist)), ["Distribution parameters: "]))
   end
end

#function modname(::Type{S}) where S<:Union{UnivariateVolatilitySpec, MeanSpec}
#    s = "$(S)"
#	lastcomma = findlast(isequal(','), s)
#    lastcomma == nothing || (s = s[1:lastcomma-1] * '}')
#	firstdot = findfirst(isequal('.'), s)
#	firstdot == nothing || (s = s[firstdot+1:end])
#	s
#end
