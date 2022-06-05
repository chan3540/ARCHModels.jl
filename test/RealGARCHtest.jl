"""
RealGARCH(d,o,p,q)

    GARCH equation

        loghₘ,ₜ = ωₘ + ∑(i=1,...,o) τ₁ᵢzₜ₋ᵢ  + ∑(j=1,...,p) τ₂ⱼ(aₜ₋ⱼ²-1) τ(zₜ) + ∑(r=1,..,q) βᵣ loghₜ₋ᵣ + γ′uₜ  for m = 1,...,d  where d is the number of periodicity parameters. 

    Measurement Equation

        log xₜ = ξ + ϕ log hₜ + δ(zₜ) + uₜ (we assume ϕ=1) 

        where δ(z)=δ₁z + δ₂(z²-1)
 
rₜ = √hₜ zₜ,  zₜ ~ N(0,1)


**Reference:**
P.R. Hansen, Zhuo Huang, H.H. Shek, 2012. Realized GARCH: A joint model for returns and realized measures of volatility. Journal of applied econometrics 

"""




using CSV, DataFrames
include("../src/ARCHModels.jl")
using .ARCHModels
using Optim
using Plots

readpath = dirname(pwd())*"\\ARCHModels.jl\\src\\data\\CoinbasePro_RV_ETH-USD.csv"
df = DataFrame(CSV.File(readpath,header = 1))
ts = df[:,1]
rts = df[:,3]
xts = df[:,2]


spec = RealGARCH{7,1,1,1}(zeros(8+6))
am = UnivariateARCHXModel(spec,rts,xts)
fitted_am = fit(am)

ht = volatilities(fitted_am).^2

plot(ts,ht)
plot!(xts)