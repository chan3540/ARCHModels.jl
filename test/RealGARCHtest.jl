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
#import package
using CSV, DataFrames
using Optim
using Plots

include("../src/ARCHModels.jl")
using .ARCHModels


# read data
readpath = dirname(pwd())*"\\ARCHModels.jl\\src\\data\\CoinbasePro_RV_ETH-USD.csv"
df = DataFrame(CSV.File(readpath,header = 1))
ts = df[:,1]
rts = df[:,3]
xts = df[:,2]

#split insample / out-of-sample
ins_n = 200
rts_ins = rts[1:7*ins_n]
xts_ins = xts[1:7*ins_n]



#In-sample / out-of-sample comparison among various models  
# in sample estimation -> out-of-sample comparison

# periodic realgarch
spec = RealGARCH{7,1,1,1}(zeros(9+6))
am = UnivariateARCHXModel(spec,rts_ins,xts_ins)
fitted_am = fit(am)
fitted_coefs = fitted_am.spec.coefs
spec = RealGARCH{7,1,1,1}(fitted_coefs)
am = UnivariateARCHXModel(spec,rts,xts)
ht_pregarch_os = (volatilities(am).^2)[7*ins_n+1:end]


# realgarch
spec = RealGARCH{1,1,1,1}(zeros(9))
am = UnivariateARCHXModel(spec,rts_ins,xts_ins)
fitted_am = fit(am)
fitted_coefs = fitted_am.spec.coefs

spec = RealGARCH{1,1,1,1}(fitted_coefs)
am = UnivariateARCHXModel(spec,rts,xts)
ht_regarch_os = (volatilities(am).^2)[7*ins_n+1:end]


spec = EGARCH{1,1,1}(zeros(4))
am = UnivariateARCHModel(spec,rts_ins)
fitted_am = fit(am)
fitted_coefs = fitted_am.spec.coefs
spec = EGARCH{1,1,1}(fitted_coefs)
am = UnivariateARCHModel(spec,rts)
ht_egarch_os = (volatilities(am).^2)[7*ins_n+1:end]

spec = TGARCH{1,1,1}(zeros(4))
am = UnivariateARCHModel(spec,rts_ins)
fitted_am = fit(am)
fitted_coefs = fitted_am.spec.coefs
spec = TGARCH{1,1,1}(fitted_coefs)
am = UnivariateARCHModel(spec,rts)
ht_tgarch_os = (volatilities(am).^2)[7*ins_n+1:end]



σt2 =rts[7*ins_n+1:end].^2 
σt2 =xts[7*ins_n+1:end] 

mse(σt2,ht_pregarch_os), mse(σt2,ht_regarch_os),mse(σt2,ht_egarch_os),mse(σt2,ht_tgarch_os)
mse(log.(σt2),log.(ht_pregarch_os)),mse(log.(σt2),log.(ht_regarch_os)),mse(log.(σt2),log.(ht_egarch_os)),mse(log.(σt2),log.(ht_tgarch_os))
qlike(σt2,ht_pregarch_os), qlike(σt2,ht_regarch_os),qlike(σt2,ht_egarch_os),qlike(σt2,ht_tgarch_os)


plot(σt2,yaxis=:log,label="RV")
plot!(ht_pregarch_os,yaxis=:log,label="pRealGARCH",alpha=0.5)
plot!(ht_regarch_os,label="RealGARCH",alpha=0.5)
plot!(ht_egarch_os,label="EGARCH",alpha=0.5)
plot!(ht_egarch_os,label="TGARCH",alpha=0.5)