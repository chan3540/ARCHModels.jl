"""
RealGARCH(d,o,p,q)

    GARCH equation

        loghₜ = ω(mod(t-1,m)+1) + ∑(i=1,...,o) τ₁ᵢzₜ₋ᵢ  + ∑(j=1,...,p) τ₂ⱼ(aₜ₋ⱼ²-1) τ(zₜ) + ∑(r=1,..,q) βᵣ loghₜ₋ᵣ + γ′uₜ  for m = 1,...,d  where d is the number of periodicity parameters. 

    Measurement Equation

        log xₜ = ξ + ϕ log hₜ + δ(zₜ) + uₜ (we set ϕ = 1 if an unbiased measure of volatility xₜ is used.)

        where δ(z)=δ₁z + δ₂(z²-1)
 
rₜ = √hₜ zₜ,  zₜ ~ N(0,1)


**Reference:**
P.R. Hansen, Zhuo Huang, H.H. Shek, 2012. Realized GARCH: A joint model for returns and realized measures of volatility. Journal of Applied Econometrics 
"""

#import package
using CSV, DataFrames
using Optim
using Plots
using PlotlyJS

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

# in sample estimation 

# periodic realgarch
# Refer to the model description for what d,o,p and q mean. 

spec = RealGARCH{7,1,1,1}(zeros(8+6))
am = UnivariateARCHXModel(spec,rts_ins,xts_ins)
fitted_am = fit(am)
fitted_coefs = fitted_am.spec.coefs
spec = RealGARCH{7,1,1,1}(fitted_coefs)
am = UnivariateARCHXModel(spec,rts,xts)
ht_pregarch_os = (volatilities(am).^2)[7*ins_n+1:end]

# realgarch
spec = RealGARCH{1,1,1,1}(zeros(8))
am = UnivariateARCHXModel(spec,rts_ins,xts_ins)
fitted_am = fit(am)
fitted_coefs = fitted_am.spec.coefs
spec = RealGARCH{1,1,1,1}(fitted_coefs)
am = UnivariateARCHXModel(spec,rts,xts)
ht_regarch_os = (volatilities(am).^2)[7*ins_n+1:end]

# egarch
spec = EGARCH{1,1,1}(zeros(4))
am = UnivariateARCHModel(spec,rts_ins)
fitted_am = fit(am)
fitted_coefs = fitted_am.spec.coefs
spec = EGARCH{1,1,1}(fitted_coefs)
am = UnivariateARCHModel(spec,rts)
ht_egarch_os = (volatilities(am).^2)[7*ins_n+1:end]

# tgarch
spec = TGARCH{1,1,1}(zeros(4))
am = UnivariateARCHModel(spec,rts_ins)
fitted_am = fit(am)
fitted_coefs = fitted_am.spec.coefs
spec = TGARCH{1,1,1}(fitted_coefs)
am = UnivariateARCHModel(spec,rts)
ht_tgarch_os = (volatilities(am).^2)[7*ins_n+1:end]


#Out-of-sample realized measures

#σt2 =rts[7*ins_n+1:end].^2  # 1 day squared return
σt2 =xts[7*ins_n+1:end] # daily rv


loss = Dict()
loss["LossFunction"] = ["MSE","logMSE","QLIKE"]
loss["pRealGARCH"] = [mse(σt2,ht_pregarch_os),mse(log.(σt2),log.(ht_pregarch_os)),qlike(σt2,ht_pregarch_os)]
loss["RealGARCH"] = [mse(σt2,ht_regarch_os),mse(log.(σt2),log.(ht_regarch_os)),qlike(σt2,ht_regarch_os)]
loss["EGARCH"] = [mse(σt2,ht_egarch_os),mse(log.(σt2),log.(ht_egarch_os)),qlike(σt2,ht_egarch_os)]
loss["TGARCH"] = [mse(σt2,ht_tgarch_os),mse(log.(σt2),log.(ht_tgarch_os)),qlike(σt2,ht_tgarch_os)]

loss_table = DataFrame(loss)
loss_table = loss_table[:,["LossFunction","TGARCH" ,"EGARCH", "RealGARCH", "pRealGARCH"]]
relative_loss_table = loss_table[:,["TGARCH" ,"EGARCH", "RealGARCH", "pRealGARCH"]] ./ loss_table[:,"TGARCH"]
relative_loss_table[!,"LossFunction"] = ["MSE","logMSE","QLIKE"]
relative_loss_table = relative_loss_table[:,["LossFunction","TGARCH" ,"EGARCH", "RealGARCH", "pRealGARCH"]]


plot(σt2,yaxis=:log,label="RV")
plot!(ht_pregarch_os,yaxis=:log,label="pRealGARCH",alpha=0.5)
plot!(ht_regarch_os,label="RealGARCH",alpha=0.5)
plot!(ht_egarch_os,label="EGARCH",alpha=0.5)
plot!(ht_egarch_os,label="TGARCH",alpha=0.5)