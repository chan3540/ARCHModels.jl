"""
pRealGARCH(P,p,q₁,q₂)

    GARCH equation

        loghₜ = ω(mod(t-1,m)+1) + ∑(r=1,..,p) βᵣ loghₜ₋ᵣ +  ∑(i=1,...,q₁) τ₁ᵢzₜ₋ᵢ  + ∑(j=1,...,q₂) τ₂ⱼ(aₜ₋ⱼ²-1) τ(zₜ) +  + γ′uₜ  for m = 1,...,P  where P is the number of periodicity parameters. 

    Measurement Equation

        log xₜ = ξ + log hₜ + δ(zₜ) + uₜ 

        where δ(z)=δ₁z + δ₂(z²-1)
 
rₜ = √hₜ zₜ,  zₜ ~ N(0,1)



**Reference:**
P.R. Hansen, Zhuo Huang, H.H. Shek, 2012. Realized GARCH: A joint model for returns and realized measures of volatility. Journal of Applied Econometrics 
"""

#import package
using CSV, DataFrames
using Optim
using Plots

include("../src/ARCHModels.jl")
using .ARCHModels


# read data
tickers = ["SPY","KO","AAPL","TSLA","JNJ","CVX"]
filename = tickers[1]*"_RMs.csv"#"CoinbasePro_RV_ETH-USD.csv"
readpath = dirname(pwd())*"\\ARCHModels.jl\\src\\data\\"*filename
df = DataFrame(CSV.File(readpath,header = 1))
#ts = df[2:end,1]
#rts = df[2:end,3]
rms = ["RV_15s","RV_2min","RV_5min","RV_10min","RV_15min","DR","RK"]

c2c_r = diff(log.(df.close))
o2c_r = (log.(df.close) .- log.(df.open))[2:end]
rm = df[2:end,rms[1]]

#split insample / out-of-sample
N₁ = 200
rts_ins = c2c_r[1:7*N₁]
xts_ins = rm[1:7*N₁]





#In-sample / out-of-sample comparison among various models  

# in sample estimation 

# periodic realgarch
#spec = pRealGARCH{7,1,1,1}(zeros(8+6))
#am = UnivariateARCHXModel(spec,rts_ins,xts_ins)
#fitted_am = fit(am)
#fitted_coefs = fitted_am.spec.coefs
#spec = pRealGARCH{7,1,1,1}(fitted_coefs)
#am = UnivariateARCHXModel(spec,rts,xts)
#ht_pregarch_os = (volatilities(am).^2)[7*N₁+1:end]

# realgarch
spec = RealGARCH{1,1,1}(zeros(8))
am = UnivariateARCHXModel(spec,rts_ins,xts_ins)
fitted_am = fit(am)
fitted_coefs = fitted_am.spec.coefs
spec = RealGARCH{1,1,1}(fitted_coefs)
am = UnivariateARCHXModel(spec,rts,xts)
ht_regarch_os = (volatilities(am).^2)[7*N₁+1:end]

# egarch
spec = EGARCH{1,1,1}(zeros(4))
am = UnivariateARCHModel(spec,rts_ins)
fitted_am = fit(am)
fitted_coefs = fitted_am.spec.coefs
spec = EGARCH{1,1,1}(fitted_coefs)
am = UnivariateARCHModel(spec,rts)


ht_egarch_os = (volatilities(am).^2)[7*N₁+1:end]

# tgarch
spec = TGARCH{1,1,1}(zeros(4))
am = UnivariateARCHModel(spec,rts_ins)
fitted_am = fit(am)
fitted_coefs = fitted_am.spec.coefs
spec = TGARCH{1,1,1}(fitted_coefs)
am = UnivariateARCHModel(spec,rts)
ht_tgarch_os = (volatilities(am).^2)[7*N₁+1:end]


#Out-of-sample realized measures

#σt2 =rts[7*N₁+1:end].^2  # 1 day squared return
σt2 =xts[7*N₁+1:end] # daily rv


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