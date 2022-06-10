"""
pRealGARCH(P,p,q₁,q₂)

    GARCH equation

        loghₜ = ω(mod(t-1,m)+1) + ∑(r=1,..,p) βᵣ loghₜ₋ᵣ +  ∑(i=1,...,q₁) τ₁ᵢzₜ₋ᵢ  + ∑(j=1,...,q₂) τ₂ⱼ(zₜ₋ⱼ²-1) +  + γ′uₜ  for m = 1,...,P  where P is the number of periodicity parameters. 

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
filename = tickers[2]*"_RMs.csv"#
readpath = dirname(pwd())*"\\ARCHModels.jl\\src\\data\\"*filename
df = DataFrame(CSV.File(readpath,header = 1))


# Add date and label (legend:upperleft )

plot(df.close)
# 0 damper / 10 damper 

rms = ["RV_15s","RV_2min","RV_5min","RV_10min","RV_15min","DR","RK"]
c2c_r = diff(log.(df.close))
o2c_r = (log.(df.close) .- log.(df.open))[2:end]

#split insample / out-of-sample
N₁ = 1000
rts = c2c_r
rts_ins = c2c_r[1:N₁]

#Plot realized measures
plot(df[:,"RK"],yaxis=:log,alpha=0.5)
plot!(df[:,"RV_15min"],alpha=0.5)
plot!(df[:,"RV_15s"],alpha=0.5)

#Out-of-sample realized measures

σt2 = rts[N₁+1:end].^2 .+0.00001 # 1 day squared return


loss = Dict()
loss["LossFunction"] = ["MSE","logMSE","QLIKE"]
fitted_coefs = Dict()
fitted_coefs["Parameter"] = ["ω","β","τ₁","τ₂","γ","ξ","δ₁","δ₂"]

for rm0 in rms
    rm = df[2:end,rm0]
    xts = rm
    xts_ins = rm[1:N₁]

    spec = RealGARCH{1,1}(zeros(8)) # RealGARCH{p,q} = pRealGARCH{p,q₁,q₂,S} where q₁=q₂ 

    am = UnivariateARCHXModel(spec,rts_ins,xts_ins)

    fitted_am = fit(am)
    fitted_coefs[rm0] = fitted_am.spec.coefs

    spec = RealGARCH{1,1}(fitted_coefs[rm0])
    am = UnivariateARCHXModel(spec,rts,xts)

    ht_realgarch_os = (volatilities(am).^2)[N₁+1:end]
    
    loss[rm0] = [mse(σt2,ht_realgarch_os),mse(log.(σt2),log.(ht_realgarch_os)),qlike(σt2,ht_realgarch_os)]
end

loss
loss_table = DataFrame(loss)

loss_table = loss_table[:,["LossFunction","RV_15s","RV_2min","RV_5min","RV_10min","RV_15min","DR","RK"]]
relative_loss_table = loss_table[:,["RV_15s","RV_2min","RV_5min","RV_10min","RV_15min","DR","RK"]] ./ loss_table[:,"RK"]
relative_loss_table[!,"LossFunction"] = ["MSE","logMSE","QLIKE"]
relative_loss_table = relative_loss_table[:,["LossFunction","RV_15s","RV_2min","RV_5min","RV_10min","RV_15min","DR","RK"]]

coef_tables = DataFrame(fitted_coefs)
coef_tables = coef_tables[:,["Parameter","RV_15s","RV_2min","RV_5min","RV_10min","RV_15min","DR","RK"]]