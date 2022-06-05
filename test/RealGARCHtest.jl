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
plot(log.(ht))
plot!(log.(rts.^2))
plot!(log.(xts))
plot(sqrt.(ht))
plot!(sqrt.(xts))
