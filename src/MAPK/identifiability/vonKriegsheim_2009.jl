using StructuralIdentifiability

ode = @ODEmodel(
    x1'(t) = NGFDeriv,
    x2'(t) = EGFDeriv,
    x3'(t) = (k2*x4(t))-(k1*x1(t)*x3(t)),
    x4'(t) = -((k2*x4(t))-(k1*x1(t)*x3(t)))-(k54*x4(t)/(k58_+x4(t))),
    x5'(t) = (k4*x6(t))+(k6*x34(t))-(k3*x2(t)*x5(t)),
    x6'(t) = (k3*x2(t)*x5(t))-(k4*x6(t))-(k5*x6(t))-(k52*x6(t)/(k53+x6(t))),
    x7'(t) = (k5*x6(t))-(k7*x7(t))+(k52*x6(t)/(k53+x6(t)))+(k56*x34(t))-(k55*x7(t)),
    x8'(t) = (k8*x4(t))+(k9*x6(t))-(k10*x8(t)),
    x9'(t) = (k48*x10(t))-(k47*x8(t)*x9(t)),
    x10'(t) = -(k48*x10(t))+(k47*x8(t)*x9(t)),
    x11'(t) = 0.0,
    x12'(t) = (k13*x13(t))-(k12*x6(t)*x12(t))-(k11*x4(t)*x12(t)),
    x13'(t) = -((k13*x13(t))-(k12*x6(t)*x12(t))-(k11*x4(t)*x12(t))),
    x14'(t) = (k16*x15(t)+k59*x15(t)*x2(t)+k60*x15(t)*x1(t))-(k14*(x13(t)/(k62_+x13(t)))*x14(t) + k63*(x13(t)/(k65_+x13(t)))*x14(t))-(k15*(x11(t)/(k67_+x11(t)))*x14(t)),
    x15'(t) = -((k16*x15(t)+k59*x15(t)*x2(t)+k60*x15(t)*x1(t))-(k14*(x13(t)/(k62_+x13(t)))*x14(t) + k63*(x13(t)/(k65_+x13(t)))*x14(t))-(k15*(x11(t)/(k67_+x11(t)))*x14(t))),
    x16'(t) = (k18*x17(t))-(k17*x16(t)*x15(t)),
    x17'(t) = -((k18*x17(t))-(k17*x16(t)*x15(t))),
    x18'(t) = (k26*x20(t)*x19(t))-(k19*x6(t)*x18(t)/(1+(x37(t))/k49))-(k20*x4(t)*x18(t)/(1+(x37(t))/k49))-(k21*x7(t)*x18(t)/(1+(x37(t))/k49))-(k22*x8(t)*x18(t)/(1+(x37(t))/k49))-(k23*x18(t))-(k24*x11(t)*x18(t))-(k25*(x13(t)/(x13(t)+k71_))*x18(t)),
    x19'(t) = -((k26*x20(t)*x19(t))-(k19*x6(t)*x18(t)/(1+(x37(t))/k49))-(k20*x4(t)*x18(t)/(1+(x37(t))/k49))-(k21*x7(t)*x18(t)/(1+(x37(t))/k49))-(k22*x8(t)*x18(t)/(1+(x37(t))/k49))-(k23*x18(t))-(k24*x11(t)*x18(t))-(k25*(x13(t)/(x13(t)+k71_))*x18(t))),
    x20'(t) = (k28*x21(t))-(k27*x27(t)*x20(t)),
    x21'(t) = -((k28*x21(t))-(k27*x27(t)*x20(t))),
    x22'(t) = (k30*x23(t))-(k29*(x19(t)/(x19(t)+k73_))*x22(t)),
    x23'(t) = -((k30*x23(t))-(k29*(x19(t)/(x19(t)+k73_))*x22(t))),
    x24'(t) = (k32*x25(t))-(k31*x23(t)*x24(t)/(k50 + x24(t) + (k50*x16(t)/k51))),
    x25'(t) = -((k32*x25(t))-(k31*x23(t)*x24(t)/(k50 + x24(t) + (k50*x16(t)/k51)))),
    x26'(t) = (k34*x27(t))-(k33*x25(t)*x26(t))-(k37*x26(t)*x31(t))+(k39*x28(t))+(k42*x30(t))*V_NUC/V_CYT,
    x27'(t) = (k33*x25(t)*x26(t))-(k34*x27(t))-(k38*x27(t)*x31(t))+(k40*x29(t))-(k41*x27(t)),
    x28'(t) = (k34*x29(t))-(k33*x25(t)*x28(t))-(k39*x28(t))+(k37*x26(t)*x31(t)),
    x29'(t) = (k33*x25(t)*x28(t))-(k34*x29(t))-(k40*x29(t))+(k38*x27(t)*x31(t)),
    x30'(t) = (k41*x27(t))*V_CYT/V_NUC-(k42*x30(t)),
    x31'(t) = (k39*x28(t))+(k40*x29(t))-(k37*x26(t)*x31(t))-(k38*x27(t)*x31(t))-(k43*x27(t)*x31(t))-(k44*x10(t)*x31(t))+(k45*x32(t))+(k46*x33(t)),
    x32'(t) = (k43*x27(t)*x31(t))-(k45*x32(t)),
    x33'(t) = (k44*x10(t)*x31(t))-(k46*x33(t)),
    x34'(t) = -(k56*x34(t))+(k55*x7(t))-(k6*x34(t)),
    x35'(t) = 0.0,
    x36'(t) = (k69*x37(t))-(k68*(x27(t)+x29(t))*x36(t)),
    x37'(t) = -((k69*x37(t))-(k68*(x27(t)+x29(t))*x36(t))),
    y1(t) = x27(t) + x29(t) + x30(t)
)

local_id = assess_local_identifiability(ode, 0.99)

# save to file for this model
# create file to save results
file = open("./local_ID_VK_2009.txt", "w")
println(file, "Locally Identifiable parameters:")
for (key, value) in local_id
    if value == 1
        print(file, "$key, ")
    end
end
println(file, "")
println(file, "Nonidentifiable parameters:")
for (key, value) in local_id
    if value == 0
        print(file, "$key, ")
    end
end
close(file)

global_id = assess_identifiability(ode, 0.95)

# create file to save results
file = open("./global_ID_VK_2009.txt", "w")
println(file, "Globally Identifiable parameters:")
for (key, value) in local_id
    if value == ":globally"
        print(file, "$key, ")
    end
end
println(file, "")
println(file, "Nonidentifiable parameters:")
for (key, value) in local_id
    if value == ":nonidentifiable"
        print(file, "$key, ")
    end
end
close(file)