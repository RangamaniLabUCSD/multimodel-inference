using StructuralIdentifiability

ode = @ODEmodel(
    x1'(t) = -(k1*x1(t)*EGF - kd1*Ptase_R*x2(t)),
    x2'(t) = (k1*x1(t)*EGF - kd1*Ptase_R*x2(t)),
    x3'(t) = -(k6*x2(t)*(x3(t)/ (K6 + x3(t))) - kd6*GAP*(x4(t)/ (D6 + x4(t)))),
    x4'(t) = (k6*x2(t)*(x3(t)/ (K6 + x3(t))) - kd6*GAP*(x4(t)/ (D6 + x4(t)))),
    x5'(t) = -(k5*x4(t)*(x5(t)/(K5 + x5(t)))*(K_NFB*K_NFB/(K_NFB*K_NFB + x12(t)*x12(t))) - kd5*Ptase_Raf*(x6(t)/(D5 + x6(t)))) - (k_PFB*x14(t)*(x5(t)/(K_PFB + x5(t)))) ,
    x6'(t) = (k5*x4(t)*(x5(t)/(K5 + x5(t)))*(K_NFB*K_NFB/(K_NFB*K_NFB + x12(t)*x12(t))) - kd5*Ptase_Raf*(x6(t)/(D5 + x6(t)))) + (k_PFB*x14(t)*(x5(t)/(K_PFB + x5(t)))),
    x7'(t) = -(k4*x6(t)*(x7(t)/(K4 + x7(t))) - kd4*Ptase_MEK*(x8(t)/(D4 + x8(t)))),
    x8'(t) = (k4*x6(t)*(x7(t)/(K4 + x7(t))) - kd4*Ptase_MEK*(x8(t)/(D4 + x8(t)))),
    x9'(t) = -(k2*x8(t)*(x9(t)/(K2 + x9(t))) - kd2*x16(t)*(x10(t)/(D2 + x10(t)))),
    x10'(t) = (k2*x8(t)*(x9(t)/(K2 + x9(t))) - kd2*x16(t)*(x10(t)/(D2 + x10(t)))),
    x11'(t) = -(k3_F*x10(t)*(x11(t)/(K3 + x11(t)))*(x2(t)*x2(t)/(K3R*K3R + x2(t)*x2(t))) - kd3*Ptase_NFB*(x12(t)/(D3 + x12(t)))),
    x12'(t) = (k3_F*x10(t)*(x11(t)/(K3 + x11(t)))*(x2(t)*x2(t)/(K3R*K3R + x2(t)*x2(t))) - kd3*Ptase_NFB*(x12(t)/(D3 + x12(t)))),
    x13'(t) = -(k7*x10(t)*x2(t)*(x13(t)/(K7 + x13(t))) - kd7*Ptase_PFB*(x14(t)/(D7 + x14(t)))),
    x14'(t) = (k7*x10(t)*x2(t)*(x13(t)/(K7 + x13(t))) - kd7*Ptase_PFB*(x14(t)/(D7 + x14(t)))),
    x15'(t) = (dusp_basal*(1 + dusp_ind*(x10(t)*x10(t)/(K_dusp + x10(t)*x10(t))))*(0.3010299956639812/T_dusp)) - (x15(t)*(0.3010299956639812/T_dusp)),
    x16'(t) = (x15(t)*(0.3010299956639812/T_dusp)) - (x16(t)*(0.3010299956639812/T_dusp)),
    y1(t) = x10(t)
)

local_id = assess_local_identifiability(ode, 0.99)

# save to file for this model
# create file to save results
file = open("./local_ID_R_2015.txt", "w")
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
file = open("./global_ID_R_2015.txt", "w")
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
