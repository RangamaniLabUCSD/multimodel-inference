using StructuralIdentifiability

ode = @ODEmodel(
    # Note we replace the E term with E 
    # for anaysis because the code cannot handle exp on the RHS 
    x1'(t) = ((k3*x3(t)-f*kn3*E*x1(t))) -((k1*x1(t)*x4(t)-kn1*x2(t))),
    x2'(t) = ((k1*x1(t)*x4(t)-kn1*x2(t))) -((f*k2*E*x2(t)))-((k4*x2(t)*x2(t)-k2_4*x5(t))),
    x3'(t) = ((f*k2*E*x2(t)))+((k3*x3(t)-f*kn3*E*x1(t)))+((k6*E*x8(t))),
    x4'(t) = -((k1*x1(t)*x4(t)-kn1*x2(t))),
    x5'(t) = ((k4*x2(t)*x2(t)-k2_4*x5(t))) -((f*k5*E*x5(t)))-((k7*f*x5(t)-kn7*x6(t))),
    x6'(t) = ((k7*f*x5(t)-kn7*x6(t)))-((k8*E*x6(t))),
    x3'(t) = ((f*k2*E*x2(t)))+((k6*E*x8(t))),
    x8'(t) = ((f*k5*E*x5(t)))+((k8*E*x6(t))) -((k6*E*x8(t))),
    x9'(t) = (((V_10*x10(t))/(K_10+x10(t))))-(((k9*2.0*(x5(t)+x8(t)+x6(t))*x9(t))/(K_9+x9(t)))),
    x10'(t) = (((k9*2.0*(x5(t)+x8(t)+x6(t))*x9(t))/(K_9+x9(t))))+(((x27(t)*x11(t)*k27)/(K_27+x11(t)))) -(((V_10*x10(t))/(K_10+x10(t))))-((k11*x10(t)*x12(t)-kn11*x11(t))),
    x11'(t) = ((k_13*x15(t)))+((k11*x10(t)*x12(t)-kn11*x11(t))) -(((x27(t)*x11(t)*k27)/(K_27+x11(t))))-((k12*x14(t)*x11(t)-kn12*x15(t))),
    x12'(t) = (((V_28*x13(t))/(K_28+x13(t))))-((k11*x10(t)*x12(t)-kn11*x11(t))),
    x13'(t) = (((x27(t)*x11(t)*k27)/(K_27+x11(t))))-(((V_28*x13(t))/(K_28+x13(t)))),
    x14'(t) = ((k15*x18(t)))-((k12*x14(t)*x11(t)-kn12*x15(t))),
    x15'(t) = ((k12*x14(t)*x11(t)-kn12*x15(t)))-((k_13*x15(t))),
    x16'(t) = ((k_13*x15(t)))+((k17*x20(t))) -((k14*x16(t)*x17(t)-kn14*x18(t)))-((k16*x16(t)*x19(t)-kn16*x20(t))),
    x17'(t) = ((k15*x18(t)))-((k14*x16(t)*x17(t)-kn14*x18(t))),
    x18'(t) = ((k14*x16(t)*x17(t)-kn14*x18(t)))-((k15*x18(t))),
    x19'(t) = (((V_18*x21(t))/(K_18+x21(t))))-((k16*x16(t)*x19(t)-kn16*x20(t))),
    x20'(t) = ((k16*x16(t)*x19(t)-kn16*x20(t)))-((k17*x20(t))),
    x21'(t) = ((k17*x20(t)))-(((V_18*x21(t))/(K_18+x21(t)))),
    x22'(t) = (((V_20*x23(t))/(K_20+x23(t))))-(((x21(t)*x22(t)*k19)/(K_19+x22(t)))),
    x23'(t) = (((x21(t)*x22(t)*k19)/(K_19+x22(t))))+(((V_22*x24(t))/(K_22+x24(t)))) -(((V_20*x23(t))/(K_20+x23(t))))-(((x21(t)*x23(t)*k21)/(K_21+x23(t)))),
    x24'(t) = (((x21(t)*x23(t)*k21)/(K_21+x23(t))))-(((V_22*x24(t))/(K_22+x24(t)))),
    x25'(t) = (((V_24*x26(t))/(K_24+x26(t))))-(((k23*x25(t)*(x23(t)+x24(t)))/(K_23+x25(t)))),
    x26'(t) = (((k23*x25(t)*(x23(t)+x24(t)))/(K_23+x25(t))))+(((V_26*x27(t))/(K_26+x27(t)))) -(((V_24*x26(t))/(K_24+x26(t))))-(((k25*x26(t)*(x23(t)+x24(t)))/(K_25+x26(t)))),
    x27'(t) = (((k25*x26(t)*(x23(t)+x24(t)))/(K_25+x26(t))))-(((V_26*x27(t))/(K_26+x27(t)))),
    y1(t) = x27(t)
)

local_id = assess_local_identifiability(ode, 0.99)

# save to file for this model
# create file to save results
file = open("./local_ID_BF_2000.txt", "w")
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
file = open("./global_ID_BF_2000.txt", "w")
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
