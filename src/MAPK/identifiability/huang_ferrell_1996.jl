using StructuralIdentifiability

ode = @ODEmodel(

    x1'(t) = (a1*(MKKK_tot - x2(t) - x1(t) - x3(t))*(E1_tot - x1(t)))-  (d1*x1(t))- (k1*x1(t)),
    x2'(t) = -(a2*x2(t)*(E2_tot - x3(t)))+ (d2*x3(t))+ (k1*x1(t))+ (k3*x4(t))+ (d3*x4(t))- (a3*x2(t)* (MKK_tot - x5(t) - x8(t) - x4(t) - x7(t) - x6(t) - x9(t) - x10(t) - x13(t)))+ (k5*x7(t))+ (d5*x7(t))- (a5*x5(t)*x2(t)),
    x3'(t) = (a2*x2(t)*(E2_tot - x3(t)))- (d2*x3(t))- (k2*x3(t)),
    x4'(t) = (a3*x2(t)* (MKK_tot - x5(t) - x8(t) - x4(t) - x7(t) - x6(t) - x9(t) - x10(t) - x13(t)))- (d3*x4(t))- (k3*x4(t)),
    x5'(t) = -(a4*x5(t)*(MKKPase_tot - x6(t) - x9(t)))+ (d4*x6(t))+ (k3*x4(t))+ (k6*x9(t))+ (d5*x7(t))- (a5*x5(t)*x2(t)),
    x6'(t) = (a4*x5(t)*(MKKPase_tot - x6(t) - x9(t)))- (d4*x6(t))- (k4*x6(t)),
    x7'(t) = (a5*x5(t)*x2(t))- (d5*x7(t))- (k5*x7(t)),
    x8'(t) = (k5*x7(t))- (a6*x8(t)*(MKKPase_tot - x6(t) - x9(t)))+ (d6*x9(t))- (a7*x8(t)*(MAPK_tot - x11(t) - x14(t) - x10(t) - x13(t) - x12(t) - x15(t)))+ (d7*x10(t))+ (k7*x10(t))+ (d9*x13(t))+ (k9*x13(t))- (a9*x11(t)*x8(t)),
    x9'(t) = (a6*x8(t)*(MKKPase_tot - x6(t) - x9(t)))- (d6*x9(t))- (k6*x9(t)),
    x10'(t) = (a7*x8(t)*(MAPK_tot - x11(t) - x14(t) - x10(t) - x13(t) - x12(t) - x15(t)))- (d7*x10(t))- (k7*x10(t)),
    x11'(t) = (k7*x10(t))- (a8*x11(t)*(MKKPase_tot - x6(t) - x9(t)))+ (d8*x12(t))- (a9*x11(t)*x8(t))+ (d9*x13(t))+ (k10*x15(t)),
    x12'(t) = (a8*x11(t)*(MKKPase_tot - x6(t) - x9(t)))- (d8*x12(t))- (k8*x12(t)),
    x13'(t) = (a9*x11(t)*x8(t))- (d9*x13(t))- (k9*x13(t)),
    x14'(t) = -(a10*x14(t)*(MKKPase_tot - x6(t) - x9(t)))+ (d10*x15(t))+ (k9*x13(t)),
    x15'(t) = (a10*x14(t)*(MKKPase_tot - x6(t) - x9(t)))- (d10*x15(t))- (k10*x15(t)),
    y1(t) = x14(t) + x15(t)
)

local_id = assess_local_identifiability(ode, 0.99)

# save to file for this model
# create file to save results
file = open("./local_ID_HF_1996.txt", "w")
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
file = open("./global_ID_HF_1996.txt", "w")
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
