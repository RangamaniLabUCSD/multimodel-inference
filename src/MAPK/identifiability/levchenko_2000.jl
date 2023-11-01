using StructuralIdentifiability

ode = @ODEmodel(
    x1'(t) = -(a1*x2(t)*x1(t)) + (d1*x3(t)) + (k1*x3(t)),
    x2'(t) = -(a1*x2(t)*x1(t)) + (d1*x3(t)) + (k2*x5(t)),
    x3'(t) = (a1*x2(t)*x1(t)) - (d1*x3(t)) - (k1*x3(t)),
    x4'(t) = -(a2*x4(t)*(RAFPase - x5(t))) + (d2*x5(t)) + (k1*x3(t)) + (k3*x7(t)) + (d3*x7(t)) - (a3*x4(t)*x6(t)) + (k5*x10(t)) + (d5*x10(t)) - (a5*x8(t)*x4(t)),
    x5'(t) = (a2*x4(t)*(RAFPase - x5(t))) - (d2*x5(t)) - (k2*x5(t)),
    x6'(t) = -(a3*x4(t)*x6(t)) + (d3*x7(t)) + (k4*x9(t)) + (kOff1*(x21(t) + x25(t) + x28(t))) - (kOn1*x6(t)*((total_scaffold - x21(t) - x23(t) - x24(t) - x25(t) - x26(t) - x27(t) - x28(t)) + x23(t) + x24(t))),
    x7'(t) = (a3*x4(t)*x6(t)) - (k3*x7(t)) - (d3*x7(t)),
    x8'(t) = -(a4*x8(t)*(MEKPase - x9(t) - x12(t))) + (d4*x9(t)) + (k3*x7(t)) + (k6*x12(t)) + (d5*x10(t)) - (a5*x8(t)*x4(t)),
    x9'(t) = (a4*x8(t)*(MEKPase - x9(t) - x12(t))) - (d4*x9(t)) - (k4*x9(t)),
    x10'(t) = (a5*x8(t)*x4(t)) - (d5*x10(t)) - (k5*x10(t)),
    x11'(t) = (k5*x10(t)) - (a6*x11(t)*(MEKPase - x9(t) - x12(t))) + (d6*x12(t)) - (a7*x11(t)*x13(t)) + (d7*x14(t)) + (k7*x14(t)) + (d9*x16(t)) + (k9*x16(t)) - (a9*x15(t)*x11(t)) + (kOff3*(x22(t) + x26(t) + x27(t))),
    x12'(t) = (a6*x11(t)*(MEKPase - x9(t) - x12(t)))  - (d6*x12(t)) - (k6*x12(t)),
    x13'(t) = -(a7*x11(t)*x13(t)) + (d7*x14(t)) + (k8*x18(t)) + (kOff2*(x23(t) + x25(t) + x26(t))) - (kOn2*x13(t)*((total_scaffold - x21(t) - x23(t) - x24(t) - x25(t) - x26(t) - x27(t) - x28(t)) + x21(t) + x22(t))),
    x14'(t) = (a7*x11(t)*x13(t)) - (d7*x14(t)) - (k7*x14(t)),
    x15'(t) = (k7*x14(t)) - (a8*x15(t)*(MAPKPase - x18(t) - x19(t))) + (d8*x18(t)) - (a9*x15(t)*x11(t)) + (d9*x16(t)) + (k10*x19(t)),
    x16'(t) = (a9*x15(t)*x11(t)) - (d9*x16(t)) - (k9*x16(t)),
    x17'(t) = -(a10*x17(t)*(MAPKPase - x18(t) - x19(t))) + (d10*x19(t)) + (k9*x16(t)) + (kOff4*(x24(t) + x27(t) + x28(t))),
    x18'(t) = (a8*x15(t)*(MAPKPase - x18(t) - x19(t)))  - (d8*x18(t)) - (k8*x18(t)),
    x19'(t) = (a10*x17(t)*(MAPKPase - x18(t) - x19(t))) - (k10*x19(t)) - (d10*x19(t)),
    # x20'(t) = -(kOn1*(total_scaffold - x21(t) - x23(t) - x24(t) - x25(t) - x26(t) - x27(t) - x28(t))*x6(t)) - (kOn2*(total_scaffold - x21(t) - x23(t) - x24(t) - x25(t) - x26(t) - x27(t) - x28(t))*x13(t)) + (kOff1*x21(t)) + (kOff2*x23(t)) + (kOff3*x22(t)) + (kOff4*x24(t)),
    x21'(t) = (kOn1*(total_scaffold - x21(t) - x23(t) - x24(t) - x25(t) - x26(t) - x27(t) - x28(t))*x6(t)) + (kOff2*x25(t)) + (kOff4*x28(t)) - (kOff1*x21(t)) - (kOn2*x21(t)*x13(t)) - (k5*x21(t)*x4(t)),
    x22'(t) = -(kOn2*x22(t)*x13(t)) + (kOff2*x26(t)) - (kOff3*x22(t)) + (kOff4*x27(t)) + (k5*x21(t)*x4(t)),
    x23'(t) = (kOff1*x25(t)) + (kOn2*(total_scaffold - x21(t) - x23(t) - x24(t) - x25(t) - x26(t) - x27(t) - x28(t))*x13(t)) + (kOff3*x26(t)) - (kOff2*x23(t)) - (kOn1*x23(t)*x6(t)),
    x24'(t) = (kOff1*x28(t)) + (kOff3*x27(t)) - (kOn1*x24(t)*x6(t)) - (kOff4*x24(t)),
    x25'(t) = (kOn1*x23(t)*x6(t)) + (kOn2*x21(t)*x13(t)) - (kOff1*x25(t)) - (kOff2*x25(t)) - (k5*x25(t)*x4(t)),
    x26'(t) = -(kOff3*x26(t)) + (kOn2*x22(t)*x13(t)) - (k9*x26(t)) + (k5*x25(t)*x4(t)) - (kOff2*x26(t)),
    x27'(t) = (k9*x26(t)) - (kOff3*x27(t)) - (kOff4*x27(t)) + (k5*x28(t)*x4(t)),
    x28'(t) = (kOn1*x24(t)*x6(t)) - (kOff1*x28(t)) - (kOff4*x28(t)) - (k5*x28(t)*x4(t)),
    y1(t) = x17(t)
)

local_id = assess_local_identifiability(ode, 0.99)

# save to file for this model
# create file to save results
file = open("./local_ID_L_2000.txt", "w")
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
file = open("./global_ID_L_2000.txt", "w")
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