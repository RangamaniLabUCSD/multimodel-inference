using StructuralIdentifiability

ode = @ODEmodel(
    x1'(t) = -(a1*x1(t)*(x1(t)act-x2(t))) + (d1*x2(t)) + (k2*x4(t)),
    x2'(t) = (a1*x1(t)*(x1(t)act-x2(t))) - (d1*x2(t)) - (k1*x2(t)),
    x3'(t) = -(a2*x3(t)*(x1(t)Pase-x4(t))) + (d2*x4(t)) + (k1*x2(t)) + (k3*x6(t)) + (d3*x6(t)) - (a3*x3(t)*x5(t)) + (k5*x9(t)) + (d5*x9(t)) - (a5*x7(t)*x3(t)),
    x4'(t) = (a2*x3(t)*(x1(t)Pase-x4(t))) - (d2*x4(t)) - (k2*x4(t)),
    x5'(t) = -(a3*x3(t)*x5(t)) + (d3*x6(t)) + (k4*x8(t)) + (kOff1*(x20(t)+x24(t)+x27(t))) - (kOn1*x5(t)*(x19(t)+x22(t)+x23(t))),
    x6'(t) = (a3*x3(t)*x5(t)) - (k3*x6(t)) - (d3*x6(t)),
    x7'(t) = -(a4*x7(t)*(x5(t)Pase-x8(t)-x11(t))) + (d4*x8(t)) + (k3*x6(t)) + (k6*x11(t)) + (d5*x9(t)) - (a5*x7(t)*x3(t)),
    x8'(t) = (a4*x7(t)*(x5(t)Pase-x8(t)-x11(t))) - (d4*x8(t)) - (k4*x8(t)),
    x9'(t) = (a5*x7(t)*x3(t)) - (d5*x9(t)) - (k5*x9(t)),
    x10'(t) = (k5*x9(t)) - (a6*x10(t)*(x5(t)Pase-x8(t)-x11(t))) + (d6*x11(t)) - (a7*x10(t)*x12(t)) + (d7*x13(t)) + (k7*x13(t)) + (d9*x15(t)) + (k9*x15(t)) - (a9*x14(t)*x10(t)) + (kOff3*(x21(t)+x25(t)+x26(t))),
    x11'(t) = (a6*x10(t)*(x5(t)Pase-x8(t)-x11(t)))  - (d6*x11(t)) - (k6*x11(t)),
    x12'(t) = -(a7*x10(t)*x12(t)) + (d7*x13(t)) + (k8*x17(t)) + (kOff2*(x22(t)+x24(t)+x25(t))) - (kOn2*x12(t)*(x19(t)+x20(t)+x21(t))),
    x13'(t) = (a7*x10(t)*x12(t)) - (d7*x13(t)) - (k7*x13(t)),
    x14'(t) = (k7*x13(t)) - (a8*x14(t)*(x12(t)Pase-x17(t)-x18(t))) + (d8*x17(t)) - (a9*x14(t)*x10(t)) + (d9*x15(t)) + (k10*x18(t)),
    x15'(t) = (a9*x14(t)*x10(t)) - (d9*x15(t)) - (k9*x15(t)),
    x16'(t) = -(a10*x16(t)*(x12(t)Pase-x17(t)-x18(t))) + (d10*x18(t)) + (k9*x15(t)) + (kOff4*(x23(t)+x26(t)+x27(t))),
    x17'(t) = (a8*x14(t)*(x12(t)Pase-x17(t)-x18(t)))  - (d8*x17(t)) - (k8*x17(t)),
    x18'(t) = (a10*x16(t)*(x12(t)Pase-x17(t)-x18(t))) - (k10*x18(t)) - (d10*x18(t)),
    x19'(t) = -(kOn1*x19(t)*x5(t)) - (kOn2*x19(t)*x12(t)) + (kOff1*x20(t)) + (kOff2*x22(t)) + (kOff3*x21(t)) + (kOff4*x23(t)),
    x20'(t) = (kOn1*x19(t)*x5(t)) + (kOff2*x24(t)) + (kOff4*x27(t)) - (kOff1*x20(t)) - (kOn2*x20(t)*x12(t)) - (k5*x20(t)*x3(t)),
    x21'(t) = -(kOn2*x21(t)*x12(t)) + (kOff2*x25(t)) - (kOff3*x21(t)) + (kOff4*x26(t)) + (k5*x20(t)*x3(t)),
    x22'(t) = (kOff1*x24(t)) + (kOn2*x19(t)*x12(t)) + (kOff3*x25(t)) - (kOff2*x22(t)) - (kOn1*x22(t)*x5(t)),
    x23'(t) = (kOff1*x27(t)) + (kOff3*x26(t)) - (kOn1*x23(t)*x5(t)) - (kOff4*x23(t)),
    x24'(t) = (kOn1*x22(t)*x5(t)) + (kOn2*x20(t)*x12(t)) - (kOff1*x24(t)) - (kOff2*x24(t)) - (k5*x24(t)*x3(t)),
    x25'(t) = -(kOff3*x25(t)) + (kOn2*x21(t)*x12(t)) - (k9*x25(t)) + (k5*x24(t)*x3(t)) - (kOff2*x25(t)),
    x26'(t) = (k9*x25(t)) - (kOff3*x26(t)) - (kOff4*x26(t)) + (k5*x27(t)*x3(t)),
    x27'(t) = (kOn1*x23(t)*x5(t)) - (kOff1*x27(t)) - (kOff4*x27(t)) - (k5*x27(t)*x3(t)),
    y1(t) = x16(t)
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