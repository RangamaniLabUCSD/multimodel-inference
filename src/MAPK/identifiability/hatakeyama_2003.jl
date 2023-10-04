using StructuralIdentifiability

ode = @ODEmodel(
    x1'(t) = -(kf1*x1(t)*x4(t)-kb1*x5(t)),
    x2'(t) = -(kf5*x8(t)*x2(t)-kb5*x9(t))+((V10*x11(t))/(k10+x11(t))),
    x3'(t) = -(kf23*x8(t)*x3(t)-kb23*x15(t))+((V26*x17(t))/(k26+x17(t))),
    x4'(t) = -(kf1*x1(t)*x4(t)-kb1*x5(t)),
    x5'(t) = (kf1*x1(t)*x4(t)-kb1*x5(t)) - 2.0*(kf2*x5(t)*x5(t)-kb2*x6(t)),
    x6'(t) = (kf2*x5(t)*x5(t)-kb2*x6(t))-(kf3*x6(t)-kb3*x8(t))+((V4*x8(t))/(k4+x8(t))),
    x7'(t) = (kf34*x8(t)-kb34*x7(t)),
    x8'(t) = (kf3*x6(t)-kb3*x8(t))-((V4*x8(t))/(k4+x8(t)))-(kf5*x8(t)*x2(t)-kb5*x9(t))+(kf8*x12(t)-kb8*x13(t)*x8(t))-(kf23*x8(t)*x3(t)-kb23*x15(t))+(kf25*x16(t)-kb25*x8(t)*x17(t))-(kf34*x8(t)-kb34*x7(t)),
    x9'(t) = (kf5*x8(t)*x2(t)-kb5*x9(t))-(kf6*x9(t)-kb6*x10(t)),
    x10'(t) = (kf6*x9(t)-kb6*x10(t))-(kf7*x10(t)*x14(t)-kb7*x12(t)),
    x11'(t) = (kf9*x13(t)-kb9*x14(t)*x11(t))-((V10*x11(t))/(k10+x11(t))) ,
    x12'(t) = (kf7*x10(t)*x14(t)-kb7*x12(t))-(kf8*x12(t)-kb8*x13(t)*x8(t)),
    x13'(t) = (kf8*x12(t)-kb8*x13(t)*x8(t))-(kf9*x13(t)-kb9*x14(t)*x11(t)),
    x14'(t) = -(kf7*x10(t)*x14(t)-kb7*x12(t))+(kf9*x13(t)-kb9*x14(t)*x11(t)),
    x15'(t) = (kf23*x8(t)*x3(t)-kb23*x15(t))-(kf24*x15(t)-kb24*x16(t)),
    x16'(t) = (kf24*x15(t)-kb24*x16(t))-(kf25*x16(t)-kb25*x8(t)*x17(t)),
    x17'(t) = (kf25*x16(t)-kb25*x8(t)*x17(t))-((V26*x17(t))/(k26+x17(t))),
    x18'(t) = ((kf11*x13(t)*x19(t))/(k11+x19(t)))-((V12*x18(t))/(k12+x18(t))),
    x19'(t) = -((kf11*x13(t)*x19(t))/(k11+x19(t)))+((V12*x18(t))/(k12+x18(t))) ,
    x20'(t) = ((V32*x29(t))/(k32*(1.0+x28(t)/k30)+x29(t)))-((kf33*PP2A*x20(t))/(k33*(1.0+x23(t)/k16+x24(t)/k18+x29(t)/k31)+x20(t))),
    x21'(t) = -((kf14*(x20(t)+E)*x21(t))/(k14+x21(t)))+((kf13*x18(t)*x22(t))/(k13+x22(t))),
    x22'(t) = ((kf14*(x20(t)+E)*x21(t))/(k14+x21(t)))-((kf13*x18(t)*x22(t))/(k13+x22(t))),
    x23'(t) = ((kf15*x21(t)*x30(t))/(k15*(1.0+x23(t)/k17)+x30(t)))-((kf16*PP2A*x23(t))/(k16*(1.0+x24(t)/k18+x29(t)/k31+x20(t)/k33)+x23(t)))-((kf17*x21(t)*x23(t))/(k17*(1.0+x30(t)/k15)+x23(t)))+((kf18*PP2A*x24(t))/(k18*(1.0+x23(t)/k16+x29(t)/k31+x20(t)/k33)+x24(t))),
    x24'(t) = ((kf17*x21(t)*x23(t))/(k17*(1.0+x30(t)/k15)+x23(t)))-((kf18*PP2A*x24(t))/(k18*(1.0+x23(t)/k16+x29(t)/k31+x20(t)/k33)+x24(t))),
    x25'(t) = ((V28*x26(t))/(k28+x26(t)))-((kf27*x17(t)*x25(t))/(k27+x25(t))),
    x26'(t) = -((V28*x26(t))/(k28+x26(t)))+((kf27*x17(t)*x25(t))/(k27+x25(t)))-(kf29*x26(t)*x27(t)-kb29*x28(t)),
    x27'(t) = -(kf29*x26(t)*x27(t)-kb29*x28(t)),
    x28'(t) = (kf29*x26(t)*x27(t)-kb29*x28(t))-((V30*x28(t))/(k30*(1.0+x29(t)/k32)+x28(t)))+((kf31*PP2A*x29(t))/(k31*(1.0+x23(t)/k16+x24(t)/k18+x20(t)/k33)+x29(t))),
    x29'(t) = ((V30*x28(t))/(k30*(1.0+x29(t)/k32)+x28(t)))-((kf31*PP2A*x29(t))/(k31*(1.0+x23(t)/k16+x24(t)/k18+x20(t)/k33)+x29(t)))-((V32*x29(t))/(k32*(1.0+x28(t)/k30)+x29(t)))+((kf33*PP2A*x20(t))/(k33*(1.0+x23(t)/k16+x24(t)/k18+x29(t)/k31)+x20(t))),
    x30'(t) = -((kf15*x21(t)*x30(t))/(k15*(1.0+x23(t)/k17)+x30(t)))+((kf16*PP2A*x23(t))/(k16*(1.0+x24(t)/k18+x29(t)/k31+x20(t)/k33)+x23(t))),
    x31'(t) = -((kf19*x24(t)*x31(t))/(k19*(1.0+x32(t)/k21)+x31(t)))+((kf20*MKP3*x32(t))/(k20*(1.0+x33(t)/k22)+x32(t))),
    x32'(t) = ((kf19*x24(t)*x31(t))/(k19*(1.0+x32(t)/k21)+x31(t)))-((kf20*MKP3*x32(t))/(k20*(1.0+x33(t)/k22)+x32(t)))-((kf21*x24(t)*x32(t))/(k21*(1.0+x31(t)/k19)+x32(t)))+((kf22*MKP3*x33(t))/(k22*(1.0+x32(t)/k20)+x33(t))),
    x33'(t) = ((kf21*x24(t)*x32(t))/(k21*(1.0+x31(t)/k19)+x32(t)))-((kf22*MKP3*x33(t))/(k22*(1.0+x32(t)/k20)+x33(t))),
    y1(t) = x33(t)
)

local_id = assess_local_identifiability(ode, 0.99)

# save to file for this model
# create file to save results
file = open("./local_ID_K_2003.txt", "w")
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
file = open("./global_ID_K_2003.txt", "w")
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
