using StructuralIdentifiability

ode = @ODEmodel(
    x1'(t) = -((5e-5*EGF)*x1(t)) +(d1*x11(t)) +(u1b*x17(t)),
    x2'(t) = -((1e-7/BIMOL)*x14(t)*x2(t)) +(u3*x15(t)) -((1e-7/BIMOL)*x17(t)*x2(t)) +(u2b*x19(t)) -((1e-4/BIMOL)*x18(t)*x2(t)) -((1e-5/BIMOL)*x19(t)*x2(t)) +(u2b*x22(t)) -((1e-4/BIMOL)*x21(t)*x2(t)) -((1e-5/BIMOL)*x22(t)*x2(t)),
    x3'(t) = -((1e-5/BIMOL)*x5(t)*x3(t)) -((1e-6/BIMOL)*x14(t)*x3(t)) -((1e-6/BIMOL)*x17(t)*x3(t)) +(u2a*x18(t)) +((1e-4/BIMOL)*x18(t)*x2(t)) +((1e-5/BIMOL)*x19(t)*x2(t)) +(u2a*x21(t)) +((1e-4/BIMOL)*x21(t)*x2(t)) +((1e-5/BIMOL)*x22(t)*x2(t)),
    x4'(t) = -((1e-5/BIMOL)*x11(t)*x4(t)) +(u1a*x14(t)) +(u1b*x17(t)) -((((3e-9/BIMOL)*ERKpp_SOS1_FB)*4)*x24(t)*x4(t)) +(q3*x25(t)),
    x5'(t) = -((1e-5/BIMOL)*x5(t)*x3(t)) +(u3*x15(t)),
    x6'(t) = -((1e-7/BIMOL)*x3(t)*x6(t)) +(d2*x13(t)) -(((6e-10/BIMOL)*ERKpp_RAF1_FB)*x24(t)*x6(t)) +(q6*x29(t)),
    x7'(t) = -(((1e-7/BIMOL)*2)*x13(t)*x7(t)) +(q1*x16(t)) -(((6e-10/BIMOL)*ERKpp_MEK_FB)*x24(t)*x7(t)) +(q4*x26(t)),
    x8'(t) = -(((3e-6/BIMOL)*2)*x20(t)*x8(t)) +(q2*x23(t)) -(((3e-6/BIMOL)*2)*x28(t)*x8(t)),
    x9'(t) = -(a0_ekarev*x24(t)*x9(t)) +(d0_ekarev*x30(t)),
    x10'(t) = -(a0_erktr*x24(t)*x10(t)) +(d0_erktr*x31(t)),
    x11'(t) = ((5e-5*EGF)*x1(t)) -(d1*x11(t)) -((1e-5/BIMOL)*x11(t)*x4(t)) +(u1a*x14(t)),
    x12'(t) = ((1e-5/BIMOL)*x5(t)*x3(t)) -(k3*x12(t)),
    x13'(t) = ((1e-7/BIMOL)*x3(t)*x6(t)) -(d2*x13(t)) -(((6e-10/BIMOL)*ERKpp_RAF1_FB)*x24(t)*x13(t)),
    x14'(t) = ((1e-5/BIMOL)*x11(t)*x4(t)) -(d1*x14(t)) -(u1a*x14(t)) -((1e-6/BIMOL)*x14(t)*x3(t)) -((1e-7/BIMOL)*x14(t)*x2(t)) +((5e-5*EGF)*x17(t)) +(u2a*x18(t)) +(u2b*x19(t)),
    x15'(t) = (k3*x12(t)) -(u3*x15(t)),
    x16'(t) = (((1e-7/BIMOL)*2)*x13(t)*x7(t)) -((1e-7/BIMOL)*x13(t)*x16(t)) -(q1*x16(t)) +((q1*2)*x20(t)) -(((6e-10/BIMOL)*ERKpp_MEK_FB)*x24(t)*x16(t)) +(q4*x27(t)),
    x17'(t) = (d1*x14(t)) -((5e-5*EGF)*x17(t)) -(u1b*x17(t)) -((1e-6/BIMOL)*x17(t)*x3(t)) -((1e-7/BIMOL)*x17(t)*x2(t)) +(u2a*x21(t)) +(u2b*x22(t)),
    x18'(t) = ((1e-6/BIMOL)*x14(t)*x3(t)) -(d1*x18(t)) -(u2a*x18(t)) +((5e-5*EGF)*x21(t)),
    x19'(t) = ((1e-7/BIMOL)*x14(t)*x2(t)) -(d1*x19(t)) -(u2b*x19(t)) +((5e-5*EGF)*x22(t)),
    x20'(t) = ((1e-7/BIMOL)*x13(t)*x16(t)) -((q1*2)*x20(t)) -(((6e-10/BIMOL)*ERKpp_MEK_FB)*x24(t)*x20(t)) +(q4*x28(t)),
    x21'(t) = (d1*x18(t)) +((1e-6/BIMOL)*x17(t)*x3(t)) -((5e-5*EGF)*x21(t)) -(u2a*x21(t)),
    x22'(t) = (d1*x19(t)) +((1e-7/BIMOL)*x17(t)*x2(t)) -((5e-5*EGF)*x22(t)) -(u2b*x22(t)),
    x23'(t) = (((3e-6/BIMOL)*2)*x20(t)*x8(t)) -((3e-6/BIMOL)*x20(t)*x23(t)) -(q2*x23(t)) +((q2*2)*x24(t)) +(((3e-6/BIMOL)*2)*x28(t)*x8(t)) -((3e-6/BIMOL)*x28(t)*x23(t)),
    x24'(t) = ((3e-6/BIMOL)*x20(t)*x23(t)) -((q2*2)*x24(t)) +((3e-6/BIMOL)*x28(t)*x23(t)),
    x25'(t) = ((((3e-9/BIMOL)*ERKpp_SOS1_FB)*4)*x24(t)*x4(t)) -((((3e-9/BIMOL)*ERKpp_SOS1_FB)*3)*x24(t)*x25(t)) -(q3*x25(t)) +((q3*2)*x32(t)),
    x26'(t) = (((6e-10/BIMOL)*ERKpp_MEK_FB)*x24(t)*x7(t)) -(((1e-7/BIMOL)*2)*x13(t)*x26(t)) +(q1*x27(t)) -(q4*x26(t)) +(q5*x27(t)),
    x27'(t) = (((6e-10/BIMOL)*ERKpp_MEK_FB)*x24(t)*x16(t)) +(((1e-7/BIMOL)*2)*x13(t)*x26(t)) -((1e-7/BIMOL)*x13(t)*x27(t)) +((q1*2)*x28(t)) -(q1*x27(t)) -(q4*x27(t)) +((q5*2)*x28(t)) -(q5*x27(t)),
    x28'(t) = (((6e-10/BIMOL)*ERKpp_MEK_FB)*x24(t)*x20(t)) +((1e-7/BIMOL)*x13(t)*x27(t)) -((q1*2)*x28(t)) -(q4*x28(t)) -((q5*2)*x28(t)),
    x29'(t) = (((6e-10/BIMOL)*ERKpp_RAF1_FB)*x24(t)*x13(t)) +(((6e-10/BIMOL)*ERKpp_RAF1_FB)*x24(t)*x6(t)) -(q6*x29(t)),
    x30'(t) = (a0_ekarev*x24(t)*x9(t)) -(d0_ekarev*x30(t)),
    x31'(t) = (a0_erktr*x24(t)*x10(t)) -(d0_erktr*x31(t)),
    x32'(t) = ((((3e-9/BIMOL)*ERKpp_SOS1_FB)*3)*x24(t)*x25(t)) -((((3e-9/BIMOL)*ERKpp_SOS1_FB)*2)*x24(t)*x32(t)) -((q3*2)*x32(t)) +((q3*3)*x33(t)),
    x33'(t) = ((((3e-9/BIMOL)*ERKpp_SOS1_FB)*2)*x24(t)*x32(t)) -(((3e-9/BIMOL)*ERKpp_SOS1_FB)*x24(t)*x33(t)) -((q3*3)*x33(t)) +((q3*4)*x34(t)),
    x34'(t) = (((3e-9/BIMOL)*ERKpp_SOS1_FB)*x24(t)*x33(t)) -((q3*4)*x34(t)),
    y1(t) = x24(t)
)


local_id = assess_local_identifiability(ode, 0.99)

# save to file for this model
# create file to save results
file = open("./local_ID_K_2017.txt", "w")
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
file = open("./global_ID_K_2017.txt", "w")
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
