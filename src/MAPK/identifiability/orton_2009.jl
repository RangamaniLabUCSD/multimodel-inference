using StructuralIdentifiability

ode = @ODEmodel(
    x1'(t) = ( - ((kcat_bRaf_Activation * x4(t) * x1(t) / (km_bRaf_Activation + x1(t)))) + ((kcat_bRaf_Deactivation * x2(t) * Raf1PPtase / (km_bRaf_Deactivation + x2(t)))) - ((kcat_bRaf_Activation_Ras * x1(t) * x21(t) / (km_bRaf_Activation_Ras + x1(t))))),
    x2'(t) = (((kcat_bRaf_Activation * x4(t) * x1(t) / (km_bRaf_Activation + x1(t)))) - ((kcat_bRaf_Deactivation * x2(t) * Raf1PPtase / (km_bRaf_Deactivation + x2(t)))) + ((kcat_bRaf_Activation_Ras * x1(t) * x21(t) / (km_bRaf_Activation_Ras + x1(t))))),
    x3'(t) = ( - ((kcat_Rap1_Activation * x6(t) * x3(t) / (km_Rap1_Activation + x3(t)))) + ((kcat_Rap1_Deactivation * x4(t) * Rap1Gap / (km_Rap1_Deactivation + x4(t))))),
    x4'(t) = (((kcat_Rap1_Activation * x6(t) * x3(t) / (km_Rap1_Activation + x3(t)))) - ((kcat_Rap1_Deactivation * x4(t) * Rap1Gap / (km_Rap1_Deactivation + x4(t))))),
    x5'(t) = ( - ((kcat_C3G_Activation * x25(t) * x5(t) / (km_C3G_Activation + x5(t)))) + ((k1_C3G_Deactivation * x6(t)))),
    x6'(t) = (((kcat_C3G_Activation * x25(t) * x5(t) / (km_C3G_Activation + x5(t)))) - ((k1_C3G_Deactivation * x6(t)))),
    x7'(t) = (((k1_EGFReceptor_Degradation * x25(t))) + ((k1_EGFReceptor_Degradation_Free * x24(t)))),
    x8'(t) = ( - ((Kcat_Akt_Activation * x11(t) * x8(t) / (km_Akt_Activation + x8(t)))) + ((k1_Akt_Deactivation * x9(t)))),
    x9'(t) = (((Kcat_Akt_Activation * x11(t) * x8(t) / (km_Akt_Activation + x8(t)))) - ((k1_Akt_Deactivation * x9(t)))),
    x10'(t) = ( - ((Kcat_PI3K_Activation_EGFR * x25(t) * x10(t) / (km_PI3K_Activation_EGFR + x10(t)))) - ((Kcat_PI3K_Activation_Ras * x10(t) * x21(t) / (km_PI3K_Activation_Ras + x10(t)))) + ((k1_PI3K_Deactivation * x11(t)))),
    x11'(t) = (((Kcat_PI3K_Activation_EGFR * x25(t) * x10(t) / (km_PI3K_Activation_EGFR + x10(t)))) + ((Kcat_PI3K_Activation_Ras * x10(t) * x21(t) / (km_PI3K_Activation_Ras + x10(t)))) - ((k1_PI3K_Deactivation * x11(t)))),
    x12'(t) = ( - ((Kcat_P90Rsk_Activation * x15(t) * x12(t) / (km_P90Rsk_Activation + x12(t)))) + ((k1_P90Rsk_Deactivation * x13(t)))),
    x13'(t) = (((Kcat_P90Rsk_Activation * x15(t) * x12(t) / (km_P90Rsk_Activation + x12(t)))) - ((k1_P90Rsk_Deactivation * x13(t)))),
    x14'(t) = ( - ((Kcat_Erk_Activation * x14(t) * x17(t) / (km_Erk_Activation + x14(t)))) + ((Kcat_Erk_Deactivation * x15(t) * PP2AActive / (km_Erk_Deactivation + x15(t))))),
    x15'(t) = (((Kcat_Erk_Activation * x14(t) * x17(t) / (km_Erk_Activation + x14(t)))) - ((Kcat_Erk_Deactivation * x15(t) * PP2AActive / (km_Erk_Deactivation + x15(t))))),
    x16'(t) = ( - ((Kcat_Mek_Activation * x19(t) * x16(t) / (km_Mek_Activation + x16(t)))) + ((Kcat_Mek_Deactivation * PP2AActive * x17(t) / (km_Mek_Deactivation + x17(t)))) - ((kcat_Mek_Activation_bRaf * x2(t) * x16(t) / (km_Mek_Activation_bRaf + x16(t))))),
    x17'(t) = (((Kcat_Mek_Activation * x19(t) * x16(t) / (km_Mek_Activation + x16(t)))) - ((Kcat_Mek_Deactivation * PP2AActive * x17(t) / (km_Mek_Deactivation + x17(t)))) + ((kcat_Mek_Activation_bRaf * x2(t) * x16(t) / (km_Mek_Activation_bRaf + x16(t))))),
    x18'(t) = ( - ((Kcat_Raf1_Activation * x21(t) * x18(t) / (km_Raf1_Activation + x18(t)))) + ((Kcat_Raf1_Deactivation * Raf1PPtase * x19(t) / (km_Raf1_Deactivation + x19(t)))) + ((Kcat_Raf1_Deactivation_Akt * x9(t) * x19(t) / (km_Raf1_Deactivation_Akt + x19(t))))),
    x19'(t) = (((Kcat_Raf1_Activation * x21(t) * x18(t) / (km_Raf1_Activation + x18(t)))) - ((Kcat_Raf1_Deactivation * Raf1PPtase * x19(t) / (km_Raf1_Deactivation + x19(t)))) - ((Kcat_Raf1_Deactivation_Akt * x9(t) * x19(t) / (km_Raf1_Deactivation_Akt + x19(t))))),
    x20'(t) = ( - ((Kcat_Ras_Activation * x23(t) * x20(t) / (km_Ras_Activation + x20(t)))) + ((Kcat_Ras_Deactivation * RasGapActive * x21(t) / (km_Ras_Deactivation + x21(t))))),
    x21'(t) = (((Kcat_Ras_Activation * x23(t) * x20(t) / (km_Ras_Activation + x20(t)))) - ((Kcat_Ras_Deactivation * RasGapActive * x21(t) / (km_Ras_Deactivation + x21(t))))),
    x22'(t) = ( - ((Kcat_Sos_Activation * x25(t) * x22(t) / (km_Sos_Activation + x22(t)))) + ((k1_Sos_Deactivation * x23(t))) + ((Kcat_Sos_Feedback_Deactivation * x13(t) * x23(t) / (km_Sos_Feedback_Deactivation + x23(t))))),
    x23'(t) = (((Kcat_Sos_Activation * x25(t) * x22(t) / (km_Sos_Activation + x22(t)))) - ((k1_Sos_Deactivation * x23(t))) - ((Kcat_Sos_Feedback_Deactivation * x13(t) * x23(t) / (km_Sos_Feedback_Deactivation + x23(t))))),
    x24'(t) = ( - ( - ((k2_EGF_Binding_Unbinding * x25(t)) - (k1_EGF_Binding_Unbinding * x24(t) * EGF))) + J_reaction_28 - ((k1_EGFReceptor_Degradation_Free * x24(t)))),
    x25'(t) = (( - ((k2_EGF_Binding_Unbinding * x25(t)) - (k1_EGF_Binding_Unbinding * x24(t) * EGF))) - ((k1_EGFReceptor_Degradation * x25(t)))),
    y1(t) = x15(t)
)


local_id = assess_local_identifiability(ode, 0.99)

# save to file for this model
# create file to save results
file = open("./local_ID_O_2009.txt", "w")
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
file = open("./global_ID_O_2009.txt", "w")
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
