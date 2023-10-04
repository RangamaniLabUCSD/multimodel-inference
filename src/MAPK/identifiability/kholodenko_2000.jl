using StructuralIdentifiability

ode = @ODEmodel(
    x1'(t) = ((v1*(MKKK_total - x1(t)))/((1+(x5(t)/KI))*(K1 + (MKKK_total - x1(t))))) - ((v2*x1(t))/(K2 + x1(t))), # NOTE we removed n from the exponent of (1+(x5(t)/KI))since it is nomainally set to 1.0
    x2'(t) = ((k3*x1(t)*(MKK_total - x2(t) - x3(t)))/(K3 + (MKK_total - x2(t) - x3(t)))) + ((v5*x3(t))/(K5 + x3(t))) - ((k4*x1(t)*x2(t))/(K4 + x2(t))) - ((v6*x2(t))/(K6 + x2(t))),
    x3'(t) = ((k4*x1(t)*x2(t))/(K4 + x2(t))) - ((v5*x3(t))/(K5 + x3(t))),
    x4'(t) = ((k7*x3(t)*(MAPK_total - x4(t) - x5(t)))/(K7 + (MAPK_total - x4(t) - x5(t)))) + ((v9*x5(t))/(K9 + x5(t))) - ((k8*x3(t)*x4(t))/(K8 + x4(t))) - ((v10*x4(t))/(K10 + x4(t))),
    x5'(t) = ((k8*x3(t)*x4(t))/(K8 + x4(t))) - ((v9*x5(t))/(K9 + x5(t))),
    y1(t) = x5(t)
)

local_id = assess_local_identifiability(ode, 0.99)

# save to file for this model
# create file to save results
file = open("./local_ID_K_2000.txt", "w")
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
file = open("./global_ID_K_2000.txt", "w")
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