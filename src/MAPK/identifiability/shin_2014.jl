using StructuralIdentifiability

ode = @ODEmodel(
    x1'(t) = ka38*EGF*(EGFR_tot - x1(t)) - kd38*x1(t),
    x2'(t) = ka39a*x1(t)*(SOS_tot - x2(t))*Grb2_tot/(1 + (x6(t)/ki39)*(x6(t)/ki39)*(x6(t)/ki39)) - kd39*x2(t),
    x3'(t) = kc40*x2(t)*(Ras_tot - x3(t)) - kc41*x3(t),
    x4'(t) = kc42*x3(t)*(Raf_tot - x4(t)) - kc43*x4(t),
    x5'(t) = kc44*x4(t)*(MEK_tot - x5(t)) - kc45*x5(t),
    x6'(t) = kc46*x5(t)*(ERK_tot - x6(t)) - kc47*x6(t),
    y1(t) = x6(t)
)

local_id = assess_local_identifiability(ode, 0.99)

# save to file for this model
# create file to save results
file = open("./local_ID_S_2014.txt", "w")
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
file = open("./global_ID_S_2014.txt", "w")
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