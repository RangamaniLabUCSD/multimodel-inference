import equinox as eqx
import jax.numpy as jnp

class brightman_fell_2000(eqx.Module):
    """ Note this is the code at https://models.physiomeproject.org/jnp.exposure/55e182564e746cc9bac6b03ad7778d4d/brightman_fell_2000.cellml/@@cellml_codegen/Python
    
    Which comes from the Brightman-Fell paper entry on the CellML site https://models.physiomeproject.org/jnp.exposure/55e182564e746cc9bac6b03ad7778d4d"""
    def __call__(self, t, y, constants):
        # unpack state
        Rs, RL, Ri, L, R2L2, R2_CPP, Li, R2i, Shc, ShcP, ShcGS, GS, GSP,\
            RasGDP, Ras_ShcGS, RasGTP, GAP, Ras_GAP, Raf, Ras_Raf, Rafa, \
            MEK, MEKP, MEKPP, ERK, ERKP, ERKPP = y
        
        # define rates
        v1 = constants[0]*Rs*L-constants[1]*RL
        dy_dt_3 = -v1
        v9 = (constants[26]*2.00000*(R2L2+R2i+R2_CPP)*Shc)/(constants[27]+Shc)
        v10 = (constants[28]*ShcP)/(constants[29]+ShcP)
        dy_dt_8 = v10-v9
        v24 = (constants[53]*ERKP)/(constants[54]+ERKP)
        v23 = (constants[51]*ERK*(MEKP+MEKPP))/(constants[52]+ERK)
        dy_dt_24 = v24-v23
        v3 = constants[10]*Ri-constants[9]*constants[8]*(constants[7]+(1.0-constants[7])*(1.0-jnp.exp(-(jnp.power(t/constants[6], 3.0)))))*Rs
        dy_dt_0 = v3-v1
        v2 = constants[5]*constants[4]*(constants[3]+(1.0-constants[3])*(1.0-jnp.exp(-(jnp.power(t/constants[2], 3.0)))))*RL
        v4 = constants[12]*RL*RL-constants[11]*R2L2
        dy_dt_1 = (v1-v2)-v4
        v26 = (constants[57]*ERKPP)/(constants[58]+ERKPP)
        v25 = (constants[55]*ERKP*(MEKP+MEKPP))/(constants[56]+ERKP)
        dy_dt_25 = ((v23+v26)-v24)-v25
        dy_dt_26 = v25-v26
        v13 = constants[34]*Ras_ShcGS
        v12 = constants[32]*RasGDP*ShcGS-constants[33]*Ras_ShcGS
        dy_dt_14 = v12-v13
        v6 = constants[19]*(constants[18]+(1.0-constants[18])*(1.0-jnp.exp(-(jnp.power(t/constants[17], 3.0)))))*R2i
        dy_dt_2 = v2+v3+v6
        dy_dt_6 = v2+v6
        v11 = constants[30]*ShcP*GS-constants[31]*ShcGS
        v27 = (ERKPP*ShcGS*constants[59])/(constants[60]+ShcGS)
        dy_dt_9 = (v27-v10)-v11
        dy_dt_10 = ((v13+v11)-v27)-v12
        v5 = constants[16]*constants[15]*(constants[14]+(1.0-constants[14])*(1.0-jnp.exp(-(jnp.power(t/constants[13], 3.0)))))*R2L2
        v7 = constants[20]*constants[21]*R2L2-constants[22]*R2_CPP
        dy_dt_4 = (v4-v5)-v7
        v28 = (constants[61]*GSP)/(constants[62]+GSP)
        dy_dt_11 = v28-v11
        dy_dt_12 = v27-v28
        v15 = constants[37]*Ras_GAP
        dy_dt_13 = v15-v12
        v14 = constants[35]*RasGTP*GAP-constants[36]*Ras_GAP
        dy_dt_16 = v15-v14
        dy_dt_17 = v14-v15
        v8 = constants[25]*(constants[24]+(1.0-constants[24])*(1.0-jnp.exp(-(jnp.power(t/constants[23], 3.0)))))*R2_CPP
        dy_dt_5 = v7-v8
        dy_dt_7 = (v5+v8)-v6
        v17 = constants[40]*Ras_Raf
        v16 = constants[38]*RasGTP*Raf-constants[39]*Ras_Raf
        dy_dt_15 = ((v13+v17)-v14)-v16
        dy_dt_19 = v16-v17
        v18 = (constants[41]*Rafa)/(constants[42]+Rafa)
        dy_dt_18 = v18-v16
        v19 = (Rafa*MEK*constants[43])/(constants[44]+MEK)
        v20 = (constants[45]*MEKP)/(constants[46]+MEKP)
        dy_dt_21 = v20-v19
        v21 = (Rafa*MEKP*constants[47])/(constants[48]+MEKP)
        dy_dt_20 = ((v17-v19)-v21)-v18
        v22 = (constants[49]*MEKPP)/(constants[50]+MEKPP)
        dy_dt_22 = ((v19+v22)-v20)-v21
        dy_dt_23 = v21-v22

        return (dy_dt_0, dy_dt_1, dy_dt_2, dy_dt_3, dy_dt_4, dy_dt_5, dy_dt_6, dy_dt_7, dy_dt_8, dy_dt_9, dy_dt_10, dy_dt_11, dy_dt_12, dy_dt_13, dy_dt_14, dy_dt_15, dy_dt_16, dy_dt_17, dy_dt_18, dy_dt_19, dy_dt_20, dy_dt_21, dy_dt_22, dy_dt_23, dy_dt_24, dy_dt_25, dy_dt_26)

    
    def get_initial_conditions(self):
        """ Function to get nominal initial conditions for the model. """
        Rs = 11100.0
        RL = 0.0
        Ri = 3900.0
        L = 1e-07
        R2L2 = 0.0
        R2_CPP = 0.0
        Li = 0.0
        R2i = 0.0
        Shc = 30000.0
        ShcP = 0.0
        ShcGS = 0.0
        GS = 20000.0
        GSP = 0.0
        RasGDP = 19800.0
        Ras_ShcGS = 0.0
        RasGTP = 200.0
        GAP = 15000.0
        Ras_GAP = 0.0
        Raf = 10000.0
        Ras_Raf = 0.0
        Rafa = 0.0
        MEK = 360000.0
        MEKP = 0.0
        MEKPP = 0.0
        ERK = 750000.0
        ERKP = 0.0
        ERKPP = 0.0

        return (Rs, RL, Ri, L, R2L2, R2_CPP, Li, R2i, Shc, ShcP, ShcGS, GS, GSP,\
                RasGDP, Ras_ShcGS, RasGTP, GAP, Ras_GAP, Raf, Ras_Raf, Rafa, \
                MEK, MEKP, MEKPP, ERK, ERKP, ERKPP)

        
    
    def get_nominal_params(self):
        """ Function to get nominal parameters for the model. """
        param_dict = {'k1': 384210000.0,
            'kn1': 0.73,
            'DT': 6.5,
            'E': 0.12,
            'k2': 0.7,
            'f': 0.2,
            'kn3': 0.7,
            'k3': 0.0484,
            'k2_4': 1e-07,
            'k4': 0.001383,
            'k5': 0.35,
            'k6': 0.35,
            'k7': 1.0,
            'kn7': 0.000347,
            'k8': 0.35,
            'k9': 12.0,
            'K_9': 6000.0,
            'V_10': 300000.0,
            'K_10': 6000.0,
            'k11': 0.002,
            'kn11': 3.81,
            'k12': 0.0163,
            'kn12': 10.0,
            'k_13': 15.0,
            'k14': 0.005,
            'kn14': 60.0,
            'k15': 720.0,
            'k16': 0.0012,
            'kn16': 3.0,
            'k17': 27.0,
            'V_18': 97000.0,
            'K_18': 6000.0,
            'k19': 50.0,
            'K_19': 9000.0,
            'V_20': 920000.0,
            'K_20': 600000.0,
            'k21': 50.0,
            'K_21': 9000.0,
            'V_22': 920000.0,
            'K_22': 600000.0,
            'k23': 8.3,
            'K_23': 90000.0,
            'V_24': 200000.0,
            'K_24': 600000.0,
            'k25': 8.3,
            'K_25': 90000.0,
            'V_26': 400000.0,
            'K_26': 600000.0,
            'k27': 1.6,
            'K_27': 600000.0,
            'V_28': 75.0,
            'K_28': 20000.0,
            'v_1': 1.0,
            'k_11': 0.0,}

        param_list = [param_dict[key] for key in param_dict]

        return param_dict, param_list

