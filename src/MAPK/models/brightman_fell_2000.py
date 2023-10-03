import equinox as eqx
import jax.numpy as jnp

class brightman_fell_2000(eqx.Module):
    """ Note this is the code at https://models.physiomeproject.org/jnp.exposure/55e182564e746cc9bac6b03ad7778d4d/brightman_fell_2000.cellml/@@cellml_codegen/Python
    
    Which comes from the Brightman-Fell paper entry on the CellML site https://models.physiomeproject.org/jnp.exposure/55e182564e746cc9bac6b03ad7778d4d"""
    # transient: bool True for transient EGF stim, False for sustained
    transient: any
    def __init__(self, transient=True): # defaults to sustained stim
        self.transient = transient
    
    def __call__(self, t, y, args):

        # unpack state
        Rs, RL, Ri, L, R2L2, R2_CPP, Li, R2i, Shc, ShcP, ShcGS, GS, GSP,\
            RasGDP, Ras_ShcGS, RasGTP, GAP, Ras_GAP, Raf, Ras_Raf, Rafa, \
            MEK, MEKP, MEKPP, ERK, ERKP, ERKPP = y
        
        # unpack parameters
        k1, kn1, DT, E, k2, f, kn3, k3, k2_4, k4, k5, k6, k7, kn7, k8, \
            k9, K_9, V_10, K_10, k11, kn11, k12, kn12, k_13, k14, kn14, k15, \
            k16, kn16, k17, V_18, K_18, k19, K_19, V_20, K_20, k21, K_21, V_22, \
            K_22, k23, K_23, V_24, K_24, k25, K_25, V_26, K_26, k27, K_27, \
            V_28, K_28 = args
        
        complex_term = E + (1 - E)*(1 - (jnp.exp(-(t/DT)**3)))
        
        # define fluxes
        v1 = k1*Rs*L-kn1*RL
        v2 = f*k2*(complex_term)*RL
        v3 = k3*Ri-f*kn3*(complex_term)*Rs
        v4 = k4*RL*RL-k2_4*R2L2
        v5 = f*k5*(complex_term)*R2L2
        v6 = k6*(complex_term)*R2i
        v7 = k7*f*R2L2-kn7*R2_CPP
        v8 = k8*(complex_term)*R2_CPP
        v9 = (k9*2.0*(R2L2+R2i+R2_CPP)*Shc)/(K_9+Shc)
        v10 = (V_10*ShcP)/(K_10+ShcP)
        v11 = k11*ShcP*GS-kn11*ShcGS
        v12 = k12*RasGDP*ShcGS-kn12*Ras_ShcGS
        v13 = k_13*Ras_ShcGS
        v14 = k14*RasGTP*GAP-kn14*Ras_GAP
        v15 = k15*Ras_GAP
        v16 = k16*RasGTP*Raf-kn16*Ras_Raf
        v17 = k17*Ras_Raf
        v18 = (V_18*Rafa)/(K_18+Rafa)
        v19 = (Rafa*MEK*k19)/(K_19+MEK)
        v20 = (V_20*MEKP)/(K_20+MEKP)
        v21 = (Rafa*MEKP*k21)/(K_21+MEKP)
        v22 = (V_22*MEKPP)/(K_22+MEKPP)
        v23 = (k23*ERK*(MEKP+MEKPP))/(K_23+ERK)
        v24 = (V_24*ERKP)/(K_24+ERKP)
        v25 = (k25*ERKP*(MEKP+MEKPP))/(K_25+ERKP)
        v26 = (V_26*ERKPP)/(K_26+ERKPP)
        v27 = (ERKPP*ShcGS*k27)/(K_27+ShcGS)
        v28 = (V_28*GSP)/(K_28+GSP)

        # rates
        dy_dt_0 = v3 -v1 # Rs
        dy_dt_1 = v1 -v2-v4 # RL
        dy_dt_2 = v2+v3+v6 # Ri
        if self.transient:
            dy_dt_3 = -v1 # L
        else:
            dy_dt_3 = 0.0
        dy_dt_4 = v4 -v5-v7 # R2L2
        dy_dt_5 = v7-v8 # R2-CPP
        dy_dt_6 = v2+v6 # Li
        dy_dt_7 = v5+v8 -v6 # R2i
        dy_dt_8 = v10-v9 # Shc
        dy_dt_9 = v9+v27 -v10-v11 # ShcP
        dy_dt_10 = v13+v11 -v27-v12 # SchGS
        dy_dt_11 = v28-v11 # GS
        dy_dt_12 = v27-v28 # GSP
        dy_dt_13 = v15-v12 # RasGDP
        dy_dt_14 = v12-v13 # Ras-ShcGS
        dy_dt_15 = v13+v17 -v14-v16 # RasGTP
        dy_dt_16 = v15-v14 # GAP
        dy_dt_17 = v14-v15 # Ras-GAP
        dy_dt_18 = v18-v16 # Raf
        dy_dt_19 = v16-v17 # Ras-Raf
        dy_dt_20 = v17-v18 # Rafa v17-v19-v21-v18
        dy_dt_21 = v20-v19 # MEK
        dy_dt_22 = v19+v22 -v20-v21 # MEKP
        dy_dt_23 = v21-v22 # MEKPP
        dy_dt_24 = v24-v23 # ERK
        dy_dt_25 = v23+v26 -v24-v25 # ERKP
        dy_dt_26 = v25-v26 # ERKPP

        return (dy_dt_0, dy_dt_1, dy_dt_2, dy_dt_3, dy_dt_4, dy_dt_5, dy_dt_6,
                 dy_dt_7, dy_dt_8, dy_dt_9, dy_dt_10, dy_dt_11, dy_dt_12, 
                 dy_dt_13, dy_dt_14, dy_dt_15, dy_dt_16, dy_dt_17, dy_dt_18, 
                 dy_dt_19, dy_dt_20, dy_dt_21, dy_dt_22, dy_dt_23, dy_dt_24, 
                 dy_dt_25, dy_dt_26)

    
    def get_initial_conditions(self):
        """ Function to get nominal initial conditions for the model. """
        y0_dict = {
            'Rs':11100.0,
            'RL':0.0,
            'Ri':3900.0,
            'L':1e-07,
            'R2L2':0.0,
            'R2_CPP':0.0,
            'Li':0.0,
            'R2i':0.0,
            'Shc':30000.0,
            'ShcP':0.0,
            'ShcGS':0.0,
            'GS':20000.0,
            'GSP':0.0,
            'RasGDP':19800.0,
            'Ras_ShcGS':0.0,
            'RasGTP':200.0,
            'GAP':15000.0,
            'Ras_GAP':0.0,
            'Raf':10000.0,
            'Ras_Raf':0.0,
            'Rafa':0.0,
            'MEK':360000.0,
            'MEKP':0.0,
            'MEKPP':0.0,
            'ERK':750000.0,
            'ERKP':0.0,
            'ERKPP':0.0,
        }

        y0_tup = tuple([y0_dict[key] for key in y0_dict.keys()])

        return y0_dict, y0_tup

        
    
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
            'K_28': 20000.0,}

        param_list = [param_dict[key] for key in param_dict]

        return param_dict, param_list

