import equinox as eqx
import jax.numpy as jnp

class hatakeyama_2003(eqx.Module):
    """ Note this is the code at https://models.physiomeproject.org/exposure/c14e77a831ec3f1e56ef03240986a8b7/hatakeyama_kimura_naka_kawasaki_yumoto_ichikawa_kim_saito_saeki_shirouzu_yokoyama_konagaya_2003.cellml/@@cellml_codegen/Python
    
    Which comes from the Hatakeyama paper entry on the CellML site https://models.physiomeproject.org/exposure/c14e77a831ec3f1e56ef03240986a8b7"""
    def __call__(self, t, y, args):
        # unpack state
        R, Shc, PI3K, HRG, R_HRG, R_HRG2, Internalisation, RP, R_Shc, R_ShP, \
            ShP, R_ShGS, ShGS, GS, R_PI3K, R_PI3Kstar, PI3Kstar, RasGTP, \
            RasGDP, Akt_PIPP, RAF_star, RAF, MEKP, MEKPP, P, PIP3, Akt, \
            Akt_PIP3, Akt_PIP, MEK, ERK, ERKP, ERKPP = y
        
        # unpack params
        kf1, kb1, kf2, kb2, kf3, kb3, kf34, kb34, V4, k4, kf5, kb5, kf6, kb6, \
            kf7, kb7, kf8, kb8, kf9, kb9, V10, k10, kf23, kb23, kf24, kb24, \
            kf25, kb25, V26, k26, kf11, k11, V12, k12, kf13, k13, kf14, k14, E, \
            kf27, k27, V28, k28, kf29, kb29, V30, k30, kf31, k31, V32, k32, \
            kf33, k33, k16, k18, PP2A, kf15, k15, kf16, kf17, k17, kf18, MKP3, \
            kf19, k19, kf20, k20, kf21, k21, kf22, k22 = args
        
        v1 = kf1*R*HRG-kb1*R_HRG
        v2 = kf2*R_HRG*R_HRG-kb2*R_HRG2
        v3 = kf3*R_HRG2-kb3*RP
        v4 = (V4*RP)/(k4+RP)
        v5 = kf5*RP*Shc-kb5*R_Shc
        v6 = kf6*R_Shc-kb6*R_ShP
        v7 = kf7*R_ShP*GS-kb7*R_ShGS # old kf7*R_ShP-kb7*R_ShGS
        v8 = kf8*R_ShGS-kb8*ShGS*RP # old kf8*R_ShP-kb8*ShGS*RP
        v9 = kf9*ShGS-kb9*GS*ShP
        v10 = (V10*ShP)/(k10+ShP)
        v11 = (kf11*ShGS*RasGDP)/(k11+RasGDP)
        v12 = (V12*RasGTP)/(k12+RasGTP)
        v13 = (kf13*RasGTP*RAF)/(k13+RAF)
        v14 = (kf14*(Akt_PIPP+E)*RAF_star)/(k14+RAF_star)
        v15 = (kf15*RAF_star*MEK)/(k15*(1.0+MEKP/k17)+MEK)
        v16 = (kf16*PP2A*MEKP)/(k16*(1.0+MEKPP/k18+Akt_PIP/k31+Akt_PIPP/k33)+MEKP)
        v17 = (kf17*RAF_star*MEKP)/(k17*(1.0+MEK/k15)+MEKP)
        v18 = (kf18*PP2A*MEKPP)/(k18*(1.0+MEKP/k16+Akt_PIP/k31+Akt_PIPP/k33)+MEKPP)
        v19 = (kf19*MEKPP*ERK)/(k19*(1.0+ERKP/k21)+ERK)
        v20 = (kf20*MKP3*ERKP)/(k20*(1.0+ERKPP/k22)+ERKP)
        v21 = (kf21*MEKPP*ERKP)/(k21*(1.0+ERK/k19)+ERKP)
        v22 = (kf22*MKP3*ERKPP)/(k22*(1.0+ERKP/k20)+ERKPP)
        v23 = kf23*RP*PI3K-kb23*R_PI3K
        v24 = kf24*R_PI3K-kb24*R_PI3Kstar
        v25 = kf25*R_PI3Kstar-kb25*RP*PI3Kstar
        v26 = (V26*PI3Kstar)/(k26+PI3Kstar)
        v27 = (kf27*PI3Kstar*P)/(k27+P)
        v28 = (V28*PIP3)/(k28+PIP3)
        v29 = kf29*PIP3*Akt-kb29*Akt_PIP3
        v30 = (V30*Akt_PIP3)/(k30*(1.0+Akt_PIP/k32)+Akt_PIP3)
        v31 = (kf31*PP2A*Akt_PIP)/(k31*(1.0+MEKP/k16+MEKPP/k18+Akt_PIPP/k33)+Akt_PIP)
        v32 = (V32*Akt_PIP)/(k32*(1.0+Akt_PIP3/k30)+Akt_PIP)
        v33 = (kf33*PP2A*Akt_PIPP)/(k33*(1.0+MEKP/k16+MEKPP/k18+Akt_PIP/k31)+Akt_PIPP)
        v34 = kf34*RP-kb34*Internalisation


        # define rates
        dy_dt_0 = -v1 # R
        dy_dt_1 = -v5+v10 # Shc
        dy_dt_2 = -v23+v26 # PI3K
        dy_dt_3 = -v1 # HRG
        dy_dt_4 = v1 - 2.0*v2 # R_HRG
        dy_dt_5 = v2-v3+v4 # R_HRG2
        dy_dt_6 = v34 # Internalisation
        dy_dt_7 = v3-v4-v5+v8-v23+v25-v34 # RP
        dy_dt_8 = v5-v6 # R_Shc
        dy_dt_9 = v6-v7 # R_ShP
        dy_dt_10 = v9-v10  # ShP
        dy_dt_11 = v7-v8 # R_ShGS
        dy_dt_12 = v8-v9 # ShGS
        dy_dt_13 = -v7+v9 # GS
        dy_dt_14 = v23-v24 # R_PI3K
        dy_dt_15 = v24-v25 # R_PI3Kstar
        dy_dt_16 = v25-v26 # PI3Kstar
        dy_dt_17 = v11-v12 # RasGTP
        dy_dt_18 = -v11+v12  # RasGDP
        dy_dt_19 = v32-v33 # Akt_PIPP
        dy_dt_20 = -v14+v13 # RAF_star
        dy_dt_21 = v14-v13 # RAF
        dy_dt_22 = v15-v16-v17+v18 # MEKP
        dy_dt_23 = v17-v18 # MEKPP
        dy_dt_24 = v28-v27 # P
        dy_dt_25 = -v28+v27-v29 # PIP3
        dy_dt_26 = -v29 # Akt
        dy_dt_27 = v29-v30+v31 # Akt_PIP3
        dy_dt_28 = v30-v31-v32+v33 # Akt_PIP
        dy_dt_29 = -v15+v16 # MEK
        dy_dt_30 = -v19+v20 # ERK
        dy_dt_31 = v19-v20-v21+v22 # ERKP
        dy_dt_32 = v21-v22 # ERKPP
        
        return (dy_dt_0, dy_dt_1, dy_dt_2, dy_dt_3, dy_dt_4, dy_dt_5, dy_dt_6, \
                    dy_dt_7, dy_dt_8, dy_dt_9, dy_dt_10, dy_dt_11, dy_dt_12, \
                    dy_dt_13, dy_dt_14, dy_dt_15, dy_dt_16, dy_dt_17, dy_dt_18, \
                    dy_dt_19, dy_dt_20, dy_dt_21, dy_dt_22, dy_dt_23, dy_dt_24, \
                    dy_dt_25, dy_dt_26, dy_dt_27, dy_dt_28, dy_dt_29, dy_dt_30, \
                    dy_dt_31, dy_dt_32)

    
    def get_initial_conditions(self):
        """ Function to get nominal initial conditions for the model. """
        R = 80.0
        Shc = 1000.0
        PI3K = 10.0
        HRG = 10.0
        R_HRG = 0.0
        R_HRG2 = 0.0
        Internalisation = 0.0
        RP = 0.0
        R_Shc = 0.0
        R_ShP = 0.0
        ShP = 0.0
        R_ShGS = 0.0
        ShGS = 0.0
        GS = 10.0
        R_PI3K = 0.0
        R_PI3Kstar = 0.0
        PI3Kstar = 0.0
        RasGTP = 0.0
        RasGDP = 120.0
        Akt_PIPP = 0.0
        RAF_star = 0.0
        RAF = 100.0
        MEKP = 0.0
        MEKPP = 0.0
        P = 800.0
        PIP3 = 0.0
        Akt = 10.0
        Akt_PIP3 = 0.0
        Akt_PIP = 0.0
        MEK = 120.0
        ERK = 1000.0
        ERKP = 0.0
        ERKPP = 0.0

        return (R, Shc, PI3K, HRG, R_HRG, R_HRG2, Internalisation, RP, R_Shc, R_ShP, ShP, R_ShGS, ShGS, GS, R_PI3K, R_PI3Kstar, PI3Kstar, RasGTP, RasGDP, Akt_PIPP, RAF_star, RAF, MEKP, MEKPP, P, PIP3, Akt, Akt_PIP3, Akt_PIP, MEK, ERK, ERKP, ERKPP)

        
    
    def get_nominal_params(self):
        """ Function to get nominal parameters for the model. """
        param_dict = {
            'kf1': 0.0012,
            'kb1': 0.00076,
            'kf2': 0.01,
            'kb2': 0.1,
            'kf3': 1.0,
            'kb3': 0.01,
            'kf34': 0.001,
            'kb34': 0.0,
            'V4': 62.5,
            'k4': 50.0,
            'kf5': 0.1,
            'kb5': 1.0,
            'kf6': 20.0,
            'kb6': 5.0,
            'kf7': 60.0,
            'kb7': 546.0,
            'kf8': 2040.0,
            'kb8': 15700.0,
            'kf9': 40.8,
            'kb9': 0.0,
            'V10': 0.0154,
            'k10': 340.0,
            'kf23': 0.1,
            'kb23': 2.0,
            'kf24': 9.85,
            'kb24': 0.0985,
            'kf25': 45.8,
            'kb25': 0.047,
            'V26': 2620.0,
            'k26': 3680.0,
            'kf11': 0.222,
            'k11': 0.181,
            'V12': 0.289,
            'k12': 0.0571,
            'kf13': 1.53,
            'k13': 11.7,
            'kf14': 0.00673,
            'k14': 8.07,
            'E': 7.0,
            'kf27': 16.9,
            'k27': 39.1,
            'V28': 17000.0,
            'k28': 9.02,
            'kf29': 507.0,
            'kb29': 234.0, # TODO check that this reaction is MM
            'V30': 20000.0,
            'k30': 80000.0,
            'kf31': 0.107,
            'k31': 4.35,
            'V32': 20000.0,
            'k32': 80000.0,
            'kf33': 0.211,
            'k33': 12.0,
            'k16': 2200.0,
            'k18': 60.0,
            'PP2A': 11.4,
            'kf15': 3.5,
            'k15': 317.0,
            'kf16': 0.058,
            'kf17': 2.9,
            'k17': 317.0,
            'kf18': 0.058,
            'MKP3': 2.4,
            'kf19': 9.5,
            'k19': 146000.0,
            'kf20': 0.3,
            'k20': 160.0,
            'kf21': 16.0,
            'k21': 146000.0,
            'kf22': 0.27,
            'k22': 60.0,
        }


        param_list = [param_dict[key] for key in param_dict]

        return param_dict, param_list

