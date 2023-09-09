import equinox as eqx
import jax.numpy as jnp

class hatakeyama_2003(eqx.Module):
    """ Note this is the code at https://models.physiomeproject.org/exposure/c14e77a831ec3f1e56ef03240986a8b7/hatakeyama_kimura_naka_kawasaki_yumoto_ichikawa_kim_saito_saeki_shirouzu_yokoyama_konagaya_2003.cellml/@@cellml_codegen/Python
    
    Which comes from the Hatakeyama paper entry on the CellML site https://models.physiomeproject.org/exposure/c14e77a831ec3f1e56ef03240986a8b7"""
    def __call__(self, t, y, constants):
        # unpack state
        R, Shc, PI3K, HRG, R_HRG, R_HRG2, Internalisation, RP, R_Shc, R_ShP, \
            ShP, R_ShGS, ShGS, GS, R_PI3K, R_PI3Kstar, PI3Kstar, RasGTP, \
            RasGDP, Akt_PIPP, RAF_star, RAF, MEKP, MEKPP, P, PIP3, Akt, \
            Akt_PIP3, Akt_PIP, MEK, ERK, ERKP, ERKPP = y
        
        # define rates
        dy_dt_0 = -constants[0]*R*HRG+constants[1]*R_HRG
        dy_dt_3 = -constants[0]*R*HRG+constants[1]*R_HRG
        dy_dt_4 = (constants[0]*R*HRG-constants[1]*R_HRG)-constants[30]*(constants[2]*R_HRG*R_HRG-constants[3]*R_HRG2)
        dy_dt_5 = ((constants[2]*R_HRG*R_HRG-constants[3]*R_HRG2)-(constants[4]*R_HRG2-constants[5]*RP))+(constants[8]*RP)/(constants[9]+RP)
        dy_dt_7 = ((((((constants[4]*R_HRG2-constants[5]*RP)-(constants[8]*RP)/(constants[9]+RP))-(constants[10]*RP*Shc-constants[11]*R_Shc))+(constants[16]*R_ShP-constants[17]*ShGS*RP))-(constants[22]*RP*PI3K-constants[23]*R_PI3K))+(constants[26]*R_PI3Kstar-constants[27]*RP*PI3Kstar))-(constants[6]*RP-constants[7]*Internalisation)
        dy_dt_6 = constants[6]*RP-constants[7]*Internalisation
        dy_dt_8 = (constants[10]*RP*Shc-constants[11]*R_Shc)-(constants[12]*R_Shc-constants[13]*R_ShP)
        dy_dt_1 = -(constants[10]*RP*Shc-constants[11]*R_Shc)+(constants[20]*ShP)/(constants[21]+ShP)
        dy_dt_9 = (constants[12]*R_Shc-constants[13]*R_ShP)-(constants[14]*R_ShP-constants[15]*R_ShGS)
        dy_dt_13 = -(constants[14]*R_ShP-constants[15]*R_ShGS)+(constants[18]*ShGS-constants[19]*GS*ShP)
        dy_dt_10 = (constants[18]*ShGS-constants[19]*GS*ShP)-(constants[20]*ShP)/(constants[21]+ShP)
        dy_dt_11 = (constants[14]*R_ShP-constants[15]*R_ShGS)-(constants[16]*R_ShP-constants[17]*ShGS*RP)
        dy_dt_12 = (constants[16]*R_ShP-constants[17]*ShGS*RP)-(constants[18]*ShGS-constants[19]*GS*ShP)
        dy_dt_14 = (constants[22]*RP*PI3K-constants[23]*R_PI3K)-(constants[24]*R_PI3K-constants[25]*R_PI3Kstar)
        dy_dt_2 = -(constants[22]*RP*PI3K-constants[23]*R_PI3K)+(constants[28]*PI3Kstar)/(constants[29]+PI3Kstar)
        dy_dt_15 = (constants[24]*R_PI3K-constants[25]*R_PI3Kstar)-(constants[26]*R_PI3Kstar-constants[27]*RP*PI3Kstar)
        dy_dt_16 = (constants[26]*R_PI3Kstar-constants[27]*RP*PI3Kstar)-(constants[28]*PI3Kstar)/(constants[29]+PI3Kstar)
        dy_dt_18 = -((constants[31]*ShGS*RasGDP)/(constants[32]+RasGDP))+(constants[33]*RasGTP)/(constants[34]+RasGTP)
        dy_dt_17 = (constants[31]*ShGS*RasGDP)/(constants[32]+RasGDP)-(constants[33]*RasGTP)/(constants[34]+RasGTP)
        dy_dt_21 = (constants[37]*(Akt_PIPP+constants[39])*RAF_star)/(constants[38]+RAF_star)-(constants[35]*RasGTP*RAF)/(constants[36]+RAF)
        dy_dt_20 = -((constants[37]*(Akt_PIPP+constants[39])*RAF_star)/(constants[38]+RAF_star))+(constants[35]*RasGTP*RAF)/(constants[36]+RAF)
        dy_dt_24 = (constants[42]*PIP3)/(constants[43]+PIP3)-(constants[40]*PI3Kstar*P)/(constants[41]+P)
        dy_dt_25 = (-((constants[42]*PIP3)/(constants[43]+PIP3))+(constants[40]*PI3Kstar*P)/(constants[41]+P))-(constants[44]*PIP3*Akt-constants[45]*Akt_PIP3)
        dy_dt_26 = -(constants[44]*PIP3*Akt-constants[45]*Akt_PIP3)
        dy_dt_27 = ((constants[44]*PIP3*Akt-constants[45]*Akt_PIP3)-(constants[46]*Akt_PIP3)/(constants[47]*(constants[57]+Akt_PIP/constants[51])+Akt_PIP3))+(constants[48]*constants[56]*Akt_PIP)/(constants[49]*(constants[57]+MEKP/constants[54]+MEKPP/constants[55]+Akt_PIPP/constants[53])+Akt_PIP)
        dy_dt_28 = (((constants[46]*Akt_PIP3)/(constants[47]*(constants[57]+Akt_PIP/constants[51])+Akt_PIP3)-(constants[48]*constants[56]*Akt_PIP)/(constants[49]*(constants[57]+MEKP/constants[54]+MEKPP/constants[55]+Akt_PIPP/constants[53])+Akt_PIP))-(constants[50]*Akt_PIP)/(constants[51]*(constants[57]+Akt_PIP3/constants[47])+Akt_PIP))+(constants[52]*constants[56]*Akt_PIPP)/(constants[53]*(constants[57]+MEKP/constants[54]+MEKPP/constants[55]+Akt_PIP/constants[49])+Akt_PIPP)
        dy_dt_19 = (constants[50]*Akt_PIP)/(constants[51]*(constants[57]+Akt_PIP3/constants[47])+Akt_PIP)-(constants[52]*constants[56]*Akt_PIPP)/(constants[53]*(constants[57]+MEKP/constants[54]+MEKPP/constants[55]+Akt_PIP/constants[49])+Akt_PIPP)
        dy_dt_29 = -((constants[59]*RAF_star*MEK)/(constants[60]*(constants[69]+MEKP/constants[64])+MEK))+(constants[61]*constants[58]*MEKP)/(constants[62]*(constants[69]+MEKPP/constants[66]+Akt_PIP/constants[67]+Akt_PIPP/constants[68])+MEKP)
        dy_dt_22 = (((constants[59]*RAF_star*MEK)/(constants[60]*(constants[69]+MEKP/constants[64])+MEK)-(constants[61]*constants[58]*MEKP)/(constants[62]*(constants[69]+MEKPP/constants[66]+Akt_PIP/constants[67]+Akt_PIPP/constants[68])+MEKP))-(constants[63]*RAF_star*MEKP)/(constants[64]*(constants[69]+MEK/constants[60])+MEKP))+(constants[65]*constants[58]*MEKPP)/(constants[66]*(constants[69]+MEKP/constants[62]+Akt_PIP/constants[67]+Akt_PIPP/constants[68])+MEKPP)
        dy_dt_23 = (constants[63]*RAF_star*MEKP)/(constants[64]*(constants[69]+MEK/constants[60])+MEKP)-(constants[65]*constants[58]*MEKPP)/(constants[66]*(constants[69]+MEKP/constants[62]+Akt_PIP/constants[67]+Akt_PIPP/constants[68])+MEKPP)
        dy_dt_30 = -((constants[71]*MEKPP*ERK)/(constants[72]*(constants[79]+ERKP/constants[76])+ERK))+(constants[73]*constants[70]*ERKP)/(constants[74]*(constants[79]+ERKPP/constants[78])+ERKP)
        dy_dt_32 = (constants[75]*MEKPP*ERKP)/(constants[76]*(constants[79]+ERK/constants[72])+ERKP)-(constants[77]*constants[70]*ERKPP)/(constants[78]*(constants[79]+ERKP/constants[74])+ERKPP)
        dy_dt_31 = (((constants[71]*MEKPP*ERK)/(constants[72]*(constants[79]+ERKP/constants[76])+ERK)-(constants[73]*constants[70]*ERKP)/(constants[74]*(constants[79]+ERKPP/constants[78])+ERKP))-(constants[75]*MEKPP*ERKP)/(constants[76]*(constants[79]+ERK/constants[72])+ERKP))+(constants[77]*constants[70]*ERKPP)/(constants[78]*(constants[79]+ERKP/constants[74])+ERKPP)

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
        RAF_star = 100.0
        RAF = 0.0
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
        param_dict = {'kf1': 0.0012,
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
            'two': 2.0,
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
            'kb29': 234.0,
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
            'one': 1.0,
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
            'k22': 60.0,}


        param_list = [param_dict[key] for key in param_dict]

        return param_dict, param_list

