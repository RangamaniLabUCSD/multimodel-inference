import equinox as eqx
import jax.numpy as jnp

class dessauges_2022(eqx.Module):
    def __call__(self, t, y, args):

        # unpack state variables
        EGFR, EGFR_star, EGFR_endo, RAS, RAS_star, RAF, RAF_star, MEK, MEK_star,\
            ERK, ERK_star, NFB, NFB_star, KTR, KTR_star = y
        
        # unpack parameters
        r12, EGF,r23, r21, r31, k12, K12, k21, K21, k34, K34, k43, \
            K43, k56, K56, k65, K65, k78, K78, k87, K87, k910, K910, \
            k_nfb, f12, F12, f21, F21, s12, s21, light, ktr_init = args

        # expressions

        # ODE rhs
        d_EGFR = -r12*EGF*EGFR + r21*EGFR_star + r31*EGFR_endo
        d_EGFR_star = r12*EGF*EGFR - (r21 + r23)*EGFR_star
        d_EGFR_endo = r23*EGFR_star - r31*EGFR_endo
        d_RAS = -(k12*EGFR_star + light)*(RAS/(K12 + RAS)) + k21*(RAS_star/(K21 + RAS_star))
        d_RAS_star = (k12*EGFR_star + light)*(RAS/(K12 + RAS)) - k21*(RAS_star/(K21 + RAS_star))
        d_RAF = -k34*RAS_star*(RAF/(K34 + RAF)) + (k_nfb*NFB_star + k43)*(RAF_star/(K43 + RAF_star))
        d_RAF_star = k34*RAS_star*(RAF/(K34 + RAF)) - (k_nfb*NFB_star + k43)*(RAF_star/(K43 + RAF_star))
        d_MEK = -k56*RAF_star*(MEK/(K56 + MEK)) + k65*(MEK_star/(K65 + MEK_star))
        d_MEK_star = k56*RAF_star*(MEK/(K56 + MEK)) - k65*(MEK_star/(K65 + MEK_star))
        d_ERK = -k78*MEK_star*(ERK/(K78 + ERK)) + k87*(ERK_star/(K87 + ERK_star))
        d_ERK_star = k78*MEK_star*(ERK/(K78 + ERK)) - k87*(ERK_star/(K87 + ERK_star))
        d_NFB = -f12*ERK_star*(NFB/(F12 + NFB)) + f21*(NFB_star/(F21 + NFB_star))
        d_NFB_star = f12*ERK_star*(NFB/(F12 + NFB)) - f21*(NFB_star/(F21 + NFB_star))
        d_KTR = -(k910*ERK_star*(KTR/(K910 + KTR)) + s12*KTR) + s21*KTR_star
        d_KTR_star = (k910*ERK_star*(KTR/(K910 + KTR)) + s12*KTR) - s21*KTR_star

        # concatenate into tuple and return
        return (d_EGFR, d_EGFR_star, d_EGFR_endo, d_RAS, d_RAS_star, d_RAF,
                d_RAF_star, d_MEK, d_MEK_star, d_ERK, d_ERK_star, d_NFB,
                d_NFB_star, d_KTR, d_KTR_star,)

    def get_nominal_params(self):
        print('WARNING: These nominal parameters are means of the priors.')

        p_dict =  {
            'r_12':343.0, 
            'EGF':1.0,
            'r_23':343.0, 
            'r_21':343.0, 
            'r_31':343.0, 
            'k_12':343.0, 
            'K_12':343.0, 
            'k_21':343.0, 
            'K_21':343.0, 
            'k_34':343.0, 
            'K_34':343.0, 
            'k_43':343.0, 
            'K_43':343.0, 
            'k_56':343.0, 
            'K_56':343.0, 
            'k_65':343.0, 
            'K_65':343.0, 
            'k_78':343.0, 
            'K_78':343.0, 
            'k_87':343.0, 
            'K_87':343.0, 
            'k910':343.0, 
            'K910':343.0, 
            'k_nfb':343.0, 
            'f12':343.0, 
            'F12':343.0, 
            'f21':343.0, 
            'F21':343.0, 
            's_12':343.0, 
            's_21':343.0, 
            'light':0.0,
            'ktr_init':0.5,
        }

        p_list = [p_dict[key] for key in p_dict.keys()]

        return p_dict, p_list



    def get_initial_conditions(self, ktr_init=0.5):

        y0_dict = {
            'EGFR': 1.0,
            'EGFR_star': 0.0,
            'EGFR_endo': 0.0,
            'RAS': 1.0,
            'RAS_star': 0.0,
            'RAF': 1.0,
            'RAF_star': 0.0,
            'MEK': 1.0,
            'MEK_star': 0.0,
            'ERK': 1.0,
            'ERK_star': 0.0,
            'NFB': 1.0,
            'NFB_star': 0.0,
            'KTR': 1 - ktr_init,
            'KTR_star': ktr_init,   
        }

        y0_tup = tuple([y0_dict[key] for key in y0_dict.keys()])

        return y0_dict, y0_tup
