import equinox as eqx
import jax.numpy as jnp

class shin_2014(eqx.Module):
    def __call__(self, t, y, args):
        # unpack state
        RE, GS, Ras_GTP, act_Raf, pp_MEK, pp_ERK = y

        # unpack parameters
        EGFR_tot, SOS_tot, Grb2_tot, Ras_tot, Raf_tot, MEK_tot, ERK_tot, ka38, \
            kd38, ka39a, ki39, kd39, kc40, kc41, kc42, kc43, kc44, kc45, kc46, \
            kc47, EGF = args


        # define algebraic equations
        EGFR = EGFR_tot - RE
        SOS = SOS_tot - GS
        Grb2 = Grb2_tot
        Ras_GDP = Ras_tot - Ras_GTP
        Raf = Raf_tot - act_Raf
        MEK = MEK_tot - pp_MEK
        ERK = ERK_tot - pp_ERK

        # ode
        # RE
        d_RE_dt = ka38*EGF*EGFR - kd38*RE
        # GS
        d_GS_dt = ka39a*RE*SOS*Grb2/(1 + (pp_ERK/ki39)**3) - kd39*GS
        # Ras_GTP
        d_Ras_GTP_dt = kc40*GS*Ras_GDP - kc41*Ras_GTP
        # act_Raf
        d_act_Raf_dt = kc42*Ras_GTP*Raf - kc43*act_Raf
        # pp_MEK
        d_pp_MEK_dt = kc44*act_Raf*MEK - kc45*pp_MEK
        # pp_ERK
        d_pp_ERK_dt = kc46*pp_MEK*ERK - kc47*pp_ERK

        return (d_RE_dt, d_GS_dt, d_Ras_GTP_dt, d_act_Raf_dt, d_pp_MEK_dt, d_pp_ERK_dt)

    
    def get_initial_conditions(self):
        """ Function to get nominal initial conditions for the model. """
        y0_dict = {'RE': 0.0, # uM
            'GS': 0.0, # uM
            'Ras_GTP': 0.0, # uM
            'act_Raf': 0.0, # uM
            'pp_MEK': 0.0, # uM
            'pp_ERK': 0.0, # uM
        }

        y0_tup = tuple([y0_dict[key] for key in y0_dict.keys()])

        return y0_dict, y0_tup

        
    
    def get_nominal_params(self):
        """ Function to get nominal parameters for the model. """
        param_dict = {
            'EGFR_tot': 1.99e-3, # uM
            'SOS_tot': 1.18e-2, # uM
            'Grb2_tot': 7.74e02, # uM
            'Ras_tot': 1.25e-1, # uM
            'Raf_tot': 2.17e-1, # uM
            'MEK_tot': 6.74e-2, # uM
            'ERK_tot': 2.26e-1, # uM
            'ka38': 3.36e1, # 1/(uM*min)
            'kd38': 6.925e3, # 1/min
            'ka39a': 1.088e5, # 1/(uM^2*min)
            'ki39': 7.716e-4, # uM
            'kd39': 1.616e-1, # 1/min
            'kc40': 4.869e2, # 1/(uM*min)
            'kc41': 5.073e1, # 1/min
            'kc42': 4.648e2, # 1/(uM*min)
            'kc43': 4.663, # 1/min
            'kc44': 4.037e2, # 1/(uM*min)
            'kc45': 6.872e-2, # 1/min
            'kc46': 7.821, # 1/(uM*min)
            'kc47': 3.905e-1, # 1/min
            'EGF': 1.0e-2, # uM NOTE: this is not in the paper, but assuming something reasonable

        }
        param_list = [param_dict[key] for key in param_dict]

        return param_dict, param_list

