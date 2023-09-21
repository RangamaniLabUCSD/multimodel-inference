import equinox as eqx

class kholodenko_2000(eqx.Module):
    def __call__(self, t, y, args):
        # unpack state
        MKKK_P, MKK_P, MKK_PP, MAPK_P, MAPK_PP = y
        
        # unpack parameters
        v1,KI,K1,n,v2,K2,k3,K3,k4,K4,v5,K5,v6,K6,k7,K7,k8,K8,v9,K9,v10,K10, \
            MKKK_total, MKK_total, MAPK_total = args

        # algebraic equations
        MKKK = MKKK_total - MKKK_P  # MKKK
        MKK = MKK_total - MKK_P - MKK_PP  # MKK
        MAPK = MAPK_total - MAPK_P - MAPK_PP  # MAPK
        
        # fluxes
        J1 = (v1*MKKK)/((1+(MAPK_PP/KI)**n)*(K1 + MKKK))
        J2 = (v2*MKKK_P)/(K2 + MKKK_P)
        J3 = (k3*MKKK_P*MKK)/(K3 + MKK)
        J4 = (k4*MKKK_P*MKK_P)/(K4 + MKK_P)
        J5 = (v5*MKK_PP)/(K5 + MKK_PP)
        J6 = (v6*MKK_P)/(K6 + MKK_P)
        J7 = (k7*MKK_PP*MAPK)/(K7 + MAPK)
        J8 = (k8*MKK_PP*MAPK_P)/(K8 + MAPK_P)
        J9 = (v9*MAPK_PP)/(K9 + MAPK_PP)
        J10 = (v10*MAPK_P)/(K10 + MAPK_P)

        # ODE rhs
        # d_MKKK = J2 - J1
        d_MKKK_P = J1 - J2
        # d_MKK = J6 - J3
        d_MKK_P = J3 + J5 - J4 - J6
        d_MKK_PP = J4 - J5
        # d_MAPK = J10 - J7
        d_MAPK_P = J7 + J9 - J8 - J10
        d_MAPK_PP = J8 - J9

        # concatenate into tuple and return
        return (d_MKKK_P, d_MKK_P, d_MKK_PP, d_MAPK_P, d_MAPK_PP)
    

    def get_nominal_params():
        return {
            'v1': 2.5,
            'KI': 9.0,
            'K1': 10.0,
            'n': 1.0,
            'v2': 0.25,
            'K2': 8.0,
            'k3': 0.025,
            'K3': 15.0,
            'k4': 0.025,
            'K4': 15.0,
            'v5': 0.75,
            'K5': 15.0,
            'v6': 0.75,
            'K6': 15.0,
            'k7': 0.025,
            'K7': 15.0,
            'k8': 0.025,
            'K8': 15.0,
            'v9': 0.5,
            'K9': 15.0,
            'v10': 0.5,
            'K10': 15.0,
            'MKKK_total': 100.0,
            'MKK_total': 300.0,
            'MAPK_total': 300.0,
        }

    def get_initial_conditions():
        ic_dict = {
            'MKKK_P': 90.0,
            'MKK_P': 10.0,
            'MKK_PP': 0.0,
            'MAPK_P': 10.0,
            'MAPK_PP': 0.0,
        }

        ic_tup = tuple([ic_dict[k] for k in ic_dict.keys()])

        return ic_dict, ic_tup