import equinox as eqx
import jax.numpy as jnp

class schoeberl_2002(eqx.Module):
    """ Note this is the code at https://models.physiomeproject.org/exposure/a7197a8d519bbee0e95256f699b26050/Schoeberl_EichlerJonsson_Gilles_Muller_2002.cellml/@@cellml_codegen/Python
    
    Which comes from the Schoeberl paper entry on the CellML site https://models.physiomeproject.org/exposure/a7197a8d519bbee0e95256f699b26050"""

    def custom_piecewise(self, cases):
        """Compute result of a piecewise function"""
        return jnp.select(cases[0::2],cases[1::2])

    def __call__(self, t, y, args):
        # unpack state
        c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, \
            c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, \
            c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, \
            c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, \
            c51, c52, c53, c54, c55, c56, c57, c58, c59, c60, c61, c62, \
            c63, c64, c65, c66, c67, c68, c69, c70, c71, c72, c73, c74, \
            c75, c76, c77, c78, c79, c80, c81, c82, c83, c84, c85, c86, \
            c87, c88, c89, c90, c91, c92, c93, c94 = y
        
        # unpack params
        k1, kd1, k2, kd2, k3, kd3, k4, kd4, k6, kd6, k8, kd8, k10b, kd10, k13, \
            k15, k16, kd16, k17, kd17, k18, kd18, k19, kd19, k20, kd20, k21, \
            kd21, k22, kd22, k23, kd23, kd24, k25, kd25, k28, kd28, k29, kd29, \
            k32, kd32, k33, kd33, k34, kd34, k35, kd35, Km36, Vm36, k37, kd37, \
            k40, kd40, k41, kd41, k42, kd42, k43, k44, kd44, k45, k47, k48, \
            kd48, k49, k50, kd50, k52, kd52, k53, k55, k56, kd56, k57, k58, \
            kd58, k59, k60, k61, RT = args
        
        # algebraic equations
        C = (RT)/(kd1/(k1*c1)+1.0)
        k5 = self.custom_piecewise([jnp.less(C , 3100.00), 1.55, jnp.greater(C , 100000.), 0.2, True, C*-1.35e-05+1.55])
        
        # fluxes
        v1 = k1*c1*c2-kd1*c3
        v2 = k2*c3*c3-kd2*c4
        v3 = k3*c4-kd3*c5
        v4 = k4*c23*c12-kd4*c7
        v5 = k5*c7
        v6 = k6*c2-kd6*c6
        v7 = k6*c5
        v8 = k8*c5*c14-kd8*c15
        v9 = k6*c23
        v10 = k10b*c6*c16-kd10*c10
        v11 = k2*c10*c10-kd2*c11
        v12 = k3*c11-kd3*c8
        v13 = k13
        v14 = k8*c8*c14-kd8*c17
        v15 = k15*c9
        v16 = k16*c22*c15-kd16*c23
        v17 = k17*c24*c23-kd17*c25
        v18 = k18*c26*c25-kd18*c27
        v19 = k19*c27-kd19*c28*c25
        v20 = k20*c25*c43-kd20*c29
        v21 = k21*c29-kd21*c25*c26
        v22 = k22*c31*c15-kd22*c32
        v23 = k23*c32-kd23*c33
        v24 = k16*c22*c33-kd24*c34
        v25 = k25*c24*c34-kd25*c35
        v26 = k18*c26*c35-kd18*c36
        v27 = k19*c36-kd19*c35*c28
        v28 = k28*c28*c41-kd28*c42
        v29 = k29*c42-kd29*c43*c45
        v30 = k20*c35*c43-kd20*c37
        v31 = k21*c37-kd21*c35*c26
        v32 = k32*c35-kd32*c38*c15
        v33 = k33*c38-kd33*c40*c30
        v34 = k34*c25-kd34*c15*c30
        v35 = k35*c30-kd35*c24*c22
        v36 = (Vm36*c40)/(Km36+c40)
        v37 = k37*c33-kd37*c15*c40
        v38 = k16*c22*c40-kd24*c39
        v39 = k37*c34-kd37*c15*c39
        v40 = k40*c24*c39-kd40*c38
        v41 = k41*c30*c33-kd41*c35
        v42 = k42*c44*c45-kd42*c46
        v43 = k43*c46
        v44 = k44*c47*c45-kd44*c48
        v45 = k45*c48
        v46 = k44*c49*c45-kd44*c50
        v47 = k47*c50
        v48 = k48*c51*c53-kd48*c52
        v49 = k49*c52
        v50 = k50*c53*c49-kd50*c54
        v51 = k49*c54
        v52 = k52*c55*c51-kd52*c56
        v53 = k53*c56
        v54 = k52*c51*c57-kd52*c58
        v55 = k55*c58
        v56 = k56*c59*c60-kd56*c61
        v57 = k57*c61
        v58 = k58*c60*c57-kd58*c62
        v59 = k59*c62
        v60 = k60*c6
        v61 = k61*c16
        v62 = k60*c8
        v63 = k16*c17*c22-kd16*c18
        v64 = k17*c24*c18-kd17*c19
        v65 = k18*c26*c19-kd18*c20
        v66 = k19*c20-kd19*c69*c19
        v67 = k20*c71*c19-kd20*c21
        v68 = k21*c21-kd21*c19*c26
        v69 = k22*c31*c17-kd22*c63
        v70 = k23*c63-kd23*c64
        v71 = k16*c22*c64-kd24*c65
        v72 = k25*c24*c65-kd25*c66
        v73 = k18*c26*c66-kd18*c67
        v74 = k19*c67-kd19*c66*c69
        v75 = k28*c69*c41-kd28*c70
        v76 = k29*c70-kd29*c71*c72
        v77 = k20*c71*c66-kd20*c68
        v78 = k21*c68-c26*kd21*c66
        v79 = k32*c66-kd32*c17*c38
        v80 = k34*c19-kd34*c17*c30
        v81 = k37*c64-kd37*c17*c40
        v82 = k37*c65-kd37*c17*c39
        v83 = k41*c30*c64-kd41*c66
        v84 = k42*c44*c72-kd42*c73
        v85 = k43*c73
        v86 = k44*c47*c72-kd44*c74
        v87 = k45*c74
        v88 = k44*c72*c75-kd44*c76
        v89 = k47*c76
        v90 = k48*c77*c53-kd48*c78
        v91 = k49*c78
        v92 = k50*c53*c75-kd50*c79
        v93 = k49*c79
        v94 = k52*c55*c77-kd52*c80
        v95 = k53*c80
        v96 = k52*c77*c81-kd52*c82
        v97 = k55*c82
        v98 = k56*c83*c60-kd56*c84
        v99 = k57*c84
        v100 = k58*c60*c81-kd58*c85
        v101 = k59*c85
        v102 = k6*c15-kd6*c17
        v103 = k6*c32-kd6*c63
        v104 = k6*c33-kd6*c64
        v105 = k6*c25-kd6*c19
        v106 = k4*c25*c12-kd4*c88
        v107 = k5*c88
        v108 = k6*c27-kd6*c20
        v109 = k4*c27*c12-kd4*c89
        v110 = k5*c89
        v111 = k6*c29-kd6*c21
        v112 = k4*c29*c12-kd4*c90
        v113 = k5*c90
        v114 = k6*c34-kd6*c65
        v115 = k4*c34*c12-kd4*c91
        v116 = k5*c91
        v117 = k6*c35-kd6*c66
        v118 = k4*c35*c12-kd4*c92
        v119 = k5*c92
        v120 = k6*c36-kd6*c67
        v121 = k4*c36*c12-kd4*c93
        v123 = k6*c37-kd6*c68
        v122 = k5*c93
        v124 = k4*c37*c12-kd4*c94
        v125 = k5*c94
        
        dy_dt_0 = 0.0
        dy_dt_1 = v13-(v1+v6)
        dy_dt_2 = v1-2.0*v2
        dy_dt_3 = v2-v3
        dy_dt_4 = v3-(v7+v8)
        dy_dt_5 = v6-(v10+v60)
        dy_dt_6 = v4-v5
        dy_dt_7 = (v7+v12)-(v14+v62)
        dy_dt_8 = (v5+v107+v110+v113+v116+v119+v122+v125)-v15
        dy_dt_9 = v10-2.0*v11
        dy_dt_10 = v11-v12
        dy_dt_11 = v15-(v4+v106+v109+v112+v115+v118+v121+v124)
        dy_dt_12 = v61
        dy_dt_13 = -(v8+v14)
        dy_dt_14 = (v8+v32+v34+v37+v39)-(v16+v22+v102)
        dy_dt_15 = -(v10+v61)
        dy_dt_16 = (v14+v79+v80+v81+v82+v102)-(v63+v69)
        dy_dt_17 = (v5+v9+v63)-v64
        dy_dt_18 = (v66+v64+v68+v105+v107)-(v65+v67+v80)
        dy_dt_19 = (v65+v108+v110)-v66
        dy_dt_20 = (v67+v111+v113)-v68
        dy_dt_21 = v35-(v16+v24+v38+v63+v71)
        dy_dt_22 = v16-(v4+v9+v17)
        dy_dt_23 = v35-(v17+v25+v40+v64+v72)
        dy_dt_24 = (v17+v19+v21)-(v18+v20+v34+v105+v106)
        dy_dt_25 = (v21+v31+v68+v78)-(v18+v26+v65+v73)
        dy_dt_26 = v18-(v19+v108+v109)
        dy_dt_27 = (v19+v27)-v28
        dy_dt_28 = v20-(v21+v111+v112)
        dy_dt_29 = (v33+v34+v80)-(v35+v41+v83)
        dy_dt_30 = v36-(v22+v69)
        dy_dt_31 = v22-(v23+v103)
        dy_dt_32 = v23-(v24+v37+v41+v104)
        dy_dt_33 = v24-(v25+v39+v114+v115)
        dy_dt_34 = (v25+v27+v31+v41)-(v117+v118+v26+v30+v32)
        dy_dt_35 = v26-(v27+v120+v121)
        dy_dt_36 = v30-(v31+v123+v124)
        dy_dt_37 = (v32+v40+v79)-v33
        dy_dt_38 = (v38+v39+v82)-v40
        dy_dt_39 = (v33+v37+v81)-(v36+v38)
        dy_dt_40 = (v43+v85)-(v28+v75)
        dy_dt_41 = v28-v29
        dy_dt_42 = v29-(v20+v30)
        dy_dt_43 = (v43+v85)-(v42+v84)
        dy_dt_44 = (v29+v45+v47)-(v42+v44+v46)
        dy_dt_45 = v42-v43
        dy_dt_46 = (v51+v93)-(v44+v86)
        dy_dt_47 = v44-v45
        dy_dt_48 = (v45+v49)-(v46+v50)
        dy_dt_49 = v46-v47
        dy_dt_50 = (v47+v53+v55)-(v48+v52+v54)
        dy_dt_51 = v48-v49
        dy_dt_52 = (v49+v51+v91+v93)-(v48+v50+v90+v92)
        dy_dt_53 = v50-v51
        dy_dt_54 = (v59+v101)-(v52+v94)
        dy_dt_55 = v52-v53
        dy_dt_56 = (v53+v57)-(v54+v58)
        dy_dt_57 = v54-v55
        dy_dt_58 = v55-v56
        dy_dt_59 = (v57+v59+v99+v101)-(v56+v58+v98+v100)
        dy_dt_60 = v56-v57
        dy_dt_61 = v58-v59
        dy_dt_62 = (v69+v103)-v70
        dy_dt_63 = (v70+v104)-(v71+v81+v83)
        dy_dt_64 = (v71+v114+v116)-(v72+v82)
        dy_dt_65 = (v72+v74+v78+v83+v117+v119)-(v73+v77+v79)
        dy_dt_66 = (v73+v120+v122)-v74
        dy_dt_67 = (v77+v123+v125)-v78
        dy_dt_68 = (v66+v74)-v75
        dy_dt_69 = v75-v76
        dy_dt_70 = v76-(v67+v77)
        dy_dt_71 = (v76+v87+v89)-(v84+v86+v88)
        dy_dt_72 = v84-v85
        dy_dt_73 = v86-v87
        dy_dt_74 = (v87+v91)-(v92+v88)
        dy_dt_75 = v88-v89
        dy_dt_76 = (v89+v95+v97)-(v90+v94+v96)
        dy_dt_77 = v90-v91
        dy_dt_78 = v92-v93
        dy_dt_79 = v94-v95
        dy_dt_80 = (v95+v99)-(v96+v100)
        dy_dt_81 = v96-v97
        dy_dt_82 = v97-v98
        dy_dt_83 = v98-v99
        dy_dt_84 = v100-v101
        dy_dt_85 = v60
        dy_dt_86 = v62
        dy_dt_87 = v106-v107
        dy_dt_88 = v109-v110
        dy_dt_89 = v112-v113
        dy_dt_90 = v115-v116
        dy_dt_91 = v118-v119
        dy_dt_92 = v121-v122
        dy_dt_93 = v124-v125
        
        return (dy_dt_0, dy_dt_1, dy_dt_2, dy_dt_3, dy_dt_4, dy_dt_5, dy_dt_6, dy_dt_7, dy_dt_8, dy_dt_9, dy_dt_10, dy_dt_11, dy_dt_12, dy_dt_13, dy_dt_14, dy_dt_15, dy_dt_16, dy_dt_17, dy_dt_18, dy_dt_19, dy_dt_20, dy_dt_21, dy_dt_22, dy_dt_23, dy_dt_24, dy_dt_25, dy_dt_26, dy_dt_27, dy_dt_28, dy_dt_29, dy_dt_30, dy_dt_31, dy_dt_32, dy_dt_33, dy_dt_34, dy_dt_35, dy_dt_36, dy_dt_37, dy_dt_38, dy_dt_39, dy_dt_40, dy_dt_41, dy_dt_42, dy_dt_43, dy_dt_44, dy_dt_45, dy_dt_46, dy_dt_47, dy_dt_48, dy_dt_49, dy_dt_50, dy_dt_51, dy_dt_52, dy_dt_53, dy_dt_54, dy_dt_55, dy_dt_56, dy_dt_57, dy_dt_58, dy_dt_59, dy_dt_60, dy_dt_61, dy_dt_62, dy_dt_63, dy_dt_64, dy_dt_65, dy_dt_66, dy_dt_67, dy_dt_68, dy_dt_69, dy_dt_70, dy_dt_71, dy_dt_72, dy_dt_73, dy_dt_74, dy_dt_75, dy_dt_76, dy_dt_77, dy_dt_78, dy_dt_79, dy_dt_80, dy_dt_81, dy_dt_82, dy_dt_83, dy_dt_84, dy_dt_85, dy_dt_86, dy_dt_87, dy_dt_88, dy_dt_89, dy_dt_90, dy_dt_91, dy_dt_92, dy_dt_93)

    
    def get_initial_conditions(self):
        """ Function to get nominal initial conditions for the model. """
        c1 = 4962.0 # EGF
        c2 = 50000.0 # EGFR
        c3 = 0.0
        c4 = 0.0
        c5 = 0.0
        c6 = 0.0
        c7 = 0.0
        c8 = 0.0
        c9 = 0.0
        c10 = 0.0
        c11 = 0.0
        c12 = 81000.0 # coated pit protein
        c13 = 0.0
        c14 = 12000.0 # GAP
        c15 = 0.0
        c16 = 0.0
        c17 = 0.0
        c18 = 0.0
        c19 = 0.0
        c20 = 0.0
        c21 = 0.0
        c22 = 11000.0 # (value from CellML exposure) # Grb2 (val in paper 5.1e4)
        c23 = 0.0
        c24 = 26300.0 # (value from CellML exposure) # Sos (val in paper 6.63e4)
        c25 = 0.0
        c26 =  72000.0 # (value from CellML exposure) # Ras-GDP (val in paper 1.14e7)
        c27 = 0.0
        c28 = 0.0
        c29 = 0.0
        c30 = 40000.0 # Grb2-Sos
        c31 = 101000.0 # Shc
        c32 = 0.0
        c33 = 0.0
        c34 = 0.0
        c35 = 0.0
        c36 = 0.0
        c37 = 0.0
        c38 = 0.0
        c39 = 0.0
        c40 = 0.0
        c41 = 40000.0 # Raf
        c42 = 0.0
        c43 = 0.0
        c44 = 40000.0 # Phosphatase1 (acts on Raf)
        c45 = 0.0
        c46 = 0.0
        c47 = 22000000.0 # MEK
        c48 = 0.0
        c49 = 0.0
        c50 = 0.0
        c51 = 0.0
        c52 = 0.0
        c53 = 40000.0 # Phosphatase2 (acts on MEK)
        c54 = 0.0
        c55 = 21000000.0 # ERK
        c56 = 0.0
        c57 = 0.0
        c58 = 0.0
        c59 = 0.0 # ERK_pp
        c60 = 10000000.0 # Phosphatase3 (acts on ERK)
        c61 = 0.0
        c62 = 0.0
        c63 = 0.0
        c64 = 0.0
        c65 = 0.0
        c66 = 0.0
        c67 = 0.0
        c68 = 0.0
        c69 = 0.0
        c70 = 0.0
        c71 = 0.0
        c72 = 0.0
        c73 = 0.0
        c74 = 0.0
        c75 = 0.0
        c76 = 0.0
        c77 = 0.0
        c78 = 0.0
        c79 = 0.0
        c80 = 0.0
        c81 = 0.0
        c82 = 0.0
        c83 = 0.0 # ERK_pp due to internalized receptor
        c84 = 0.0
        c85 = 0.0
        c86 = 0.0
        c87 = 0.0
        c88 = 0.0
        c89 = 0.0
        c90 = 0.0
        c91 = 0.0
        c92 = 0.0
        c93 = 0.0
        c94 = 0.0

        return (c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, 
                c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, 
                c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38,  
                c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, 
                c51, c52, c53, c54, c55, c56, c57, c58, c59, c60, c61, c62, 
                c63, c64, c65, c66, c67, c68, c69, c70, c71, c72, c73, c74, 
                c75, c76, c77, c78, c79, c80, c81, c82, c83, c84, c85, c86, 
                c87, c88, c89, c90, c91, c92, c93, c94)

        
    
    def get_nominal_params(self):
        """ Function to get nominal parameters for the model. """
        # param_dict = {'k1': 3e7, # this is the param dict from Supp Table 1 in the paper
        #     'kd1': 3.8e-3,
        #     'k2': 1e7,
        #     'kd2': 0.1,
        #     'k3': 1,
        #     'kd3': 0.01,
        #     'k4': 1.73e-7,
        #     'kd4': 1.66e-3,
        #     'k6': 5e-5,
        #     'kd6': 5e-3,
        #     'k8': 1e6,
        #     'kd8': 0.2,
        #     'k10b': 1.4e5,
        #     'kd10': 0.011,
        #     'k13': 2.17,
        #     'k15': 1e4,
        #     'k16': 0.055,
        #     'kd16': 16.5,
        #     'k17': 1e7,
        #     'kd17': 0.06,
        #     'k18': 1.5e7,
        #     'kd18': 1.3,
        #     'k19': 0.5,
        #     'kd19': 1e5,
        #     'k20': 2.1e6,
        #     'kd20': 0.4,
        #     'k21': 0.023,
        #     'kd21': 2.2e5,
        #     'k22': 2.1e7,
        #     'kd22': 0.1,
        #     'k23': 6.0,
        #     'kd23': 0.6,
        #     'kd24': 1e7,
        #     'k25': 1e7,
        #     'kd25': 0.0214,
        #     'k28': 1e6,
        #     'kd28': 0.0053,
        #     'k29': 1.0,
        #     'kd29': 7e5,
        #     'k32': 0.1,
        #     'kd32': 2.4e5,
        #     'k33': 0.2,
        #     'kd33': 2.1e7,
        #     'k34': 0.03,
        #     'kd34': 4.5e6,
        #     'k35': 0.0015,
        #     'kd35': 4.5e6,
        #     'Km36': 340,
        #     'Vm36': 1.7,
        #     'k37': 0.3,
        #     'kd37': 9e5,
        #     'k40': 3e7,
        #     'kd40': 0.064,
        #     'k41': 3e7,
        #     'kd41': 0.0429,
        #     'k42': 7.17e7,
        #     'kd42': 0.2,
        #     'k43': 1.0,
        #     'k44': 1.11e7,
        #     'kd44': 0.01833,
        #     'k45': 3.5,
        #     'k47': 2.9,
        #     'k48': 1.43e7,
        #     'kd48': 0.8,
        #     'k49': 0.058,
        #     'k50': 2.5e5,
        #     'kd50': 0.5,
        #     'k52': 1.1e5,
        #     'kd52': 0.033,
        #     'k53': 16.0,
        #     'k55': 5.7,
        #     'k56': 1.45e7,
        #     'kd56': 0.6,
        #     'k57': 0.27,
        #     'k58': 5e6,
        #     'kd58': 0.5,
        #     'k59': 0.3,
        #     'k60': 6.67e-4,
        #     'k61': 1.67e-4,
        #     'RT': 50000.0,
        #     'v13': 130.2,}
        
        param_dict = {'k1': 0.003, # this is the param dict from the CellML exposure
            'kd1': 0.228,
            'k2': 0.001,
            'kd2': 6.0,
            'k3': 60.0,
            'kd3': 0.6,
            'k4': 1.038e-05,
            'kd4': 0.0996,
            'k6': 0.003,
            'kd6': 0.3,
            'k8': 0.0001,
            'kd8': 12.0,
            'k10b': 3.25581,
            'kd10': 0.66,
            'k13': 130.2,
            'k15': 600000.0,
            'k16': 0.001,
            'kd16': 16.5,
            'k17': 0.001,
            'kd17': 3.6,
            'k18': 0.0015,
            'kd18': 78.0,
            'k19': 30.0,
            'kd19': 1e-05,
            'k20': 0.00021,
            'kd20': 24.0,
            'k21': 1.38,
            'kd21': 2.2e-05,
            'k22': 0.0021,
            'kd22': 6.0,
            'k23': 360.0,
            'kd23': 36.0,
            'kd24': 33.0,
            'k25': 0.001,
            'kd25': 1.284,
            'k28': 0.0001,
            'kd28': 0.318,
            'k29': 60.0,
            'kd29': 7e-05,
            'k32': 6.0,
            'kd32': 2.4e-05,
            'k33': 12.0,
            'kd33': 0.0021,
            'k34': 1.8,
            'kd34': 0.00045,
            'k35': 0.09,
            'kd35': 0.00045,
            'Km36': 200000000000000.0,
            'Vm36': 61200.0,
            'k37': 18.0,
            'kd37': 9e-05,
            'k40': 0.003,
            'kd40': 3.84,
            'k41': 0.003,
            'kd41': 2.574,
            'k42': 0.007,
            'kd42': 12.0,
            'k43': 60.0,
            'k44': 0.00111,
            'kd44': 1.0998,
            'k45': 210.0,
            'k47': 174.0,
            'k48': 0.00143,
            'kd48': 48.0,
            'k49': 3.48,
            'k50': 2.5e-05,
            'kd50': 30.0,
            'k52': 0.00534,
            'kd52': 1.98,
            'k53': 960.0,
            'k55': 342.0,
            'k56': 0.00145,
            'kd56': 36.0,
            'k57': 16.2,
            'k58': 0.0005,
            'kd58': 30.0,
            'k59': 18.0,
            'k60': 0.04002,
            'k61': 0.01002,
            'RT': 50000.0,}

        param_list = [param_dict[key] for key in param_dict]

        return param_dict, param_list

