import equinox as eqx
import jax.numpy as jnp

class schoeberl_2002(eqx.Module):
    """ Note this is the code at https://models.physiomeproject.org/exposure/a7197a8d519bbee0e95256f699b26050/Schoeberl_EichlerJonsson_Gilles_Muller_2002.cellml/@@cellml_codegen/Python
    
    Which comes from the Schoeberl paper entry on the CellML site https://models.physiomeproject.org/exposure/a7197a8d519bbee0e95256f699b26050"""

    def custom_piecewise(self, cases):
        """Compute result of a piecewise function"""
        return jnp.select(cases[0::2],cases[1::2])

    def __call__(self, t, y, constants):
        # unpack state
        c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, \
            c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, \
            c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, \
            c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, \
            c51, c52, c53, c54, c55, c56, c57, c58, c59, c60, c61, c62, \
            c63, c64, c65, c66, c67, c68, c69, c70, c71, c72, c73, c74, \
            c75, c76, c77, c78, c79, c80, c81, c82, c83, c84, c85, c86, \
            c87, c88, c89, c90, c91, c92, c93, c94 = y
        
        dy_dt_0 = constants[81]
        v1 = constants[0]*c1*c2-constants[1]*c3
        v2 = constants[2]*c3*c3-constants[3]*c4
        dy_dt_2 = v1-2.0*v2
        v3 = constants[4]*c4-constants[5]*c5
        dy_dt_3 = v2-v3
        v6 = constants[8]*c2-constants[9]*c6
        dy_dt_1 = constants[80]-(v1+v6)
        v7 = constants[8]*c5
        v8 = constants[10]*c5*c14-constants[11]*c15
        dy_dt_4 = v3-(v7+v8)
        v10 = constants[12]*c6*c16-constants[13]*c10
        v60 = constants[77]*c6
        dy_dt_5 = v6-(v10+v60)
        v11 = constants[2]*c10*c10-constants[3]*c11
        dy_dt_9 = v10-2.00000*v11
        v61 = constants[78]*c16
        dy_dt_12 = v61
        dy_dt_15 = -(v10+v61)
        dy_dt_85 = v60
        v12 = constants[4]*c11-constants[5]*c8
        dy_dt_10 = v11-v12
        v14 = constants[10]*c8*c14-constants[11]*c17
        dy_dt_13 = -(v8+v14)
        v62 = constants[77]*c8
        dy_dt_7 = (v7+v12)-(v14+v62)
        dy_dt_86 = v62
        v4 = constants[6]*c23*c12-constants[7]*c7
        v16 = constants[16]*c22*c15-constants[17]*c23
        v9 = constants[8]*c23
        v17 = constants[18]*c24*c23-constants[19]*c25
        dy_dt_22 = v16-(v4+v9+v17)
        v19 = constants[22]*c27-constants[23]*c28*c25
        v27 = constants[22]*c36-constants[23]*c35*c28
        v28 = constants[35]*c28*c41-constants[36]*c42
        dy_dt_27 = (v19+v27)-v28
        v29 = constants[37]*c42-constants[38]*c43*c45
        dy_dt_41 = v28-v29
        v20 = constants[24]*c25*c43-constants[25]*c29
        v30 = constants[24]*c35*c43-constants[25]*c37
        dy_dt_42 = v29-(v20+v30)
        v43 = constants[57]*c46
        v42 = constants[55]*c44*c45-constants[56]*c46
        dy_dt_45 = v42-v43
        v44 = constants[58]*c47*c45-constants[59]*c48
        v45 = constants[60]*c48
        dy_dt_47 = v44-v45
        v46 = constants[58]*c49*c45-constants[59]*c50
        v47 = constants[61]*c50
        dy_dt_44 = (v29+v45+v47)-(v42+v44+v46)
        dy_dt_49 = v46-v47
        v49 = constants[64]*c52
        v48 = constants[62]*c51*c53-constants[63]*c52
        dy_dt_51 = v48-v49
        v50 = constants[65]*c53*c49-constants[66]*c54
        dy_dt_48 = (v45+v49)-(v46+v50)
        v51 = constants[64]*c54
        dy_dt_53 = v50-v51
        v52 = constants[67]*c55*c51-constants[68]*c56
        v53 = constants[69]*c56
        dy_dt_55 = v52-v53
        v54 = constants[67]*c51*c57-constants[68]*c58
        v55 = constants[70]*c58
        dy_dt_50 = (v47+v53+v55)-(v48+v52+v54)
        dy_dt_57 = v54-v55
        v56 = constants[71]*c59*c60-constants[72]*c61
        dy_dt_58 = v55-v56
        v57 = constants[73]*c61
        dy_dt_60 = v56-v57
        v58 = constants[74]*c60*c57-constants[75]*c62
        dy_dt_56 = (v53+v57)-(v54+v58)
        v59 = constants[76]*c62
        dy_dt_61 = v58-v59
        v22 = constants[28]*c31*c15-constants[29]*c32
        v69 = constants[28]*c31*c17-constants[29]*c63
        v36 = (constants[48]*c40)/(constants[47]+c40)
        dy_dt_30 = v36-(v22+v69)
        v63 = constants[16]*c17*c22-constants[17]*c18
        v24 = constants[16]*c22*c33-constants[32]*c34
        v35 = constants[45]*c30-constants[46]*c24*c22
        v38 = constants[16]*c22*c40-constants[32]*c39
        v71 = constants[16]*c22*c64-constants[32]*c65
        dy_dt_21 = v35-(v16+v24+v38+v63+v71)
        v23 = constants[30]*c32-constants[31]*c33
        v103 = constants[8]*c32-constants[9]*c63
        dy_dt_31 = v22-(v23+v103)
        v70 = constants[30]*c63-constants[31]*c64
        dy_dt_62 = (v69+v103)-v70
        v64 = constants[18]*c24*c18-constants[19]*c19
        v72 = constants[33]*c24*c65-constants[34]*c66
        v25 = constants[33]*c24*c34-constants[34]*c35
        v40 = constants[51]*c24*c39-constants[52]*c38
        dy_dt_23 = v35-(v17+v25+v40+v64+v72)
        v66 = constants[22]*c20-constants[23]*c69*c19
        v75 = constants[35]*c69*c41-constants[36]*c70
        v74 = constants[22]*c67-constants[23]*c66*c69
        dy_dt_68 = (v66+v74)-v75
        v76 = constants[37]*c70-constants[38]*c71*c72
        dy_dt_69 = v75-v76
        v67 = constants[24]*c71*c19-constants[25]*c21
        v77 = constants[24]*c71*c66-constants[25]*c68
        dy_dt_70 = v76-(v67+v77)
        v65 = constants[20]*c26*c19-constants[21]*c20
        v68 = constants[26]*c21-constants[27]*c19*c26
        v18 = constants[20]*c26*c25-constants[21]*c27
        v21 = constants[26]*c29-constants[27]*c25*c26
        v26 = constants[20]*c26*c35-constants[21]*c36
        v31 = constants[26]*c37-constants[27]*c35*c26
        v78 = constants[26]*c68-c26*constants[27]*c66
        v73 = constants[20]*c26*c66-constants[21]*c67
        dy_dt_25 = (v21+v31+v68+v78)-(v18+v26+v65+v73)
        v85 = constants[57]*c73
        dy_dt_40 = (v43+v85)-(v28+v75)
        v84 = constants[55]*c44*c72-constants[56]*c73
        dy_dt_43 = (v43+v85)-(v42+v84)
        dy_dt_72 = v84-v85
        v32 = constants[39]*c35-constants[40]*c38*c15
        v79 = constants[39]*c66-constants[40]*c17*c38
        v33 = constants[41]*c38-constants[42]*c40*c30
        dy_dt_37 = (v32+v40+v79)-v33
        v86 = constants[58]*c47*c72-constants[59]*c74
        v87 = constants[60]*c74
        dy_dt_73 = v86-v87
        v37 = constants[49]*c33-constants[50]*c15*c40
        v81 = constants[49]*c64-constants[50]*c17*c40
        dy_dt_39 = (v33+v37+v81)-(v36+v38)
        v88 = constants[58]*c72*c75-constants[59]*c76
        v89 = constants[61]*c76
        dy_dt_71 = (v76+v87+v89)-(v84+v86+v88)
        dy_dt_75 = v88-v89
        v39 = constants[49]*c34-constants[50]*c15*c39
        v82 = constants[49]*c65-constants[50]*c17*c39
        dy_dt_38 = (v38+v39+v82)-v40
        v34 = constants[43]*c25-constants[44]*c15*c30
        v102 = constants[8]*c15-constants[9]*c17
        dy_dt_14 = (v8+v32+v34+v37+v39)-(v16+v22+v102)
        v80 = constants[43]*c19-constants[44]*c17*c30
        dy_dt_16 = (v14+v79+v80+v81+v82+v102)-(v63+v69)
        v41 = constants[53]*c30*c33-constants[54]*c35
        v83 = constants[53]*c30*c64-constants[54]*c66
        dy_dt_29 = (v33+v34+v80)-(v35+v41+v83)
        v90 = constants[62]*c77*c53-constants[63]*c78
        v91 = constants[64]*c78
        dy_dt_77 = v90-v91
        v104 = constants[8]*c33-constants[9]*c64
        dy_dt_32 = v23-(v24+v37+v41+v104)
        dy_dt_63 = (v70+v104)-(v71+v81+v83)
        v92 = constants[65]*c53*c75-constants[66]*c79
        dy_dt_74 = (v87+v91)-(v92+v88)
        v94 = constants[67]*c55*c77-constants[68]*c80
        v95 = constants[69]*c80
        dy_dt_79 = v94-v95
        v106 = constants[6]*c25*c12-constants[7]*c88
        v105 = constants[8]*c25-constants[9]*c19
        dy_dt_24 = (v17+v19+v21)-(v18+v20+v34+v105+v106)
        v93 = constants[64]*c79
        dy_dt_46 = (v51+v93)-(v44+v86)
        dy_dt_52 = (v49+v51+v91+v93)-(v48+v50+v90+v92)
        dy_dt_78 = v92-v93
        v96 = constants[67]*c77*c81-constants[68]*c82
        v97 = constants[70]*c82
        dy_dt_76 = (v89+v95+v97)-(v90+v94+v96)
        dy_dt_81 = v96-v97
        v109 = constants[6]*c27*c12-constants[7]*c89
        v108 = constants[8]*c27-constants[9]*c20
        dy_dt_26 = v18-(v19+v108+v109)
        v98 = constants[71]*c83*c60-constants[72]*c84
        dy_dt_82 = v97-v98
        v99 = constants[73]*c84
        dy_dt_83 = v98-v99
        v112 = constants[6]*c29*c12-constants[7]*c90
        v111 = constants[8]*c29-constants[9]*c21
        dy_dt_28 = v20-(v21+v111+v112)
        v100 = constants[74]*c60*c81-constants[75]*c85
        dy_dt_80 = (v95+v99)-(v96+v100)
        v101 = constants[76]*c85
        dy_dt_54 = (v59+v101)-(v52+v94)
        dy_dt_59 = (v57+v59+v99+v101)-(v56+v58+v98+v100)
        dy_dt_84 = v100-v101
        v115 = constants[6]*c34*c12-constants[7]*c91
        v114 = constants[8]*c34-constants[9]*c65
        dy_dt_33 = v24-(v25+v39+v114+v115)
        v118 = constants[6]*c35*c12-constants[7]*c92
        v117 = constants[8]*c35-constants[9]*c66
        dy_dt_34 = (v25+v27+v31+v41)-(v117+v118+v26+v30+v32)
        v121 = constants[6]*c36*c12-constants[7]*c93
        v120 = constants[8]*c36-constants[9]*c67
        dy_dt_35 = v26-(v27+v120+v121)
        v15 = constants[15]*c9
        v124 = constants[6]*c37*c12-constants[7]*c94
        dy_dt_11 = v15-(v4+v106+v109+v112+v115+v118+v121+v124)
        v123 = constants[8]*c37-constants[9]*c68
        dy_dt_36 = v30-(v31+v123+v124)
        C = (constants[79]*1.0)/(constants[1]/(constants[0]*c1)+1.0)
        k5 = self.custom_piecewise([jnp.less(C , 3100.00), 1.55, jnp.greater(C , 100000.), 0.2, True, (C*-1.35e-05+1.55)/1.0])
        v5 = k5*c7
        dy_dt_6 = v4-v5
        dy_dt_17 = (v5+v9+v63)-v64
        v107 = k5*c88
        dy_dt_18 = (v66+v64+v68+v105+v107)-(v65+v67+v80)
        dy_dt_87 = v106-v107
        v110 = k5*c89
        dy_dt_19 = (v65+v108+v110)-v66
        dy_dt_88 = v109-v110
        v113 = k5*c90
        dy_dt_20 = (v67+v111+v113)-v68
        dy_dt_89 = v112-v113
        v116 = k5*c91
        dy_dt_64 = (v71+v114+v116)-(v72+v82)
        dy_dt_90 = v115-v116
        v119 = k5*c92
        dy_dt_65 = (v72+v74+v78+v83+v117+v119)-(v73+v77+v79)
        dy_dt_91 = v118-v119
        v122 = k5*c93
        dy_dt_66 = (v73+v120+v122)-v74
        dy_dt_92 = v121-v122
        v125 = k5*c94
        dy_dt_8 = (v5+v107+v110+v113+v116+v119+v122+v125)-v15
        dy_dt_67 = (v77+v123+v125)-v78
        dy_dt_93 = v124-v125

        return (dy_dt_0, dy_dt_1, dy_dt_2, dy_dt_3, dy_dt_4, dy_dt_5, dy_dt_6, dy_dt_7, dy_dt_8, dy_dt_9, dy_dt_10, dy_dt_11, dy_dt_12, dy_dt_13, dy_dt_14, dy_dt_15, dy_dt_16, dy_dt_17, dy_dt_18, dy_dt_19, dy_dt_20, dy_dt_21, dy_dt_22, dy_dt_23, dy_dt_24, dy_dt_25, dy_dt_26, dy_dt_27, dy_dt_28, dy_dt_29, dy_dt_30, dy_dt_31, dy_dt_32, dy_dt_33, dy_dt_34, dy_dt_35, dy_dt_36, dy_dt_37, dy_dt_38, dy_dt_39, dy_dt_40, dy_dt_41, dy_dt_42, dy_dt_43, dy_dt_44, dy_dt_45, dy_dt_46, dy_dt_47, dy_dt_48, dy_dt_49, dy_dt_50, dy_dt_51, dy_dt_52, dy_dt_53, dy_dt_54, dy_dt_55, dy_dt_56, dy_dt_57, dy_dt_58, dy_dt_59, dy_dt_60, dy_dt_61, dy_dt_62, dy_dt_63, dy_dt_64, dy_dt_65, dy_dt_66, dy_dt_67, dy_dt_68, dy_dt_69, dy_dt_70, dy_dt_71, dy_dt_72, dy_dt_73, dy_dt_74, dy_dt_75, dy_dt_76, dy_dt_77, dy_dt_78, dy_dt_79, dy_dt_80, dy_dt_81, dy_dt_82, dy_dt_83, dy_dt_84, dy_dt_85, dy_dt_86, dy_dt_87, dy_dt_88, dy_dt_89, dy_dt_90, dy_dt_91, dy_dt_92, dy_dt_93)

    
    def get_initial_conditions(self):
        """ Function to get nominal initial conditions for the model. """
        c1 = 4962.0
        c2 = 50000.0
        c3 = 0.0
        c4 = 0.0
        c5 = 0.0
        c6 = 0.0
        c7 = 0.0
        c8 = 0.0
        c9 = 0.0
        c10 = 0.0
        c11 = 0.0
        c12 = 81000.0
        c13 = 0.0
        c14 = 12000.0
        c15 = 0.0
        c16 = 0.0
        c17 = 0.0
        c18 = 0.0
        c19 = 0.0
        c20 = 0.0
        c21 = 0.0
        c22 = 11000.0
        c23 = 0.0
        c24 = 26300.0
        c25 = 0.0
        c26 = 72000.0
        c27 = 0.0
        c28 = 0.0
        c29 = 0.0
        c30 = 40000.0
        c31 = 101000.0
        c32 = 0.0
        c33 = 0.0
        c34 = 0.0
        c35 = 0.0
        c36 = 0.0
        c37 = 0.0
        c38 = 0.0
        c39 = 0.0
        c40 = 0.0
        c41 = 40000.0
        c42 = 0.0
        c43 = 0.0
        c44 = 40000.0
        c45 = 0.0
        c46 = 0.0
        c47 = 22000000.0
        c48 = 0.0
        c49 = 0.0
        c50 = 0.0
        c51 = 0.0
        c52 = 0.0
        c53 = 40000.0
        c54 = 0.0
        c55 = 21000000.0
        c56 = 0.0
        c57 = 0.0
        c58 = 0.0
        c59 = 0.0
        c60 = 10000000.0
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
        c83 = 0.0
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

        return (c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, \ 
                c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, \
                c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, \ 
                c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, \
                c51, c52, c53, c54, c55, c56, c57, c58, c59, c60, c61, c62, \
                c63, c64, c65, c66, c67, c68, c69, c70, c71, c72, c73, c74, \
                c75, c76, c77, c78, c79, c80, c81, c82, c83, c84, c85, c86, \
                c87, c88, c89, c90, c91, c92, c93, c94)

        
    
    def get_nominal_params(self):
        """ Function to get nominal parameters for the model. """
        param_dict = {'k1': 0.003,
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
            'RT': 50000.0,
            'v13': 130.2,
            'constants[81]': 0.0}

        param_list = [param_dict[key] for key in param_dict]

        return param_dict, param_list

