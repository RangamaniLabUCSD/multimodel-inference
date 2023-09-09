import equinox as eqx

class huang_ferrell_1996(eqx.Module):
    """ Note this is based on the code in https://models.physiomeproject.org/exposure/3227298fc9cc3afc14d058750142c69c/huang_ferrell_1996.cellml/@@cellml_codegen/Python
    
    Which comes from the Huang-Ferrell 1996 paper entry on the CellML site https://models.physiomeproject.org/exposure/3227298fc9cc3afc14d058750142c69c/huang_ferrell_1996.cellml/view"""
    def __call__(self, t, y, constants):
        # unpack state
        mos, mosstar, mose1, mosstare2, mekstarmosstar, mekmosstar, mekstar, mekstarstar, \
            mekstarmekpase, mekstarstarmekpase, mapkmekstarstar, mapkmekstarstar, mek, \
            mapkstar, mapkstarstar, mapkstarmapkpase, mapkstarstarmapkpase, mapk = y
        
        
        dy_dt_0 = -constants[7]*(((((constants[0]-mosstar)-mose1)-mosstare2)-mekmosstar)-mekstarmosstar)*(constants[1]-mose1)+constants[17]*mose1+constants[28]*mosstare2
        dy_dt_2 = constants[7]*(((((constants[0]-mosstar)-mose1)-mosstare2)-mekmosstar)-mekstarmosstar)*(constants[1]-mose1)-(constants[17]+constants[27])*mose1
        dy_dt_1 = (((-constants[8]*mosstar*(constants[2]-mosstare2)+constants[18]*mosstare2+constants[27]*mose1+(constants[29]+constants[19])*mekmosstar)-constants[9]*mosstar*((((((((constants[3]-mekstar)-mekstarstar)-mekstarmekpase)-mekstarstarmekpase)-mekmosstar)-mekstarmosstar)-mapkmekstarstar)-mapkmekstarstar))+(constants[31]+constants[21])*mekstarmosstar)-constants[11]*mekstar*mosstar
        dy_dt_3 = constants[8]*mosstar*(constants[2]-mosstare2)-(constants[18]+constants[28])*mosstare2
        dy_dt_12 = -constants[9]*((((((((constants[3]-mekstar)-mekstarstar)-mekstarmekpase)-mekstarstarmekpase)-mekmosstar)-mekstarmosstar)-mapkmekstarstar)-mapkmekstarstar)*mosstar+constants[19]*mekmosstar+constants[30]*mekstarmekpase
        dy_dt_5 = constants[9]*((((((((constants[3]-mekstar)-mekstarstar)-mekstarmekpase)-mekstarstarmekpase)-mekmosstar)-mekstarmosstar)-mapkmekstarstar)-mapkmekstarstar)*mosstar-(constants[19]+constants[29])*mekmosstar
        dy_dt_6 = (-constants[10]*mekstar*((constants[4]-mekstarmekpase)-mekstarstarmekpase)+constants[20]*mekstarmekpase+constants[29]*mekmosstar+constants[32]*mekstarstarmekpase+constants[21]*mekstarmosstar)-constants[11]*mekstar*mosstar
        dy_dt_8 = constants[10]*mekstar*((constants[4]-mekstarmekpase)-mekstarstarmekpase)-(constants[20]+constants[30])*mekstarmekpase
        dy_dt_4 = constants[11]*mekstar*mosstar-(constants[21]+constants[31])*mekstarmosstar
        dy_dt_7 = ((((constants[31]*mekstarmosstar-constants[12]*mekstarstar*((constants[4]-mekstarmekpase)-mekstarstarmekpase))+constants[22]*mekstarstarmekpase)-constants[13]*mekstarstar*((((((constants[5]-mapkstar)-mapkstarstar)-mapkstarmapkpase)-mapkstarstarmapkpase)-mapkmekstarstar)-mapkmekstarstar))+(constants[23]+constants[33])*mapkmekstarstar+(constants[25]+constants[35])*mapkmekstarstar)-constants[15]*mapkstar*mekstarstar
        dy_dt_9 = constants[12]*mekstarstar*((constants[4]-mekstarmekpase)-mekstarstarmekpase)-(constants[22]+constants[32])*mekstarstarmekpase
        dy_dt_17 = -constants[13]*((((((constants[5]-mapkstar)-mapkstarstar)-mapkstarmapkpase)-mapkstarstarmapkpase)-mapkmekstarstar)-mapkmekstarstar)*mekstarstar+constants[23]*mapkmekstarstar+constants[34]*mapkstarmapkpase
        dy_dt_10 = constants[13]*((((((constants[5]-mapkstar)-mapkstarstar)-mapkstarmapkpase)-mapkstarstarmapkpase)-mapkmekstarstar)-mapkmekstarstar)*mekstarstar-(constants[23]+constants[33])*mapkmekstarstar
        dy_dt_13 = (((constants[33]*mapkmekstarstar-constants[14]*mapkstar*((constants[6]-mapkstarmapkpase)-mapkstarstarmapkpase))+constants[24]*mapkstarmapkpase)-constants[15]*mapkstar*mekstarstar)+constants[25]*mapkmekstarstar+constants[36]*mapkstarstarmapkpase
        dy_dt_11 = constants[15]*mapkstar*mekstarstar-(constants[25]+constants[35])*mapkmekstarstar
        dy_dt_14 = -constants[16]*mapkstarstar*((constants[6]-mapkstarmapkpase)-mapkstarstarmapkpase)+constants[26]*mapkstarstarmapkpase+constants[35]*mapkmekstarstar
        dy_dt_15 = constants[14]*mapkstar*((constants[6]-mapkstarmapkpase)-mapkstarstarmapkpase)-(constants[24]+constants[34])*mapkstarmapkpase
        dy_dt_16 = constants[16]*mapkstarstar*((constants[6]-mapkstarmapkpase)-mapkstarstarmapkpase)-(constants[26]+constants[36])*mapkstarstarmapkpase
        
        return (dy_dt_0, dy_dt_1, dy_dt_2, dy_dt_3, dy_dt_4, dy_dt_5, dy_dt_6, \
                dy_dt_7, dy_dt_8, dy_dt_9, dy_dt_10, dy_dt_11, dy_dt_12, \
                dy_dt_13, dy_dt_14, dy_dt_15, dy_dt_16, dy_dt_17)
    
    def get_initial_conditions(self):
        mos = 0.003
        mosstar = 0
        mose1 = 0
        mosstare2 = 0
        mekstarmosstar = 0
        mekmosstar = 0
        mekstar = 0
        mekstarstar = 0
        mekstarmekpase = 0
        mekstarstarmekpase = 0
        mapkmekstarstar = 0
        mapkmekstarstar = 0
        mek = 1.2
        mapkstar = 0
        mapkstarstar = 0
        mapkstarmapkpase = 0
        mapkstarstarmapkpase = 0
        mapk = 1.2

        return (mos, mosstar, mose1, mosstare2, mekstarmosstar, mekmosstar, mekstar, mekstarstar, \
                mekstarmekpase, mekstarstarmekpase, mapkmekstarstar, mapkmekstarstar, mek, \
                    mapkstar, mapkstarstar, mapkstarmapkpase, mapkstarstarmapkpase, mapk)
    
    def get_nominal_params(self):
        param_dict = {'mostot': 0.003,
            'e1tot': 0.0003,
            'e2tot': 0.0003,
            'mektot': 1.2,
            'mekpasetot': 0.0003,
            'mapktot': 1.2,
            'mapkpasetot': 0.12,
            'a1': 1000.0,
            'a2': 1000.0,
            'a3': 1000.0,
            'a4': 1000.0,
            'a5': 1000.0,
            'a6': 1000.0,
            'a7': 1000.0,
            'a8': 1000.0,
            'a9': 1000.0,
            'a10': 1000.0,
            'd1': 150.0,
            'd2': 150.0,
            'd3': 150.0,
            'd4': 150.0,
            'd5': 150.0,
            'd6': 150.0,
            'd7': 150.0,
            'd8': 150.0,
            'd9': 150.0,
            'd10': 150.0,
            'k1': 150.0,
            'k2': 150.0,
            'k3': 150.0,
            'k4': 150.0,
            'k5': 150.0,
            'k6': 150.0,
            'k7': 150.0,
            'k8': 150.0,
            'k9': 150.0,
            'k10': 150.0,}
        
        param_list = [param_dict[key] for key in param_dict]

        return param_dict, param_list

