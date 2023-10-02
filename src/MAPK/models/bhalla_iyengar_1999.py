import equinox as eqx
import jax.numpy as jnp

class bhalla_iyengar_1999(eqx.Module):
    """ 
Which comes from the Bhalla and Inyengar paper entry on the CellML site https://models.physiomeproject.org/e/7e3/bhalla_iyengar_1999_a.cellml/view"""

    def __call__(self, t, y, constants):

        # unpack state
        PKC, GAP, Raf, GEF_star, CaM_GEF, Gbg_GEF, CaGqPLC, CaPLC, NgCaM, GEF, \
            Ng, Gbg, Ga_GDP, mGluR, PLC, Ca2CaM, Ca3CaM, Ng_star, Gabg, R, \
            Gabg_R, Glu_synapse, Gabg_GluR, Glu, Ga_GTP, GqPLC, GEF_inact, \
            Ca4CAM, CaM, SHCstar_SOS_GRB2, MAPK_star, SOS, MAPKK_star_star, \
            MAPKK_star, Raf_star, GTPRasRaf_star, Raf_star_star, GTP_RAS, \
            GDP_RAS, GAPstar, MAPKK, MAPK, MAPK_tyr, EGF_EGFR, CaPLCg, SHC, \
            SHCstar, SOSstar_GRB2, SOS_GRB2, GRB2, SOSstar, EGF_EGFR_INTERNAL, \
            EGF, EGFR, CaPLCg_star, PLCg, PLCg_star, PKC_i, DAGPKC, AADAGPKC, \
            DAGCaPKC, CaPKC, AA, APC, PIP2_star, PLA2_cyt, PLA2_star, CaPLA2, \
            PIP2CaPLA2, PIP2PLA2, DAGCaPLA2, CaPLA2_star, PIP2, DAG, IP3, \
            Inositol, PC = y
        
        # unpack constants
        Ca, GTP, PKA, kfD1, kbD1, kfD2, kbD2, kfD3, kbD3, kfD4, kbD4, \
            kfD5, kbD5, kfD6, kfD7, kfD8, kfD9, kfG1, kbG1, kfG2, kbG2, kfG3, \
            kbG3, kfG4, kbG4, kfG5, kfB2, kbB2, kmB3, kfB5, kbB5, kfB7, kbB7, \
            kfB8, kbB8, kmB1, VmaxB1, kmB6, VmaxB6, kfM1, kbM1, kfM2, kbM2, kfM3, \
            kbM3, kfM4, kbM4, kfM6, kmM5, VmaxM5, kmM7, VmaxM7, kmM8, VmaxM8, \
            kmH1, PP2A, kfB12, kbB12, kmB9, VmaxB9, kmB10, VmaxB10, kmB11, \
            VmaxB11, kmB13, VmaxB13, kfB4, kbB4, VmaxB3, kmA7, VmaxA7, kfH5, \
            kbH5, kmH2, kmH3, kmH4, VmaxH1, VmaxH2, VmaxH3, VmaxH4, kmA9, kmH8, \
            kmH9, kmH6, kmH7, VmaxH6, VmaxH7, VmaxH8, VmaxH9, MKP1, kmH10, \
            kmH11, kmH12, kmH13, VmaxH10, VmaxH11, VmaxH12, VmaxH13, kmA3, \
            kfA4, kbA4, kfA5, kbA5, kfA6, kbA6, kfA8, kbA8, kfA10, kbA10, \
            VmaxA3, VmaxA9, kmF4, kfA1, kbA1, kfA2, kbA2, kfF1, kbF1, kfF3, \
            kfF5, kbF5, VmaxF4, kfK1, kbK1, kfK2, kbK2, kfK3, kbK3, kfK4, kbK4, \
            kfK5, kbK5, kfK6, kbK6, kfK7, kbK7, kfK8, kbK8, kfK9, kbK9, kfK10, \
            kbK10, kfE1, kbE1, kfE3, kbE3, kfE5, kbE5, kfE7, kbE7, kfE10, kbE10, \
            kfE11, kbE11, kmE9, VmaxE9, kfE13, kbE13, kmE2, kmE4, kmE6, kmE8, \
            kmE12, VmaxE2, VmaxE4, VmaxE6, VmaxE8, VmaxE12, kmF2, kmF6, VmaxF2, \
            VmaxF6, kfG8, kfG9, kmG6, VmaxG6, kmG7, VmaxG7 = constants

        # rates
        d_Ga_GDP = (Ga_GTP*kfD7-Ga_GDP*kfD8*Gbg)+CaGqPLC*kfG5
        d_mGluR = ((kfD5*Gabg_GluR-kbD5*Gabg*mGluR)-(kfD3*mGluR-kbD3*Glu*R))+kfD6*Gabg_GluR
        d_Gabg = (((Ga_GDP*Gbg*kfD8-(Gabg*R*kfD4-Gabg_R*kbD4))+kfD5*Gabg_GluR)-kbD5*Gabg*mGluR)-kfD9*Gabg*GTP
        d_R = (kfD3*mGluR-kbD3*Glu*R)-(Gabg*R*kfD4-Gabg_R*kbD4)
        d_Gbg = (-Ga_GDP*Gbg*kfD8+kfD6*Gabg_GluR+kfD9*Gabg*GTP)-(kfB8*GEF*Gbg-kbB8*Gbg_GEF)
        d_Gabg_R = ((Gabg*R*kfD4-Gabg_R*kbD4)+kfD2*Gabg_GluR)-Gabg_R*Glu*kbD2
        d_Glu_synapse = -(kfD1*Glu_synapse-kbD1*Glu)
        d_Gabg_GluR = (-(kfD2*Gabg_GluR-Gabg_R*Glu*kbD2)-(kfD5*Gabg_GluR-kbD5*Gabg*mGluR))-kfD6*Gabg_GluR
        d_Glu = ((((kfD1*Glu_synapse-kbD1*Glu)+kfD2*Gabg_GluR)-Gabg_R*Glu*kbD2)+kfD3*mGluR)-kbD3*Glu*R
        d_Ga_GTP = ((kfD9*Gabg*GTP-Ga_GTP*kfD7)-(PLC*Ga_GTP*kfG2-GqPLC*kbG2))-(CaGqPLC*kbG4+CaPLC*Ga_GTP*kfG4)
        d_PLC = -(PLC*Ca*kfG1-CaPLC*kbG1)-(PLC*Ga_GTP*kfG2-GqPLC*kbG2)
        d_GqPLC = (PLC*Ga_GTP*kfG2-GqPLC*kbG2)-(Ca*GqPLC*kfG3-CaGqPLC*kbG3)
        d_CaPLC = ((PLC*Ca*kfG1-CaPLC*kbG1)+CaGqPLC*kfG5)-(-CaGqPLC*kbG4+CaPLC*Ga_GTP*kfG4)
        d_CaGqPLC = (((GqPLC*Ca*kfG3-CaGqPLC*kbG3)+CaPLC*Ga_GTP*kfG4)-CaGqPLC*kbG4)-CaGqPLC*kfG5
        d_GEF_inact = (PKA*VmaxB1*GEF)/(kmB1+GEF)-kfB2*GEF_inact
        d_GEF = (((-((((PKA*VmaxB1*GEF)/(kmB1+GEF)+(PKC*VmaxB6*GEF)/(kmB6*(1.0+GAP/kmB3+Ng/kmM7+Raf/kmH1+NgCaM/kmM8)+GEF))-kfB2*GEF_inact)-kfB7*GEF_star)-kfB5*Ca4CAM*GEF)+kbB5*CaM_GEF)-kfB8*GEF*Gbg)+kbB8*Gbg_GEF
        d_CaM_GEF = -kbB5*CaM_GEF+kfB5*GEF*Ca4CAM
        d_Ca4CAM = (-(-kbB5*CaM_GEF+kfB5*GEF*Ca4CAM)+Ca*Ca3CaM*kfM3)-kbM3*Ca4CAM
        d_Gbg_GEF = -(-kfB8*GEF*Gbg+kbB8*Gbg_GEF)
        d_GEF_star = (PKC*VmaxB6*GEF)/(kmB6*(1.0+GAP/kmB3+Ng/kmM7+Raf/kmH1+NgCaM/kmM8)+GEF)-kfB7*GEF_star
        d_CaM = (-(kfM1*Ca*Ca*CaM-kbM1*Ca2CaM)-(kfM4*Ng*CaM-kbM4*NgCaM))+(NgCaM*PKC*VmaxM8)/(NgCaM+kmM8)
        d_NgCaM = (kfM4*Ng*CaM-kbM4*NgCaM)-(NgCaM*PKC*VmaxM8)/(NgCaM+kmM8*(1.0+GAP/kmB3+GEF/kmB6+Raf/kmH1+Ng/kmM7))
        d_Ca2CaM = (kfM1*Ca*Ca*CaM-kbM1*Ca2CaM)-(Ca*Ca2CaM*kfM2-kbM2*Ca3CaM)
        d_Ca3CaM = (Ca*Ca2CaM*kfM2-kbM2*Ca3CaM)-(Ca*Ca3CaM*kfM3-kbM3*Ca4CAM)
        d_Ng_star = (((NgCaM*PKC*VmaxM8)/(NgCaM+kmM8*(1.0+GAP/kmB3+GEF/kmB6+Raf/kmH1+Ng/kmM7))-kfM6*Ng_star)-(VmaxM5*Ng_star)/(Ng_star+kmM5))+(Ng*PKC*VmaxM7)/(Ng+kmM7*(1.0+GAP/kmB3+GEF/kmB6+Raf/kmH1+NgCaM/kmM8))
        d_Ng = (kfM6*Ng_star+(VmaxM5*Ng_star)/(Ng_star+kmM5))-(Ng*PKC*VmaxM7)/(Ng+kmM7*(1.0+GAP/kmB3+GEF/kmB6+Raf/kmH1+NgCaM/kmM8))
        d_GTP_RAS = ((((-GAP*VmaxB13*GTP_RAS)/(kmB13+GTP_RAS)-kfB12*GTP_RAS)+(GEF_star*VmaxB10*GDP_RAS)/(kmB10+GDP_RAS)+(Gbg_GEF*VmaxB11*GDP_RAS)/(kmB11+GDP_RAS)+(CaM_GEF*VmaxB9*GDP_RAS)/(kmB9+GDP_RAS)+(SHCstar_SOS_GRB2*VmaxA7*GDP_RAS)/(kmA7+GDP_RAS))-GTP_RAS*Raf_star*kfH5)+GTPRasRaf_star*kbH5
        d_GDP_RAS = -(((-GAP*VmaxB13*GTP_RAS)/(kmB13+GTP_RAS)-kfB12*GTP_RAS)+(GEF_star*VmaxB10*GDP_RAS)/(kmB10+GDP_RAS)+(Gbg_GEF*VmaxB11*GDP_RAS)/(kmB11+GDP_RAS)+(CaM_GEF*VmaxB9*GDP_RAS)/(kmB9+GDP_RAS))-(SHCstar_SOS_GRB2*VmaxA7*GDP_RAS)/(kmA7+GDP_RAS)
        d_GAP = (-PKC*VmaxB3*GAP)/(kmB3*(1.0+Raf/kmH1+GEF/kmB6+Ng/kmM7+NgCaM/kmM8)+GAP)+kfB4*GAPstar
        d_GAPstar = (PKC*VmaxB3*GAP)/(kmB3*(1.0+Raf/kmH1+GEF/kmB6+Ng/kmM7+NgCaM/kmM8)+GAP)-kfB4*GAPstar
        d_GTPRasRaf_star = GTP_RAS*Raf_star*kfH5-GTPRasRaf_star*kbH5
        d_Raf = (-PKC*Raf*VmaxH1)/(kmH1*(1.0+GAP/kmB3+GEF/kmB6+Ng/kmM7+NgCaM/kmM8)+Raf)+(Raf_star*PP2A*VmaxH3)/(kmH3*(1.0+Raf_star_star/kmH4+MAPKK_star_star/kmH9+MAPKK_star/kmH8)+Raf_star)
        d_Raf_star = (((PKC*Raf*VmaxH1)/(kmH1*(1.0+GAP/kmB3+GEF/kmB6+Ng/kmM7+NgCaM/kmM8)+Raf)-(Raf_star*PP2A*VmaxH3)/(kmH3*(1.0+Raf_star_star/kmH4+MAPKK_star_star/kmH9+MAPKK_star/kmH8)+Raf_star))-(Raf_star*MAPK_star*VmaxH2)/(kmH2*(1.0+SOS/kmA9)+Raf_star))+(Raf_star_star*PP2A*VmaxH4)/(kmH4*(1.0+Raf_star/kmH3+MAPKK_star_star/kmH9+MAPKK_star/kmH8)+Raf_star_star)
        d_Raf_star_star = (Raf_star*MAPK_star*VmaxH2)/(kmH2*(1.0+SOS/kmA9)+Raf_star)-(Raf_star_star*PP2A*VmaxH4)/(kmH4*(1.0+Raf_star/kmH3+MAPKK_star_star/kmH9+MAPKK_star/kmH8)+Raf_star_star)
        d_MAPKK = (-MAPKK*VmaxH6*GTPRasRaf_star)/(kmH6*(1.0+MAPKK_star/kmH7)+MAPKK)+(MAPKK_star*VmaxH8*PP2A)/(kmH8*(1.0+Raf_star/kmH3+Raf_star_star/kmH4+MAPKK_star_star/kmH9)+MAPKK_star)
        d_MAPKK_star = (((MAPKK*VmaxH6*GTPRasRaf_star)/(kmH6*(1.0+MAPKK_star/kmH7)+MAPKK)-(MAPKK_star*VmaxH8*PP2A)/(kmH8*(1.0+Raf_star/kmH3+Raf_star_star/kmH4+MAPKK_star_star/kmH9)+MAPKK_star))-(MAPKK_star*VmaxH7*GTPRasRaf_star)/(kmH7*(1.0+MAPKK/kmH6)+MAPKK_star))+(MAPKK_star_star*VmaxH9*PP2A)/(kmH9*(1.0+Raf_star/kmH3+Raf_star_star/kmH4+MAPKK_star/kmH8)+MAPKK_star_star)
        d_MAPKK_star_star = (MAPKK_star*VmaxH7*GTPRasRaf_star)/(kmH7*(1.0+MAPKK/kmH6)+MAPKK_star)-(MAPKK_star_star*VmaxH9*PP2A)/(kmH9*(1.0+Raf_star/kmH3+Raf_star_star/kmH4+MAPKK_star/kmH8)+MAPKK_star_star)
        d_MAPK = (-MAPK*VmaxH10*MAPKK_star_star)/(kmH10*(1.0+MAPK_tyr/kmH11)+MAPK)+(MAPK_tyr*VmaxH12*MKP1)/(kmH12*(1.0+MAPK_star/kmH13)+MAPK_tyr)
        d_MAPK_tyr = (((MAPK*VmaxH10*MAPKK_star_star)/((kmH10*MAPK_tyr)/kmH11+MAPK)-(MAPK_tyr*VmaxH12*MKP1)/(kmH12*(1.0+MAPK_star/kmH13)+MAPK_tyr))-(MAPK_tyr*VmaxH11*MAPKK_star_star)/(kmH11*(1.0+MAPK_tyr/kmH12)+MAPK_tyr))+(MAPK_star*VmaxH13*MKP1)/(kmH13*(1.0+MAPK_tyr/kmH12)+MAPK_star)
        d_MAPK_star = (MAPK_tyr*VmaxH11*MAPKK_star_star)/(kmH11*(1.0+MAPK/kmH10)+MAPK_tyr)-(MAPK_star*VmaxH13*MKP1)/(kmH13*(1.0+MAPK_tyr/kmH12)+MAPK_star)
        d_SHCstar = (((EGF_EGFR*VmaxA3*SHC)/(kmA3*(1.0+CaPLCg/kmF4)+SHC)-kfA6*SHCstar*SOS_GRB2)+kbA6*SHCstar_SOS_GRB2)-kfA4*SHCstar
        d_SHC = (kfA4*SHCstar-kbA4*SHC)-(EGF_EGFR*VmaxA3*SHC)/(kmA3*(1.0+CaPLCg/kmF4)+SHC)
        d_SHCstar_SOS_GRB2 = kfA6*SHCstar*SOS_GRB2-kbA6*SHCstar_SOS_GRB2
        d_SOS_GRB2 = ((-kfA6*SHCstar*SOS_GRB2+kbA6*SHCstar_SOS_GRB2)-kbA5*SOS_GRB2)+kfA5*SOS*GRB2
        d_SOS = (((kfA8*SOSstar-kbA8*SOS)-(MAPK_star*VmaxA9*SOS)/(kmA9*(1.0+Raf_star/kmH2)+SOS))+kbA5*SOS_GRB2)-kfA5*SOS*GRB2
        d_SOSstar = ((-kfA8*SOSstar+kbA8*SOS+(MAPK_star*VmaxA9*SOS)/(kmA9*(1.0+Raf_star/kmH2)+SOS))-kfA10*SOSstar*GRB2)+kbA10*SOSstar_GRB2
        d_SOSstar_GRB2 = kfA10*SOSstar*GRB2-kbA10*SOSstar_GRB2
        d_GRB2 = ((kbA5*SOS_GRB2-kfA5*SOS*GRB2)-kfA10*SOSstar*GRB2)+kbA10*SOSstar_GRB2
        d_EGF = -kfA1*EGF*EGFR+kbA1*EGF_EGFR
        d_EGFR = -kfA1*EGF*EGFR+kbA1*EGF_EGFR
        d_EGF_EGFR = ((kfA1*EGF*EGFR-kbA1*EGF_EGFR)-kfA2*EGF_EGFR)+kbA2*EGF_EGFR_INTERNAL
        d_EGF_EGFR_INTERNAL = kfA2*EGF_EGFR-kbA2*EGF_EGFR_INTERNAL
        d_PLCg = -Ca*PLCg*kfF1+CaPLCg*kbF1
        d_CaPLCg = (-(-Ca*PLCg*kfF1+CaPLCg*kbF1)+kfF3*CaPLCg_star)-(VmaxF4*EGF_EGFR*CaPLCg)/(kmF4*(1.0+SHC/kmA3)+CaPLCg)
        d_CaPLCg_star = (-(kfF3*CaPLCg_star-(VmaxF4*EGF_EGFR*CaPLCg)/(kmF4+CaPLCg))+Ca*PLCg_star*kfF5)-kbF5*CaPLCg_star
        d_PLCg_star = -Ca*PLCg_star*kfF5+kbF5*CaPLCg_star
        d_PKC_i = ((((((-PKC_i*kfK1+PKC*kbK1)-PKC_i*AA*kfK2)+PKC*kbK2)-PKC_i*DAG*kfK9)+DAGPKC*kbK9)-PKC_i*Ca*kfK7)+CaPKC*kbK7
        d_DAGPKC = ((PKC_i*DAG*kfK9-DAGPKC*kbK9)-DAGPKC*AA*kfK10)+AADAGPKC*kbK10
        d_AADAGPKC = ((DAGPKC*AA*kfK10-AADAGPKC*kbK10)-AADAGPKC*kfK6)+PKC*kbK6
        d_CaPKC = ((((((PKC_i*Ca*kfK7-CaPKC*kbK7)-CaPKC*kfK3)+PKC*kbK3)-CaPKC*AA*kfK4)+PKC*kbK4)-DAG*CaPKC*kfK8)+DAGCaPKC*kbK8
        d_DAGCaPKC = ((DAG*CaPKC*kfK8-DAGCaPKC*kbK8)-DAGCaPKC*kfK5)+PKC*kbK5
        d_PKC = ((((((((((PKC_i*kfK1-PKC*kbK1)+PKC_i*AA*kfK2)-PKC*kbK2)+CaPKC*kfK3)-PKC*kbK3)+CaPKC*AA*kfK4)-PKC*kbK4)+DAGCaPKC*kfK5)-PKC*kbK5)+AADAGPKC*kfK6)-PKC*kbK6
        d_AA = ((((((-PKC_i*AA*kfK2+PKC*kbK2)-CaPKC*AA*kfK4)+PKC*kbK4)-DAGPKC*AA*kfK10)+AADAGPKC*kbK10)-AA*kfE13)+(CaPLA2_star*VmaxE12*APC)/(kmE12+APC)+(CaPLA2*VmaxE4*APC)/(kmE4+APC)+(DAGCaPLA2*VmaxE8*APC)/(kmE8+APC)+(PIP2CaPLA2*VmaxE6*APC)/(kmE6+APC)+(PIP2PLA2*VmaxE2*APC)/(kmE2+APC)
        d_PLA2_cyt = ((((PLA2_star*kfE10-PIP2_star*PLA2_cyt*kfE1)+PIP2PLA2*kbE1)-Ca*PLA2_cyt*kfE3)+CaPLA2*kbE3)-(MAPK*VmaxE9*PLA2_cyt)/(kmE9+PLA2_cyt)
        d_PIP2PLA2 = PIP2_star*PLA2_cyt*kfE1-PIP2PLA2*kbE1
        d_PIP2_star = ((-PIP2_star*PLA2_cyt*kfE1+PIP2PLA2*kbE1)-PIP2_star*CaPLA2*kfE5)+PIP2CaPLA2*kbE5
        d_PLA2_star = ((-PLA2_star*kfE10+(MAPK*VmaxE9*PLA2_cyt)/(kmE9+PLA2_cyt))-Ca*PLA2_star*kfE11)+CaPLA2_star*kbE11
        d_CaPLA2_star = Ca*PLA2_star*kfE11-CaPLA2_star*kbE11
        d_CaPLA2 = ((((Ca*PLA2_cyt*kfE3-CaPLA2*kbE3)-DAG*CaPLA2*kfE7)+DAGCaPLA2*kbE7)-PIP2_star*CaPLA2*kfE5)+PIP2CaPLA2*kbE5
        d_PIP2CaPLA2 = PIP2_star*CaPLA2*kfE5-PIP2CaPLA2*kbE5
        d_DAGCaPLA2 = DAG*CaPLA2*kfE7-DAGCaPLA2*kbE7
        d_APC = AA*kfE13-((CaPLA2_star*VmaxE12*APC)/(kmE12+APC)+(CaPLA2*VmaxE4*APC)/(kmE4+APC)+(DAGCaPLA2*VmaxE8*APC)/(kmE8+APC)+(PIP2CaPLA2*VmaxE6*APC)/(kmE6+APC)+(PIP2PLA2*VmaxE2*APC)/(kmE2+APC))
        d_PIP2 = (((-CaPLCg*VmaxF2*PIP2)/(kmF2+PIP2)-(CaPLCg_star*VmaxF6*PIP2)/(kmF6+PIP2))-(CaPLC*VmaxG7 *PIP2)/(kmG7+PIP2))-(CaGqPLC*VmaxG6*PIP2)/(kmG6+PIP2)
        d_DAG = (((((((((CaPLCg*VmaxF2*PIP2)/(kmF2+PIP2)+(CaPLCg_star*VmaxF6*PIP2)/(kmF6+PIP2)+(CaPLC*VmaxG7 *PIP2)/(kmG7+PIP2))-kfG8*DAG)+(CaGqPLC*VmaxG6*PIP2)/(kmG6+PIP2))-PKC_i*DAG*kfK9)+DAGPKC*kbK9)-DAG*CaPKC*kfK8)+DAGCaPKC*kbK8)-DAG*CaPLA2*kfE7)+DAGCaPLA2*kbE7
        d_IP3 = (((CaPLCg*VmaxF2*PIP2)/(kmF2+PIP2)+(CaPLCg_star*VmaxF6*PIP2)/(kmF6+PIP2)+(CaGqPLC*VmaxG6*PIP2)/(kmG6+PIP2))-kfG9*IP3)+(CaPLC*VmaxG7 *PIP2)/(kmG7+PIP2)
        d_PC = kfG8*DAG
        d_Inositol  = kfG9*IP3


        return (d_PKC, d_GAP, d_Raf, d_GEF_star, d_CaM_GEF, d_Gbg_GEF, d_CaGqPLC, 
                d_CaPLC, d_NgCaM, d_GEF, d_Ng, d_Gbg, d_Ga_GDP, d_mGluR, d_PLC, 
                d_Ca2CaM, d_Ca3CaM, d_Ng_star, d_Gabg, d_R, d_Gabg_R, d_Glu_synapse, d_Gabg_GluR, d_Glu, d_Ga_GTP, d_GqPLC, d_GEF_inact, d_Ca4CAM, d_CaM, d_SHCstar_SOS_GRB2, d_MAPK_star, d_SOS, d_MAPKK_star_star, 
                d_MAPKK_star, d_Raf_star, d_GTPRasRaf_star, d_Raf_star_star, 
                d_GTP_RAS, d_GDP_RAS, d_GAPstar, d_MAPKK, d_MAPK, d_MAPK_tyr, 
                d_EGF_EGFR, d_CaPLCg, d_SHC, d_SHCstar, d_SOSstar_GRB2, 
                d_SOS_GRB2, d_GRB2, d_SOSstar, d_EGF_EGFR_INTERNAL, d_EGF, 
                d_EGFR, d_CaPLCg_star, d_PLCg, d_PLCg_star, d_PKC_i, d_DAGPKC, 
                d_AADAGPKC, d_DAGCaPKC, d_CaPKC, d_AA, d_APC, d_PIP2_star, 
                d_PLA2_cyt, d_PLA2_star, d_CaPLA2, d_PIP2CaPLA2, d_PIP2PLA2, 
                d_DAGCaPLA2, d_CaPLA2_star, d_PIP2, d_DAG, d_IP3, d_Inositol, d_PC)


    
    def get_initial_conditions(self):
        """ Function to get nominal initial conditions for the model. """
        ic_dict = {'PKC':0.0,
        'GAP':0.002,
        'Raf':0.2,
        'GEF_star':0.0,
        'CaM_GEF':0.0,
        'Gbg_GEF':0.0,
        'CaGqPLC':0.0,
        'CaPLC':0.0,
        'NgCaM':0.0,
        'GEF':0.0,
        'Ng':10.0,
        'Gbg':0.1,
        'Ga_GDP':1.0,
        'mGluR':0.3,
        'PLC':0.8,
        'Ca2CaM':0.0,
        'Ca3CaM':0.0,
        'Ng_star':0.0,
        'Gabg':0.0,
        'R':0.0,
        'Gabg_R':0.0,
        'Glu_synapse':0.0,
        'Gabg_GluR':0.0,
        'Glu':0.0,
        'Ga_GTP':0.0,
        'GqPLC':0.0,
        'GEF_inact':0.1,
        'Ca4CAM':0.0,
        'CaM':20.0,
        'SHCstar_SOS_GRB2':0.0,
        'MAPK_star':0.0,
        'SOS':0.1,
        'MAPKK_star_star':0.0,
        'MAPKK_star':0.0,
        'Raf_star':0.0,
        'GTPRasRaf_star':0.0,
        'Raf_star_star':0.0,
        'GTP_RAS':0.0,
        'GDP_RAS':0.2,
        'GAPstar':0.0,
        'MAPKK':0.18,
        'MAPK':0.36,
        'MAPK_tyr':0.0,
        'EGF_EGFR':0.0,
        'CaPLCg':0.0,
        'SHC':0.5,
        'SHCstar':0.0,
        'SOSstar_GRB2':0.0,
        'SOS_GRB2':0.0,
        'GRB2':1.0,
        'SOSstar':0.0,
        'EGF_EGFR_INTERNAL':0.0,
        'EGF':166.67,
        'EGFR':0.16667,
        'CaPLCg_star':0.0,
        'PLCg':0.82,
        'PLCg_star':0.0,
        'PKC_i':1.0,
        'DAGPKC':0.0,
        'AADAGPKC':0.0,
        'DAGCaPKC':0.0,
        'CaPKC':0.0,
        'AA':0.0,
        'APC':30.0,
        'PIP2_star':2.5,
        'PLA2_cyt':0.4,
        'PLA2_star':0.0,
        'CaPLA2':0.0,
        'PIP2CaPLA2':0.0,
        'PIP2PLA2':0.0,
        'DAGCaPLA2':0.0,
        'CaPLA2_star':0.0,
        'PIP2':10.0,
        'DAG':0.0,
        'IP3':0.0,
        'Inositol':0.0,
        'PC':0.0,}

        ic_tup = tuple([ic_dict[key] for key in ic_dict.keys()])
        
        return ic_dict, ic_tup

        
    
    def get_nominal_params(self):
        """ Function to get nominal parameters for the model. """

        param_dict = {'Ca':1.0,
                'GTP':0.5,
                'PKA':0.1,
                'kfD1':500.0,
                'kbD1':1000.0,
                'kfD2':2.8e-05,
                'kbD2':0.1,
                'kfD3':2.8e-05,
                'kbD3':10.0,
                'kfD4':1e-06,
                'kbD4':1.0,
                'kfD5':1e-08,
                'kbD5':0.0001,
                'kfD6':0.01,
                'kfD7':0.0133,
                'kfD8':1e-05,
                'kfD9':0.0001,
                'kfG1':5e-06,
                'kbG1':1.0,
                'kfG2':4.2e-06,
                'kbG2':1.0,
                'kfG3':5e-05,
                'kbG3':1.0,
                'kfG4':4.2e-06,
                'kbG4':1.0,
                'kfG5':0.0133,
                'kfB2':1.0,
                'kbB2':0.0,
                'kmB3':3.333333333333,
                'kfB5':0.0001,
                'kbB5':1.0,
                'kfB7':1.0,
                'kbB7':0.0,
                'kfB8':1e-05,
                'kbB8':1.0,
                'kmB1':7.5,
                'VmaxB1':9.0,
                'kmB6':3.33333333333,
                'VmaxB6':4.0,
                'kfM1':2e-10,
                'kbM1':72.0,
                'kfM2':6e-06,
                'kbM2':10.0,
                'kfM3':7.75e-07,
                'kbM3':10.0,
                'kfM4':5e-07,
                'kbM4':1.0,
                'kfM6':0.005,
                'kmM5':10.012,
                'VmaxM5':0.67,
                'kmM7':28.626667,
                'VmaxM7':0.58,
                'kmM8':28.595,
                'VmaxM8':0.35,
                'kmH1':66.666666667,
                'PP2A':0.224,
                'kfB12':0.0001,
                'kbB12':0.0,
                'kmB9':0.50505,
                'VmaxB9':0.02,
                'kmB10':0.50505,
                'VmaxB10':0.02,
                'kmB11':0.50505,
                'VmaxB11':0.02,
                'kmB13':1.0104,
                'VmaxB13':10.0,
                'kfB4':0.1,
                'kbB4':0.0,
                'VmaxB3':4.0,
                'kmA7':0.50505,
                'VmaxA7':0.02,
                'kfH5':4e-05,
                'kbH5':0.5,
                'kmH2':25.64166667,
                'kmH3':15.6565,
                'kmH4':15.6565,
                'VmaxH1':4.0,
                'VmaxH2':10.0,
                'VmaxH3':6.0,
                'VmaxH4':6.0,
                'kmA9':2.564166667,
                'kmH8':15.6565,
                'kmH9':15.6565,
                'kmH6':0.159091667,
                'kmH7':0.159091667,
                'VmaxH6':0.105,
                'VmaxH7':0.105,
                'VmaxH8':6.0,
                'VmaxH9':6.0,
                'MKP1':0.032,
                'kmH10':0.046296667,
                'kmH11':0.046296667,
                'kmH12':0.066666667,
                'kmH13':0.066666667,
                'VmaxH10':0.15,
                'VmaxH11':0.15,
                'VmaxH12':1.0,
                'VmaxH13':1.0,
                'kmA3':0.83333333333,
                'kfA4':0.0016667,
                'kbA4':0.0,
                'kfA5':4.1667e-08,
                'kbA5':0.0168,
                'kfA6':8.333e-07,
                'kbA6':0.1,
                'kfA8':0.001,
                'kbA8':0.0,
                'kfA10':4.1667e-08,
                'kbA10':0.0168,
                'VmaxA3':0.2,
                'VmaxA9':10.0,
                'kmF4':0.33333333333,
                'kfA1':7e-06,
                'kbA1':0.25,
                'kfA2':0.002,
                'kbA2':0.00033,
                'kfF1':0.0003,
                'kbF1':10.0,
                'kfF3':0.05,
                'kfF5':2e-05,
                'kbF5':10.0,
                'VmaxF4':0.2,
                'kfK1':1.0,
                'kbK1':50.0,
                'kfK2':2e-10,
                'kbK2':0.1,
                'kfK3':1.2705,
                'kbK3':3.5026,
                'kfK4':2e-09,
                'kbK4':0.1,
                'kfK5':1.0,
                'kbK5':0.1,
                'kfK6':2.0,
                'kbK6':0.2,
                'kfK7':1e-06,
                'kbK7':0.5,
                'kfK8':1.3333e-08,
                'kbK8':8.6348,
                'kfK9':1e-09,
                'kbK9':0.1,
                'kfK10':3e-08,
                'kbK10':2.0,
                'kfE1':2e-09,
                'kbE1':0.5,
                'kfE3':1.6667e-07,
                'kbE3':0.1,
                'kfE5':2e-08,
                'kbE5':0.1,
                'kfE7':5e-09,
                'kbE7':4.0,
                'kfE10':0.17,
                'kbE10':0.0,
                'kfE11':1e-05,
                'kbE11':0.1,
                'kmE9':25.64166667,
                'VmaxE9':20.0,
                'kfE13':0.4,
                'kbE13':0.0,
                'kmE2':20.0,
                'kmE4':20.0,
                'kmE6':20.0,
                'kmE8':20.0,
                'kmE12':20.0,
                'VmaxE2':11.04,
                'VmaxE4':5.4,
                'VmaxE6':36.0,
                'VmaxE8':60.0,
                'VmaxE12':120.0,
                'kmF2':97.0,
                'kmF6':19.79166667,
                'VmaxF2':14.0,
                'VmaxF6':57.0,
                'kfG8':0.15,
                'kfG9':2.5,
                'kmG6':5.0,
                'VmaxG6':48.0,
                'kmG7':19.84166667,
                'VmaxG7':10.0,}

        param_list = [param_dict[key] for key in param_dict]

        return param_dict, param_list

