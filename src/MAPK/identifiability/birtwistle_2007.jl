using StructuralIdentifiability

ode = @ODEmodel(
    x1'(t) = -((((((kon1 * x1(t) * x3(t)) - (EGF_off * x7(t))) + (kon86 * x1(t) * x107(t))) - (EGF_off * x110(t))) / VeVc)),
    x2'(t) = -((((((((kon87 * x2(t) * x109(t)) - (HRGoff_4 * x111(t))) + (kon2 * x2(t) * x5(t))) - (HRGoff_3 * x8(t))) + (kon3 * x2(t) * x6(t))) - (HRGoff_4 * x9(t))) / VeVc)),
    x3'(t) = -(((kon1 * x1(t) * x3(t)) - (EGF_off * x7(t))) / VeVc) -((((kf81 * x3(t) * x76(t)) / (Kmf81 + x3(t))) - ((Vmaxr81 * x107(t)) / (Kmr81 + x107(t))))),
    x4'(t) = -(((kon5 * x7(t) * x4(t)) - (koff5 * x11(t)))) -(((kon6 * x8(t) * x4(t)) - (koff6 * x12(t)))) -(((kon8 * x9(t) * x4(t)) - (koff8 * x14(t)))) -((((kf82 * x4(t) * x76(t)) / (Kmf82 + x4(t))) - ((Vmaxr82 * x108(t)) / (Kmr82 + x108(t))))),
    x5'(t) = -(((kon2 * x2(t) * x5(t)) - (HRGoff_3 * x8(t)))),
    x6'(t) = -(((kon3 * x2(t) * x6(t)) - (HRGoff_4 * x9(t)))) -((((kf83 * x6(t) * x76(t)) / (Kmf83 + x6(t))) - ((Vmaxr83 * x109(t)) / (Kmr83 + x109(t))))),
    x7'(t) = (((((kon1 * x1(t) * x3(t)) - (EGF_off * x7(t))) / VeVc) ) - (2.0 * (((kon4 * x7(t) * x7(t)) - (koff4 * x10(t)))) ) - ((((kon5 * x7(t) * x4(t)) - (koff5 * x11(t)))) ) - ((((kon61 * x8(t) * x7(t)) - (koff61 * x83(t)))) ) - ((((kon62 * x9(t) * x7(t)) - (koff62 * x84(t)))) ) - (((((kf84 * x7(t) * x76(t)) / (Kmf84 + x7(t))) - ((Vmaxr84 * x110(t)) / (Kmr84 + x110(t))))) )),
    x8'(t) = (((((kon2 * x2(t) * x5(t)) - (HRGoff_3 * x8(t)))) ) - ((((kon6 * x8(t) * x4(t)) - (koff6 * x12(t)))) ) - ((((kon7 * x8(t) * x9(t)) - (koff7 * x13(t)))) ) - ((((kon61 * x8(t) * x7(t)) - (koff61 * x83(t)))) )),
    x9'(t) = (((((kon3 * x2(t) * x6(t)) - (HRGoff_4 * x9(t)))) ) - ((((kon7 * x8(t) * x9(t)) - (koff7 * x13(t)))) ) - ((((kon8 * x9(t) * x4(t)) - (koff8 * x14(t)))) ) - (2.0 * (((kon9 * x9(t) * x9(t)) - (koff9 * x15(t)))) ) - ((((kon62 * x9(t) * x7(t)) - (koff62 * x84(t)))) ) - (((((kf85 * x9(t) * x76(t)) / (Kmf85 + x9(t))) - ((Vmaxr85 * x111(t)) / (Kmr85 + x111(t))))) )),
    x10'(t) = (((((kon4 * x7(t) * x7(t)) - (koff4 * x10(t)))) ) - (((((kf10 * x10(t)) - ((VmaxPY * x16(t)) / (KmPY + x16(t)))) - (kPTP10 * x106(t) * x16(t)))) )),
    x11'(t) = (((((kon5 * x7(t) * x4(t)) - (koff5 * x11(t)))) ) - (((((kf11 * x11(t)) - ((VmaxPY * x17(t)) / (KmPY + x17(t)))) - (kPTP11 * x106(t) * x17(t)))) )),
    x12'(t) = (((((kon6 * x8(t) * x4(t)) - (koff6 * x12(t)))) ) - (((((kf12 * x12(t)) - ((VmaxPY * x18(t)) / (KmPY + x18(t)))) - (kPTP12 * x106(t) * x18(t)))) )),
    x13'(t) = (((((kon7 * x8(t) * x9(t)) - (koff7 * x13(t)))) ) - (((((kf13 * x13(t)) - ((VmaxPY * x19(t)) / (KmPY + x19(t)))) - (kPTP13 * x106(t) * x19(t)))) )),
    x14'(t) = (((((kon8 * x9(t) * x4(t)) - (koff8 * x14(t)))) ) - (((((kf14 * x14(t)) - ((VmaxPY * x20(t)) / (KmPY + x20(t)))) - (kPTP14 * x106(t) * x20(t)))) )),
    x15'(t) = (((((kon9 * x9(t) * x9(t)) - (koff9 * x15(t)))) ) - (((((kf15 * x15(t)) - ((VmaxPY * x21(t)) / (KmPY + x21(t)))) - (kPTP15 * x106(t) * x21(t)))) )),
    x16'(t) = ((((((kf10 * x10(t)) - ((VmaxPY * x16(t)) / (KmPY + x16(t)))) - (kPTP10 * x106(t) * x16(t)))) ) - ((((4.0 * kon16 * x16(t) * x22(t)) - (koff16 * (x50(t) / (x50(t) + x58(t) + x57(t) + x79(t) + eps)) * x28(t)))) ) - ((((8.0 * kon17 * x16(t) * x23(t)) - (koff17 * (x51(t) / (x51(t) + x55(t) + x59(t) + eps)) * x29(t)))) ) - ((((2.0 * kon18 * x16(t) * x25(t)) - (koff18 * (x53(t) / (x53(t) + x70(t) + eps)) * x30(t)))) ) - ((((4.0 * kon73 * x16(t) * x96(t)) - (koff73 * x97(t)))) ) - (((kdeg * x16(t))) )),
    x17'(t) = ((((((kf11 * x11(t)) - ((VmaxPY * x17(t)) / (KmPY + x17(t)))) - (kPTP11 * x106(t) * x17(t)))) ) - ((((3.0 * kon19 * x17(t) * x22(t)) - (koff19 * (x50(t) / (x50(t) + x58(t) + x57(t) + x79(t) + eps)) * x31(t)))) ) - ((((6.0 * kon20 * x17(t) * x23(t)) - (koff20 * (x51(t) / (x51(t) + x55(t) + x59(t) + eps)) * x32(t)))) ) - ((((2.0 * kon21 * x17(t) * x25(t)) - (koff21 * (x53(t) / (x53(t) + x70(t) + eps)) * x33(t)))) ) - ((((3.0 * kon74 * x17(t) * x96(t)) - (koff74 * x98(t)))) )),
    x18'(t) = ((((((kf12 * x12(t)) - ((VmaxPY * x18(t)) / (KmPY + x18(t)))) - (kPTP12 * x106(t) * x18(t)))) ) - ((((3.0 * kon22 * x18(t) * x22(t)) - (koff22 * (x50(t) / (x50(t) + x58(t) + x57(t) + x79(t) + eps)) * x34(t)))) ) - ((((3.0 * kon23 * x18(t) * x23(t)) - (koff23 * (x51(t) / (x51(t) + x55(t) + x59(t) + eps)) * x35(t)))) ) - ((((3.0 * kon24 * x18(t) * x24(t)) - (koff24 * x36(t)))) ) - ((((2.0 * kon25 * x18(t) * x25(t)) - (koff25 * (x53(t) / (x53(t) + x70(t) + eps)) * x37(t)))) ) - ((((2.0 * kon75 * x18(t) * x96(t)) - (koff75 * x99(t)))) )),
    x19'(t) = ((((((kf13 * x13(t)) - ((VmaxPY * x19(t)) / (KmPY + x19(t)))) - (kPTP13 * x106(t) * x19(t)))) ) - ((((4.0 * kon26 * x19(t) * x22(t)) - (koff26 * (x50(t) / (x50(t) + x58(t) + x57(t) + x79(t) + eps)) * x38(t)))) ) - ((((3.0 * kon27 * x19(t) * x23(t)) - (koff27 * (x51(t) / (x51(t) + x55(t) + x59(t) + eps)) * x39(t)))) ) - ((((4.0 * kon28 * x19(t) * x24(t)) - (koff28 * x40(t)))) ) - ((((2.0 * kon29 * x19(t) * x25(t)) - (koff29 * (x53(t) / (x53(t) + x70(t) + eps)) * x41(t)))) ) - ((((2.0 * kon76 * x19(t) * x96(t)) - (koff76 * x100(t)))) )),
    x20'(t) = ((((((kf14 * x14(t)) - ((VmaxPY * x20(t)) / (KmPY + x20(t)))) - (kPTP14 * x106(t) * x20(t)))) ) - ((((3.0 * kon30 * x20(t) * x22(t)) - (koff30 * (x50(t) / (x50(t) + x58(t) + x57(t) + x79(t) + eps)) * x42(t)))) ) - ((((4.0 * kon31 * x20(t) * x23(t)) - (koff31 * (x51(t) / (x51(t) + x55(t) + x59(t) + eps)) * x43(t)))) ) - ((((1.0 * kon32 * x20(t) * x24(t)) - (koff32 * x44(t)))) ) - ((((2.0 * kon33 * x20(t) * x25(t)) - (koff33 * (x53(t) / (x53(t) + x70(t) + eps)) * x45(t)))) ) - ((((2.0 * kon77 * x20(t) * x96(t)) - (koff77 * x101(t)))) )),
    x21'(t) = ((((((kf15 * x15(t)) - ((VmaxPY * x21(t)) / (KmPY + x21(t)))) - (kPTP15 * x106(t) * x21(t)))) ) - ((((4.0 * kon34 * x21(t) * x22(t)) - (koff34 * (x50(t) / (x50(t) + x58(t) + x57(t) + x79(t) + eps)) * x46(t)))) ) - ((((4.0 * kon35 * x21(t) * x23(t)) - (koff35 * (x51(t) / (x51(t) + x55(t) + x59(t) + eps)) * x47(t)))) ) - ((((2.0 * kon36 * x21(t) * x24(t)) - (koff36 * x48(t)))) ) - ((((2.0 * kon37 * x21(t) * x25(t)) - (koff37 * (x53(t) / (x53(t) + x70(t) + eps)) * x49(t)))) ) - ((((2.0 * kon78 * x21(t) * x96(t)) - (koff78 * x102(t)))) )),
    x22'(t) = ( - ((((4.0 * kon16 * x16(t) * x22(t)) - (koff16 * (x50(t) / (x50(t) + x58(t) + x57(t) + x79(t) + eps)) * x28(t)))) ) - ((((3.0 * kon19 * x17(t) * x22(t)) - (koff19 * (x50(t) / (x50(t) + x58(t) + x57(t) + x79(t) + eps)) * x31(t)))) ) - ((((3.0 * kon22 * x18(t) * x22(t)) - (koff22 * (x50(t) / (x50(t) + x58(t) + x57(t) + x79(t) + eps)) * x34(t)))) ) - ((((4.0 * kon26 * x19(t) * x22(t)) - (koff26 * (x50(t) / (x50(t) + x58(t) + x57(t) + x79(t) + eps)) * x38(t)))) ) - ((((3.0 * kon30 * x20(t) * x22(t)) - (koff30 * (x50(t) / (x50(t) + x58(t) + x57(t) + x79(t) + eps)) * x42(t)))) ) - ((((4.0 * kon34 * x21(t) * x22(t)) - (koff34 * (x50(t) / (x50(t) + x58(t) + x57(t) + x79(t) + eps)) * x46(t)))) ) - ((((kon42 * x55(t) * x22(t)) - (koff42 * x59(t) * (x50(t) / (x50(t) + x58(t) + x57(t) + x79(t) + eps))))) ) - ((((kon57 * x63(t) * x22(t)) - (koff57 * x80(t)))) ) - ((((4.0 * kon65 * x85(t) * x22(t)) - (koff65 * (x50(t) / (x50(t) + x58(t) + x57(t) + x79(t) + eps)) * x87(t)))) ) - ((((4.0 * kon69 * x86(t) * x22(t)) - (koff69 * (x50(t) / (x50(t) + x58(t) + x57(t) + x79(t) + eps)) * x91(t)))) ) + (((kdeg * x28(t))) )),
    x23'(t) = ( - ((((8.0 * kon17 * x16(t) * x23(t)) - (koff17 * (x51(t) / (x51(t) + x55(t) + x59(t) + eps)) * x29(t)))) ) - ((((6.0 * kon20 * x17(t) * x23(t)) - (koff20 * (x51(t) / (x51(t) + x55(t) + x59(t) + eps)) * x32(t)))) ) - ((((3.0 * kon23 * x18(t) * x23(t)) - (koff23 * (x51(t) / (x51(t) + x55(t) + x59(t) + eps)) * x35(t)))) ) - ((((3.0 * kon27 * x19(t) * x23(t)) - (koff27 * (x51(t) / (x51(t) + x55(t) + x59(t) + eps)) * x39(t)))) ) - ((((4.0 * kon31 * x20(t) * x23(t)) - (koff31 * (x51(t) / (x51(t) + x55(t) + x59(t) + eps)) * x43(t)))) ) - ((((4.0 * kon35 * x21(t) * x23(t)) - (koff35 * (x51(t) / (x51(t) + x55(t) + x59(t) + eps)) * x47(t)))) ) - ((((3.0 * kon43 * x56(t) * x23(t)) - (koff43 * x60(t) * (x51(t) / (x51(t) + x55(t) + x59(t) + eps))))) ) - ((((5.0 * kon66 * x85(t) * x23(t)) - (koff66 * (x51(t) / (x51(t) + x55(t) + x59(t) + eps)) * x88(t)))) ) - ((((6.0 * kon70 * x86(t) * x23(t)) - (koff70 * (x51(t) / (x51(t) + x55(t) + x59(t) + eps)) * x92(t)))) ) + (((kdeg * x29(t))) )),
    x24'(t) = ( - ((((3.0 * kon24 * x18(t) * x24(t)) - (koff24 * x36(t)))) ) - ((((4.0 * kon28 * x19(t) * x24(t)) - (koff28 * x40(t)))) ) - ((((1.0 * kon32 * x20(t) * x24(t)) - (koff32 * x44(t)))) ) - ((((2.0 * kon36 * x21(t) * x24(t)) - (koff36 * x48(t)))) ) - ((((3.0 * kon44 * x56(t) * x24(t)) - (koff44 * x61(t)))) ) - ((((3.0 * kon67 * x85(t) * x24(t)) - (koff67 * x89(t)))) ) - ((((1.0 * kon71 * x86(t) * x24(t)) - (koff71 * x93(t)))) )),
    x25'(t) = ( - ((((2.0 * kon18 * x16(t) * x25(t)) - (koff18 * (x53(t) / (x53(t) + x70(t) + eps)) * x30(t)))) ) - ((((2.0 * kon21 * x17(t) * x25(t)) - (koff21 * (x53(t) / (x53(t) + x70(t) + eps)) * x33(t)))) ) - ((((2.0 * kon25 * x18(t) * x25(t)) - (koff25 * (x53(t) / (x53(t) + x70(t) + eps)) * x37(t)))) ) - ((((2.0 * kon29 * x19(t) * x25(t)) - (koff29 * (x53(t) / (x53(t) + x70(t) + eps)) * x41(t)))) ) - ((((2.0 * kon33 * x20(t) * x25(t)) - (koff33 * (x53(t) / (x53(t) + x70(t) + eps)) * x45(t)))) ) - ((((2.0 * kon37 * x21(t) * x25(t)) - (koff37 * (x53(t) / (x53(t) + x70(t) + eps)) * x49(t)))) ) - ((((2.0 * kon45 * x56(t) * x25(t)) - (koff45 * x62(t) * (x53(t) / (x53(t) + x70(t) + eps))))) ) - ((((2.0 * kon68 * x85(t) * x25(t)) - (koff68 * (x53(t) / (x53(t) + x70(t) + eps)) * x90(t)))) ) - ((((2.0 * kon72 * x86(t) * x25(t)) - (koff72 * (x53(t) / (x53(t) + x70(t) + eps)) * x94(t)))) ) + (((kdeg * x30(t))) )),
    x26'(t) = ( - ((((kon40 * x50(t) * x26(t)) - (koff40 * x57(t)))) ) - (((((kf54 * x26(t) * x76(t)) / (Kmf54 + x26(t))) - ((Vmaxr54 * x77(t)) / (Kmr54 + x77(t))))) ) - ((((kon58 * x80(t) * x26(t)) - (koff58 * x81(t)))) ) - ((((kon60 * x58(t) * x26(t)) - (koff60 * x79(t)))) )),
    x27'(t) = ( - ((((kon41 * x50(t) * x27(t)) - (koff41 * x58(t) * (x54(t) / (eps + x54(t) + x56(t) + x60(t) + x62(t) + x61(t) + x105(t)))))) ) - ((((kon46 * x65(t) * x27(t)) - (koff46 * x63(t) * (x54(t) / (eps + x54(t) + x56(t) + x60(t) + x62(t) + x61(t) + x105(t)))))) ) - (((((kf55 * x27(t) * x76(t)) / (Kmf55 + x27(t))) - ((Vmaxr55 * x78(t)) / (Kmr55 + x78(t))))) ) - ((((kon59 * x57(t) * x27(t)) - (koff59 * x79(t) * (x54(t) / (eps + x54(t) + x56(t) + x60(t) + x62(t) + x61(t) + x105(t)))))) )),
    x28'(t) = (((((4.0 * kon16 * x16(t) * x22(t)) - (koff16 * (x50(t) / (x50(t) + x58(t) + x57(t) + x79(t) + eps)) * x28(t)))) ) - (((kdeg * x28(t))) )),
    x29'(t) = (((((8.0 * kon17 * x16(t) * x23(t)) - (koff17 * (x51(t) / (x51(t) + x55(t) + x59(t) + eps)) * x29(t)))) ) - (((kdeg * x29(t))) )),
    x30'(t) = (((((2.0 * kon18 * x16(t) * x25(t)) - (koff18 * (x53(t) / (x53(t) + x70(t) + eps)) * x30(t)))) ) - (((kdeg * x30(t))) )),
    x31'(t) = ((((3.0 * kon19 * x17(t) * x22(t)) - (koff19 * (x50(t) / (x50(t) + x58(t) + x57(t) + x79(t) + eps)) * x31(t)))) ),
    x32'(t) = ((((6.0 * kon20 * x17(t) * x23(t)) - (koff20 * (x51(t) / (x51(t) + x55(t) + x59(t) + eps)) * x32(t)))) ),
    x33'(t) = ((((2.0 * kon21 * x17(t) * x25(t)) - (koff21 * (x53(t) / (x53(t) + x70(t) + eps)) * x33(t)))) ),
    x34'(t) = ((((3.0 * kon22 * x18(t) * x22(t)) - (koff22 * (x50(t) / (x50(t) + x58(t) + x57(t) + x79(t) + eps)) * x34(t)))) ),
    x35'(t) = ((((3.0 * kon23 * x18(t) * x23(t)) - (koff23 * (x51(t) / (x51(t) + x55(t) + x59(t) + eps)) * x35(t)))) ),
    x36'(t) = ((((3.0 * kon24 * x18(t) * x24(t)) - (koff24 * x36(t)))) ),
    x37'(t) = ((((2.0 * kon25 * x18(t) * x25(t)) - (koff25 * (x53(t) / (x53(t) + x70(t) + eps)) * x37(t)))) ),
    x38'(t) = ((((4.0 * kon26 * x19(t) * x22(t)) - (koff26 * (x50(t) / (x50(t) + x58(t) + x57(t) + x79(t) + eps)) * x38(t)))) ),
    x39'(t) = ((((3.0 * kon27 * x19(t) * x23(t)) - (koff27 * (x51(t) / (x51(t) + x55(t) + x59(t) + eps)) * x39(t)))) ),
    x40'(t) = ((((4.0 * kon28 * x19(t) * x24(t)) - (koff28 * x40(t)))) ),
    x41'(t) = ((((2.0 * kon29 * x19(t) * x25(t)) - (koff29 * (x53(t) / (x53(t) + x70(t) + eps)) * x41(t)))) ),
    x42'(t) = ((((3.0 * kon30 * x20(t) * x22(t)) - (koff30 * (x50(t) / (x50(t) + x58(t) + x57(t) + x79(t) + eps)) * x42(t)))) ),
    x43'(t) = ((((4.0 * kon31 * x20(t) * x23(t)) - (koff31 * (x51(t) / (x51(t) + x55(t) + x59(t) + eps)) * x43(t)))) ),
    x44'(t) = ((((1.0 * kon32 * x20(t) * x24(t)) - (koff32 * x44(t)))) ),
    x45'(t) = ((((2.0 * kon33 * x20(t) * x25(t)) - (koff33 * (x53(t) / (x53(t) + x70(t) + eps)) * x45(t)))) ),
    x46'(t) = ((((4.0 * kon34 * x21(t) * x22(t)) - (koff34 * (x50(t) / (x50(t) + x58(t) + x57(t) + x79(t) + eps)) * x46(t)))) ),
    x47'(t) = ((((4.0 * kon35 * x21(t) * x23(t)) - (koff35 * (x51(t) / (x51(t) + x55(t) + x59(t) + eps)) * x47(t)))) ),
    x48'(t) = ((((2.0 * kon36 * x21(t) * x24(t)) - (koff36 * x48(t)))) ),
    x49'(t) = ((((2.0 * kon37 * x21(t) * x25(t)) - (koff37 * (x53(t) / (x53(t) + x70(t) + eps)) * x49(t)))) ),
    x50'(t) = (((((4.0 * kon16 * x16(t) * x22(t)) - (koff16 * (x50(t) / (x50(t) + x58(t) + x57(t) + x79(t) + eps)) * x28(t)))) ) + ((((3.0 * kon19 * x17(t) * x22(t)) - (koff19 * (x50(t) / (x50(t) + x58(t) + x57(t) + x79(t) + eps)) * x31(t)))) ) + ((((3.0 * kon22 * x18(t) * x22(t)) - (koff22 * (x50(t) / (x50(t) + x58(t) + x57(t) + x79(t) + eps)) * x34(t)))) ) + ((((4.0 * kon26 * x19(t) * x22(t)) - (koff26 * (x50(t) / (x50(t) + x58(t) + x57(t) + x79(t) + eps)) * x38(t)))) ) + ((((3.0 * kon30 * x20(t) * x22(t)) - (koff30 * (x50(t) / (x50(t) + x58(t) + x57(t) + x79(t) + eps)) * x42(t)))) ) + ((((4.0 * kon34 * x21(t) * x22(t)) - (koff34 * (x50(t) / (x50(t) + x58(t) + x57(t) + x79(t) + eps)) * x46(t)))) ) - ((((kon40 * x50(t) * x26(t)) - (koff40 * x57(t)))) ) - ((((kon41 * x50(t) * x27(t)) - (koff41 * x58(t) * (x54(t) / (eps + x54(t) + x56(t) + x60(t) + x62(t) + x61(t) + x105(t)))))) ) + ((((kon42 * x55(t) * x22(t)) - (koff42 * x59(t) * (x50(t) / (x50(t) + x58(t) + x57(t) + x79(t) + eps))))) ) + ((((4.0 * kon65 * x85(t) * x22(t)) - (koff65 * (x50(t) / (x50(t) + x58(t) + x57(t) + x79(t) + eps)) * x87(t)))) ) + ((((4.0 * kon69 * x86(t) * x22(t)) - (koff69 * (x50(t) / (x50(t) + x58(t) + x57(t) + x79(t) + eps)) * x91(t)))) ) - (((kdeg * x28(t))) )),
    x51'(t) = (((((8.0 * kon17 * x16(t) * x23(t)) - (koff17 * (x51(t) / (x51(t) + x55(t) + x59(t) + eps)) * x29(t)))) ) + ((((6.0 * kon20 * x17(t) * x23(t)) - (koff20 * (x51(t) / (x51(t) + x55(t) + x59(t) + eps)) * x32(t)))) ) + ((((3.0 * kon23 * x18(t) * x23(t)) - (koff23 * (x51(t) / (x51(t) + x55(t) + x59(t) + eps)) * x35(t)))) ) + ((((3.0 * kon27 * x19(t) * x23(t)) - (koff27 * (x51(t) / (x51(t) + x55(t) + x59(t) + eps)) * x39(t)))) ) + ((((4.0 * kon31 * x20(t) * x23(t)) - (koff31 * (x51(t) / (x51(t) + x55(t) + x59(t) + eps)) * x43(t)))) ) + ((((4.0 * kon35 * x21(t) * x23(t)) - (koff35 * (x51(t) / (x51(t) + x55(t) + x59(t) + eps)) * x47(t)))) ) - (((((kf38 * x51(t) * (x16(t) + x17(t) + x18(t) + x20(t) + x19(t) + x21(t) + x85(t) + x86(t))) - ((VmaxPY * x55(t)) / (KmPY + x55(t)))) - (kPTP38 * x106(t) * x55(t)))) ) + ((((3.0 * kon43 * x56(t) * x23(t)) - (koff43 * x60(t) * (x51(t) / (x51(t) + x55(t) + x59(t) + eps))))) ) + ((((5.0 * kon66 * x85(t) * x23(t)) - (koff66 * (x51(t) / (x51(t) + x55(t) + x59(t) + eps)) * x88(t)))) ) + ((((6.0 * kon70 * x86(t) * x23(t)) - (koff70 * (x51(t) / (x51(t) + x55(t) + x59(t) + eps)) * x92(t)))) ) - (((kdeg * x29(t))) )),
    x52'(t) = (((((3.0 * kon24 * x18(t) * x24(t)) - (koff24 * x36(t)))) ) + ((((4.0 * kon28 * x19(t) * x24(t)) - (koff28 * x40(t)))) ) + ((((1.0 * kon32 * x20(t) * x24(t)) - (koff32 * x44(t)))) ) + ((((2.0 * kon36 * x21(t) * x24(t)) - (koff36 * x48(t)))) ) + ((((3.0 * kon44 * x56(t) * x24(t)) - (koff44 * x61(t)))) ) + ((((3.0 * kon67 * x85(t) * x24(t)) - (koff67 * x89(t)))) ) + ((((1.0 * kon71 * x86(t) * x24(t)) - (koff71 * x93(t)))) )),
    x53'(t) = (((((2.0 * kon18 * x16(t) * x25(t)) - (koff18 * (x53(t) / (x53(t) + x70(t) + eps)) * x30(t)))) ) + ((((2.0 * kon21 * x17(t) * x25(t)) - (koff21 * (x53(t) / (x53(t) + x70(t) + eps)) * x33(t)))) ) + ((((2.0 * kon25 * x18(t) * x25(t)) - (koff25 * (x53(t) / (x53(t) + x70(t) + eps)) * x37(t)))) ) + ((((2.0 * kon29 * x19(t) * x25(t)) - (koff29 * (x53(t) / (x53(t) + x70(t) + eps)) * x41(t)))) ) + ((((2.0 * kon33 * x20(t) * x25(t)) - (koff33 * (x53(t) / (x53(t) + x70(t) + eps)) * x45(t)))) ) + ((((2.0 * kon37 * x21(t) * x25(t)) - (koff37 * (x53(t) / (x53(t) + x70(t) + eps)) * x49(t)))) ) + ((((2.0 * kon45 * x56(t) * x25(t)) - (koff45 * x62(t) * (x53(t) / (x53(t) + x70(t) + eps))))) ) - (((((kf50 * x53(t) * (x16(t) + x17(t) + x18(t) + x20(t) + x19(t) + x21(t) + x85(t) + x86(t))) - ((VmaxPY * x70(t)) / (KmPY + x70(t)))) - (kPTP50 * x106(t) * x70(t)))) ) + ((((2.0 * kon68 * x85(t) * x25(t)) - (koff68 * (x53(t) / (x53(t) + x70(t) + eps)) * x90(t)))) ) + ((((2.0 * kon72 * x86(t) * x25(t)) - (koff72 * (x53(t) / (x53(t) + x70(t) + eps)) * x94(t)))) ) - (((kdeg * x30(t))) )),
    x54'(t) = ( - (((((kf39 * x54(t) * (x16(t) + x17(t) + x18(t) + x20(t) + x19(t) + x21(t) + x85(t) + x86(t))) - ((VmaxPY * x56(t)) / (KmPY + x56(t)))) - (kPTP39 * x106(t) * x56(t)))) ) + ((((kon41 * x50(t) * x27(t)) - (koff41 * x58(t) * (x54(t) / (eps + x54(t) + x56(t) + x60(t) + x62(t) + x61(t) + x105(t)))))) ) + ((((kon46 * x65(t) * x27(t)) - (koff46 * x63(t) * (x54(t) / (eps + x54(t) + x56(t) + x60(t) + x62(t) + x61(t) + x105(t)))))) ) + ((((kon59 * x57(t) * x27(t)) - (koff59 * x79(t) * (x54(t) / (eps + x54(t) + x56(t) + x60(t) + x62(t) + x61(t) + x105(t)))))) )),
    x55'(t) = ((((((kf38 * x51(t) * (x16(t) + x17(t) + x18(t) + x20(t) + x19(t) + x21(t) + x85(t) + x86(t))) - ((VmaxPY * x55(t)) / (KmPY + x55(t)))) - (kPTP38 * x106(t) * x55(t)))) ) - ((((kon42 * x55(t) * x22(t)) - (koff42 * x59(t) * (x50(t) / (x50(t) + x58(t) + x57(t) + x79(t) + eps))))) )),
    x56'(t) = ((((((kf39 * x54(t) * (x16(t) + x17(t) + x18(t) + x20(t) + x19(t) + x21(t) + x85(t) + x86(t))) - ((VmaxPY * x56(t)) / (KmPY + x56(t)))) - (kPTP39 * x106(t) * x56(t)))) ) - ((((3.0 * kon43 * x56(t) * x23(t)) - (koff43 * x60(t) * (x51(t) / (x51(t) + x55(t) + x59(t) + eps))))) ) - ((((3.0 * kon44 * x56(t) * x24(t)) - (koff44 * x61(t)))) ) - ((((2.0 * kon45 * x56(t) * x25(t)) - (koff45 * x62(t) * (x53(t) / (x53(t) + x70(t) + eps))))) ) - ((((2.0 * kon88 * x56(t) * x96(t)) - (koff88 * x105(t)))) )),
    x57'(t) = (((((kon40 * x50(t) * x26(t)) - (koff40 * x57(t)))) ) - ((((kon59 * x57(t) * x27(t)) - (koff59 * x79(t) * (x54(t) / (eps + x54(t) + x56(t) + x60(t) + x62(t) + x61(t) + x105(t)))))) )),
    x58'(t) = (((((kon41 * x50(t) * x27(t)) - (koff41 * x58(t) * (x54(t) / (eps + x54(t) + x56(t) + x60(t) + x62(t) + x61(t) + x105(t)))))) ) - ((((kon60 * x58(t) * x26(t)) - (koff60 * x79(t)))) )),
    x59'(t) = ((((kon42 * x55(t) * x22(t)) - (koff42 * x59(t) * (x50(t) / (x50(t) + x58(t) + x57(t) + x79(t) + eps))))) ),
    x60'(t) = ((((3.0 * kon43 * x56(t) * x23(t)) - (koff43 * x60(t) * (x51(t) / (x51(t) + x55(t) + x59(t) + eps))))) ),
    x61'(t) = ((((3.0 * kon44 * x56(t) * x24(t)) - (koff44 * x61(t)))) ),
    x62'(t) = ((((2.0 * kon45 * x56(t) * x25(t)) - (koff45 * x62(t) * (x53(t) / (x53(t) + x70(t) + eps))))) ),
    x63'(t) = (((((kon46 * x65(t) * x27(t)) - (koff46 * x63(t) * (x54(t) / (eps + x54(t) + x56(t) + x60(t) + x62(t) + x61(t) + x105(t)))))) ) - ((((kon57 * x63(t) * x22(t)) - (koff57 * x80(t)))) )),
    x64'(t) = - (((((kf48 * (1.0 - (x95(t) * (x16(t) / (x16(t) + x17(t) + x18(t) + x20(t) + x19(t) + x21(t) + x85(t) + x86(t) + eps)))) * x52(t) * x64(t)) / (Kmf48 + x64(t))) - ((3.0 * PTEN * x65(t)) / (Kmr48 + x65(t))))) ),
    x65'(t) = ( - ((((kon46 * x65(t) * x27(t)) - (koff46 * x63(t) * (x54(t) / (eps + x54(t) + x56(t) + x60(t) + x62(t) + x61(t) + x105(t)))))) ) + (((((kf48 * (1.0 - (x95(t) * (x16(t) / (x16(t) + x17(t) + x18(t) + x20(t) + x19(t) + x21(t) + x85(t) + x86(t) + eps)))) * x52(t) * x64(t)) / (Kmf48 + x64(t))) - ((3.0 * PTEN * x65(t)) / (Kmr48 + x65(t))))) )),
    x66'(t) = - (((((kf47 * x65(t) * x66(t)) / (Kmf47 + x66(t))) - ((Vmaxr47 * x67(t)) / (Kmr47 + x67(t))))) ),
    x67'(t) = (((((kf47 * x65(t) * x66(t)) / (Kmf47 + x66(t))) - ((Vmaxr47 * x67(t)) / (Kmr47 + x67(t))))) ),
    x68'(t) = - (((((((kf49 * x82(t) * x68(t)) / (Kmf49 + x68(t))) - ((kr49 * x53(t) * x69(t)) / (Kmr49 + x69(t)))) - ((kr49b * x70(t) * x69(t)) / (Kmr49b + x69(t)))) - (kcon49 * x69(t)))) ),
    x69'(t) = (((((((kf49 * x82(t) * x68(t)) / (Kmf49 + x68(t))) - ((kr49 * x53(t) * x69(t)) / (Kmr49 + x69(t)))) - ((kr49b * x70(t) * x69(t)) / (Kmr49b + x69(t)))) - (kcon49 * x69(t)))) ),
    x70'(t) = (((((kf50 * x53(t) * (x16(t) + x17(t) + x18(t) + x20(t) + x19(t) + x21(t) + x85(t) + x86(t))) - ((VmaxPY * x70(t)) / (KmPY + x70(t)))) - (kPTP50 * x106(t) * x70(t)))) ),
    x71'(t) = - (((((kf51 * x69(t) * x71(t)) / (Kmf51 + x71(t))) - ((Vmaxr51 * x72(t)) / (Kmrb51 + x72(t))))) ),
    x72'(t) = (((((kf51 * x69(t) * x71(t)) / (Kmf51 + x71(t))) - ((Vmaxr51 * x72(t)) / (Kmrb51 + x72(t))))) ),
    x73'(t) = - (((((kf52 * x72(t) * x73(t)) / (Kmf52 + x73(t))) - ((Vmaxr52 * x74(t)) / (Kmr52 + x74(t))))) ),
    x74'(t) = ((((((kf52 * x72(t) * x73(t)) / (Kmf52 + x73(t))) - ((Vmaxr52 * x74(t)) / (Kmr52 + x74(t))))) ) - ((((kon89 * x75(t) * x74(t)) - (koff89 * x113(t)))) ) + (((kcat90 * x113(t))) ) - ((((kon91 * x112(t) * x74(t)) - (koff91 * x114(t)))) ) + (((kcat92 * x114(t))) )),
    x75'(t) = ( - ((((kon89 * x75(t) * x74(t)) - (koff89 * x113(t)))) ) + (((kcat96 * x117(t))) )),
    x76'(t) = ((((kcat92 * x114(t))) ) - ((((kon93 * x76(t) * x115(t)) - (koff93 * x116(t)))) )),
    x77'(t) = (((((kf54 * x26(t) * x76(t)) / (Kmf54 + x26(t))) - ((Vmaxr54 * x77(t)) / (Kmr54 + x77(t))))) ),
    x78'(t) = (((((kf55 * x27(t) * x76(t)) / (Kmf55 + x27(t))) - ((Vmaxr55 * x78(t)) / (Kmr55 + x78(t))))) ),
    x79'(t) = (((((kon59 * x57(t) * x27(t)) - (koff59 * x79(t) * (x54(t) / (eps + x54(t) + x56(t) + x60(t) + x62(t) + x61(t) + x105(t)))))) ) + ((((kon60 * x58(t) * x26(t)) - (koff60 * x79(t)))) )),
    x80'(t) = (((((kon57 * x63(t) * x22(t)) - (koff57 * x80(t)))) ) - ((((kon58 * x80(t) * x26(t)) - (koff58 * x81(t)))) )),
    x81'(t) = ((((kon58 * x80(t) * x26(t)) - (koff58 * x81(t)))) ),
    x82'(t) = (((((kon40 * x50(t) * x26(t)) - (koff40 * x57(t)))) ) + ((((kon58 * x80(t) * x26(t)) - (koff58 * x81(t)))) ) + ((((kon60 * x58(t) * x26(t)) - (koff60 * x79(t)))) )),
    x83'(t) = (((((kon61 * x8(t) * x7(t)) - (koff61 * x83(t)))) ) - (((((kf63 * x83(t)) - ((VmaxPY * x85(t)) / (KmPY + x85(t)))) - (kPTP63 * x106(t) * x85(t)))) )),
    x84'(t) = (((((kon62 * x9(t) * x7(t)) - (koff62 * x84(t)))) ) - (((((kf64 * x84(t)) - ((VmaxPY * x86(t)) / (KmPY + x86(t)))) - (kPTP64 * x106(t) * x86(t)))) )),
    x85'(t) = ((((((kf63 * x83(t)) - ((VmaxPY * x85(t)) / (KmPY + x85(t)))) - (kPTP63 * x106(t) * x85(t)))) ) - ((((4.0 * kon65 * x85(t) * x22(t)) - (koff65 * (x50(t) / (x50(t) + x58(t) + x57(t) + x79(t) + eps)) * x87(t)))) ) - ((((5.0 * kon66 * x85(t) * x23(t)) - (koff66 * (x51(t) / (x51(t) + x55(t) + x59(t) + eps)) * x88(t)))) ) - ((((3.0 * kon67 * x85(t) * x24(t)) - (koff67 * x89(t)))) ) - ((((2.0 * kon68 * x85(t) * x25(t)) - (koff68 * (x53(t) / (x53(t) + x70(t) + eps)) * x90(t)))) ) - ((((3.0 * kon79 * x85(t) * x96(t)) - (koff79 * x103(t)))) )),
    x86'(t) = ((((((kf64 * x84(t)) - ((VmaxPY * x86(t)) / (KmPY + x86(t)))) - (kPTP64 * x106(t) * x86(t)))) ) - ((((4.0 * kon69 * x86(t) * x22(t)) - (koff69 * (x50(t) / (x50(t) + x58(t) + x57(t) + x79(t) + eps)) * x91(t)))) ) - ((((6.0 * kon70 * x86(t) * x23(t)) - (koff70 * (x51(t) / (x51(t) + x55(t) + x59(t) + eps)) * x92(t)))) ) - ((((1.0 * kon71 * x86(t) * x24(t)) - (koff71 * x93(t)))) ) - ((((2.0 * kon72 * x86(t) * x25(t)) - (koff72 * (x53(t) / (x53(t) + x70(t) + eps)) * x94(t)))) ) - ((((3.0 * kon80 * x86(t) * x96(t)) - (koff80 * x104(t)))) )),
    x87'(t) = ((((4.0 * kon65 * x85(t) * x22(t)) - (koff65 * (x50(t) / (x50(t) + x58(t) + x57(t) + x79(t) + eps)) * x87(t)))) ),
    x88'(t) = ((((5.0 * kon66 * x85(t) * x23(t)) - (koff66 * (x51(t) / (x51(t) + x55(t) + x59(t) + eps)) * x88(t)))) ),
    x89'(t) = ((((3.0 * kon67 * x85(t) * x24(t)) - (koff67 * x89(t)))) ),
    x90'(t) = ((((2.0 * kon68 * x85(t) * x25(t)) - (koff68 * (x53(t) / (x53(t) + x70(t) + eps)) * x90(t)))) ),
    x91'(t) = ((((4.0 * kon69 * x86(t) * x22(t)) - (koff69 * (x50(t) / (x50(t) + x58(t) + x57(t) + x79(t) + eps)) * x91(t)))) ),
    x92'(t) = ((((6.0 * kon70 * x86(t) * x23(t)) - (koff70 * (x51(t) / (x51(t) + x55(t) + x59(t) + eps)) * x92(t)))) ),
    x93'(t) = ((((1.0 * kon71 * x86(t) * x24(t)) - (koff71 * x93(t)))) ),
    x94'(t) = ((((2.0 * kon72 * x86(t) * x25(t)) - (koff72 * (x53(t) / (x53(t) + x70(t) + eps)) * x94(t)))) ),
    x95'(t) = (((a98 * ( - x95(t) + b98))) ),
    x96'(t) = ( - ((((4.0 * kon73 * x16(t) * x96(t)) - (koff73 * x97(t)))) ) - ((((3.0 * kon74 * x17(t) * x96(t)) - (koff74 * x98(t)))) ) - ((((2.0 * kon75 * x18(t) * x96(t)) - (koff75 * x99(t)))) ) - ((((2.0 * kon76 * x19(t) * x96(t)) - (koff76 * x100(t)))) ) - ((((2.0 * kon77 * x20(t) * x96(t)) - (koff77 * x101(t)))) ) - ((((2.0 * kon78 * x21(t) * x96(t)) - (koff78 * x102(t)))) ) - ((((3.0 * kon79 * x85(t) * x96(t)) - (koff79 * x103(t)))) ) - ((((3.0 * kon80 * x86(t) * x96(t)) - (koff80 * x104(t)))) ) - ((((2.0 * kon88 * x56(t) * x96(t)) - (koff88 * x105(t)))) ) + (((kdeg * x97(t))) )),
    x97'(t) = (((((4.0 * kon73 * x16(t) * x96(t)) - (koff73 * x97(t)))) ) - (((kdeg * x97(t))) )),
    x98'(t) = ((((3.0 * kon74 * x17(t) * x96(t)) - (koff74 * x98(t)))) ),
    x99'(t) = ((((2.0 * kon75 * x18(t) * x96(t)) - (koff75 * x99(t)))) ),
    x100'(t) = ((((2.0 * kon76 * x19(t) * x96(t)) - (koff76 * x100(t)))) ),
    x101'(t) = ((((2.0 * kon77 * x20(t) * x96(t)) - (koff77 * x101(t)))) ),
    x102'(t) = ((((2.0 * kon78 * x21(t) * x96(t)) - (koff78 * x102(t)))) ),
    x103'(t) = ((((3.0 * kon79 * x85(t) * x96(t)) - (koff79 * x103(t)))) ),
    x104'(t) = ((((3.0 * kon80 * x86(t) * x96(t)) - (koff80 * x104(t)))) ),
    x105'(t) = ((((2.0 * kon88 * x56(t) * x96(t)) - (koff88 * x105(t)))) ),
    x106'(t) = (((((4.0 * kon73 * x16(t) * x96(t)) - (koff73 * x97(t)))) ) + ((((3.0 * kon74 * x17(t) * x96(t)) - (koff74 * x98(t)))) ) + ((((2.0 * kon75 * x18(t) * x96(t)) - (koff75 * x99(t)))) ) + ((((2.0 * kon76 * x19(t) * x96(t)) - (koff76 * x100(t)))) ) + ((((2.0 * kon77 * x20(t) * x96(t)) - (koff77 * x101(t)))) ) + ((((2.0 * kon78 * x21(t) * x96(t)) - (koff78 * x102(t)))) ) + ((((3.0 * kon79 * x85(t) * x96(t)) - (koff79 * x103(t)))) ) + ((((3.0 * kon80 * x86(t) * x96(t)) - (koff80 * x104(t)))) ) + ((((2.0 * kon88 * x56(t) * x96(t)) - (koff88 * x105(t)))) ) - (((kdeg * x97(t))) )),
    x107'(t) = ((((((kf81 * x3(t) * x76(t)) / (Kmf81 + x3(t))) - ((Vmaxr81 * x107(t)) / (Kmr81 + x107(t))))) ) - ((((kon86 * x1(t) * x107(t)) - (EGF_off * x110(t)))) )),
    x108'(t) = (((((kf82 * x4(t) * x76(t)) / (Kmf82 + x4(t))) - ((Vmaxr82 * x108(t)) / (Kmr82 + x108(t))))) ),
    x109'(t) = ((((((kf83 * x6(t) * x76(t)) / (Kmf83 + x6(t))) - ((Vmaxr83 * x109(t)) / (Kmr83 + x109(t))))) ) - ((((kon87 * x2(t) * x109(t)) - (HRGoff_4 * x111(t)))) )),
    x110'(t) = ((((((kf84 * x7(t) * x76(t)) / (Kmf84 + x7(t))) - ((Vmaxr84 * x110(t)) / (Kmr84 + x110(t))))) ) + ((((kon86 * x1(t) * x107(t)) - (EGF_off * x110(t)))) )),
    x111'(t) = ((((((kf85 * x9(t) * x76(t)) / (Kmf85 + x9(t))) - ((Vmaxr85 * x111(t)) / (Kmr85 + x111(t))))) ) + ((((kon87 * x2(t) * x109(t)) - (HRGoff_4 * x111(t)))) )),
    x112'(t) = ((((kcat90 * x113(t))) ) - ((((kon91 * x112(t) * x74(t)) - (koff91 * x114(t)))) ) + (((kcat94 * x116(t))) ) - ((((kon95 * x112(t) * x115(t)) - (koff95 * x117(t)))) )),
    x113'(t) = (((((kon89 * x75(t) * x74(t)) - (koff89 * x113(t)))) ) - (((kcat90 * x113(t))) )),
    x114'(t) = (((((kon91 * x112(t) * x74(t)) - (koff91 * x114(t)))) ) - (((kcat92 * x114(t))) )),
    x115'(t) = ( - ((((kon93 * x76(t) * x115(t)) - (koff93 * x116(t)))) ) + (((kcat94 * x116(t))) ) - ((((kon95 * x112(t) * x115(t)) - (koff95 * x117(t)))) ) + (((kcat96 * x117(t))) )),
    x116'(t) = (((((kon93 * x76(t) * x115(t)) - (koff93 * x116(t)))) ) - (((kcat94 * x116(t))) )),
    x117'(t) = (((((kon95 * x112(t) * x115(t)) - (koff95 * x117(t)))) ) - (((kcat96 * x117(t))) )),
    y1(t) = x76(t) + x116(t)
)

local_id = assess_local_identifiability(ode, 0.95)

# save to file for this model
# create file to save results
file = open("./local_ID_B_2007.txt", "w")
println(file, "Locally Identifiable parameters:")
for (key, value) in local_id
    if value == 1
        print(file, "$key, ")
    end
end
println(file, "")
println(file, "Nonidentifiable parameters:")
for (key, value) in local_id
    if value == 0
        print(file, "$key, ")
    end
end
close(file)

global_id = assess_identifiability(ode, 0.95)

# create file to save results
file = open("./global_ID_B_2007.txt", "w")
println(file, "Globally Identifiable parameters:")
for (key, value) in local_id
    if value == ":globally"
        print(file, "$key, ")
    end
end
println(file, "")
println(file, "Nonidentifiable parameters:")
for (key, value) in local_id
    if value == ":nonidentifiable"
        print(file, "$key, ")
    end
end
close(file)