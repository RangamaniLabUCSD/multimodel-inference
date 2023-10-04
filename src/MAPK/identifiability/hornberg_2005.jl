using StructuralIdentifiability

ode = @ODEmodel(
    x1'(t) = cnstnts95,
    x2'(t) = (k13*1.0-kd13 *x2(t))-((k1*x1(t)*x2(t)-kd1*x3(t))+(k6*x2(t)-kd6*x6(t))),
    x3'(t) = (k1*x1(t)*x2(t)-kd1*x3(t))-2.0*(k2*x3(t)*x3(t)-kd2*x4(t)),
    x4'(t) = (k2*x3(t)*x3(t)-kd2*x4(t))-(k3*x4(t)*1.0-kd3*x5(t)),
    x5'(t) = (k3*x4(t)*1.0-kd3*x5(t))-((k6*x5(t)-kd6*x8(t))+(k8*x5(t)*x14(t)-kd8*x15(t))),
    x6'(t) = (k6*x2(t)-kd6*x6(t))-((k10b*x6(t)*x16(t)-kd10*x10(t))+(k60*x6(t)-kd60 *x86(t))),
    x7'(t) = (k4*x23(t)*x12(t)-kd4*x7(t))+(k5 *x18(t)*x9(t)-kd5*x7(t)),
    x8'(t) = ((k6*x5(t)-kd6*x8(t))+(k3*x11(t)-kd3*x8(t)))-((k8*x8(t)*x14(t)-kd8*x17(t))+(k60*x8(t)-kd60 *x87(t))),
    x9'(t) = -((k5 *x18(t)*x9(t)-kd5*x7(t))+(k15*x9(t)-kd15 *x12(t))+(k5 *x9(t)*x19(t)-kd5*x88(t))+(k5 *x9(t)*x20(t)-kd5*x89(t))+(k5 *x9(t)*x21(t)-kd5*x90(t))+(k5 *x9(t)*x65(t)-kd5*x91(t))+(k5 *x9(t)*x66(t)-kd5*x92(t))+(k5 *x9(t)*x67(t)-kd5*x93(t))+(k5 *x68(t)*x9(t)-kd5*x94(t))),
    x10'(t) = (k10b*x6(t)*x16(t)-kd10*x10(t))-2.0*(k2*x10(t)*x10(t)-kd2*x11(t)),
    x11'(t) = (k2*x10(t)*x10(t)-kd2*x11(t))-(k3*x11(t)-kd3*x8(t)),
    x12'(t) = (k15*x9(t)-kd15 *x12(t))-((k4*x23(t)*x12(t)-kd4*x7(t))+(k4*x25(t)*x12(t)-kd4*x88(t))+(k4*x27(t)*x12(t)-kd4*x89(t))+(k4*x29(t)*x12(t)-kd4*x90(t))+(k4*x34(t)*x12(t)-kd4*x91(t))+(k4*x35(t)*x12(t)-kd4*x92(t))+(k4*x36(t)*x12(t)-kd4*x93(t))+(k4*x37(t)*x12(t)-kd4*x94(t))),
    x13'(t) = (k61*x16(t)-kd61 *x13(t)),
    x14'(t) = -((k8*x5(t)*x14(t)-kd8*x15(t))+(k8*x8(t)*x14(t)-kd8*x17(t))),
    x15'(t) = (k8*x5(t)*x14(t)-kd8*x15(t))-((k16*x22(t)*x15(t)-kd63*x23(t))+(k22*x31(t)*x15(t)-kd22*x32(t))+(k32*x38(t)*x15(t)-kd32*x35(t))+(k34*x15(t)*x30(t)-kd34*x25(t))+(k37*x15(t)*x40(t)-kd37*x33(t))+(k37*x15(t)*x39(t)-kd37*x34(t))+(k6*x15(t)-kd6*x17(t))),
    x16'(t) = -((k10b*x6(t)*x16(t)-kd10*x10(t))+(k61*x16(t)-kd61 *x13(t))),
    x17'(t) = ((k8*x8(t)*x14(t)-kd8*x17(t))+(k6*x15(t)-kd6*x17(t)))-((k16*x17(t)*x22(t)-kd63*x18(t))+(k22*x31(t)*x17(t)-kd22*x63(t))+(k32*x17(t)*x38(t)-kd32*x66(t))+(k34*x17(t)*x30(t)-kd34*x19(t))+(k37*x17(t)*x40(t)-kd37*x64(t))+(k37*x17(t)*x39(t)-kd37*x65(t))+(k60*x17(t)-kd60 *x87(t))),
    x18'(t) = ((k6*x23(t)-kd6*x18(t))+(k16*x17(t)*x22(t)-kd63*x18(t)))-((k5 *x18(t)*x9(t)-kd5*x7(t))+(k17*x24(t)*x18(t)-kd17*x19(t))+(k60*x18(t)-kd60 *x87(t))),
    x19'(t) = ((k34*x17(t)*x30(t)-kd34*x19(t))+(k17*x24(t)*x18(t)-kd17*x19(t))+(k6*x25(t)-kd6*x19(t)))-((k18*x26(t)*x19(t)-kd18*x20(t))+(k19*x69(t)*x19(t)-kd19*x20(t))+(k20*x71(t)*x19(t)-kd20*x21(t))+(k21*x19(t)*x26(t)-kd21*x21(t))+(k5 *x9(t)*x19(t)-kd5*x88(t))+(k126*x83(t)*x19(t)-kd126*x96(t))+(k60*x19(t)-kd60 *x87(t))),
    x20'(t) = ((k18*x26(t)*x19(t)-kd18*x20(t))+(k19*x69(t)*x19(t)-kd19*x20(t))+(k6*x27(t)-kd6*x20(t)))-((k5 *x9(t)*x20(t)-kd5*x89(t))+(k60*x20(t)-kd60 *x87(t))),
    x21'(t) = ((k20*x71(t)*x19(t)-kd20*x21(t))+(k21*x19(t)*x26(t)-kd21*x21(t))+(k6*x29(t)-kd6*x21(t)))-((k5 *x9(t)*x21(t)-kd5*x90(t))+(k60*x21(t)-kd60 *x87(t))),
    x22'(t) = -((k16*x22(t)*x15(t)-kd63*x23(t))+(k16*x22(t)*x33(t)-kd24*x34(t))+(k35*x24(t)*x22(t)-kd35*x30(t))+(k16*x22(t)*x40(t)-kd24*x39(t))+(k16*x17(t)*x22(t)-kd63*x18(t))+(k16*x22(t)*x64(t)-kd24*x65(t))),
    x23'(t) = (k16*x22(t)*x15(t)-kd63*x23(t))-((k4*x23(t)*x12(t)-kd4*x7(t))+(k6*x23(t)-kd6*x18(t))+(k17*x24(t)*x23(t)-kd17*x25(t))),
    x24'(t) = -((k17*x24(t)*x23(t)-kd17*x25(t))+(k25*x24(t)*x34(t)-kd25*x35(t))+(k35*x24(t)*x22(t)-kd35*x30(t))+(k40*x24(t)*x39(t)-kd40*x38(t))+(k17*x24(t)*x18(t)-kd17*x19(t))+(k25*x24(t)*x65(t)-kd25*x66(t))+(k126*x59(t)*x24(t)-kd126*x101(t))+(k126*x83(t)*x24(t)-kd126*x102(t))),
    x25'(t) = ((k17*x24(t)*x23(t)-kd17*x25(t))+(k34*x15(t)*x30(t)-kd34*x25(t)))-((k18*x26(t)*x25(t)-kd18*x27(t))+(k19*x28(t)*x25(t)-kd19*x27(t))+(k20*x25(t)*x43(t)-kd20*x29(t))+(k21*x25(t)*x26(t)-kd21*x29(t))+(k6*x25(t)-kd6*x19(t))+(k4*x25(t)*x12(t)-kd4*x88(t))+(k126*x59(t)*x25(t)-kd126*x95(t))),
    x26'(t) = -((k18*x26(t)*x25(t)-kd18*x27(t))+(k21*x25(t)*x26(t)-kd21*x29(t))+(k18*x26(t)*x35(t)-kd18*x36(t))+(k21*x35(t)*x26(t)-kd21*x37(t))+(k18*x26(t)*x19(t)-kd18*x20(t))+(k21*x19(t)*x26(t)-kd21*x21(t))+(k18*x26(t)*x66(t)-kd18*x67(t))+(k21*x66(t)*x26(t)-kd21*x68(t))),
    x27'(t) = ((k18*x26(t)*x25(t)-kd18*x27(t))+(k19*x28(t)*x25(t)-kd19*x27(t)))-((k6*x27(t)-kd6*x20(t))+(k4*x27(t)*x12(t)-kd4*x89(t))),
    x28'(t) = -((k19*x28(t)*x25(t)-kd19*x27(t))+(k19*x35(t)*x28(t)-kd19*x36(t))+(k28*x28(t)*x41(t)-kd28*x42(t))),
    x29'(t) = ((k20*x25(t)*x43(t)-kd20*x29(t))+(k21*x25(t)*x26(t)-kd21*x29(t)))-((k6*x29(t)-kd6*x21(t))+(k4*x29(t)*x12(t)-kd4*x90(t))),
    x30'(t) = (k35*x24(t)*x22(t)-kd35*x30(t))-((k33*x40(t)*x30(t)-kd33*x38(t))+(k34*x15(t)*x30(t)-kd34*x25(t))+(k41*x30(t)*x33(t)-kd41*x35(t))+(k34*x17(t)*x30(t)-kd34*x19(t))+(k41*x30(t)*x64(t)-kd41*x66(t))),
    x31'(t) = (k36*x40(t)-kd36 *x31(t))-((k22*x31(t)*x15(t)-kd22*x32(t))+(k22*x31(t)*x17(t)-kd22*x63(t))),
    x32'(t) = (k22*x31(t)*x15(t)-kd22*x32(t))-((k23*x32(t)-kd23*x33(t))+(k6*x32(t)-kd6*x63(t))),
    x33'(t) = ((k23*x32(t)-kd23*x33(t))+(k37*x15(t)*x40(t)-kd37*x33(t)))-((k16*x22(t)*x33(t)-kd24*x34(t))+(k41*x30(t)*x33(t)-kd41*x35(t))+(k6*x33(t)-kd6*x64(t))),
    x34'(t) = ((k37*x15(t)*x39(t)-kd37*x34(t))+(k16*x22(t)*x33(t)-kd24*x34(t)))-((k25*x24(t)*x34(t)-kd25*x35(t))+(k6*x34(t)-kd6*x65(t))+(k4*x34(t)*x12(t)-kd4*x91(t))),
    x35'(t) = ((k25*x24(t)*x34(t)-kd25*x35(t))+(k32*x38(t)*x15(t)-kd32*x35(t))+(k41*x30(t)*x33(t)-kd41*x35(t)))-((k6*x35(t)-kd6*x66(t))+(k4*x35(t)*x12(t)-kd4*x92(t))+(k126*x59(t)*x35(t)-kd126*x97(t))+(k18*x26(t)*x35(t)-kd18*x36(t))+(k19*x35(t)*x28(t)-kd19*x36(t))+(k20*x35(t)*x43(t)-kd20*x37(t))+(k21*x35(t)*x26(t)-kd21*x37(t))),
    x36'(t) = ((k19*x35(t)*x28(t)-kd19*x36(t))+(k18*x26(t)*x35(t)-kd18*x36(t)))-((k6*x36(t)-kd6*x67(t))+(k4*x36(t)*x12(t)-kd4*x93(t))),
    x37'(t) = ((k20*x35(t)*x43(t)-kd20*x37(t))+(k21*x35(t)*x26(t)-kd21*x37(t)))-((k6*x37(t)-kd6*x68(t))+(k4*x37(t)*x12(t)-kd4*x94(t))),
    x38'(t) = ((k33*x40(t)*x30(t)-kd33*x38(t))+(k40*x24(t)*x39(t)-kd40*x38(t)))-((k32*x38(t)*x15(t)-kd32*x35(t))+(k32*x17(t)*x38(t)-kd32*x66(t))),
    x39'(t) = (k16*x22(t)*x40(t)-kd24*x39(t))-((k37*x15(t)*x39(t)-kd37*x34(t))+(k40*x24(t)*x39(t)-kd40*x38(t))+(k37*x17(t)*x39(t)-kd37*x65(t))),
    x40'(t) = -((k33*x40(t)*x30(t)-kd33*x38(t))+(k36*x40(t)-kd36 *x31(t))+(k37*x15(t)*x40(t)-kd37*x33(t))+(k16*x22(t)*x40(t)-kd24*x39(t))+(k37*x17(t)*x40(t)-kd37*x64(t))),
    x41'(t) = -((k28*x28(t)*x41(t)-kd28*x42(t))+(k43 *x41(t)*x44(t)-kd43*x46(t))+(k28*x69(t)*x41(t)-kd28*x70(t))+(k43 *x41(t)*x44(t)-kd43*x73(t))),
    x42'(t) = (k28*x28(t)*x41(t)-kd28*x42(t))+(k29*x43(t)*x45(t)-kd29*x42(t)),
    x43'(t) = -((k29*x43(t)*x45(t)-kd29*x42(t))+(k20*x25(t)*x43(t)-kd20*x29(t))+(k20*x35(t)*x43(t)-kd20*x37(t))),
    x44'(t) = -((k42*x44(t)*x45(t)-kd42*x46(t))+(k43 *x41(t)*x44(t)-kd43*x46(t))+(k42*x44(t)*x72(t)-kd42*x73(t))+(k43 *x41(t)*x44(t)-kd43*x73(t))),
    x45'(t) = -((k29*x43(t)*x45(t)-kd29*x42(t))+(k42*x44(t)*x45(t)-kd42*x46(t))+(k44*x47(t)*x45(t)-kd52*x48(t))+(k45 *x49(t)*x45(t)-kd45*x48(t))+(k44*x49(t)*x45(t)-kd52*x50(t))+(k47 *x51(t)*x45(t)-kd47*x50(t))),
    x46'(t) = (k42*x44(t)*x45(t)-kd42*x46(t))+(k43 *x41(t)*x44(t)-kd43*x46(t)),
    x47'(t) = -((k44*x47(t)*x45(t)-kd52*x48(t))+(k49 *x47(t)*x53(t)-kd49*x54(t))+(k44*x47(t)*x72(t)-kd52*x74(t))+(k49 *x47(t)*x53(t)-kd49*x79(t))),
    x48'(t) = (k44*x47(t)*x45(t)-kd52*x48(t))+(k45 *x49(t)*x45(t)-kd45*x48(t)),
    x49'(t) = -((k45 *x49(t)*x45(t)-kd45*x48(t))+(k44*x49(t)*x45(t)-kd52*x50(t))+(k49 *x49(t)*x53(t)-kd49*x52(t))+(k50*x53(t)*x49(t)-kd50*x54(t))),
    x50'(t) = (k44*x49(t)*x45(t)-kd52*x50(t))+(k47 *x51(t)*x45(t)-kd47*x50(t)),
    x51'(t) = -((k47 *x51(t)*x45(t)-kd47*x50(t))+(k53 *x51(t)*x57(t)-kd53*x56(t))+(k55 *x59(t)*x51(t)-kd55*x58(t))+(k48*x51(t)*x53(t)-kd48*x52(t))+(k52*x55(t)*x51(t)-kd44*x56(t))+(k52*x51(t)*x57(t)-kd44*x58(t))),
    x52'(t) = (k48*x51(t)*x53(t)-kd48*x52(t))+(k49 *x49(t)*x53(t)-kd49*x52(t)),
    x53'(t) = -((k48*x51(t)*x53(t)-kd48*x52(t))+(k49 *x49(t)*x53(t)-kd49*x52(t))+(k50*x53(t)*x49(t)-kd50*x54(t))+(k49 *x47(t)*x53(t)-kd49*x54(t))+(k48*x77(t)*x53(t)-kd48*x78(t))+(k49 *x75(t)*x53(t)-kd49*x78(t))+(k50*x53(t)*x75(t)-kd50*x79(t))+(k49 *x47(t)*x53(t)-kd49*x79(t))),
    x54'(t) = (k50*x53(t)*x49(t)-kd50*x54(t))+(k49 *x47(t)*x53(t)-kd49*x54(t)),
    x55'(t) = -((k52*x55(t)*x51(t)-kd44*x56(t))+(k57 *x55(t)*x60(t)-kd57*x62(t))+(k52*x55(t)*x77(t)-kd44*x80(t))+(k57 *x55(t)*x60(t)-kd57*x85(t))),
    x56'(t) = (k52*x55(t)*x51(t)-kd44*x56(t))+(k53 *x51(t)*x57(t)-kd53*x56(t)),
    x57'(t) = -((k53 *x51(t)*x57(t)-kd53*x56(t))+(k52*x51(t)*x57(t)-kd44*x58(t))+(k57 *x57(t)*x60(t)-kd57*x61(t))+(k58*x60(t)*x57(t)-kd58*x62(t))),
    x58'(t) = (k52*x51(t)*x57(t)-kd44*x58(t))+(k55 *x59(t)*x51(t)-kd55*x58(t)),
    x59'(t) = -((k55 *x59(t)*x51(t)-kd55*x58(t))+(k56*x59(t)*x60(t)-kd56*x61(t))+(k126*x59(t)*x25(t)-kd126*x95(t))+(k126*x59(t)*x35(t)-kd126*x97(t))+(k126*x59(t)*x24(t)-kd126*x101(t))+(k127 *x59(t)*x99(t)-kd127*x95(t))+(k127 *x59(t)*x99(t)-kd127*x97(t))+(k127 *x59(t)*x103(t)-kd127*x101(t))),
    x60'(t) = -((k56*x59(t)*x60(t)-kd56*x61(t))+(k57 *x57(t)*x60(t)-kd57*x61(t))+(k58*x60(t)*x57(t)-kd58*x62(t))+(k57 *x55(t)*x60(t)-kd57*x62(t))+(k56*x83(t)*x60(t)-kd56*x84(t))+(k57 *x81(t)*x60(t)-kd57*x84(t))+(k58*x60(t)*x81(t)-kd58*x85(t))+(k57 *x55(t)*x60(t)-kd57*x85(t))),
    x61'(t) = (k56*x59(t)*x60(t)-kd56*x61(t))+(k57 *x57(t)*x60(t)-kd57*x61(t)),
    x62'(t) = (k58*x60(t)*x57(t)-kd58*x62(t))+(k57 *x55(t)*x60(t)-kd57*x62(t)),
    x63'(t) = ((k22*x31(t)*x17(t)-kd22*x63(t))+(k6*x32(t)-kd6*x63(t)))-((k23*x63(t)-kd23*x64(t))+(k60*x63(t)-kd60 *x87(t))),
    x64'(t) = ((k23*x63(t)-kd23*x64(t))+(k37*x17(t)*x40(t)-kd37*x64(t))+(k6*x33(t)-kd6*x64(t)))-((k16*x22(t)*x64(t)-kd24*x65(t))+(k41*x30(t)*x64(t)-kd41*x66(t))+(k60*x64(t)-kd60 *x87(t))),
    x65'(t) = ((k16*x22(t)*x64(t)-kd24*x65(t))+(k37*x17(t)*x39(t)-kd37*x65(t))+(k6*x34(t)-kd6*x65(t)))-((k25*x24(t)*x65(t)-kd25*x66(t))+(k5 *x9(t)*x65(t)-kd5*x91(t))+(k60*x65(t)-kd60 *x87(t))),
    x66'(t) = ((k25*x24(t)*x65(t)-kd25*x66(t))+(k32*x17(t)*x38(t)-kd32*x66(t))+(k41*x30(t)*x64(t)-kd41*x66(t))+(k6*x35(t)-kd6*x66(t)))-((k18*x26(t)*x66(t)-kd18*x67(t))+(k19*x66(t)*x69(t)-kd19*x67(t))+(k20*x71(t)*x66(t)-kd20*x68(t))+(k21*x66(t)*x26(t)-kd21*x68(t))+(k5 *x9(t)*x66(t)-kd5*x92(t))+(k126*x83(t)*x66(t)-kd126*x98(t))+(k60*x66(t)-kd60 *x87(t))),
    x67'(t) = ((k18*x26(t)*x66(t)-kd18*x67(t))+(k19*x66(t)*x69(t)-kd19*x67(t))+(k6*x36(t)-kd6*x67(t)))-((k5 *x9(t)*x67(t)-kd5*x93(t))+(k60*x67(t)-kd60 *x87(t))),
    x68'(t) = ((k20*x71(t)*x66(t)-kd20*x68(t))+(k21*x66(t)*x26(t)-kd21*x68(t))+(k6*x37(t)-kd6*x68(t)))-((k5 *x68(t)*x9(t)-kd5*x94(t))+(k60*x68(t)-kd60 *x87(t))),
    x69'(t) = -((k19*x69(t)*x19(t)-kd19*x20(t))+(k19*x66(t)*x69(t)-kd19*x67(t))+(k28*x69(t)*x41(t)-kd28*x70(t))),
    x70'(t) = (k28*x69(t)*x41(t)-kd28*x70(t))+(k29*x71(t)*x72(t)-kd29*x70(t)),
    x71'(t) = -((k20*x71(t)*x19(t)-kd20*x21(t))+(k29*x71(t)*x72(t)-kd29*x70(t))+(k20*x71(t)*x66(t)-kd20*x68(t))),
    x72'(t) = -((k29*x71(t)*x72(t)-kd29*x70(t))+(k42*x44(t)*x72(t)-kd42*x73(t))+(k44*x47(t)*x72(t)-kd52*x74(t))+(k45 *x75(t)*x72(t)-kd45*x74(t))+(k44*x72(t)*x75(t)-kd52*x76(t))+(k47 *x72(t)*x77(t)-kd47*x76(t))),
    x73'(t) = (k42*x44(t)*x72(t)-kd42*x73(t))+(k43 *x41(t)*x44(t)-kd43*x73(t)),
    x74'(t) = (k44*x47(t)*x72(t)-kd52*x74(t))+(k45 *x75(t)*x72(t)-kd45*x74(t)),
    x75'(t) = -((k45 *x75(t)*x72(t)-kd45*x74(t))+(k44*x72(t)*x75(t)-kd52*x76(t))+(k49 *x75(t)*x53(t)-kd49*x78(t))+(k50*x53(t)*x75(t)-kd50*x79(t))),
    x76'(t) = (k44*x72(t)*x75(t)-kd52*x76(t))+(k47 *x72(t)*x77(t)-kd47*x76(t)),
    x77'(t) = -((k47 *x72(t)*x77(t)-kd47*x76(t))+(k48*x77(t)*x53(t)-kd48*x78(t))+(k52*x55(t)*x77(t)-kd44*x80(t))+(k53 *x81(t)*x77(t)-kd53*x80(t))+(k52*x77(t)*x81(t)-kd44*x82(t))+(k55 *x83(t)*x77(t)-kd55*x82(t))),
    x78'(t) = (k48*x77(t)*x53(t)-kd48*x78(t))+(k49 *x75(t)*x53(t)-kd49*x78(t)),
    x79'(t) = (k50*x53(t)*x75(t)-kd50*x79(t))+(k49 *x47(t)*x53(t)-kd49*x79(t)),
    x80'(t) = (k52*x55(t)*x77(t)-kd44*x80(t))+(k53 *x81(t)*x77(t)-kd53*x80(t)),
    x81'(t) = -((k53 *x81(t)*x77(t)-kd53*x80(t))+(k52*x77(t)*x81(t)-kd44*x82(t))+(k57 *x81(t)*x60(t)-kd57*x84(t))+(k58*x60(t)*x81(t)-kd58*x85(t))),
    x82'(t) = (k52*x77(t)*x81(t)-kd44*x82(t))+(k55 *x83(t)*x77(t)-kd55*x82(t)),
    x83'(t) = -((k55 *x83(t)*x77(t)-kd55*x82(t))+(k56*x83(t)*x60(t)-kd56*x84(t))+(k126*x83(t)*x19(t)-kd126*x96(t))+(k126*x83(t)*x66(t)-kd126*x98(t))+(k126*x83(t)*x24(t)-kd126*x102(t))+(k127 *x83(t)*x100(t)-kd127*x96(t))+(k127 *x83(t)*x100(t)-kd127*x98(t))+(k127 *x83(t)*x103(t)-kd127*x102(t))),
    x84'(t) = (k56*x83(t)*x60(t)-kd56*x84(t))+(k57 *x81(t)*x60(t)-kd57*x84(t)),
    x85'(t) = (k58*x60(t)*x81(t)-kd58*x85(t))+(k57 *x55(t)*x60(t)-kd57*x85(t)),
    x86'(t) = (k60*x6(t)-kd60 *x86(t)),
    x87'(t) = (k60*x8(t)-kd60 *x87(t))+(k60*x17(t)-kd60 *x87(t))+(k60*x18(t)-kd60 *x87(t))+(k60*x19(t)-kd60 *x87(t))+(k60*x20(t)-kd60 *x87(t))+(k60*x21(t)-kd60 *x87(t))+(k60*x63(t)-kd60 *x87(t))+(k60*x64(t)-kd60 *x87(t))+(k60*x65(t)-kd60 *x87(t))+(k60*x66(t)-kd60 *x87(t))+(k60*x67(t)-kd60 *x87(t))+(k60*x68(t)-kd60 *x87(t)),
    x88'(t) = (k4*x25(t)*x12(t)-kd4*x88(t))+(k5 *x9(t)*x19(t)-kd5*x88(t)),
    x89'(t) = (k4*x27(t)*x12(t)-kd4*x89(t))+(k5 *x9(t)*x20(t)-kd5*x89(t)),
    x90'(t) = (k4*x29(t)*x12(t)-kd4*x90(t))+(k5 *x9(t)*x21(t)-kd5*x90(t)),
    x91'(t) = (k4*x34(t)*x12(t)-kd4*x91(t))+(k5 *x9(t)*x65(t)-kd5*x91(t)),
    x92'(t) = (k4*x35(t)*x12(t)-kd4*x92(t))+(k5 *x9(t)*x66(t)-kd5*x92(t)),
    x93'(t) = (k4*x36(t)*x12(t)-kd4*x93(t))+(k5 *x9(t)*x67(t)-kd5*x93(t)),
    x94'(t) = (k4*x37(t)*x12(t)-kd4*x94(t))+(k5 *x68(t)*x9(t)-kd5*x94(t)),
    x95'(t) = (k126*x59(t)*x25(t)-kd126*x95(t))+(k127 *x59(t)*x99(t)-kd127*x95(t)),
    x96'(t) = (k126*x83(t)*x19(t)-kd126*x96(t))+(k127 *x83(t)*x100(t)-kd127*x96(t)),
    x97'(t) = (k126*x59(t)*x35(t)-kd126*x97(t))+(k127 *x59(t)*x99(t)-kd127*x97(t)),
    x98'(t) = (k126*x83(t)*x66(t)-kd126*x98(t))+(k127 *x83(t)*x100(t)-kd127*x98(t)),
    x99'(t) = -((k127 *x59(t)*x99(t)-kd127*x95(t))+(k127 *x59(t)*x99(t)-kd127*x97(t))),
    x100'(t) = -((k127 *x83(t)*x100(t)-kd127*x96(t))+(k127 *x83(t)*x100(t)-kd127*x98(t))),
    x101'(t) = (k126*x59(t)*x24(t)-kd126*x101(t))+(k127 *x59(t)*x103(t)-kd127*x101(t)),
    x102'(t) = (k126*x83(t)*x24(t)-kd126*x102(t))+(k127 *x83(t)*x103(t)-kd127*x102(t)),
    x103'(t) = -((k127 *x59(t)*x103(t)-kd127*x101(t))+(k127 *x83(t)*x103(t)-kd127*x102(t))),
    y1(t) = x59(t) + x83(t)
)

local_id = assess_local_identifiability(ode, 0.99)

# save to file for this model
# create file to save results
file = open("./local_ID_H_2005.txt", "w")
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
file = open("./global_ID_H_2005.txt", "w")
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
