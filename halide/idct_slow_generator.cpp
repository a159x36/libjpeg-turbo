

#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <iostream>
#include <string>
#include <vector>
#include <Halide.h>
using std::vector;
using std::string;

using namespace Halide;
using namespace ConciseCasts;
using namespace std;

//#define USE32BIT

#ifdef USE32BIT
#define CONST_BITS  13
#define PASS1_BITS 2
#define PRECISION i32
#else
#define CONST_BITS  0
//#define PASS1_BITS 4
#define MULT_BITS  15
#define PRECISION i16
#endif

int pass1_bits=4;

//#define DEQUANTIZE(coef,quantval)  PRECISION((i16(coef)) * i16(quantval))
#define DEQUANTIZE(coef,quantval)  PRECISION((PRECISION(coef)<<pass1_bits) * PRECISION(quantval))
Target::Arch arch;

Expr operator*(Expr e,double f) {
    if(f<0) {e=-e;f=-f;}
    int wh=(int)f;
    Expr e1;
    f=(f-wh);
    if(arch == Target::X86 && f<0.5) {
        e1=i16(Expr(f*(1<<16)));
        return i16((i32(i16(e))*i32(e1))/(1<<16))+e*wh;
    }
    e1=i16(Expr(f*(1<<15)));
    if(arch == Target::ARM ) {
        return i16_sat((i32(i16(e))*i32(e1)+(1<<14))/(1<<15))+e*wh;
    } else {
        return i16((i32(i16(e))*i32(e1)+(1<<14))/(1<<15))+e*wh;
    }
    return -1;
}

#define DESCALE(x,n)  (((x) + (1 << ((n)-1))) >> (n))

class IdctGenerator : public Halide::Generator<IdctGenerator> {
    public:
        GeneratorParam<int32_t> dctsize{"dctsize", 8};
        Input <Buffer<int16_t>> in{"in", 3};
        Input <Buffer<int16_t>> quant{"quant", 2};
        Output <Buffer<uint8_t>> out{"out", 3};
        
        

        Func ws,dequant,wsT,op,q16;
        Var x,y,z, zo, zi;
        
#define PASS1_BITS 0

        Func idct8_dim1_orig1(Func f) {
            Func g;
            
            FuncRef tf[8]={f(x,0,z),f(x,1,z),f(x,2,z),f(x,3,z),f(x,4,z),f(x,5,z),f(x,6,z),f(x,7,z)};
            Func t[25];
            
            vector<Var> v={x,z};
            FuncRef tmp[4]={t[0](v),t[1](v),t[2](v),t[3](v)};
            FuncRef tmp1[4]={t[4](v),t[5](v),t[6](v),t[7](v)};
            FuncRef tmp3[4]={t[8](v),t[9](v),t[10](v),t[11](v)};
            FuncRef tmp4[4]={t[12](v),t[13](v),t[14](v),t[15](v)};
            
            FuncRef z1[5]={t[16](v),t[17](v),t[18](v),t[19](v),t[20](v)};//,t[21](v)};
            FuncRef z2[4]={t[21](v),t[22](v),t[23](v),t[24](v)};
            
            Expr zz = (tf[2]+tf[6]) * 0.541196100;
            tmp[2] = zz - tf[6] * 1.847759065;
            tmp[3] = zz + tf[2] * 0.765366865;

            tmp[0] = (tf[0] + tf[4]);
            tmp[1] = (tf[0] - tf[4]);

            tmp1[0] = tmp[0] + tmp[3];
            tmp1[1] = tmp[1] + tmp[2];
            tmp1[2] = tmp[1] - tmp[2];
            tmp1[3] = tmp[0] - tmp[3];
            

            z1[1]=tf[7]+tf[1];
            z1[2]=tf[5]+tf[3];
            z1[3]=tf[7]+tf[3];
            z1[4]=tf[5]+tf[1];
            z1[0] = (z1[3] + z1[4]) * 1.175875602;

            tmp3[0] =  tf[7] * 0.298631336; /* sqrt(2) * (-c1+c3+c5-c7) */
            tmp3[1] = tf[5] * 2.053119869; /* sqrt(2) * ( c1+c3-c5+c7) */
            tmp3[2] = 4*tf[3]- tf[3] * (4.0-3.072711026); /* sqrt(2) * ( c1+c3+c5-c7) */
            tmp3[3] =  tf[1] * 1.501321110; /* sqrt(2) * ( c1+c3-c5-c7) */

            z2[0]=z1[1]*0.899976223;
            z2[1]=z1[2]*2.562915447;
            z2[2]=z1[0]-z1[3]*1.961570560;
            z2[3]=z1[0]-z1[4]*0.390180644;

            tmp4[0] = tmp3[0] - z2[0] + z2[2];
            tmp4[1] = tmp3[1] - z2[1] + z2[3];
            tmp4[2] = tmp3[2] - z2[1] + z2[2];
            tmp4[3] = tmp3[3] - z2[0] + z2[3];
            
            for(int i=0;i<25;i++)
              t[i].compute_at(out,z).vectorize(x);

            g(x,y,z)=PRECISION(select( y==0,tmp1[0] + tmp4[3], y==7,tmp1[0] - tmp4[3],
                                         y==1,tmp1[1] + tmp4[2], y==6,tmp1[1] - tmp4[2], 
                                         y==2,tmp1[2] + tmp4[1], y==5,tmp1[2] - tmp4[1],
                                         y==3,tmp1[3] + tmp4[0],      tmp1[3] - tmp4[0])); 
            return g;
        }
        void generate() {
            arch=get_target().arch;
            cout<<"size="<<int(dctsize)<<"\n";
            pass1_bits=4;
            int pass2_bits=0;
            q16(x,y)=quant(x,y)<<pass1_bits;
            dequant(x,y,z)=in(x,y,z)*q16(x,y);
            ws=idct8_dim1_orig1(dequant);
            wsT(y,x,z)=ws(x,y,z);            
            op=idct8_dim1_orig1(wsT);
            out(x,y,z)= u8_sat((op(x,y,z)+(1<<(pass1_bits+2))>>(pass1_bits+3))+128);

        }
        void schedule() { 
            out.dim(0).set_bounds(0,dctsize);
            out.dim(1).set_bounds(0,dctsize);
            out.dim(0).set_stride(1);
            out.dim(2).set_stride(dctsize);
            out.dim(2).set_min(0);
            in.dim(0).set_bounds(0,8);
            in.dim(1).set_bounds(0,8);
            in.dim(0).set_stride(1);
            in.dim(1).set_stride(8);
            in.dim(2).set_min(0);
            in.dim(2).set_stride(64);
            quant.dim(0).set_bounds(0,8);
            quant.dim(1).set_bounds(0,8);
            quant.dim(0).set_stride(1);
            quant.dim(1).set_stride(8);

            q16.compute_root();
            dequant.compute_at(out,z).vectorize(x);
            ws.compute_at(out,z).reorder_storage(y, x).vectorize(x).unroll(y);
        //    wsT.compute_at(out,z).vectorize(x).unroll(y);
            op.compute_at(out,z).reorder_storage(x, y).vectorize(x).unroll(y);
            out.compute_root().vectorize(x).unroll(y);
        }
};

HALIDE_REGISTER_GENERATOR(IdctGenerator,idct_slow);

