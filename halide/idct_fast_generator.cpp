#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <ostream>
#include <string>
#include <vector>
#include <Halide.h>
using std::vector;
using std::string;

using namespace Halide;
using namespace ConciseCasts;

#define IFAST_SCALE_BITS  4

Target::Arch arch;

bool hasavx=false;

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
    private:
        Input <Buffer<int16_t>> in{"in", 3};
        Input <Buffer<int16_t>> quant{"quant", 2};
        Output <Buffer<uint8_t>> out{"out", 3};

        Func ws,dequant,wsT,op,opT;
        Var x,y,z,zo,zi,yi,fused;

        vector<FuncRef> get_func_refs(Func fx, int N) {
            vector<FuncRef> refs;
            for (int i = 0; i < N; i++) {
                    refs.push_back(fx(x,i,z));
            }
            return refs;
        }


        Func idct8_dim1(Func f,const char *name) {

			vector<FuncRef> I=get_func_refs(f,8);

            Func g(name);
            vector<FuncRef> X=get_func_refs(g,14);
            g(x,y,z)=undef(Int(16));

            X[0] = I[0] + I[4];
            X[1] = I[0] - I[4];
            X[3] = I[2] + I[6];
            X[2] =(I[2] - I[6]) * sqrt(2) - X[3];

            X[8] = X[0] + X[3];
            X[0] = X[0] - X[3];
            X[3] = X[1] + X[2];
            X[1] = X[1] - X[2];

            X[4] = I[5] - I[3];
            X[5] = I[1] + I[7];
            X[6] = I[1] - I[7];
            X[7] = I[5] + I[3];

            X[9] = (X[4] + X[6]) * sqrt(2+sqrt(2));
            X[10] = X[5] + X[7];
            X[11] = X[9] - X[4] * sqrt(4+sqrt(8)) - X[10];
            X[12] = (X[5] - X[7]) * sqrt(2) - X[11];
            X[13] = X[6] * sqrt(4-sqrt(8)) - X[9] + X[12];

            X[2]=X[1] + X[12];
            X[5]=X[1] - X[12];
            X[1]=X[3] + X[11];
            X[6]=X[3] - X[11];
            X[3]=X[0] - X[13];
            X[4]=X[0] + X[13];
            X[0]=X[8] + X[10];
            X[7]=X[8] - X[10];
            return g;
        }

        public:
        void generate() {
            arch=get_target().arch;
            hasavx=false;
            dequant(x,y,z)=in(x,y,z)*quant(x,y);
            ws=idct8_dim1(dequant,"ws");
            wsT(y,x,z)=ws(x,y,z);
            op=idct8_dim1(wsT,"op");
            out(x,y,z)= u8_sat((op(x,y,z)/2+(257<<(IFAST_SCALE_BITS+1)))>>(IFAST_SCALE_BITS+2));
        }

        void schedule() {
            out.dim(0).set_bounds(0,8);
            out.dim(1).set_bounds(0,8);
            out.dim(0).set_stride(1);
            out.dim(2).set_stride(8);
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

            if(!auto_schedule) {
                ws.compute_at(wsT,z).unroll(y);
                for (int i = 0; i < ws.num_update_definitions(); i++)
                    ws.update(i).vectorize(x);
                wsT.compute_at(out,z).reorder_storage(y, x).vectorize(x).unroll(y);
                op.compute_at(out,z).unroll(y);
                for (int i = 0; i < op.num_update_definitions(); i++)
                    op.update(i).vectorize(x);
                out.compute_root().vectorize(x).unroll(y);
			} else {
                out.dim(0).set_estimate(0, 8);
                out.dim(1).set_estimate(0, 8);
                out.dim(2).set_estimate(0, 500);
                in.dim(0).set_estimate(0, 8);
                in.dim(1).set_estimate(0, 8);
                in.dim(2).set_estimate(0, 500);
                quant.dim(0).set_estimate(0, 8);
                quant.dim(1).set_estimate(0, 8);
                
            }
        }
};

HALIDE_REGISTER_GENERATOR(IdctGenerator,idct_fast);

