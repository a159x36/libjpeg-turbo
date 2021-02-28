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


class FdctGenerator : public Halide::Generator<FdctGenerator> {
    private:
        Input <Buffer<uint8_t>> in{"in", 3};
        Input <Buffer<int16_t>> quant{"quant", 2};
        Output <Buffer<int16_t>> out{"out", 3};

        Func ws,dequant,wsT,op,opT;
        Var x,y,z,zo,zi,yi,fused;

        vector<FuncRef> get_func_refs(Func fx, int N) {
            vector<FuncRef> refs;
            for (int i = 0; i < N; i++) {
                    refs.push_back(fx(x,i,z));
            }
            return refs;
        }

        const double c6=cos((3.0*M_PI)/8.0);
        const double c2=cos((1.0*M_PI)/8.0);
        const double c4=cos((2.0*M_PI)/8.0);

        Func fdct8_dim1(Func f,const char *name) {

			vector<FuncRef> I=get_func_refs(f,8);

            Func g(name);
            vector<FuncRef> X=get_func_refs(g,8);
            g(x,y,z)=undef(Int(16));

            Expr t0 = I[0] + I[7];
            Expr t7 = I[0] - I[7];
            Expr t1 = I[1] + I[6];
            Expr t6 = I[1] - I[6];
            Expr t2 = I[2] + I[5];
            Expr t5 = I[2] - I[5];
            Expr t3 = I[3] + I[4];
            Expr t4 = I[3] - I[4];

            Expr t10 = t0 + t3;        /* phase 2 */
            Expr t13 = t0 - t3;
            Expr t11 = t1 + t2;
            Expr t12 = t1 - t2;

            X[0] =  t10 + t11;
            X[4] =  t10 - t11;

            Expr z1 = (t12 + t13) * c4; // z1
            X[2] = t13 + z1;
            X[6] = t13 - z1;

            t10 = t4 + t5;
            t11 = t5 + t6; 
            t12 = t6 + t7;

            Expr z5 = (t10 - t12) *  c6; // z5
            Expr z2 = (t10 * (c2-c6)) + z5; // z2
            Expr z4 = (t12 * (c2+c6)) + z5; // z4
            Expr z3 = t11 * c4; // z3

            Expr z11 = t7 + z3; // z11
            Expr z13 = t7 - z3; // z13

            X[5] = z13 + z2;
            X[3] = z13 - z2;
            X[1] = z11 + z4;
            X[7] = z11 - z4;
            return g;
        }
        public:
        void generate() {
            arch=get_target().arch;
            dequant(x,y,z)=(i16(in(x,y,z))-128)<<4;
            ws=fdct8_dim1(dequant,"ws");
            wsT(y,x,z)=ws(x,y,z)>>2;
            op=fdct8_dim1(wsT,"op");
            if(arch == Target::ARM)
                out(y,x,z) = i16_sat((i32(i16(op(x,y,z)))*i32(quant(x,y))+(1<<14))/(1<<15));
            else
                out(y,x,z) = i16((i32(i16(op(x,y,z)))*i32(quant(x,y))+(1<<14))/(1<<15));
        }

        void schedule() {
            out.dim(0).set_bounds(0,8); 
            out.dim(1).set_bounds(0,8);
            out.dim(0).set_stride(1);
            out.dim(1).set_stride(8);
            out.dim(2).set_stride(64);
            out.dim(2).set_min(0);
            in.dim(0).set_bounds(0,8);
            in.dim(1).set_bounds(0,8);
            in.dim(0).set_stride(1);
            in.dim(2).set_min(0);
            in.dim(2).set_stride(8);
            quant.dim(0).set_bounds(0,8);
            quant.dim(1).set_bounds(0,8);
            quant.dim(0).set_stride(1);
            quant.dim(1).set_stride(8);
 
            if(!auto_schedule) {
                ws.compute_at(wsT,z).unroll(y).vectorize(x);
                for (int i = 0; i < ws.num_update_definitions(); i++)   
                    ws.update(i).vectorize(x);
                wsT.compute_at(out,z).reorder_storage(y, x).vectorize(x).unroll(y);
                op.compute_at(out,z).unroll(y).vectorize(x);
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

HALIDE_REGISTER_GENERATOR(FdctGenerator,fdct_fast);

