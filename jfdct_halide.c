/*
 * jidctint.c
 *
 * This file was part of the Independent JPEG Group's software.
 * Copyright (C) 1991-1998, Thomas G. Lane.
 * Modification developed 2002-2009 by Guido Vollbeding.
 * libjpeg-turbo Modifications:
 * Copyright (C) 2015, D. R. Commander.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains a slow-but-accurate integer implementation of the
 * inverse DCT (Discrete Cosine Transform).  In the IJG code, this routine
 * must also perform dequantization of the input coefficients.
 *
 * A 2-D IDCT can be done by 1-D IDCT on each column followed by 1-D IDCT
 * on each row (or vice versa, but it's more convenient to emit a row at
 * a time).  Direct algorithms are also available, but they are much more
 * complex and seem not to be any faster when reduced to code.
 *
 * This implementation is based on an algorithm described in
 *   C. Loeffler, A. Ligtenberg and G. Moschytz, "Practical Fast 1-D DCT
 *   Algorithms with 11 Multiplications", Proc. Int'l. Conf. on Acoustics,
 *   Speech, and Signal Processing 1989 (ICASSP '89), pp. 988-991.
 * The primary algorithm described there uses 11 multiplies and 29 adds.
 * We use their alternate method with 12 multiplies and 32 adds.
 * The advantage of this method is that no data path contains more than one
 * multiplication; this allows a very simple and accurate implementation in
 * scaled fixed-point arithmetic, with a minimal number of shifts.
 *
 * We also provide IDCT routines with various output sample block sizes for
 * direct resolution reduction or enlargement without additional resampling:
 * NxN (N=1...16) pixels for one 8x8 input DCT block.
 *
 * For N<8 we simply take the corresponding low-frequency coefficients of
 * the 8x8 input DCT block and apply an NxN point IDCT on the sub-block
 * to yield the downscaled outputs.
 * This can be seen as direct low-pass downsampling from the DCT domain
 * point of view rather than the usual spatial domain point of view,
 * yielding significant computational savings and results at least
 * as good as common bilinear (averaging) spatial downsampling.
 *
 * For N>8 we apply a partial NxN IDCT on the 8 input coefficients as
 * lower frequencies and higher frequencies assumed to be zero.
 * It turns out that the computational effort is similar to the 8x8 IDCT
 * regarding the output size.
 * Furthermore, the scaling and descaling is the same for all IDCT sizes.
 *
 * CAUTION: We rely on the FIX() macro except for the N=1,2,4,8 cases
 * since there would be too many additional constants to pre-calculate.
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"
#include "jdct.h"               /* Private declarations for DCT subsystem */

//#include <HalideRuntime.h>
#include "halide/fdct.h"

/*
 * Perform dequantization and inverse DCT on one block of coefficients.
 */

static halide_buffer_t inbuffer;//,quantbuffer,outbuffer;
static halide_dimension_t dims[2];//,outdims[2];

GLOBAL(void)
jpeg_fdct_halide (DCTELEM *data) {


  if(dims[0].extent!=8) {
  dims[0].min=0;
  dims[0].extent=8;
  dims[0].stride=1;
  dims[1].min=0;
  dims[1].extent=8;
  dims[1].stride=8;

  inbuffer.dimensions=2;
  inbuffer.dim=dims;
  inbuffer.type.code=halide_type_int;
  inbuffer.type.bits=16;
  inbuffer.type.lanes=1;
  }

//  outdims[1].stride=output_buf[1]-output_buf[0];

  inbuffer.host=(uint8_t*)data;

  fdct(&inbuffer,&inbuffer);

}
