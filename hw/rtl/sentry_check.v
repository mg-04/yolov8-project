// sentry_check -- one weight-bound check, combinational.
//
// The whole defence, in hardware: read a weight's exponent field, compare it
// against its unit's stored bound, substitute if it fails. Mantissa and sign
// pass through untouched in the exponent modes, so no floating-point hardware
// is involved -- the check is integer comparison on one byte.
//
// MODE selects the bound, matching src/weightSentry.py exactly:
//   0  exp 1-sided   e <= emax                              1 B/unit
//   1  exp 2-sided   e <= emax && (e >= emin || x == 0)     2 B/unit
//   2  signed        lo <= w <= hi   (fp compare)           8 B/unit
//   3  signed+floor  mode 2 && (e >= emin || x == 0)        9 B/unit
//
// REPAIR selects what happens on a failure:
//   0  substitute exponent 0 (crush toward zero), which is what the published
//      Algorithm 1 provably does anyway -- see the zero-substitution result
//   1  Algorithm 1 as published: toggle exponent bits MSB-first, take the
//      first candidate that lands in range, else clamp at emax
// Synthesising both prices the search the software result showed is redundant.
//
// Float comparison uses the sign-magnitude -> monotonic-unsigned mapping, so
// mode 2/3 needs an integer comparator of full word width rather than an FPU.
//
// Widths: EXP_W/MANT_W = 8/23 for FP32, 5/10 for FP16.
//
// Simplification, deliberate: the no-candidate fallback substitutes emax in
// every mode. Python clamps to lo/hi in the signed modes. Detection -- the
// measured quantity -- is exact; only this tail differs.

module sentry_check #(
    parameter EXP_W  = 8,
    parameter MANT_W = 23,
    parameter MODE   = 0,
    parameter REPAIR = 0
)(
    input  wire [EXP_W+MANT_W:0] w_in,
    input  wire [EXP_W-1:0]      emax,
    input  wire [EXP_W-1:0]      emin,
    input  wire [EXP_W+MANT_W:0] lo,
    input  wire [EXP_W+MANT_W:0] hi,
    output wire [EXP_W+MANT_W:0] w_out,
    output wire                  flag
);
    localparam W = EXP_W + MANT_W + 1;

    // IEEE-754 sign-magnitude ordering -> unsigned ordering. For a negative
    // value invert every bit; for a positive one set the sign bit. Unsigned
    // compare on the result then matches float compare on the original.
    function [W-1:0] mono;
        input [W-1:0] f;
        begin
            mono = f[W-1] ? ~f : (f | {1'b1, {(W-1){1'b0}}});
        end
    endfunction

    function in_range;
        input [W-1:0] f;
        reg [EXP_W-1:0] e;
        reg             zer;
        begin
            e   = f[W-2 -: EXP_W];
            zer = (f[W-2:0] == {(W-1){1'b0}});     // +0.0 or -0.0
            if (MODE == 0)
                in_range = (e <= emax);
            else if (MODE == 1)
                in_range = (e <= emax) && ((e >= emin) || zer);
            else if (MODE == 2)
                in_range = (mono(f) >= mono(lo)) && (mono(f) <= mono(hi));
            else
                in_range = (mono(f) >= mono(lo)) && (mono(f) <= mono(hi))
                           && ((e >= emin) || zer);
        end
    endfunction

    wire ok = in_range(w_in);
    assign flag = ~ok;

    wire [W-1:0] crush  = {w_in[W-1], {EXP_W{1'b0}},  w_in[MANT_W-1:0]};
    wire [W-1:0] clamp  = {w_in[W-1], emax,           {MANT_W{1'b1}}};

    generate
    if (REPAIR == 0) begin : g_zero
        assign w_out = ok ? w_in : crush;
    end else begin : g_search
        wire [EXP_W*W-1:0] cand;
        wire [EXP_W-1:0]   hit;
        genvar k;
        for (k = 0; k < EXP_W; k = k + 1) begin : g_cand
            assign cand[k*W +: W] =
                w_in ^ ({{(W-1){1'b0}}, 1'b1} << (MANT_W + k));
            assign hit[k] = in_range(cand[k*W +: W]);
        end
        // Algorithm 1 searches MSB-first and takes the first hit, so the
        // highest set bit of `hit` wins. Unrolled: a priority encoder, one
        // cycle, not the eight iterations the pseudocode implies.
        reg [W-1:0] fixed;
        reg         found;
        integer     kk;
        always @(*) begin
            fixed = clamp;
            found = 1'b0;
            for (kk = EXP_W-1; kk >= 0; kk = kk - 1)
                if (!found && hit[kk]) begin
                    fixed = cand[kk*W +: W];
                    found = 1'b1;
                end
        end
        assign w_out = ok ? w_in : fixed;
    end
    endgenerate
endmodule
