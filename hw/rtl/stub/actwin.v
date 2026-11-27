// STUB. Owner: unassigned.
// Sliding activation window, (P+k-1)*k deep for ic-outermost order. Primary
// source is real activations held in the weight macro's spare words; a shaped
// index hash behind a config bit covers unlimited inputs. Raw LFSR bits
// reinterpreted as FP16 are a bad distribution (1/32 inf or NaN) and are the
// fallback of last resort.
module actwin #(
    parameter P = 8,
    parameter K = 3
)(
    input  wire            clk,
    input  wire            rst,
    input  wire            shift_en,
    input  wire [15:0]     act_in,
    output wire [P*16-1:0] act_out
);
    assign act_out = {(P*16){1'bx}};
    wire stub_active = shift_en;
    `include "stub_guard.vh"
endmodule
