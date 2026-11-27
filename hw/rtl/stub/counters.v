// STUB. Owner: unassigned.
// The 2x2 verdict matrix (caught / missed / false / clean) plus a per-bit
// histogram, 5 exponent bit positions x 4 verdicts = 20 counters of 32 bits.
// `missed` is the number the chip exists to produce and `false` should be
// structurally impossible, so a non-zero false count is a bug indicator.
// Triplicate these: they are the published result.
module counters #(
    parameter N = 24
)(
    input  wire        clk,
    input  wire        rst,
    input  wire        clr,
    input  wire        ev_caught,
    input  wire        ev_missed,
    input  wire        ev_false,
    input  wire        ev_clean,
    input  wire [2:0]  ev_bit,
    input  wire [4:0]  sel,
    output wire [31:0] rdata
);
    assign rdata = 32'bx;
    wire stub_active = (ev_caught || ev_missed || ev_false || ev_clean);
    `include "stub_guard.vh"
endmodule
