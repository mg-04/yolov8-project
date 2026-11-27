// STUB. Owner: unassigned.
// P parallel FP16 multiply-accumulate lanes with wide accumulators. A
// truncating multiplier is acceptable and smaller: the golden comparison needs
// determinism, not IEEE correctness, and integer arithmetic is exactly
// reproducible in the Python model.
module mac_array #(
    parameter P = 8,
    parameter ACC_W = 24
)(
    input  wire                clk,
    input  wire                rst,
    input  wire                en,
    input  wire                clr,
    input  wire [15:0]         weight,
    input  wire [P*16-1:0]     act,
    output wire [P*ACC_W-1:0]  acc
);
    assign acc = {(P*ACC_W){1'bx}};
    wire stub_active = en;
    `include "stub_guard.vh"
endmodule
