// STUB. Owner: unassigned.
// Golden tile store and comparator. Pass 1 writes, pass 2 compares; one drain
// at the end of the pass, so a single comparator suffices. Reports how many
// elements differ and the largest difference. A difference under beam is
// re-run: an injected fault reproduces deterministically, an upset does not.
module golden #(
    parameter P = 8,
    parameter ACC_W = 24
)(
    input  wire               clk,
    input  wire               rst,
    input  wire               store,
    input  wire               compare,
    input  wire [P*ACC_W-1:0] acc,
    output wire [15:0]        n_differ,
    output wire [ACC_W-1:0]   max_delta,
    output wire               any_differ
);
    assign n_differ   = 16'bx;
    assign max_delta  = {ACC_W{1'bx}};
    assign any_differ = 1'bx;
    wire stub_active = (store || compare);
    `include "stub_guard.vh"
endmodule
