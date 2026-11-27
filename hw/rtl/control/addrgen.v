// TODO STUB. Owner: unassigned.
// Walk the loop nest ic-outermost (444 B of state against 984 B for
// oc-outermost) and emit the flat weight address with the output-channel index
// that selects its bound. The bound index is NOT a shift of the flat address:
// weights per channel is Cin_s*k*k = 144 for a 16-channel slice, not a power
// of two, which is why weight_read_check takes bound_addr separately.
module addrgen #(
    parameter ADDR_W = 12,
    parameter BOUND_ADDR_W = 4
)(
    input  wire                    clk,
    input  wire                    rst,
    input  wire                    read_en,
    input  wire [15:0]             n_weights,
    input  wire [8:0]              n_oc,
    output wire [ADDR_W-1:0]       weight_addr,     // weight address to read
    output wire [BOUND_ADDR_W-1:0] bound_addr,      // index into bound table for its group
    output wire                    last
);
    assign weight_addr = {ADDR_W{1'bx}};
    assign bound_addr  = {BOUND_ADDR_W{1'bx}};
    assign last        = 1'bx;
    wire stub_active = read_en;
    `include "stub_guard.vh"
endmodule
