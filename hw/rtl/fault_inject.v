// Region gate: weights and activations share one macro, so a mistyped inject
// address must not corrupt an activation word and score as a weight fault.
module fault_inject #(
    parameter WIDTH    = 22,
    parameter ADDR_W   = 12,
    parameter ACT_BASE = (1 << ADDR_W)
)(
    input  wire [WIDTH-1:0]  word_in,
    // additional check gate of current read vs injection address
    input  wire [ADDR_W-1:0] rd_addr,
    input  wire [ADDR_W-1:0] inject_addr,
    input  wire              inject_act_region,
    input  wire [WIDTH-1:0]  flip_mask,
    input  wire              enable,
    output wire [WIDTH-1:0]  word_out,
    output wire              hit
);
    wire read_is_act = ({1'b0, rd_addr} >= ACT_BASE[ADDR_W:0]);
    wire armed_is_act = ({1'b0, inject_addr} >= ACT_BASE[ADDR_W:0]);

    wire addr_match = (rd_addr == inject_addr);
    wire region_ok  = (read_is_act == inject_act_region) &&
                      (armed_is_act == inject_act_region);

    assign hit      = enable && addr_match && region_ok;
    assign word_out = word_in ^ (hit ? flip_mask : {WIDTH{1'b0}});
endmodule
