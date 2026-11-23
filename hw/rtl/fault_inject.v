// Read-path fault injection for controlled experiments. One mask bit toggles
// the corresponding SRAM word bit, including the SECDED check bits.
module fault_inject #(
    parameter WIDTH = 22
)(
    input  wire [WIDTH-1:0] word_in,
    input  wire [WIDTH-1:0] flip_mask,
    input  wire             enable,
    output wire [WIDTH-1:0] word_out
);
    assign word_out = word_in ^ (enable ? flip_mask : {WIDTH{1'b0}});
endmodule
