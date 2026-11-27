// Simulation model of the weight memory. NOT a stub and NOT synthesised: it
// stands in for a vendor macro, so it behaves correctly and is replaced by the
// macro at synthesis rather than being implemented by anyone.
//
// SAED32 has no macro this shape. Its single-port instances stop at 1024 words
// deep (32x50 64x8 64x32 64x34 64x128 128x8 128x46 128x48 256x8 256x32 256x46
// 256x48 512x8 512x32 512x128 1024x8), so the real memory is composed from
// several. Note 128x46 and 256x46: 46 bits holds exactly two 22-bit words with
// two spare, which halves the depth and the macro count and returns two words
// per cycle. Choosing between that and 3-wide 1024x8 is an open task.
module sram_22b #(
    parameter ADDR_W = 12,
    parameter DEPTH  = (1 << ADDR_W)
)(
    input  wire              clk,
    input  wire              en,
    input  wire              we,
    input  wire [ADDR_W-1:0] addr,
    input  wire [21:0]       wdata,
    output reg  [21:0]       rdata
);
    reg [21:0] mem [0:DEPTH-1];

    always @(posedge clk) begin
        if (en) begin
            if (we) mem[addr] <= wdata;
            rdata <= we ? wdata : mem[addr];
        end
    end

    task preload(input [ADDR_W-1:0] a, input [21:0] d);
        mem[a] = d;
    endtask
endmodule
