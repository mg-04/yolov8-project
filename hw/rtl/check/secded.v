// Extended Hamming SECDED for an FP16 SRAM word. The data bits are packed
// at [15:0], the five Hamming check bits at [20:16], and overall parity at 21.
module secded_encode (
    input  wire [15:0] data_in,
    output wire [21:0] word_out
);
    wire [4:0] parity;

    // Masks select data at the non-power-of-two Hamming positions 1..21.
    assign parity[0] = ^(data_in & 16'had5b);
    assign parity[1] = ^(data_in & 16'h366d);
    assign parity[2] = ^(data_in & 16'hc78e);
    assign parity[3] = ^(data_in & 16'h07f0);
    assign parity[4] = ^(data_in & 16'hf800);
    assign word_out = {^{parity, data_in}, parity, data_in};
endmodule

module secded_decode (
    input  wire [21:0] word_in,
    output wire [15:0] data_out,
    output wire [4:0]  syndrome,
    output wire        corrected,
    output wire        uncorrectable,
    output reg  [21:0] error_mask
);
    wire [21:0] regenerated;
    wire odd_parity = ^word_in;
    secded_encode encode (.data_in(word_in[15:0]), .word_out(regenerated));

    assign syndrome = word_in[20:16] ^ regenerated[20:16];
    // A zero syndrome with odd parity identifies the overall parity bit.
    // Syndromes above 21 cannot identify a bit in this shortened code.
    assign corrected = odd_parity && (syndrome <= 5'd21);
    assign uncorrectable = (syndrome != 5'd0) && !corrected;
    assign data_out = word_in[15:0] ^ error_mask[15:0];

    integer position;
    integer data_bit;
    integer parity_bit;
    always @* begin
        error_mask = 22'b0;
        data_bit = 0;
        parity_bit = 16;
        if (corrected) begin
            error_mask[21] = (syndrome == 5'd0);
            for (position = 1; position <= 21; position = position + 1) begin
                if ((position & (position - 1)) == 0) begin
                    error_mask[parity_bit] = (syndrome == position[4:0]);
                    parity_bit = parity_bit + 1;
                end else begin
                    error_mask[data_bit] = (syndrome == position[4:0]);
                    data_bit = data_bit + 1;
                end
            end
        end else begin
            position = 0;
        end
    end
endmodule
