// Combinational read path for an external FP16 + SECDED SRAM. The ECC branch
// supplies the reference result and the range-check branch feeds the datapath.
module weight_read_check #(
    parameter NUM_BOUNDS = 72,
    parameter GROUP_SHIFT = 7,
    parameter TWO_SIDED = 0,
    parameter REPAIR = 0,
    parameter BOUND_ADDR_W = NUM_BOUNDS > 1 ? $clog2(NUM_BOUNDS) : 1
)(
    input  wire                               clk,
    input  wire                               rst,
    input  wire                               bound_wr_en,
    input  wire [BOUND_ADDR_W-1:0]            bound_wr_addr,
    input  wire [4:0]                         bound_wr_emin,
    input  wire [4:0]                         bound_wr_emax,
    input  wire                               read_valid,
    input  wire [BOUND_ADDR_W+GROUP_SHIFT-1:0] weight_addr,
    input  wire [21:0]                        sram_word,
    input  wire                               inject_enable,
    input  wire [21:0]                        inject_mask,
    input  wire                               bypass_sentry,
    output wire [15:0]                        weight_out,
    output wire                               weight_valid,
    output wire                               sentry_flag,
    output wire                               bound_valid,
    output wire                               bound_disagreement,
    output wire [15:0]                        golden_weight,
    output wire                               golden_valid,
    output wire                               weight_matches_golden,
    output wire                               ecc_corrected,
    output wire                               ecc_uncorrectable,
    output wire [4:0]                         ecc_syndrome,
    output wire [21:0]                        ecc_error_mask
);
    wire [21:0] read_word;
    wire [4:0] emin, emax;
    wire bounds_ready, disagreement;
    wire [15:0] sentry_weight;
    wire out_of_range;
    wire corrected, uncorrectable;
    wire [4:0] syndrome;
    wire [21:0] error_mask;

    fault_inject inject (
        .word_in(sram_word), .flip_mask(inject_mask),
        .enable(read_valid && inject_enable), .word_out(read_word)
    );
    secded_decode ecc (
        .word_in(read_word), .data_out(golden_weight), .syndrome(syndrome),
        .corrected(corrected), .uncorrectable(uncorrectable), .error_mask(error_mask)
    );
    bound_table #(
        .EXP_W(5), .DEPTH(NUM_BOUNDS), .TWO_SIDED(TWO_SIDED), .ADDR_W(BOUND_ADDR_W)
    ) bounds (
        .clk(clk), .rst(rst), .wr_en(bound_wr_en), .wr_addr(bound_wr_addr),
        .wr_emin(bound_wr_emin), .wr_emax(bound_wr_emax),
        .rd_addr(weight_addr[BOUND_ADDR_W+GROUP_SHIFT-1:GROUP_SHIFT]),
        .rd_emin(emin), .rd_emax(emax), .rd_valid(bounds_ready), .disagreement(disagreement)
    );
    sentry_check #(.EXP_W(5), .MANT_W(10), .MODE(TWO_SIDED ? 1 : 0), .REPAIR(REPAIR)) sentry (
        .w_in(read_word[15:0]), .emin(emin), .emax(emax), .lo(16'b0), .hi(16'b0),
        .w_out(sentry_weight), .flag(out_of_range)
    );

    assign weight_out = bypass_sentry ? read_word[15:0] : sentry_weight;
    assign weight_valid = read_valid && (bypass_sentry || bounds_ready);
    assign sentry_flag = read_valid && bounds_ready && out_of_range;
    assign bound_valid = read_valid && bounds_ready;
    assign bound_disagreement = read_valid && disagreement;
    assign golden_valid = read_valid && !uncorrectable;
    assign weight_matches_golden = weight_valid && golden_valid && (weight_out == golden_weight);
    assign ecc_corrected = read_valid && corrected;
    assign ecc_uncorrectable = read_valid && uncorrectable;
    assign ecc_syndrome = read_valid ? syndrome : 5'b0;
    assign ecc_error_mask = read_valid ? error_mask : 22'b0;
endmodule
