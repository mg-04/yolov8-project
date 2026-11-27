// Chip top. Real at the boundary, hollow inside: the pad-level ports, the
// pipeline registers and the control plane are final, and the datapath blocks
// are stubs that kill a simulation if they are clocked. Integrate one real
// block at a time and keep `make lint-top` passing.
//
// Pipeline, II=1:
//   S0  address generate        weight address, bound index
//   S1  SRAM read               22-bit word, bound registered here
//   S2  inject, then check      address-gated XOR; SECDED || bound check
//   S3  MAC                     P lanes, accumulate, compare against golden
//
// The injector compares the address of the word in flight, so the address must
// be pipelined to match the SRAM latency. Feeding the live S0 address would
// corrupt a neighbour and the counters could not tell.
//
// VDD_chk / VSS_chk are a separate supply domain covering sentry_check and the
// bound table, so energy per check is measurable; they do not appear as RTL
// ports. Signal pads below total nine.
module top #(
    parameter P            = 8,
    parameter ACC_W        = 24,
    parameter ADDR_W       = 12,
    parameter NUM_BOUNDS   = 16,
    parameter ACT_BASE     = 2304,
    parameter TWO_SIDED    = 0,
    parameter REPAIR       = 0,
    parameter BOUND_ADDR_W = NUM_BOUNDS > 1 ? $clog2(NUM_BOUNDS) : 1
)(
    input  wire clk,
    input  wire rst_n,
    input  wire spi_sclk,
    input  wire spi_mosi,
    input  wire spi_csn,
    output wire spi_miso,
    input  wire start_pin,
    output wire busy_pin,
    output wire done_pin
);
    wire rst = ~rst_n;

    // ---- configuration ----------------------------------------------------
    wire [1:0]              mac_src;
    wire                    ecc_correct_en;
    wire                    inject_enable, inject_act_region;
    wire [ADDR_W-1:0]       inject_addr;
    wire [21:0]             inject_mask;
    wire [15:0]             n_weights;
    wire [8:0]              n_oc;
    wire [4:0]              cnt_sel;
    wire                    bound_wr_en;
    wire [BOUND_ADDR_W-1:0] bound_wr_addr;
    wire [4:0]              bound_wr_emin, bound_wr_emax;
    wire                    cfg_start, clr_counters;
    wire [31:0]             cnt_rdata;

    wire                    busy, done, bound_disagreement;
    wire [15:0] status = {12'b0, bound_disagreement, 1'b1, done, busy};

    spi_cfg #(.ADDR_W(ADDR_W), .BOUND_ADDR_W(BOUND_ADDR_W)) cfg (
        .clk(clk), .rst(rst),
        .sclk(spi_sclk), .mosi(spi_mosi), .csn(spi_csn), .miso(spi_miso),
        .mac_src(mac_src), .ecc_correct_en(ecc_correct_en),
        .inject_enable(inject_enable), .inject_act_region(inject_act_region),
        .inject_addr(inject_addr), .inject_mask(inject_mask),
        .n_weights(n_weights), .n_oc(n_oc), .cnt_sel(cnt_sel),
        .bound_wr_en(bound_wr_en), .bound_wr_addr(bound_wr_addr),
        .bound_wr_emin(bound_wr_emin), .bound_wr_emax(bound_wr_emax),
        .start(cfg_start), .clr_counters(clr_counters),
        .status(status), .cnt_rdata(cnt_rdata)
    );

    // ---- control ----------------------------------------------------------
    wire        read_en, pass_active, drain;
    wire [15:0] beat;

    ctrl #(.ADDR_W(ADDR_W)) sequencer (
        .clk(clk), .rst(rst), .start(cfg_start | start_pin),
        .n_weights(n_weights),
        .read_en(read_en), .pass_active(pass_active), .drain(drain),
        .busy(busy), .done(done), .beat(beat)
    );
    assign busy_pin = busy;
    assign done_pin = done;

    // ---- S0: address generate --------------------------------------------
    wire [ADDR_W-1:0]       s0_waddr;
    wire [BOUND_ADDR_W-1:0] s0_baddr;
    wire                    s0_last;

    addrgen #(.ADDR_W(ADDR_W), .BOUND_ADDR_W(BOUND_ADDR_W)) ag (
        .clk(clk), .rst(rst), .read_en(read_en),
        .n_weights(n_weights), .n_oc(n_oc),
        .weight_addr(s0_waddr), .bound_addr(s0_baddr), .last(s0_last)
    );

    // ---- S1: SRAM read ----------------------------------------------------
    // The address travels with the data so S2 injects into the word in flight.
    reg  [ADDR_W-1:0]       s1_waddr;
    reg  [BOUND_ADDR_W-1:0] s1_baddr;
    reg                     s1_valid;
    wire [21:0]             s1_word;

    always @(posedge clk) begin
        if (rst) begin
            s1_waddr <= 0; s1_baddr <= 0; s1_valid <= 1'b0;
        end else begin
            s1_waddr <= s0_waddr;
            s1_baddr <= s0_baddr;
            s1_valid <= read_en;
        end
    end

    sram_22b #(.ADDR_W(ADDR_W)) wmem (
        .clk(clk), .en(read_en | s1_valid), .we(1'b0),
        .addr(s0_waddr), .wdata(22'b0), .rdata(s1_word)
    );

    // ---- S2: inject and check --------------------------------------------
    wire [15:0] s2_weight;
    wire        s2_valid, s2_flag, s2_inject_hit, s2_bound_valid;
    wire [15:0] s2_golden;
    wire        s2_golden_valid, s2_match, s2_ecc_corr, s2_ecc_unc;
    wire [4:0]  s2_syndrome;
    wire [21:0] s2_error_mask;

    weight_read_check #(
        .NUM_BOUNDS(NUM_BOUNDS), .TWO_SIDED(TWO_SIDED), .REPAIR(REPAIR),
        .ADDR_W(ADDR_W), .ACT_BASE(ACT_BASE)
    ) rdpath (
        .clk(clk), .rst(rst),
        .bound_wr_en(bound_wr_en), .bound_wr_addr(bound_wr_addr),
        .bound_wr_emin(bound_wr_emin), .bound_wr_emax(bound_wr_emax),
        .read_valid(s1_valid), .weight_addr(s1_waddr), .bound_addr(s1_baddr),
        .sram_word(s1_word),
        .inject_enable(inject_enable), .inject_addr(inject_addr),
        .inject_act_region(inject_act_region), .inject_mask(inject_mask),
        .mac_src(mac_src),
        .weight_out(s2_weight), .weight_valid(s2_valid),
        .sentry_flag(s2_flag), .inject_hit(s2_inject_hit),
        .bound_valid(s2_bound_valid), .bound_disagreement(bound_disagreement),
        .golden_weight(s2_golden), .golden_valid(s2_golden_valid),
        .weight_matches_golden(s2_match),
        .ecc_corrected(s2_ecc_corr), .ecc_uncorrectable(s2_ecc_unc),
        .ecc_syndrome(s2_syndrome), .ecc_error_mask(s2_error_mask)
    );

    // ---- S3: MAC and golden comparison -----------------------------------
    wire [P*16-1:0]    act;
    wire [P*ACC_W-1:0] acc;

    actwin #(.P(P), .K(3)) win (
        .clk(clk), .rst(rst), .shift_en(s2_valid),
        .act_in(16'h3c00), .act_out(act)
    );

    mac_array #(.P(P), .ACC_W(ACC_W)) macs (
        .clk(clk), .rst(rst), .en(s2_valid), .clr(cfg_start),
        .weight(s2_weight), .act(act), .acc(acc)
    );

    // Pass 1 stores the golden tile, pass 2 compares against it. Which pass is
    // running is the injection arm itself, so no extra state is needed.
    wire [15:0]      n_differ;
    wire [ACC_W-1:0] max_delta;
    wire             any_differ;

    golden #(.P(P), .ACC_W(ACC_W)) gold (
        .clk(clk), .rst(rst),
        .store(drain & ~inject_enable), .compare(drain & inject_enable),
        .acc(acc),
        .n_differ(n_differ), .max_delta(max_delta), .any_differ(any_differ)
    );

    // ---- event classification --------------------------------------------
    // `false` is structurally impossible: the bound cannot flag a word that was
    // never injected. A non-zero false count is therefore a bug indicator, not
    // a measurement, which is why it is counted rather than assumed away.
    wire hit   = s2_inject_hit;
    wire ev_caught = s2_valid &  hit &  s2_flag;
    wire ev_missed = s2_valid &  hit & ~s2_flag;
    wire ev_false  = s2_valid & ~hit &  s2_flag;
    wire ev_clean  = s2_valid & ~hit & ~s2_flag;

    // Which exponent bit was armed, for the per-bit histogram.
    wire [2:0] ev_bit = inject_mask[14] ? 3'd4 :
                        inject_mask[13] ? 3'd3 :
                        inject_mask[12] ? 3'd2 :
                        inject_mask[11] ? 3'd1 : 3'd0;

    counters #(.N(24)) cnt (
        .clk(clk), .rst(rst), .clr(clr_counters),
        .ev_caught(ev_caught), .ev_missed(ev_missed),
        .ev_false(ev_false), .ev_clean(ev_clean), .ev_bit(ev_bit),
        .sel(cnt_sel), .rdata(cnt_rdata)
    );

    // Tie-off so an unimplemented datapath does not look like a clean result.
    wire _unused = &{1'b0, s0_last, pass_active, beat, ecc_correct_en,
                     s2_golden, s2_golden_valid, s2_match, s2_ecc_corr,
                     s2_ecc_unc, s2_syndrome, s2_error_mask, s2_bound_valid,
                     n_differ, max_delta, any_differ, 1'b0};
endmodule
