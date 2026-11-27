// TODO: understand this protocol and check with Ken. Can be deferred for now

// SPI slave and configuration register file. Frame is 24 bits, MSB first:
// one R/W bit, seven address bits, sixteen data bits. Reads shift the
// addressed register out on MISO during the data phase of the same frame.
//
// Sampled in the core clock domain, not on SCLK, so the core must be held in
// reset while configuration is loaded; only start/done then cross domains.
module spi_cfg #(
    parameter ADDR_W = 12,
    parameter BOUND_ADDR_W = 4
)(
    input  wire        clk,
    input  wire        rst,

    input  wire        sclk,
    input  wire        mosi,
    input  wire        csn,
    output wire        miso,

    output reg  [1:0]  mac_src,
    output reg         ecc_correct_en,
    output reg         inject_enable,
    output reg         inject_act_region,
    output reg  [ADDR_W-1:0] inject_addr,
    output reg  [21:0] inject_mask,
    output reg  [15:0] n_weights,
    output reg  [8:0]  n_oc,
    output reg  [4:0]  cnt_sel,

    output reg         bound_wr_en,
    output reg  [BOUND_ADDR_W-1:0] bound_wr_addr,
    output reg  [4:0]  bound_wr_emin,
    output reg  [4:0]  bound_wr_emax,

    // Weight/activation memory load. Address auto-increments on each data
    // write, so a full load is two frames per word after one R_MEMA.
    output reg         mem_wr_en,
    output reg  [ADDR_W-1:0] mem_addr,
    output reg  [21:0] mem_wdata,

    output reg         start,
    output reg         clr_counters,

    input  wire [15:0] status,
    input  wire [31:0] cnt_rdata
);
    localparam [6:0] R_CTRL = 7'h00, R_MODE   = 7'h01, R_INJC  = 7'h02,
                     R_INJA = 7'h03, R_MSKL   = 7'h04, R_MSKH  = 7'h05,
                     R_BNDA = 7'h06, R_BNDD   = 7'h07, R_NWT   = 7'h08,
                     R_NOC  = 7'h09, R_CNTSEL = 7'h0a,
                     R_MEMA = 7'h0b, R_MEMDL  = 7'h0c, R_MEMDH = 7'h0d,
                     R_STAT = 7'h40, R_CNTLO  = 7'h41, R_CNTHI = 7'h42;

    reg  [1:0]  sclk_q;
    reg         csn_q;
    wire        sclk_rise = (sclk_q[1:0] == 2'b01);
    always @(posedge clk) begin
        sclk_q <= {sclk_q[0], sclk};
        csn_q  <= csn;
    end

    // Header is latched when its eighth bit arrives; data is decoded from the
    // post-shift value when the twenty-fourth arrives. Reading `sh` before the
    // shift would take every field one bit early.
    reg [23:0] sh;
    reg [4:0]  nbit;
    reg        rw_q;
    reg [6:0]  addr_q;
    reg [15:0] rdata;
    reg [ADDR_W-1:0] mem_ptr;

    wire [23:0] sh_next = {sh[22:0], mosi};

    // The master samples MISO two core clocks after the rising edge, by which
    // point nbit has advanced, so bit i of the data phase sits at nbit == 24-i.
    assign miso = (!csn_q && !rw_q && nbit >= 5'd9) ? rdata[24 - nbit] : 1'b0;

    always @(posedge clk) begin
        if (rst) begin
            sh <= 0; nbit <= 0; rw_q <= 0; addr_q <= 0; rdata <= 0;
            mac_src <= 2'd0; ecc_correct_en <= 1'b1;
            inject_enable <= 0; inject_act_region <= 0;
            inject_addr <= 0; inject_mask <= 0;
            n_weights <= 0; n_oc <= 0; cnt_sel <= 0;
            bound_wr_en <= 0; bound_wr_addr <= 0;
            bound_wr_emin <= 0; bound_wr_emax <= 0;
            mem_wr_en <= 0; mem_addr <= 0; mem_wdata <= 0; mem_ptr <= 0;
            start <= 0; clr_counters <= 0;
        end else begin
            start        <= 1'b0;
            clr_counters <= 1'b0;
            bound_wr_en  <= 1'b0;
            mem_wr_en    <= 1'b0;

            if (csn_q) begin
                nbit <= 0;
                rw_q <= 1'b0;
            end else if (sclk_rise) begin
                sh   <= sh_next;
                nbit <= nbit + 1'b1;

                if (nbit == 5'd7) begin
                    rw_q   <= sh_next[7];
                    addr_q <= sh_next[6:0];
                    case (sh_next[6:0])
                        R_STAT:  rdata <= status;
                        R_CNTLO: rdata <= cnt_rdata[15:0];
                        R_CNTHI: rdata <= cnt_rdata[31:16];
                        default: rdata <= 16'h0000;
                    endcase
                end

                if (nbit == 5'd23 && rw_q) begin
                    case (addr_q)
                        R_CTRL: begin
                            start        <= sh_next[0];
                            clr_counters <= sh_next[1];
                        end
                        R_MODE: begin
                            mac_src        <= sh_next[1:0];
                            ecc_correct_en <= sh_next[2];
                        end
                        R_INJC: begin
                            inject_enable     <= sh_next[0];
                            inject_act_region <= sh_next[1];
                        end
                        R_INJA:   inject_addr        <= sh_next[ADDR_W-1:0];
                        R_MSKL:   inject_mask[15:0]  <= sh_next[15:0];
                        R_MSKH:   inject_mask[21:16] <= sh_next[5:0];
                        R_BNDA:   bound_wr_addr      <= sh_next[BOUND_ADDR_W-1:0];
                        R_BNDD: begin
                            bound_wr_emax <= sh_next[4:0];
                            bound_wr_emin <= sh_next[12:8];
                            bound_wr_en   <= 1'b1;
                        end
                        R_NWT:    n_weights <= sh_next[15:0];
                        R_NOC:    n_oc      <= sh_next[8:0];
                        R_CNTSEL: cnt_sel   <= sh_next[4:0];
                        R_MEMA:   mem_ptr           <= sh_next[ADDR_W-1:0];
                        R_MEMDL:  mem_wdata[15:0]   <= sh_next[15:0];
                        // mem_addr latches the pre-increment pointer, so the
                        // write lands where the host asked.
                        R_MEMDH: begin
                            mem_wdata[21:16] <= sh_next[5:0];
                            mem_addr         <= mem_ptr;
                            mem_ptr          <= mem_ptr + 1'b1;
                            mem_wr_en        <= 1'b1;
                        end
                        default: ;
                    endcase
                end
            end
        end
    end
endmodule
