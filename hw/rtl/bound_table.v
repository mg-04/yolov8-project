// bound table with exp bounds only, compiled into flops
// currently statically compiled
// TODO: make it dynamically configurable (cost may be insane in TMR tho)

// Configuration writes at the clock edge. Reads and majority voting are
// combinational. Only the ceiling is stored when TWO_SIDED is zero.
module bound_table #(
    parameter EXP_W = 5,
    parameter DEPTH = 72,
    parameter TWO_SIDED = 0,
    parameter ADDR_W = DEPTH > 1 ? $clog2(DEPTH) : 1
)(
    input  wire              clk,
    input  wire              rst,
    input  wire              wr_en,
    input  wire [ADDR_W-1:0] wr_addr,
    input  wire [EXP_W-1:0]  wr_emin,
    input  wire [EXP_W-1:0]  wr_emax,
    input  wire [ADDR_W-1:0] rd_addr,
    output wire [EXP_W-1:0]  rd_emin,
    output wire [EXP_W-1:0]  rd_emax,
    output wire              rd_valid,
    output wire              disagreement
);
    localparam BOUND_W = TWO_SIDED ? 2*EXP_W : EXP_W;
    localparam [ADDR_W-1:0] LAST_ADDR = DEPTH - 1;

    wire write_row = wr_en && (wr_addr <= LAST_ADDR);
    wire read_row = rd_addr <= LAST_ADDR;
    wire [BOUND_W-1:0] wr_bounds;
    wire [BOUND_W:0] bank_a_word, bank_b_word, bank_c_word;
    wire [BOUND_W:0] voted = (bank_a_word & bank_b_word) |
                            (bank_a_word & bank_c_word) |
                            (bank_b_word & bank_c_word);

    if (TWO_SIDED) begin : g_min_max
        assign wr_bounds = {wr_emin, wr_emax};
        assign rd_emin = rd_valid ? voted[2*EXP_W-1:EXP_W] : {EXP_W{1'b0}};
    end else begin : g_max_only
        assign wr_bounds = wr_emax;
        assign rd_emin = {EXP_W{1'b0}};
    end

    assign rd_valid = voted[BOUND_W];
    assign rd_emax = rd_valid ? voted[EXP_W-1:0] : {EXP_W{1'b0}};
    assign disagreement = (bank_a_word != bank_b_word) || (bank_a_word != bank_c_word);

    // Retain separate banks through synthesis so the vote covers stored upsets.
    (* keep = 1, keep_hierarchy = 1, dont_touch = "true" *)
    bound_table_bank #(BOUND_W, DEPTH, ADDR_W) bank_a (
        clk, rst, write_row, wr_addr, wr_bounds, read_row, rd_addr, bank_a_word
    );
    (* keep = 1, keep_hierarchy = 1, dont_touch = "true" *)
    bound_table_bank #(BOUND_W, DEPTH, ADDR_W) bank_b (
        clk, rst, write_row, wr_addr, wr_bounds, read_row, rd_addr, bank_b_word
    );
    (* keep = 1, keep_hierarchy = 1, dont_touch = "true" *)
    bound_table_bank #(BOUND_W, DEPTH, ADDR_W) bank_c (
        clk, rst, write_row, wr_addr, wr_bounds, read_row, rd_addr, bank_c_word
    );
endmodule

module bound_table_bank #(
    parameter WIDTH = 5,
    parameter DEPTH = 72,
    parameter ADDR_W = 7
)(
    input  wire              clk,
    input  wire              rst,
    input  wire              wr_en,
    input  wire [ADDR_W-1:0] wr_addr,
    input  wire [WIDTH-1:0]  wr_data,
    input  wire              rd_en,
    input  wire [ADDR_W-1:0] rd_addr,
    output wire [WIDTH:0]    rd_word
);
    reg [WIDTH-1:0] entries [0:DEPTH-1];
    reg [DEPTH-1:0] valid;

    always @(posedge clk) begin
        if (rst)
            valid <= {DEPTH{1'b0}};
        else if (wr_en) begin
            entries[wr_addr] <= wr_data;
            valid[wr_addr] <= 1'b1;
        end
    end

    assign rd_word = (rd_en && valid[rd_addr]) ? {1'b1, entries[rd_addr]}
                                               : {(WIDTH+1){1'b0}};
endmodule
