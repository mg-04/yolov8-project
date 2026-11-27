// Pass sequencer. One pass reads every weight of the loaded slice exactly once
// at II=1, then drains. Golden is stored once per campaign, not per site.
module ctrl #(
    parameter ADDR_W = 12,
    // Last read_en to last accumulator update: S1, S2, S3, acc reg.
    parameter PIPE_DEPTH = 4
)(
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    input  wire [15:0] n_weights,

    output reg         read_en,
    output reg         pass_active,
    output reg         drain,
    output reg         busy,
    output reg         done,
    output reg  [15:0] beat
);
    localparam [1:0] S_IDLE = 2'd0, S_RUN = 2'd1, S_DRAIN = 2'd2, S_DONE = 2'd3;
    reg [1:0] st;

    // The read port is one cycle ahead of the check, so the last weight is
    // still in flight when `beat` reaches n_weights.
    localparam [15:0] DRAIN_CYCLES = PIPE_DEPTH;
    reg [15:0] dcnt;

    always @(posedge clk) begin
        if (rst) begin
            st <= S_IDLE; beat <= 0; dcnt <= 0;
            read_en <= 0; pass_active <= 0; drain <= 0; busy <= 0; done <= 0;
        end else begin
            read_en     <= 1'b0;
            drain       <= 1'b0;
            case (st)
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start && n_weights != 16'd0) begin
                        st <= S_RUN; beat <= 0; busy <= 1'b1;
                        done <= 1'b0; pass_active <= 1'b1;
                    end
                end
                S_RUN: begin
                    read_en <= 1'b1;
                    beat    <= beat + 1'b1;
                    if (beat + 1'b1 >= n_weights) begin
                        st <= S_DRAIN; dcnt <= 0;
                    end
                end
                // One pulse at pipe-empty, not a level: golden captures once.
                S_DRAIN: begin
                    dcnt <= dcnt + 1'b1;
                    if (dcnt + 1'b1 >= DRAIN_CYCLES) begin
                        drain <= 1'b1;
                        st <= S_DONE; pass_active <= 1'b0;
                    end
                end
                S_DONE: begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    st   <= S_IDLE;
                end
                default: st <= S_IDLE;
            endcase
        end
    end
endmodule
