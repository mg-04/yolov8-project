// Smoke test for the chip top while the datapath is stubbed. It proves the
// control plane is real: SPI frames land in the config registers, a bound write
// reaches the table, status reads back, and the sequencer starts and finishes.
// It must NOT clock a stub out of reset, so the pass is deliberately short and
// `n_weights` is never set before the checks that do not need it.
module tb_top;
    localparam ADDR_W = 12;

    reg clk = 0;
    always #5 clk = !clk;

    reg rst_n = 0, sclk = 0, mosi = 0, csn = 1, start_pin = 0;
    wire miso, busy_pin, done_pin;

    top #(.ADDR_W(ADDR_W), .NUM_BOUNDS(16), .ACT_BASE(2304)) dut (
        .clk(clk), .rst_n(rst_n),
        .spi_sclk(sclk), .spi_mosi(mosi), .spi_csn(csn), .spi_miso(miso),
        .start_pin(start_pin), .busy_pin(busy_pin), .done_pin(done_pin)
    );

    // One 24-bit frame: rw, 7 address bits, 16 data bits, MSB first.
    task automatic spi_frame(input bit rw, input [6:0] a, input [15:0] d,
                             output [15:0] got);
        reg [23:0] tx;
        integer i;
        begin
            tx  = {rw, a, d};
            got = 16'h0000;
            csn = 0;
            repeat (4) @(posedge clk);
            for (i = 23; i >= 0; i = i - 1) begin
                mosi = tx[i];
                repeat (2) @(posedge clk);
                sclk = 1;
                repeat (2) @(posedge clk);
                if (i < 16) got = {got[14:0], miso};
                sclk = 0;
            end
            repeat (4) @(posedge clk);
            csn = 1;
            repeat (4) @(posedge clk);
        end
    endtask

    reg [15:0] rb;

    initial begin
        repeat (4) @(posedge clk);
        rst_n = 1;
        repeat (4) @(posedge clk);

        if (busy_pin || done_pin)
            $fatal(1, "idle after reset must not be busy or done");

        // MODE: mac_src = ECC (2), ecc_correct_en = 1
        spi_frame(1'b1, 7'h01, 16'h0006, rb);
        if (dut.mac_src !== 2'd2 || dut.ecc_correct_en !== 1'b1)
            $fatal(1, "MODE write did not land: mac_src=%0d ecc=%0b",
                   dut.mac_src, dut.ecc_correct_en);

        // The three MAC sources must all be reachable, or the baseline
        // comparison cannot be run on silicon.
        spi_frame(1'b1, 7'h01, 16'h0000, rb);
        if (dut.mac_src !== 2'd0) $fatal(1, "mac_src 0 unreachable");
        spi_frame(1'b1, 7'h01, 16'h0001, rb);
        if (dut.mac_src !== 2'd1) $fatal(1, "mac_src 1 unreachable");

        // Injection setup: one armed site, weight region, bit 14.
        spi_frame(1'b1, 7'h03, 16'h0123, rb);
        spi_frame(1'b1, 7'h04, 16'h4000, rb);
        spi_frame(1'b1, 7'h05, 16'h0000, rb);
        spi_frame(1'b1, 7'h02, 16'h0001, rb);
        if (dut.inject_addr !== 12'h123 || dut.inject_mask !== 22'h004000 ||
            dut.inject_enable !== 1'b1 || dut.inject_act_region !== 1'b0)
            $fatal(1, "injection setup did not land: addr=%h mask=%h en=%b",
                   dut.inject_addr, dut.inject_mask, dut.inject_enable);
        if (dut.ev_bit !== 3'd4)
            $fatal(1, "bit-14 mask must select histogram bin 4, got %0d", dut.ev_bit);

        // A bound write must reach the table and read back through the vote.
        spi_frame(1'b1, 7'h06, 16'h0005, rb);
        spi_frame(1'b1, 7'h07, 16'h0D11, rb);
        repeat (4) @(posedge clk);
        if (dut.rdpath.bounds.bank_a.entries[5] !== 5'h11)
            $fatal(1, "bound write did not reach bank_a: %h",
                   dut.rdpath.bounds.bank_a.entries[5]);
        if (dut.bound_disagreement !== 1'b0)
            $fatal(1, "freshly written banks must agree");

        // A memory load must land, and the address must auto-increment so a
        // full slice costs two frames per word rather than three.
        spi_frame(1'b1, 7'h0b, 16'h0020, rb);
        spi_frame(1'b1, 7'h0c, 16'hBEEF, rb);
        spi_frame(1'b1, 7'h0d, 16'h0015, rb);
        spi_frame(1'b1, 7'h0c, 16'h1234, rb);
        spi_frame(1'b1, 7'h0d, 16'h002A, rb);
        repeat (4) @(posedge clk);
        if (dut.wmem.mem[12'h020] !== 22'h15BEEF)
            $fatal(1, "memory load did not land: %h", dut.wmem.mem[12'h020]);
        if (dut.wmem.mem[12'h021] !== 22'h2A1234)
            $fatal(1, "memory address did not auto-increment: %h",
                   dut.wmem.mem[12'h021]);

        // STATUS must read back over MISO.
        spi_frame(1'b0, 7'h40, 16'h0000, rb);
        if (rb[1] !== 1'b0 || rb[2] !== 1'b1)
            $fatal(1, "STATUS readback wrong: %h", rb);

        // STATUS bit 0 is `busy`, the LSB of the read phase. Read it during a
        // long pass so a dropped LSB cannot pass as a zero.
        spi_frame(1'b1, 7'h08, 16'd512, rb);
        spi_frame(1'b1, 7'h00, 16'h0001, rb);
        spi_frame(1'b0, 7'h40, 16'h0000, rb);
        if (rb[0] !== 1'b1)
            $fatal(1, "STATUS bit 0 (busy) did not read back: %h", rb);
        wait (done_pin === 1'b1);
        spi_frame(1'b1, 7'h00, 16'h0002, rb);

        // Start a pass. n_weights = 1 keeps it to a single beat, which is
        // enough to see busy rise and done assert; the stubs will kill the
        // simulation if the datapath is genuinely clocked, which is the point.
        spi_frame(1'b1, 7'h08, 16'h0001, rb);
        spi_frame(1'b1, 7'h00, 16'h0001, rb);

        fork
            begin
                repeat (400) @(posedge clk);
                $fatal(1, "pass never completed");
            end
            begin
                // `done` is sticky so the host can poll STATUS for it, which
                // means this must wait on the level, not on an edge.
                wait (done_pin === 1'b1);
                spi_frame(1'b0, 7'h40, 16'h0000, rb);
                if (rb[1] !== 1'b1)
                    $fatal(1, "done must be visible in STATUS, got %h", rb);
                $display("PASS: top smoke test -- SPI config, bound write, status readback, 3 MAC sources, sequencer completed");
                $finish;
            end
        join
    end
endmodule
