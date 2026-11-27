module tb_weight_read_check;
    reg clk = 0;
    always #5 clk = !clk;

    reg rst = 1;
    reg bound_wr_en = 0;
    reg [1:0] bound_wr_addr = 0;
    reg [4:0] bound_wr_emin = 0;
    reg [4:0] bound_wr_emax = 0;
    reg read_valid = 1;
    reg [8:0] weight_addr = 0;
    reg inject_enable = 0;
    reg [21:0] inject_mask = 0;
    reg [8:0]  inject_addr = 0;
    reg        inject_act_region = 0;
    reg [1:0] mac_src = 2'd0;   // 0 sentry, 1 raw, 2 ecc
    reg [15:0] payload = 16'h3d55;
    wire [21:0] sram_word;

    wire [15:0] weight_out, golden_weight, search_weight;
    wire weight_valid, sentry_flag, bound_valid, golden_valid;
    wire ecc_corrected, ecc_uncorrectable, search_flag, search_valid;
    wire weight_matches_golden, search_matches_golden;
    wire inject_hit, search_inject_hit;
    wire search_disagreement;
    wire [4:0] ecc_syndrome;
    wire [21:0] ecc_error_mask;

    secded_encode encode (.data_in(payload), .word_out(sram_word));
    weight_read_check #(.NUM_BOUNDS(3), .ADDR_W(9)) dut (
        .clk(clk), .rst(rst), .bound_wr_en(bound_wr_en), .bound_wr_addr(bound_wr_addr),
        .bound_wr_emin(bound_wr_emin), .bound_wr_emax(bound_wr_emax),
        .read_valid(read_valid), .weight_addr(weight_addr),
        .bound_addr(weight_addr[8:7]), .sram_word(sram_word),
        .inject_enable(inject_enable), .inject_addr(inject_addr),
        .inject_act_region(inject_act_region),
        .inject_mask(inject_mask), .mac_src(mac_src),
        .weight_out(weight_out), .weight_valid(weight_valid), .sentry_flag(sentry_flag),
        .inject_hit(inject_hit),
        .bound_valid(bound_valid), .bound_disagreement(),
        .golden_weight(golden_weight), .golden_valid(golden_valid),
        .weight_matches_golden(weight_matches_golden),
        .ecc_corrected(ecc_corrected), .ecc_uncorrectable(ecc_uncorrectable),
        .ecc_syndrome(ecc_syndrome), .ecc_error_mask(ecc_error_mask)
    );
    weight_read_check #(.NUM_BOUNDS(3), .ADDR_W(9), .TWO_SIDED(1), .REPAIR(1)) search (
        .clk(clk), .rst(rst), .bound_wr_en(bound_wr_en), .bound_wr_addr(bound_wr_addr),
        .bound_wr_emin(bound_wr_emin), .bound_wr_emax(bound_wr_emax),
        .read_valid(read_valid), .weight_addr(weight_addr),
        .bound_addr(weight_addr[8:7]), .sram_word(sram_word),
        .inject_enable(inject_enable), .inject_addr(inject_addr),
        .inject_act_region(inject_act_region),
        .inject_mask(inject_mask), .mac_src(2'd0),
        .weight_out(search_weight), .weight_valid(search_valid), .sentry_flag(search_flag),
        .inject_hit(search_inject_hit),
        .bound_valid(), .bound_disagreement(search_disagreement),
        .golden_weight(), .golden_valid(), .weight_matches_golden(search_matches_golden),
        .ecc_corrected(), .ecc_uncorrectable(),
        .ecc_syndrome(), .ecc_error_mask()
    );

    task write_bound(input [1:0] addr, input [4:0] floor, input [4:0] ceiling);
        @(negedge clk);
        bound_wr_addr = addr;
        bound_wr_emin = floor;
        bound_wr_emax = ceiling;
        bound_wr_en = 1;
        @(negedge clk);
        bound_wr_en = 0;
    endtask

    task check_ecc(input [15:0] value, input [5:0] expected_parity);
        integer first_bit, second_bit;
        payload = value;
        inject_enable = 0;
        inject_mask = 22'h3fffff;
        #1;
        if (sram_word !== {expected_parity, value} ||
            {golden_valid, ecc_corrected, ecc_uncorrectable} !== 3'b100 ||
            golden_weight !== value || ecc_syndrome !== 0 || ecc_error_mask !== 0)
            $fatal(1, "clean SECDED word %h", value);

        inject_enable = 1;
        for (first_bit = 0; first_bit < 22; first_bit++) begin
            inject_mask = 22'd1 << first_bit;
            #1;
            if ({golden_valid, ecc_corrected, ecc_uncorrectable} !== 3'b110 ||
                golden_weight !== value || ecc_error_mask !== inject_mask ||
                ((ecc_syndrome == 0) !== (first_bit == 21)))
                $fatal(1, "single flip at bit %d for %h", first_bit, value);
            for (second_bit = first_bit + 1; second_bit < 22; second_bit++) begin
                inject_mask = (22'd1 << first_bit) | (22'd1 << second_bit);
                #1;
                if ({golden_valid, ecc_corrected, ecc_uncorrectable} !== 3'b001 ||
                    ecc_error_mask !== 0 || ecc_syndrome === 5'd0 || weight_matches_golden)
                    $fatal(1, "double flip at bits %d/%d for %h", first_bit, second_bit, value);
            end
        end
        inject_enable = 0;
    endtask

    initial begin
        @(negedge clk);
        rst = 0;
        #1;
        if ({weight_valid, sentry_flag, bound_valid, search_valid} !== 4'b0 || !golden_valid)
            $fatal(1, "a read before programming must not reach the defended datapath");

        write_bound(0, 13, 17);
        write_bound(1, 13, 14);
        write_bound(2, 13, 17);
        write_bound(3, 13, 17);
        #1;
        if (!weight_valid || sentry_flag || search_flag || weight_out !== payload ||
            !weight_matches_golden || !search_matches_golden)
            $fatal(1, "clean configured read");
        weight_addr = 127;
        #1;
        if (!weight_valid || sentry_flag)
            $fatal(1, "last weight of first group");
        weight_addr = 128;
        #1;
        if (!weight_valid || !sentry_flag)
            $fatal(1, "first weight of next group");
        weight_addr = 383;
        #1;
        if (!weight_valid || sentry_flag)
            $fatal(1, "last valid weight address");
        weight_addr = 384;
        #1;
        if ({weight_valid, sentry_flag, bound_valid, search_valid} !== 4'b0)
            $fatal(1, "out-of-range address must not alias a bound");
        mac_src = 2'd1;
        #1;
        if (!weight_valid || weight_out !== payload || bound_valid || sentry_flag)
            $fatal(1, "raw-source read must pass through without a bound");
        mac_src = 2'd0;
        weight_addr = 0;

        force search.bounds.bank_b_word = 11'd0;
        #1;
        if (!search_disagreement || !search_valid || search_flag || search_weight !== payload)
            $fatal(1, "one bad bound bank must not change the result");
        release search.bounds.bank_b_word;
        #1;
        if (search_disagreement !== 0)
            $fatal(1, "the three unfaulted banks should agree");

        inject_enable = 1;
        inject_mask = 22'h004000;
        #1;
        if (!ecc_corrected || golden_weight !== payload || !sentry_flag ||
            weight_out !== 16'h0155 || !search_flag || search_weight !== payload ||
            weight_matches_golden || !search_matches_golden)
            $fatal(1, "parallel repair and reference after an exponent flip");
        mac_src = 2'd1;
        #1;
        if (weight_out !== 16'h7d55 || !sentry_flag || golden_weight !== payload)
            $fatal(1, "raw source must forward the faulted word and preserve flags");
        // The ECC baseline must be runnable, not just observable.
        mac_src = 2'd2;
        #1;
        if (weight_out !== golden_weight || weight_out !== payload || !ecc_corrected)
            $fatal(1, "the ECC source must deliver the corrected word to the MACs");
        if (!sentry_flag)
            $fatal(1, "the bound check must keep reporting while ECC drives the MACs");
        mac_src = 2'd0;

        payload = 16'h4400;
        #1;
        if (sentry_flag || weight_out !== 16'h0400 ||
            !search_flag || search_weight !== payload)
            $fatal(1, "only the two-sided checker catches a drop below the floor");
        inject_enable = 0;
        payload = 16'h8000;
        #1;
        if (search_flag || search_weight !== payload)
            $fatal(1, "negative zero must pass the exponent floor");

        // The armed site is a single address, not "every read while enabled".
        payload = 16'h3d55;
        inject_enable = 1;
        inject_mask = 22'h004000;
        inject_addr = 0;
        weight_addr = 0;
        #1;
        if (!inject_hit || !sentry_flag || weight_matches_golden)
            $fatal(1, "the armed address must be injected");
        weight_addr = 1;
        #1;
        if (inject_hit || sentry_flag || !weight_matches_golden || weight_out !== payload)
            $fatal(1, "a neighbouring address must be left alone");
        inject_addr = 1;
        #1;
        if (!inject_hit || !sentry_flag)
            $fatal(1, "re-arming must follow the address");
        inject_addr = 300;
        weight_addr = 300;
        inject_act_region = 1;
        #1;
        if (inject_hit || sentry_flag)
            $fatal(1, "an activation-region arm must not fire on a weight read");
        inject_act_region = 0;
        #1;
        if (!inject_hit)
            $fatal(1, "same site, weight region, must fire");
        inject_enable = 0;
        inject_addr = 0;
        weight_addr = 0;
        #1;
        if (inject_hit)
            $fatal(1, "disabled injection must never fire");

        check_ecc(16'h0000, 6'h00);
        check_ecc(16'hffff, 6'h1e);
        check_ecc(16'h3d55, 6'h15);
        check_ecc(16'hb955, 6'h2f);

        read_valid = 0;
        inject_enable = 1;
        inject_mask = 22'h004000;
        #1;
        if ({weight_valid, sentry_flag, bound_valid, golden_valid, ecc_corrected,
             ecc_uncorrectable, ecc_syndrome, ecc_error_mask} !== 33'b0)
            $fatal(1, "invalid SRAM reads must not report events");
        $display("PASS: read path, TMR bound vote, address-gated injection, 88 single flips, 924 double flips");
        $finish;
    end
endmodule
