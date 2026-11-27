// Included by every stub, after it defines `stub_active`. A stub that is
// actually exercised must be impossible to miss, so it kills the simulation
// instead of returning a plausible value. Reset and configuration still work,
// so the top can be brought up and smoke-tested with stubs in place.
// +define+STUB_WARN downgrades the kill to a once-per-stub warning, for
// integration bring-up.
`ifndef SYNTHESIS
reg stub_seen = 1'b0;
always @(posedge clk) begin
    if (!rst && stub_active && !stub_seen) begin
        stub_seen <= 1'b1;
`ifdef STUB_WARN
        $display("STUB EXERCISED: %m at %0t (continuing, outputs are X)", $time);
`else
        $fatal(1, "STUB EXERCISED: %m at %0t -- not implemented", $time);
`endif
    end
end
`endif
