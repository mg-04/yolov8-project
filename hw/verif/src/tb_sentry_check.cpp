// Drives sentry_check with vectors generated from src/weightSentry.py and
// checks the flag against Python's verdict. Only `flag` is compared: detection
// is the quantity every measurement in this project rests on, while the repair
// path's no-candidate fallback is deliberately simplified in the RTL.
#include "Vsentry_check.h"
#include "verilated.h"
#include <cstdio>

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    if (argc < 2) { fprintf(stderr, "usage: %s <vectors>\n", argv[0]); return 2; }
    FILE *f = fopen(argv[1], "r");
    if (!f) { perror(argv[1]); return 2; }

    Vsentry_check *dut = new Vsentry_check;
    unsigned w, emax, emin, lo, hi, want;
    long n = 0, bad = 0;
    while (fscanf(f, "%x %x %x %x %x %u", &w, &emax, &emin, &lo, &hi, &want) == 6) {
        dut->w_in = w; dut->emax = emax; dut->emin = emin;
        dut->lo = lo;  dut->hi = hi;
        dut->eval();
        ++n;
        if ((unsigned)dut->flag != want) {
            if (bad < 5)
                printf("    MISMATCH w=%08x emax=%02x emin=%02x lo=%08x hi=%08x"
                       "  rtl=%u py=%u\n", w, emax, emin, lo, hi,
                       (unsigned)dut->flag, want);
            ++bad;
        }
    }
    fclose(f);
    printf("%7ld vectors  %ld mismatches  %s\n", n, bad, bad ? "FAIL" : "OK");
    delete dut;
    return bad ? 1 : 0;
}
