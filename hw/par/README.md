# Place and route

Empty until the Innovus flow starts. Following `ece725/digital/gcd`, this
becomes:

```
par/src/INPUT    design and library inputs
par/src/PLUG     flow hooks
par/FLOW/        the Cadence flow kit         (gitignored)
```

What has to land here:

- floorplan and macro placement for the weight SRAM
- power intent for the isolated `VDD_chk` domain, including isolation cells
  and level shifters on every boundary crossing
- pad ring and IO assignment
- timing constraints, STA reports, DRC/LVS results

Nothing below the gate level exists yet. The area numbers in `syn/` are
generic gates from Yosys, which order the variants correctly but are not
µm² and carry no timing.
