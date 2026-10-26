# Partitioned execution: moving layers across a trust boundary

The fault-injection results reduce to one actionable claim: a small, dense set of
high-magnitude weights carries disproportionate adversarial risk, and two of the
layers holding them sit at the network's input and output boundaries. This directory
builds the machinery to actually run those layers somewhere else — eventually on FPGA
fabric the host cannot address.

**The point is not speed.** It is to prove the cut is *numerically invisible* before
any hardware exists. A partition that changes the output by one ULP is a partition
whose later FPGA kernel cannot be validated, because you no longer have an oracle.

Library: [../../src/shield.py](../../src/shield.py). Test: `split_test.py`.

## The ladder

Each rung changes exactly one thing, so each can fail for exactly one reason.

| rung | where the layers run | transport | status |
|---|---|---|---|
| **1a** | same process, same GPU | `LocalTransport` (a function call) | **passing**, bit-exact |
| **1b** | same process, over a loopback socket | `SocketTransport("127.0.0.1")` | **passing**, bit-exact |
| **2** | a second machine on the LAN | `SocketTransport("10.0.0.2")` | not run — needs Ethernet to the Jetson |
| **3a** | the FPGA board's ARM cores (PS) | same socket, `device="cpu"` | not run — needs a board |
| **3b** | FPGA fabric (PL) | XDMA over PCIe | not started |

Rungs 1a and 1b are what `split_test.py` runs today.

## Files

| file | what it is |
|---|---|
| [../../src/shield.py](../../src/shield.py) | the library: one stub, interchangeable transports, the surgery |
| `split_test.py` | runs rungs 1a and 1b and demands bit-identical detections. Exits 0/1 |

```bash
./analysis/partition/split_test.py          # 64 images
./analysis/partition/split_test.py 256
```

## `shield.py`

Five pieces. The design goal is that the model code never changes — ultralytics must
not be able to tell a partitioned model from a whole one.

### `Remote(nn.Module)` — the stub

A stand-in that sits in the slot where a module used to be, looks like a module to
everything around it, and does no work:

```python
class Remote(nn.Module):
    def forward(self, x):
        return self.transport.call(self.tag, x)
```

No weights, no convolution. `_predict_once` does `x = m(x)` and gets a tensor back; it
cannot tell whether `m` computed it or phoned it in. Same for `Detect` calling
`self.dfl(...)`.

### `split(core, tags) -> {tag: real_module}` — the surgery

**Mutates `core` in place.** For each tag it resolves the dotted path, replaces the
module with a `Remote`, and returns the evicted originals.

```python
real = shield.split(core, ["model.0", "model.22.dfl"])
# core.model[0] is now a Remote; the real Conv only exists in `real`
```

The same function serves both sides of the cut. On the client you keep `core` and
discard the return value's contents; on the server you keep the return value and
discard `core`.

**The one subtle part.** `BaseModel._predict_once` routes the graph with each module's
own attributes:

```python
for m in self.model:
    if m.f != -1:
        x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
    x = m(x)
    y.append(x if m.i in self.save else None)
```

- `.f` (*from*) — `-1` for the previous layer, an int for layer N, or a **list**
  (`model.11` has `f=[-1, 6]`, concatenating the previous output with layer 6's).
- `.i` (*index*) — decides whether the output is saved for a later skip connection.

A bare `nn.Module` has neither, so a naive swap raises `AttributeError` or, worse,
**silently reroutes the graph**. `_GRAPH_ATTRS = ("f", "i", "type", "np")` carries them
across, guarded by `hasattr`. `.type` and `.np` are cosmetic (used by `model.info()`);
`.np` does not exist in the pinned ultralytics version, so that one is a no-op.

**`.forward` is deliberately not copied.** `fuse()` rebinds `forward` as an *instance*
attribute pointing at `forward_fuse`. The real module keeps that binding and carries it
to the server, which is correct. If the stub inherited it, the stub would run the
original Conv locally and ignore the transport entirely — a partition that silently
does not partition.

### `attach(core, transport, tags=None) -> n` — wiring the destination

`split()` creates stubs with `transport=None`, so they are dead until attached. This
walks the module tree, finds every `Remote`, and sets one attribute:

```python
shield.attach(core, shield.LocalTransport(real))                 # 1a
shield.attach(core, shield.SocketTransport("127.0.0.1", 9801))   # 1b
```

It is separate from `split()` because the two answer different questions at different
times. `split()` says **what** moves — a property of the model, identical on both
machines. `attach()` says **where it goes** — client-only, and the destination does not
exist yet when you split (you must split before starting the server, because the server
needs `real`).

`tags=None` routes every stub to one place. Pass a list to route selectively:

```python
shield.attach(core, fpga,    tags=["model.0"])
shield.attach(core, enclave, tags=["model.22.dfl"])
```

### Transports

```python
class LocalTransport:                      # rung 1a
    def call(self, tag, x):
        return self.modules[tag](x)        # dict lookup + call. Nothing leaves the GPU.

class SocketTransport:                     # rungs 1b and 2
    def call(self, tag, x):
        _sendall(self.sock, pack(tag, x))
        _, y = recv(self.sock)
        return y.to(self.device)
```

`call()` is **synchronous**: `x = m(x)` does not return until the remote module has
finished and the result is back on the GPU. That is deliberate for the test — it makes
the result deterministic and comparable to golden — and it is the first thing to break
in a real deployment.

`device` on the client is where results land (`"cuda"` on a Jetson); `device` on
`serve()` is where the real modules run (`"cpu"` on a board with no CUDA).

### `pack` / `recv` — the wire format

A socket carries bytes, so sending a tensor means writing down the dtype, rank, shape
and payload length explicitly. Disagreement between the two ends produces wrong numbers
with no error, which is exactly the class of bug rung 1b exists to catch.

```
<H  tag length        <B  ndim        <B  dtype code        <ndim i  shape
<Q  payload bytes     tag             payload
```

Two details that matter:

- **`.contiguous()` in `pack`.** A transposed view's storage order is not its logical
  order, and `.numpy().tobytes()` would ship the wrong bytes without complaint.
- **`_recvall` loops.** `socket.recv()` may return short reads, and a tensor payload
  here is 6.5 MB.

### `serve(modules, host, port, device)`

Holds the real modules and answers calls until the client disconnects. The tag travels
with the tensor because one socket serves every cut, and `modules[tag]` is how the
server knows whether this 4.9 MB is for `model.0` or `model.22.dfl`.

## How `split_test.py` works

Three runs over the same images, demanding bit-identical detections:

```
core = load()                  ->  GOLD          23 real modules
core = load(); split; attach   ->  rung 1a       21 real + 2 stubs, local
core = load(); split; serve; attach -> rung 1b    21 real + 2 stubs, socket
```

`load()` is called fresh each time because `split()` mutates in place — reusing the
golden `core` would compare a partitioned model against itself and pass with zero
information.

`detect_all` is byte-for-byte the same code in all three runs, so any difference
between `GOLD` and a test run can only come from the two cut slots. It calls
`fi_lib.make_detector(core)`, which drives `core` directly rather than
`YOLO.predict()` — `predict()` wraps the model in `AutoBackend`, whose internal
modules are *different objects*, so the stubs would be silently discarded and the test
would validate nothing.

Rung 1b fakes two machines inside one process by running `serve` in a daemon thread.
`serve` is an infinite loop (`while True: srv.accept()`), so calling it directly would
hang before reaching the client code.

### Per frame, in rung 1b

| step | code | bytes |
|---|---|---|
| `_predict_once` iteration 0 hits the stub | `tasks.py:198` | |
| client `pack` → `.cpu()` → `sendall` | `shield.py:93` | 4.92 MB out |
| server `recv` → `.to(device)` → real Conv | `shield.py:179` | |
| server `pack` → `sendall` | `shield.py:182` | 6.55 MB back |
| iterations 1–21 run locally on the GPU | | |
| iteration 22 `Detect` calls `self.dfl(...)`, hits the second stub | `head.py:215` | 2.15 MB out, 0.13 MB back |

**2 round trips, 4 tensors, 13.75 MB per frame.** Note the second round trip happens
*inside* a module's forward rather than at the top-level loop; `Remote` looks like a
module from any call site, so the partition does not care.

## Measured: rung 1

64 images, `coco-val2017-sub1024.yaml`, RTX 4070.

| | ms/image | delta | reading |
|---|---|---|---|
| monolithic | 2.60 | — | |
| 1a in-process | 2.54 | **−0.06** | the cut itself is free |
| 1b loopback socket | 12.78 | **+10.24** | the cost of serialisation |

Bit-exact **301/301** detections on both rungs; zero count mismatches, `max|box|` and
`max|conf|` literally 0.0.

The −0.06 ms is noise, and it is the result that matters before building hardware: an
extra Python dispatch per cut costs nothing. The +10.24 ms is 13.75 MB through CPU
memory twice, i.e. **1.34 GB/s** on loopback, and it is an upper bound on the real
penalty — a deployment pipelines instead of blocking.

## Choosing the cut points

`TAGS` in `split_test.py`. Tags are dotted paths resolvable from `core`; `split()`
handles both list indices and attribute names:

```python
"model.0"            # core.model[0]
"model.22.dfl"       # core.model[22].dfl
"model.22.dfl.conv"  # nested deeper
```

**The constraint is that forward must be `tensor -> tensor`**, because the wire
protocol ships one tensor each way. For YOLOv8n that leaves **81 splittable modules**
and 5 that are not:

| not splittable | why |
|---|---|
| `model.11`, `model.14`, `model.17`, `model.20` | `Concat` — takes a *list* of tensors |
| `model.22` | `Detect` — list in, tuple out |

Which is why the head cut is `model.22.dfl` rather than `model.22`. Splitting a Concat
or the whole head would need a list-of-tensors protocol.

To enumerate candidates: `[n for n, _ in core.named_modules()]`.

### Why these two

| layer | params | `\|w\| >= 2` | density |
|---|---|---|---|
| `model.0.conv` | 432 w + 16 b | 187 weights | **43.3%** |
| `model.22.dfl.conv` | 16 | 14 | **87.5%** |

These are the only two layers where high-magnitude weights are *dense* rather than
outliers, and both sit at a network boundary — so the trust crossing coincides with
I/O that happens anyway, rather than forcing a round trip through the middle of the
network. Cost is 448 parameters, **1.8 KB, 1.02% of FLOPs**.

**Be careful with the coverage claim.** Against the 453-value weight-only target set
these two layers cover 201, or 44%. Against the full target set — **1,481 values, of
which 1,028 (69.4%) are biases** — they cover only **14%**. The harness samples all
parameters, so 14% is the honest number for random-fault coverage. The 44% figure is
only valid for a weight-only threat model. See
[../multibit/README.md](../multibit/README.md) for the bias/weight pool split.

## Rung 2: a second machine

One argument changes on the client. On the server, build the model locally and pull the
same two modules out:

```python
# server
m = YOLO("yolov8n.pt"); m.model.fuse()
core = m.model.cpu().eval()
real = shield.split(core, ["model.0", "model.22.dfl"])
del core
shield.serve(real, host="0.0.0.0", port=9801, device="cpu")

# client
tr = shield.SocketTransport("10.0.0.2", 9801)
```

**New failure mode rung 1b cannot catch.** In 1b both halves are in one process, so
`real` and the server's `modules` are the *same objects in the same GPU memory* —
`.to()` on an already-resident module returns the same object. Nothing is really
loaded, so 1b tests activation transfer only. At rung 2 the server builds its own
model, and if it forgets `.fuse()` or loads a different checkpoint, the weights
silently differ. That is what rung 2's bit-exactness check is really guarding.

Bandwidth, using the **marginal** boundary traffic of 8.84 MB/frame (total 13.75 MB
less the 4.92 MB input image, which has to reach the accelerator anyway):

| link | throughput | fps, pipelined |
|---|---|---|
| 1 GbE | 125 MB/s | 14 |
| PCIe Gen3 ×4 | 3.94 GB/s theoretical | ~446 |
| PCIe Gen3 ×4 | 3.0–3.5 GB/s realistic | 340–396 |

Serialised rather than pipelined, holding a 10% latency budget, needs ~6.7 GB/s — so
pipelining is not optional past Gigabit Ethernet.

## Rung 3a: the board's ARM cores

A ZCU104 / ZCU102 / KV260 is a **ZynqMP**: ARM Cortex-A53 cores (the PS, processing
system) beside FPGA fabric (the PL, programmable logic) on one die. The PS runs
PetaLinux with real Python and PyTorch, so the rung-2 server script above runs
unchanged with `device="cpu"`.

Still bit-exact — same PyTorch kernels, just on A53 instead of x86. **Take this rung.**
It is the first point where the weights genuinely live on a separate chip, it buys the
whole networking, addressing and deployment story before any HDL exists, and it is an
afternoon's work.

## Rung 3b: the kernel in fabric

Here the FPGA side stops being a program and becomes three separate things.

**1. The DMA bridge.** The host's PCIe talks to an XDMA (or QDMA) IP block
instantiated in the fabric. On the host it appears as character devices:

```
/dev/xdma0_h2c_0     host -> card, activations in
/dev/xdma0_c2h_0     card -> host, results out
/dev/xdma0_user      control registers (start, done, config)
```

A fourth transport implements the same two-method interface, so nothing above it
changes:

```python
class XdmaTransport:
    def call(self, tag, x):
        self.h2c.write(pack(tag, x)); ...
```

**2. The kernel.** `model.0` is `Conv2d(3, 16, k=3, s=2)` + SiLU with BatchNorm
already fused — 432 weights, 16 biases, 44.2 M MACs. Written in HLS (C++ that Vitis
compiles to RTL), with the shape baked in at compile time and the weights living in
BRAM:

```cpp
void conv0(const float in[3*640*640], float out[16*320*320]) {
  #pragma HLS INTERFACE m_axi port=in  bundle=gmem0
  #pragma HLS INTERFACE m_axi port=out bundle=gmem1
  static const float W[16][3][3][3] = { /* 432 values */ };
  static const float B[16]          = { /* 16 values  */ };
  for (int oc = 0; oc < 16; oc++)
    for (int oy = 0; oy < 320; oy++)
      for (int ox = 0; ox < 320; ox++) {
        #pragma HLS PIPELINE II=1
        float a = B[oc];
        for (int ic = 0; ic < 3; ic++)
          for (int ky = 0; ky < 3; ky++)
            for (int kx = 0; kx < 3; kx++) { /* stride 2, pad 1 */ }
        out[...] = a / (1.0f + expf(-a));          // SiLU
      }
}
```

`dfl` is smaller still: a softmax over 16 channels and a fixed 1×1 projection holding
the integers 0–15, 537 K MACs.

**3. Control.** A register poke to start, a done bit to poll or an interrupt, and the
weights loaded once at bitstream configuration — not per frame.

### Two things to expect

**It will be I/O-bound, not compute-bound.** Both kernels together are 44.7 M MACs.
At 200 MHz with 256 MACs/cycle that is ~0.9 ms, against ~2.2–2.9 ms just to move
8.84 MB over Gen3 ×4. The FPGA is not buying throughput here — **it is buying a trust
domain the host cannot address.** That is the research claim, and it should be stated
as such rather than dressed up as an accelerator result.

**Bit-exactness ends at this rung.** HLS floating-point operations are individually
IEEE-754 compliant, but a hardware pipeline accumulates in a different *order* than
cuDNN, and fp32 addition is not associative. Expect differences in the last few
mantissa bits. This is precisely why rungs 1 and 2 must be exact: when rung 3b differs,
the only way to attribute the difference to the kernel rather than to the plumbing is
to know the plumbing contributes nothing.

### Blocked on

**Which board ECE provides.** ZCU104 / ZCU102 / KV260 are ZynqMP and take the 3a→3b
path above. An Alveo is a PCIe card and skips 3a entirely, but the Jetson AGX Xavier
devkit's 65 W supply rules out a 75 W+ card without an external PSU.

Jetson side, verified: kernel `5.10.216-tegra`, 6 PCIe controllers with C5
(`pcie@141a0000`, ×8) enabled, `CONFIG_PCIE_TEGRA194_HOST=y`,
`CONFIG_PCI_ENDPOINT=y`, `CONFIG_ARM_SMMU=y`. PCIe works (a Marvell 9171 SATA
controller enumerates). **No XDMA driver is present** — it has to be built against the
Tegra kernel. `eth0` is DOWN with no cable, so the only current path is the USB
`l4tbr0` link at 192.168.55.1, which is why rung 2 needs Ethernet before anything else.

## Known gaps

- **`compare()` passes a class-label flip.** [split_test.py:72](split_test.py#L72)
  skips a row whose class differs without touching `dbox`/`dconf`, so `ok` stays
  `True` while `exact < nb` — the printout would read `bit-exact 299/301` next to
  `PASS`. Tightening the predicate to `ok = nd == 0 and exact == nb` closes it; the
  counter is already there, it is just not in the condition.
- **The `TAGS` comment at [split_test.py:34](split_test.py#L34) is wrong.** It reads
  "`model.0` holds 43.3% of the `|w| >= 2` population", conflating density with
  coverage. 43.3% is the density *within* the layer; coverage of the full 1,481-value
  target set is 13.1%.
- No standalone server script yet — rung 2 needs `shield_server.py` with
  `device="cpu"`.
- `_DTYPES` covers fp32 / fp16 / int8 only. Quantised deployment would need more.
- The transport is one blocking round trip per cut. Pipelining is a separate piece of
  work and is what the fps figures above assume.
