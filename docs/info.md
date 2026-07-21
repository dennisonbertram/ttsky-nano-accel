<!---
This file is used to generate your project datasheet.
-->

## How it works

This is a streaming port of [nano_accel](https://github.com/dennisonbertram) — an
INT4 language-model inference accelerator — squeezed into Tiny Tapeout by moving
every model tensor off-chip into the QSPI PSRAM Pmod.

It runs a char-level MLP (context 8, embed 24, two hidden layers, vocab 128)
fully autoregressively: gather the embeddings of the last 8 tokens, run three
INT4-weight / INT8-activation matvec layers, take the argmax as the next token,
feed it back, repeat. The numerics contract per output channel is

    y = clamp(round((acc + bias) * M[c]) >> sh)

with per-channel fixed-point scales M[c] precomputed by the quantizing
compiler (`sw/build.py` + `tt/sw/pack.py` in the parent repo).

Because a single shared QSPI bus delivers at most one INT4 weight per SCK
cycle, a wide MAC array would starve: this design uses **one** INT4xINT8
multiplier feeding 8 accumulators (one weight block streams 8 output channels
interleaved), and the drain multiply is bit-serial — the whole datapath runs
at QSPI line rate. Activations spill to a PSRAM scratch region; the head
layer fuses argmax into the drain so no logit buffer exists. The model's
dimensions, layer shifts, segment addresses, prompt, and QSPI read-sampling
latency all come from a 64-byte header in the PSRAM image, so one chip can
run any model that fits in 8 MB (the parent repo's default model is 208 KiB;
the checked-in test image is a 28 KiB variant).

Measured in simulation: the 28 KiB test model runs 232,159 clocks/token
(~205 tokens/s at the 47.6 MHz nominal clock); the parent repo's full
208 KiB model (hidden 512x2) runs 1,520,137 clocks/token (~31 tokens/s).
Both are bit-exact against the golden integer model, and both are faster
than a human reads.

## How to test

1. Build a PSRAM image: in the parent repo, `make vectors` then
   `python3 tt/sw/pack.py` (or use the checked-in `test/psram.hex`).
2. Preload PSRAM A on the QSPI Pmod with the image (e.g. from the demo board
   RP2040, with the design held in reset; leave the PSRAM in SPI mode).
3. Release reset with `ui[1:0] = 00`. The design reads the image header and
   starts generating.
4. When `uo[7]` (VALID) rises, `uo[6:0]` holds the next ASCII token. Raise
   `ui[0]` (ACK) to consume it; VALID drops, then rises with the next token.
   Set `ui[1]` (FREE_RUN) to generate at full speed without handshaking.
5. After the header-configured number of tokens the design halts with VALID
   low. Compare the token stream against `test/expected.hex` — it matches the
   golden integer model bit-for-bit.

Clocking: 47.6 MHz (21 ns) nominal. Keep clk >= ~37 MHz so QSPI bursts respect the
PSRAM's 8 us CS-low refresh limit (tCEM). For slow bring-up clocks, assert
`ui[2]` (SLOW_BOOT) so the header read samples with latency 0, and set header
byte 52 to 0 in the image; at speed, leave `ui[2]` low and header latency 1.

## External hardware

[QSPI Pmod](https://github.com/mole99/qspi-pmod) on the bidirectional pins,
using PSRAM A (CS1) only. A host (demo board RP2040) preloads the PSRAM image
before releasing reset.
