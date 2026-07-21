#!/usr/bin/env python3
"""pack.py - pack out/*.hex into a single PSRAM image for the Tiny Tapeout
streaming variant of nano_accel (tt/src/).

Everything (dims, shifts, prompt, token count) is recovered from the hex
files emitted by sw/build.py, so this stays decoupled from build.py.

PSRAM image layout (all little-endian, regions 64-byte aligned):
  0x00   header (64 B):
         u16 G, u16 n1(=C*E), u16 H, u16 V, u8 E, u8 sh1, u8 sh2, u8 sh3,
         u32 emb_base, u32 w1_base, u32 w2_base, u32 w3_base, u32 bm_base,
         u32 x_base, u32 h1_base, u32 h2_base, u8 prompt[8],
         u8 qspi_read_latency, pad to 64
  emb    V rows, stride 32 B (E used, rest zero) -> row addr = base + tok*32
  wN     per block of CH output channels, per chunk of K inputs:
         K*CH nibbles, channel-fastest (nibble seq = w[c0][k],w[c0+1][k],...),
         packed low-nibble-first into bytes
  bm     per channel, 8 B: i32 bias, u16 M, u16 pad  (L1, L2, L3 contiguous)
  x/h1/h2  activation scratch (zeroed)

Page-safety: every burst the RTL issues inside these regions is aligned so it
never crosses a 1 KiB PSRAM page (see asserts below).
"""
import argparse, os, sys

CH, K = 8, 16          # must match localparams in tt/src/nano_tt_core.v

ap = argparse.ArgumentParser()
ap.add_argument("--hexdir", default="out")
ap.add_argument("--out", default="tt/test/psram.hex")
ap.add_argument("--lat", type=int, default=1,
                help="QSPI read sample latency in SCK cycles (header byte 52)")
args = ap.parse_args()
hd = args.hexdir

def lines(name):
    with open(os.path.join(hd, name)) as f:
        return [l.strip() for l in f if l.strip()]

def s4(v):  return v - 16 if v >= 8 else v          # signed INT4
def s8(v):  return v - 256 if v >= 128 else v       # signed INT8
def s32(v): return v - (1 << 32) if v >= (1 << 31) else v

# ---------------- recover program parameters from isram.hex ----------------
isram = [int(l, 16) for l in lines("isram.hex")]
def field(w, lo, bits): return (w >> lo) & ((1 << bits) - 1)
assert field(isram[0], 0, 4) == 1, "isram[0] must be SETLEN"
gather, l1, l2, head = isram[1], isram[2], isram[3], isram[4]
assert field(gather, 0, 4) == 2 and field(l1, 0, 4) == 3
C_, E_ = field(gather, 16, 16), field(gather, 32, 16)
n1, H  = field(l1, 16, 16), field(l1, 32, 16)
V      = field(head, 32, 16)
sh1, sh2, sh3 = field(l1, 8, 5), field(l2, 8, 5), field(head, 8, 5)
G = (len(isram) - 2) // 5
assert len(isram) == 2 + 5 * G and field(isram[-1], 0, 4) == 7
assert n1 == C_ * E_ and C_ == 8 and E_ <= 32
assert n1 % K == 0 and H % K == 0 and H % CH == 0 and V % CH == 0

# ---------------- recover tensors ----------------
# wsram.hex: rows (output channels) of n nibbles, element k at nibble k
wwords = lines("wsram.hex")
def take_rows(words, n, m):
    wpr = n // 64                                   # 256-bit words per row
    rows = []
    for c in range(m):
        nibs = []
        for kw in range(wpr):
            w = int(words[c * wpr + kw], 16)
            nibs += [s4((w >> (4 * i)) & 0xF) for i in range(64)]
        rows.append(nibs)
    return rows, words[m * wpr:]
W1, rest = take_rows(wwords, n1, H)                 # W1[c][k]
W2, rest = take_rows(rest, H, H)
W3, rest = take_rows(rest, H, V)
assert not rest

flat = []
for l in lines("esram.hex"):
    w = int(l, 16)
    flat += [s8((w >> (8 * i)) & 0xFF) for i in range(4)]
emb = [flat[t * E_:(t + 1) * E_] for t in range(V)]

biases = [s32(int(l, 16)) for l in lines("bsram.hex")]
scales = [int(l, 16) for l in lines("msram.hex")]
b1, b2, b3 = biases[:H], biases[H:2 * H], biases[2 * H:2 * H + V]
M1, M2, M3 = scales[:H], scales[H:2 * H], scales[2 * H:2 * H + V]
prompt = [int(l, 16) for l in lines("tsram.hex")]
expected = [int(l, 16) for l in lines("expected.hex")]
assert len(prompt) == C_ and len(expected) == G

# ---------------- build image ----------------
img = bytearray()
def align(a):
    while len(img) % a: img.append(0)
def u16(v): img.extend([v & 0xFF, (v >> 8) & 0xFF])
def u32(v): img.extend([(v >> (8 * i)) & 0xFF for i in range(4)])

def w_stream(W, n, m):
    """blocked, chunked, channel-fastest nibble stream -> bytes"""
    nibs = []
    for c0 in range(0, m, CH):
        for k in range(n):                          # chunk order == k order
            for c in range(c0, c0 + CH):
                nibs.append(W[c][k] & 0xF)
    return bytes((nibs[i] | (nibs[i + 1] << 4)) for i in range(0, len(nibs), 2))

streams = [w_stream(W1, n1, H), w_stream(W2, H, H), w_stream(W3, H, V)]

img.extend(b"\x00" * 64)                            # header placeholder
emb_base = len(img)
for t in range(V):
    row = bytes(v & 0xFF for v in emb[t]) + b"\x00" * (32 - E_)
    img.extend(row)
w_bases = []
for s in streams:
    align(64); w_bases.append(len(img)); img.extend(s)
align(64); bm_base = len(img)
for b, M in ((b1, M1), (b2, M2), (b3, M3)):
    for c in range(len(b)):
        u32(b[c]); u16(M[c]); u16(0)
align(256); x_base = len(img); img.extend(b"\x00" * n1)
align(64);  h1_base = len(img); img.extend(b"\x00" * H)
align(64);  h2_base = len(img); img.extend(b"\x00" * H)
align(64)

hdr = bytearray()
img2, img = img, hdr                                # redirect emit helpers
u16(G); u16(n1); u16(H); u16(V)
img.extend([E_, sh1, sh2, sh3])
for v in [emb_base] + w_bases + [bm_base, x_base, h1_base, h2_base]: u32(v)
img.extend(prompt)
img.append(args.lat & 3)          # byte 52: QSPI read sample latency
img.extend(b"\x00" * (64 - len(img)))
img = img2
img[:64] = hdr

# page-safety: no RTL burst may cross a 1 KiB boundary
assert emb_base % 32 == 0
for wb in w_bases: assert wb % 64 == 0
assert bm_base % 8 == 0 and x_base % 256 == 0
assert x_base // 1024 == (x_base + n1 - 1) // 1024
assert h1_base % 64 == 0 and h2_base % 64 == 0

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as f:
    f.write("\n".join("%02x" % b for b in img) + "\n")
with open(os.path.join(os.path.dirname(args.out), "expected.hex"), "w") as f:
    f.write("\n".join("%02x" % t for t in expected) + "\n")

print(f"[pack] dims: C={C_} E={E_} n1={n1} H={H} V={V}  sh={sh1}/{sh2}/{sh3}  G={G}")
print(f"[pack] emb@{emb_base:#x} w1@{w_bases[0]:#x} w2@{w_bases[1]:#x} "
      f"w3@{w_bases[2]:#x} bm@{bm_base:#x} x@{x_base:#x} h1@{h1_base:#x} h2@{h2_base:#x}")
print(f"[pack] image {len(img)} bytes -> {args.out}")
