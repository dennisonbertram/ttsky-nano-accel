# SPDX-FileCopyrightText: © 2026 Dennison Bertram
# SPDX-License-Identifier: Apache-2.0
#
# Runs the full autoregressive generation against the behavioral PSRAM model
# (preloaded with psram.hex, built by tt/sw/pack.py) and checks every token
# bit-exactly against the golden integer model (expected.hex).

from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles


def load_expected():
    path = Path(__file__).parent / "expected.hex"
    return [int(line, 16) for line in path.read_text().split()]


@cocotb.test()
async def test_generate(dut):
    expected = load_expected()
    dut._log.info(f"expecting {len(expected)} tokens")

    clock = Clock(dut.clk, 15, unit="ns")  # ~66 MHz
    cocotb.start_soon(clock.start())

    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1

    got = []
    for i, exp in enumerate(expected):
        # wait for VALID (uo[7]); poll in chunks to keep the sim fast
        for _ in range(20000):
            await ClockCycles(dut.clk, 100)
            if (dut.uo_out.value.to_unsigned() >> 7) & 1:
                break
        else:
            raise AssertionError(f"timeout waiting for token {i}")

        tok = dut.uo_out.value.to_unsigned() & 0x7F
        got.append(tok)
        dut._log.info(f"token {i}: {tok:#04x} {chr(tok)!r} (expect {exp:#04x})")
        assert tok == exp, f"token {i}: got {tok:#04x}, expected {exp:#04x}"

        # ACK handshake on ui[0]
        dut.ui_in.value = 1
        await ClockCycles(dut.clk, 4)
        dut.ui_in.value = 0
        await ClockCycles(dut.clk, 4)

    text = "".join(chr(t) for t in got)
    dut._log.info(f"generated: {text!r} — bit-exact match")
