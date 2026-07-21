/*
 * tt_um_dennisonbertram_nano_accel — Tiny Tapeout top for the streaming
 * nano-LM accelerator. See nano_tt_core.v for the architecture and
 * tt/sw/pack.py for the PSRAM image the QSPI Pmod must be preloaded with.
 *
 * Pinout:
 *   ui[0]  ACK        host handshake: rising edge consumes the current token
 *   ui[1]  FREE_RUN   1 = generate at full speed without waiting for ACK
 *   ui[2]  SLOW_BOOT  1 = header read uses QSPI sample latency 0 (slow clk)
 *   uo[6:0] TOKEN     generated token (vocab is 128, tokens are 7-bit ASCII)
 *   uo[7]  VALID      high while TOKEN holds an unconsumed token
 *   uio[*] QSPI Pmod (CS0 flash / SD0 / SD1 / SCK / SD2 / SD3 / CS1 / CS2)
 *          — model data streams from PSRAM A (CS1)
 *
 * SPDX-License-Identifier: Apache-2.0
 */

`default_nettype none

module tt_um_dennisonbertram_nano_accel (
    input  wire [7:0] ui_in,    // Dedicated inputs
    output wire [7:0] uo_out,   // Dedicated outputs
    input  wire [7:0] uio_in,   // IOs: Input path
    output wire [7:0] uio_out,  // IOs: Output path
    output wire [7:0] uio_oe,   // IOs: Enable path (active high: 0=input, 1=output)
    input  wire       ena,      // always 1 when the design is powered, so you can ignore it
    input  wire       clk,      // clock
    input  wire       rst_n     // reset_n - low to reset
);
    wire        qs_start, qs_wr, qs_busy;
    wire [22:0] qs_addr;
    wire [6:0]  qs_len;
    wire [1:0]  qs_lat;
    wire        rvalid, wnext;
    wire [7:0]  rdata, wdata;
    wire        tok_valid;
    wire [6:0]  tok_out;

    wire       sck, cs_n;
    wire [3:0] sd_out, sd_oe;
    wire [3:0] sd_in = {uio_in[5], uio_in[4], uio_in[2], uio_in[1]};

    nano_tt_core core (
        .clk(clk), .rst_n(rst_n),
        .ack(ui_in[0]), .free_run(ui_in[1]), .slow_boot(ui_in[2]),
        .tok_valid(tok_valid), .tok_out(tok_out),
        .qs_start(qs_start), .qs_wr(qs_wr),
        .qs_addr(qs_addr), .qs_len(qs_len), .qs_lat(qs_lat),
        .qs_busy(qs_busy),
        .rvalid(rvalid), .rdata(rdata),
        .wnext(wnext), .wdata(wdata)
    );

    qspi_ctrl qspi (
        .clk(clk), .rst_n(rst_n),
        .start(qs_start), .wr(qs_wr), .addr(qs_addr), .len(qs_len),
        .lat(qs_lat), .busy(qs_busy),
        .rvalid(rvalid), .rdata(rdata),
        .wnext(wnext), .wdata(wdata),
        .sck(sck), .cs_n(cs_n),
        .sd_out(sd_out), .sd_oe(sd_oe), .sd_in(sd_in)
    );

    assign uo_out = {tok_valid, tok_out};

    //           CS2   CS1   SD3        SD2        SCK  SD1        SD0        CS0
    assign uio_out = {1'b1, cs_n, sd_out[3], sd_out[2], sck, sd_out[1], sd_out[0], 1'b1};
    assign uio_oe  = {1'b1, 1'b1, sd_oe[3],  sd_oe[2],  1'b1, sd_oe[1], sd_oe[0],  1'b1};

    wire _unused = &{ena, ui_in[7:3], uio_in[7:6], uio_in[3], uio_in[0], 1'b0};
endmodule
