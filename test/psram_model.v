/*
 * psram_model.v — behavioral APS6404L PSRAM for simulation.
 * Implements exactly the SPI-mode command subset qspi_ctrl.v uses:
 *   0xEB fast read quad: serial cmd, quad addr, 6 dummy, quad out
 *   0x38 quad write:     serial cmd, quad addr, quad in
 * Per the datasheet, read data launches tACLK after SCK FALLING edges (high
 * nibble first); DELAY_NS models tACLK plus the TT mux round trip, sized so
 * the controller's sample latency of 1 SCK cycle is exercised in sim.
 * Bursts wrap inside 1 KiB pages like the real chip.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

`default_nettype none
`timescale 1ns / 1ps

module psram_model #(
    parameter INIT_FILE = "psram.hex",
    parameter DELAY_NS  = 20       // tACLK + TT mux round trip; sized for the
                                   // 15ns-clk / 30ns-SCK testbenches so lat=1
                                   // sampling has margin on both sides
) (
    input  wire       sck,
    input  wire       cs_n,
    input  wire [3:0] sd_in,     // master-driven values
    output reg  [3:0] sd_q,      // model-driven values
    output reg        sd_drive   // model is driving SD[3:0]
);
    reg [7:0] mem [0:8388607];   // 8 MB
    initial $readmemh(INIT_FILE, mem);

    reg [7:0]  cmd;
    reg [22:0] addr;
    reg [15:0] rise;             // rising-edge counter within CS-low
    reg [3:0]  hi_nib;
    reg        half;

    // command / address / write data are sampled on rising edges
    always @(posedge sck or posedge cs_n) begin
        if (cs_n) begin
            rise <= 0; half <= 0; cmd <= 0; addr <= 0;
        end else begin
            rise <= rise + 16'd1;
            if (rise < 8) begin
                cmd <= {cmd[6:0], sd_in[0]};
            end else if (rise < 14) begin
                addr <= {addr[18:0], sd_in};   // 6 nibbles, 24b (top bit drops)
            end else if (cmd == 8'h38) begin
                if (!half) begin
                    hi_nib <= sd_in;
                    half <= 1;
                end else begin
                    mem[addr] <= {hi_nib, sd_in};
                    addr <= {addr[22:10], addr[9:0] + 10'd1}; // 1K page wrap
                    half <= 0;
                end
            end
        end
    end

    // read data launches DELAY_NS after falling edges: nibble j goes out on
    // the falling edge of SCK cycle 19+j (cycles 14..19 are the 6 dummies,
    // whose last falling edge already carries the first nibble).
    reg        rd_half;
    always @(negedge sck or posedge cs_n) begin
        if (cs_n) begin
            sd_drive <= 0;
            rd_half  <= 0;
            sd_q     <= 4'hx;
        end else if (cmd == 8'hEB && rise >= 16'd20) begin
            sd_drive <= #(DELAY_NS) 1;
            if (!rd_half) begin
                sd_q    <= #(DELAY_NS) mem[addr][7:4];  // high nibble first
                rd_half <= 1;
            end else begin
                sd_q    <= #(DELAY_NS) mem[addr][3:0];
                addr    <= {addr[22:10], addr[9:0] + 10'd1};
                rd_half <= 0;
            end
        end
    end
endmodule
