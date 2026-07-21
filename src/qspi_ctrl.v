/*
 * qspi_ctrl.v — single-slave QSPI burst engine for the TT QSPI Pmod PSRAM
 * (APS6404L, RAM A / CS1). Uses SPI-mode commands, which work from power-up:
 *   read  0xEB: serial cmd (8 SCK), quad addr (6 SCK), 6 dummy, quad data out
 *   write 0x38: serial cmd (8 SCK), quad addr (6 SCK), quad data in
 * SCK = clk/2, mode 0: outputs change on SCK falling edges, PSRAM samples on
 * rising. The PSRAM launches read nibbles tACLK (<=5.5ns) after SCK FALLING
 * edges; adding the TT mux round trip (~20ns), the rising edge on which a
 * nibble should be captured depends on the SCK period, so the sample latency
 * `lat` (in SCK cycles) is a runtime input, fed from the PSRAM image header:
 * lat=1 suits 66 MHz clk / 33 MHz SCK; use lat=0 below ~20 MHz SCK. The read
 * data phase runs 2*len+lat SCK cycles, capturing on every rise after the
 * first `lat` rises.
 * Callers must keep bursts inside one 1 KiB page (APS6404 wraps at pages)
 * and short enough for tCEM (CS low <= 8us): a 64-byte read burst is ~4.5us
 * at 66 MHz clk; below ~37 MHz clk shorten bursts or expect corruption.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

`default_nettype none

module qspi_ctrl (
    input  wire        clk,
    input  wire        rst_n,

    input  wire        start,    // pulse; latches wr/addr/len
    input  wire        wr,
    input  wire [22:0] addr,
    input  wire [6:0]  len,      // bytes, 1..64
    input  wire [1:0]  lat,      // read sample latency in SCK cycles (see note)
    output reg         busy,

    output reg         rvalid,   // one pulse per read byte
    output reg  [7:0]  rdata,
    output reg         wnext,    // pulse: write byte consumed, advance source
    input  wire [7:0]  wdata,    // must be valid combinationally while writing

    output reg         sck,
    output reg         cs_n,
    output reg  [3:0]  sd_out,
    output reg  [3:0]  sd_oe,
    input  wire [3:0]  sd_in
);
    localparam P_IDLE = 3'd0, P_CMD = 3'd1, P_ADDR = 3'd2, P_DUMMY = 3'd3,
               P_DATA = 3'd4, P_END = 3'd5, P_TAIL = 3'd6;
    reg [2:0]  phase;
    reg        lwr;
    reg [1:0]  llat;      // lat latched at burst start: the core may update
                          // its lat register mid-burst (header byte 52)
    reg [23:0] laddr;
    reg [7:0]  nib_total;  // data nibbles: 2*len
    reg [7:0]  cnt;        // counts SCK cycles within each phase
    reg [6:0]  cmd_sh;     // remaining command bits after the first
    reg [3:0]  first_nib;
    reg        half;       // read: high nibble captured / write: low pending

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            phase <= P_IDLE; busy <= 0; sck <= 0; cs_n <= 1;
            sd_out <= 4'h0; sd_oe <= 4'h0; rvalid <= 0; wnext <= 0;
            rdata <= 0; lwr <= 0; llat <= 0; laddr <= 0; nib_total <= 0;
            cnt <= 0; cmd_sh <= 0; first_nib <= 0; half <= 0;
        end else begin
            rvalid <= 0; wnext <= 0;

            case (phase)
            P_IDLE: if (start) begin
                busy      <= 1;
                lwr       <= wr;
                llat      <= lat;
                laddr     <= {1'b0, addr};
                nib_total <= {len, 1'b0};
                cs_n      <= 0;
                // present first command bit now; first SCK rise samples it
                sd_out[0] <= ~wr;               // MSB: 0xEB=1..., 0x38=0...
                cmd_sh    <= wr ? 7'h38 : 7'h6B; // remaining 7 bits
                sd_oe     <= 4'b0001;
                cnt       <= 0;
                half      <= 0;
                phase     <= P_CMD;
            end

            P_CMD: begin
                if (!sck) sck <= 1;             // rise: PSRAM samples
                else begin                      // fall: present next bit
                    sck <= 0;
                    sd_out[0] <= cmd_sh[6];
                    cmd_sh <= {cmd_sh[5:0], 1'b0};
                    cnt <= cnt + 8'd1;
                    if (cnt == 8'd7) begin      // 8 command bits sampled
                        sd_out <= laddr[23:20]; // first address nibble
                        laddr  <= {laddr[19:0], 4'h0};
                        sd_oe  <= 4'hF;
                        cnt    <= 0;
                        phase  <= P_ADDR;
                    end
                end
            end

            P_ADDR: begin
                if (!sck) sck <= 1;
                else begin
                    sck <= 0;
                    sd_out <= laddr[23:20];
                    laddr  <= {laddr[19:0], 4'h0};
                    cnt <= cnt + 8'd1;
                    if (cnt == 8'd5) begin      // 6 address nibbles sampled
                        cnt <= 0;
                        if (lwr) begin
                            phase <= P_DATA;    // sd_out already holds hi nibble
                            sd_out <= wdata[7:4];
                            first_nib <= wdata[3:0];
                            wnext  <= 1;
                        end else begin
                            phase <= P_DUMMY;
                            sd_oe <= 4'h0;
                        end
                    end
                end
            end

            P_DUMMY: begin
                if (!sck) sck <= 1;
                else begin
                    sck <= 0;
                    cnt <= cnt + 8'd1;
                    if (cnt == 8'd5) begin      // 6 wait cycles done
                        cnt   <= 0;
                        phase <= P_DATA;
                    end
                end
            end

            P_DATA: if (lwr) begin
                if (!sck) sck <= 1;
                else begin
                    sck <= 0;
                    cnt <= cnt + 8'd1;
                    if (cnt == nib_total - 8'd1) phase <= P_END;
                    else if (cnt[0]) begin      // even nibbles done: next byte
                        sd_out    <= wdata[7:4];
                        first_nib <= wdata[3:0];
                        wnext     <= 1;
                    end else
                        sd_out <= first_nib;    // odd: low nibble
                end
            end else begin
                // Read: nibble k reaches our pads during data-phase cycle
                // k+lat, so run nib_total+lat SCK cycles and capture on every
                // rise from the lat-th onward.
                if (!sck) begin
                    sck <= 1;
                    if (cnt >= {6'd0, llat}) begin
                        if (!half) begin
                            first_nib <= sd_in; // high nibble arrives first
                            half <= 1;
                        end else begin
                            rdata  <= {first_nib, sd_in};
                            rvalid <= 1;
                            half   <= 0;
                        end
                    end
                end else begin
                    sck <= 0;
                    cnt <= cnt + 8'd1;
                    if (cnt == nib_total - 8'd1 + {6'd0, llat}) phase <= P_END;
                end
            end

            P_END: begin
                cs_n  <= 1;
                sd_oe <= 4'h0;
                phase <= P_TAIL;                // CS high >= 2 clk covers tCPH
            end

            P_TAIL: begin
                busy  <= 0;
                phase <= P_IDLE;
            end

            default: phase <= P_IDLE;
            endcase
        end
    end
endmodule
