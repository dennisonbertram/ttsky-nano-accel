// tb_local.v — self-checking plain-iverilog testbench (no cocotb needed).
// Instantiates the TT top + PSRAM model, runs the full token generation via
// the ACK handshake, and compares against expected.hex bit-exactly.
// Run from tt/test/: iverilog -g2012 -o tb_local ../src/*.v psram_model.v tb_local.v && vvp tb_local

`default_nettype none
`timescale 1ns / 1ps

module tb_local;
    localparam MAX_G = 64;

    reg clk = 0, rst_n = 0;
    reg [7:0] ui_in = 0;
    wire [7:0] uo_out, uio_out, uio_oe;

    always #7.5 clk = ~clk;   // 66 MHz internal / 33 MHz SCK (silicon nominal)

    // QSPI bus between DUT and PSRAM model
    wire        sck   = uio_out[3];
    wire        cs1_n = uio_out[6];
    wire [3:0]  sd_m  = {uio_out[5], uio_out[4], uio_out[2], uio_out[1]};
    wire [3:0]  psram_q;
    wire        psram_drive;
    wire [3:0]  sd_bus = psram_drive ? psram_q : sd_m;
    wire [7:0]  uio_in = {2'b00, sd_bus[3], sd_bus[2], 1'b0, sd_bus[1], sd_bus[0], 1'b0};

    tt_um_dennisonbertram_nano_accel dut (
        .ui_in(ui_in), .uo_out(uo_out),
        .uio_in(uio_in), .uio_out(uio_out), .uio_oe(uio_oe),
        .ena(1'b1), .clk(clk), .rst_n(rst_n)
    );

    psram_model #(.INIT_FILE("psram.hex")) psram (
        .sck(sck), .cs_n(cs1_n), .sd_in(sd_m),
        .sd_q(psram_q), .sd_drive(psram_drive)
    );

    reg [7:0] expected [0:MAX_G-1];
    integer g, errors, ntok;
    integer t_start, t_tok, t_prev;

    initial begin
        ntok = 0;
        for (g = 0; g < MAX_G; g = g + 1) expected[g] = 8'hxx;
        $readmemh("expected.hex", expected);

        #23 rst_n = 1;
        t_start = $time;
        $write("streaming = \"");

        // count expected tokens (contiguous non-x entries)
        begin : count
            for (g = 0; g < MAX_G; g = g + 1)
                if (expected[g] === 8'hxx) begin ntok = g; disable count; end
            ntok = MAX_G;
        end

        errors = 0;
        t_prev = $time;
        for (g = 0; g < ntok; g = g + 1) begin
            wait (uo_out[7] === 1'b1);
            t_tok = $time;
            $write("%c", {1'b0, uo_out[6:0]});
            if ({1'b0, uo_out[6:0]} !== expected[g]) begin
                errors = errors + 1;
                $display("\nMISMATCH token %0d: got 0x%02x expected 0x%02x",
                         g, uo_out[6:0], expected[g]);
            end
            // ack handshake
            @(posedge clk); ui_in[0] <= 1;
            wait (uo_out[7] === 1'b0 || g == ntok - 1);
            @(posedge clk); ui_in[0] <= 0;
            @(posedge clk);
            t_prev = t_tok;
        end
        $display("\"");
        $display("----------------------------------------------------");
        $display("tokens            = %0d", ntok);
        $display("cycles/token      = %0d", ((t_tok - t_start) / 15) / ntok);
        $display("tokens/s @47.6MHz = %0d",
                 47619047 / (((t_tok - t_start) / 15) / ntok));
        if (errors == 0) $display("PASS: bit-exact match with golden model");
        else             $display("FAIL: %0d/%0d token mismatches", errors, ntok);
        $finish;
    end

    initial begin
        #400000000;  // 40M cycles timeout
        $display("TIMEOUT");
        $finish;
    end
endmodule
