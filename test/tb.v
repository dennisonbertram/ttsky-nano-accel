`default_nettype none
`timescale 1ns / 1ps

/* Testbench: instantiates the TT top plus a behavioral PSRAM model wired to
   the QSPI Pmod pins, so cocotb (test.py) only has to drive clk/rst/ui_in
   and watch tokens appear on uo_out. */
module tb ();

  // Dump the signals to a FST file. You can view it with gtkwave or surfer.
  initial begin
    $dumpfile("tb.fst");
    $dumpvars(0, tb);
    #1;
  end

  // Wire up the inputs and outputs:
  reg clk;
  reg rst_n;
  reg ena;
  reg [7:0] ui_in;
  wire [7:0] uo_out;
  wire [7:0] uio_out;
  wire [7:0] uio_oe;
`ifdef GL_TEST
  wire VPWR = 1'b1;
  wire VGND = 1'b0;
`endif

  // QSPI Pmod: PSRAM A (CS1) behavioral model on the uio pins
  wire       sck   = uio_out[3];
  wire       cs1_n = uio_out[6];
  wire [3:0] sd_m  = {uio_out[5], uio_out[4], uio_out[2], uio_out[1]};
  wire [3:0] psram_q;
  wire       psram_drive;
  wire [3:0] sd_bus = psram_drive ? psram_q : sd_m;
  wire [7:0] uio_in = {2'b00, sd_bus[3], sd_bus[2], 1'b0, sd_bus[1], sd_bus[0], 1'b0};

  psram_model #(
      .INIT_FILE("psram.hex")
  ) psram (
      .sck(sck), .cs_n(cs1_n), .sd_in(sd_m),
      .sd_q(psram_q), .sd_drive(psram_drive)
  );

  tt_um_dennisonbertram_nano_accel user_project (

      // Include power ports for the Gate Level test:
`ifdef GL_TEST
      .VPWR(VPWR),
      .VGND(VGND),
`endif

      .ui_in  (ui_in),    // Dedicated inputs
      .uo_out (uo_out),   // Dedicated outputs
      .uio_in (uio_in),   // IOs: Input path
      .uio_out(uio_out),  // IOs: Output path
      .uio_oe (uio_oe),   // IOs: Enable path (active high: 0=input, 1=output)
      .ena    (ena),      // enable - goes high when design is selected
      .clk    (clk),      // clock
      .rst_n  (rst_n)     // not reset
  );

endmodule
