/*
 * nano_tt_core.v — streaming nano-LM inference core for Tiny Tapeout.
 *
 * Same numerics as rtl/nano_accel.sv, but every model tensor lives in an
 * external QSPI PSRAM (image layout: tt/sw/pack.py). A single INT4xINT8
 * multiplier fed at QSPI line rate accumulates into CH=8 channels per pass;
 * the drain y = clamp(round((acc + bias) * M[c]) >> sh) is bit-serial (the
 * bias/scale burst for the next channel dwarfs it in cycles). The head layer
 * fuses argmax into the drain, so no logit buffer exists at all.
 *
 * Per token: GATHER embeddings of the 8-token window into the x scratch
 * region, then 3 matvec layers streamed block-by-block, then emit the argmax
 * token through the handshake interface and slide the window.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

`default_nettype none

module nano_tt_core #(
    parameter INIT_WAIT = 64
) (
    input  wire        clk,
    input  wire        rst_n,

    input  wire        ack,       // host handshake: rising edge consumes token
    input  wire        free_run,  // 1: don't wait for ack
    input  wire        slow_boot, // 1: header read uses sample latency 0
    output reg         tok_valid,
    output reg  [6:0]  tok_out,

    // qspi_ctrl
    output reg         qs_start,
    output reg         qs_wr,
    output reg  [22:0] qs_addr,
    output reg  [6:0]  qs_len,
    output reg  [1:0]  qs_lat,
    input  wire        qs_busy,
    input  wire        rvalid,
    input  wire [7:0]  rdata,
    input  wire        wnext,
    output wire [7:0]  wdata
);
    localparam CH = 8, K = 16, C = 8;

    localparam S_BOOT = 4'd0,  S_HDR = 4'd1,  S_GRD = 4'd2,  S_GWR = 4'd3,
               S_LSET = 4'd4,  S_XRD = 4'd5,  S_WRD = 4'd6,  S_BMRD = 4'd7,
               S_MULT = 4'd8,  S_FIN = 4'd9,  S_YWR = 4'd10, S_TOK = 4'd11,
               S_TOKW = 4'd12, S_DONE = 4'd13;
    reg [3:0] state;

    // ---------------- header-loaded configuration ----------------
    reg [15:0] cfg_g, cfg_n1, cfg_h, cfg_v;
    reg [7:0]  cfg_e;
    reg [4:0]  cfg_sh1, cfg_sh2, cfg_sh3;
    reg [22:0] emb_base, w1_base, w2_base, w3_base, bm_base,
               x_base, h1_base, h2_base;

    // ---------------- current-layer registers ----------------
    reg [1:0]  lay;
    reg [15:0] n_cur, m_cur;
    reg [22:0] x_src;
    reg [4:0]  sh_cur;
    reg        relu_cur, head_cur;
    wire [5:0] ck_total = n_cur[9:4];    // n/K chunks
    wire [7:0] bk_total = m_cur[10:3];   // m/CH blocks

    // ---------------- loop counters / pointers ----------------
    reg [7:0]  boot_cnt;
    reg [5:0]  ck;
    reg [7:0]  bk;
    reg [2:0]  gi, c;
    reg [5:0]  goff;
    reg [4:0]  glen;
    reg [15:0] tok_cnt;
    reg [22:0] w_ptr, bm_ptr, xw_ptr;
    reg [5:0]  bcnt;                     // read-byte index within a burst
    reg [3:0]  widx;                     // write-byte index within a burst

    // ---------------- datapath ----------------
    reg signed [31:0] acc [0:CH-1];
    reg [7:0]  xf [0:K-1];               // x chunk in, y bytes out
    reg [6:0]  win [0:C-1];              // token window, oldest first
    reg [7:0]  wbyte;
    reg [1:0]  mac_ph;
    reg [3:0]  kidx;
    reg [2:0]  cidx;
    reg signed [31:0] bm_b;
    reg [15:0] bm_m;
    reg signed [47:0] mand, pb;
    reg [15:0] mplier;
    reg [4:0]  scnt;
    reg        fin_rnd;
    reg signed [47:0] am_best;
    reg [6:0]  am_idx;

    // burst micro-driver
    reg        burst_req;
    reg [1:0]  bph;
    wire       burst_done = (bph == 2'd2) && !qs_busy;

    assign wdata = xf[widx];

    // xf[kidx] is registered before the MAC: kidx only changes on the second
    // nibble of a weight byte and the next byte's first nibble is >=2 clocks
    // away, so the one-cycle-late read is always settled in time. This keeps
    // the 16:1 byte mux off the multiply-accumulate critical path.
    reg signed [7:0]   xq_r;
    wire signed [3:0]  wq   = (mac_ph == 2'd2) ? wbyte[3:0] : wbyte[7:4];
    wire signed [11:0] prod = wq * xq_r;

    // acc[c] is likewise registered ahead of the bias add: c settles when the
    // bias/scale burst is issued, tens of clocks before accb is captured.
    reg signed [31:0]  acc_c_r;
    wire signed [32:0] accb  = acc_c_r + bm_b;
    wire signed [47:0] rnd   = (sh_cur == 0) ? 48'sd0
                              : (48'sd1 <<< (sh_cur - 5'd1));
    wire signed [47:0] yrelu = (relu_cur && pb < 0) ? 48'sd0 : pb;
    wire [7:0]         y8    = (yrelu > 48'sd127)  ? 8'd127 :
                               (yrelu < -48'sd128) ? 8'h80  : yrelu[7:0];
    wire [6:0]         chan_g = {bk[3:0], c};

    reg ack_d;
    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_BOOT; boot_cnt <= 0;
            qs_start <= 0; qs_wr <= 0; qs_addr <= 0; qs_len <= 0;
            qs_lat <= 2'd1;
            burst_req <= 0; bph <= 0;
            tok_valid <= 0; tok_out <= 0; ack_d <= 0;
            cfg_g <= 0; cfg_n1 <= 0; cfg_h <= 0; cfg_v <= 0; cfg_e <= 0;
            cfg_sh1 <= 0; cfg_sh2 <= 0; cfg_sh3 <= 0;
            emb_base <= 0; w1_base <= 0; w2_base <= 0; w3_base <= 0;
            bm_base <= 0; x_base <= 0; h1_base <= 0; h2_base <= 0;
            lay <= 0; n_cur <= 0; m_cur <= 0; x_src <= 0; sh_cur <= 0;
            relu_cur <= 0; head_cur <= 0;
            ck <= 0; bk <= 0; gi <= 0; c <= 0; goff <= 0; glen <= 0;
            tok_cnt <= 0; w_ptr <= 0; bm_ptr <= 0; xw_ptr <= 0;
            bcnt <= 0; widx <= 0;
            wbyte <= 0; mac_ph <= 0; kidx <= 0; cidx <= 0; xq_r <= 0;
            acc_c_r <= 0;
            bm_b <= 0; bm_m <= 0; mand <= 0; pb <= 0; mplier <= 0;
            scnt <= 0; fin_rnd <= 0;
            am_best <= 0; am_idx <= 0;
            for (i = 0; i < CH; i = i + 1) acc[i] <= 0;
            for (i = 0; i < K; i = i + 1) xf[i] <= 0;
            for (i = 0; i < C; i = i + 1) win[i] <= 0;
        end else begin
            qs_start <= 0;
            ack_d <= ack;
            xq_r <= xf[kidx];
            acc_c_r <= acc[c];

            // burst micro-driver: bph 0=idle/issue, 1=wait busy, 2=in flight
            if (burst_req) begin
                if (bph == 2'd0) begin qs_start <= 1; bph <= 2'd1; end
                else if (bph == 2'd1 && qs_busy) bph <= 2'd2;
                else if (burst_done) begin bph <= 2'd0; burst_req <= 0; end
            end

            // read-byte demux (runs under whichever burst is in flight)
            if (rvalid) begin
                bcnt <= bcnt + 6'd1;
                case (state)
                S_HDR: begin
                    case (bcnt)
                    6'd0:  cfg_g[7:0]   <= rdata;  6'd1:  cfg_g[15:8]  <= rdata;
                    6'd2:  cfg_n1[7:0]  <= rdata;  6'd3:  cfg_n1[15:8] <= rdata;
                    6'd4:  cfg_h[7:0]   <= rdata;  6'd5:  cfg_h[15:8]  <= rdata;
                    6'd6:  cfg_v[7:0]   <= rdata;  6'd7:  cfg_v[15:8]  <= rdata;
                    6'd8:  cfg_e        <= rdata;
                    6'd9:  cfg_sh1      <= rdata[4:0];
                    6'd10: cfg_sh2      <= rdata[4:0];
                    6'd11: cfg_sh3      <= rdata[4:0];
                    6'd12: emb_base[7:0]   <= rdata; 6'd13: emb_base[15:8] <= rdata;
                    6'd14: emb_base[22:16] <= rdata[6:0];
                    6'd16: w1_base[7:0]    <= rdata; 6'd17: w1_base[15:8]  <= rdata;
                    6'd18: w1_base[22:16]  <= rdata[6:0];
                    6'd20: w2_base[7:0]    <= rdata; 6'd21: w2_base[15:8]  <= rdata;
                    6'd22: w2_base[22:16]  <= rdata[6:0];
                    6'd24: w3_base[7:0]    <= rdata; 6'd25: w3_base[15:8]  <= rdata;
                    6'd26: w3_base[22:16]  <= rdata[6:0];
                    6'd28: bm_base[7:0]    <= rdata; 6'd29: bm_base[15:8]  <= rdata;
                    6'd30: bm_base[22:16]  <= rdata[6:0];
                    6'd32: x_base[7:0]     <= rdata; 6'd33: x_base[15:8]   <= rdata;
                    6'd34: x_base[22:16]   <= rdata[6:0];
                    6'd36: h1_base[7:0]    <= rdata; 6'd37: h1_base[15:8]  <= rdata;
                    6'd38: h1_base[22:16]  <= rdata[6:0];
                    6'd40: h2_base[7:0]    <= rdata; 6'd41: h2_base[15:8]  <= rdata;
                    6'd42: h2_base[22:16]  <= rdata[6:0];
                    default: ;
                    endcase
                    if (bcnt >= 6'd44 && bcnt <= 6'd51)
                        win[bcnt - 6'd44] <= rdata[6:0];
                    if (bcnt == 6'd52) qs_lat <= rdata[1:0];
                end
                S_GRD, S_XRD: xf[bcnt[3:0]] <= rdata;
                S_WRD: begin wbyte <= rdata; mac_ph <= 2'd2; end
                S_BMRD: case (bcnt)
                    6'd0: bm_b[7:0]   <= rdata;  6'd1: bm_b[15:8]  <= rdata;
                    6'd2: bm_b[23:16] <= rdata;  6'd3: bm_b[31:24] <= rdata;
                    6'd4: bm_m[7:0]   <= rdata;  6'd5: bm_m[15:8]  <= rdata;
                    default: ;
                endcase
                default: ;
                endcase
            end

            // MAC: two nibbles per weight byte, channel-fastest stream order
            if (mac_ph != 0) begin
                acc[cidx] <= acc[cidx] + prod;
                cidx <= cidx + 3'd1;
                if (cidx == 3'd7) kidx <= kidx + 4'd1;
                mac_ph <= mac_ph - 2'd1;
            end

            case (state)
            S_BOOT: begin
                boot_cnt <= boot_cnt + 8'd1;
                if (boot_cnt == INIT_WAIT[7:0]) begin
                    qs_wr <= 0; qs_addr <= 23'd0; qs_len <= 7'd53;
                    qs_lat <= slow_boot ? 2'd0 : 2'd1;
                    bcnt <= 0; burst_req <= 1;
                    state <= S_HDR;
                end
            end

            S_HDR: if (burst_done) begin
                bm_ptr  <= bm_base;
                xw_ptr  <= x_base;
                am_best <= 48'sh800000000000;
                gi <= 0; goff <= 0; tok_cnt <= 0;
                qs_wr <= 0; qs_addr <= emb_base + {9'd0, win[0], 5'd0};
                qs_len <= (cfg_e >= K[7:0]) ? K[6:0] : cfg_e[6:0];
                glen   <= (cfg_e >= K[7:0]) ? K[4:0] : cfg_e[4:0];
                bcnt <= 0; burst_req <= 1;
                state <= S_GRD;
            end

            S_GRD: if (burst_done) begin
                qs_wr <= 1; qs_addr <= xw_ptr; qs_len <= {2'd0, glen};
                widx <= 0; burst_req <= 1;
                state <= S_GWR;
            end

            S_GWR: if (burst_done) begin
                xw_ptr <= xw_ptr + {18'd0, glen};
                if ({2'd0, goff} + {1'd0, glen} == {2'd0, cfg_e[5:0]}) begin
                    goff <= 0;
                    if (gi == 3'd7) begin
                        lay <= 0;
                        state <= S_LSET;
                    end else begin
                        gi <= gi + 3'd1;
                        qs_wr <= 0;
                        qs_addr <= emb_base + {9'd0, win[gi + 3'd1], 5'd0};
                        qs_len <= (cfg_e >= K[7:0]) ? K[6:0] : cfg_e[6:0];
                        glen   <= (cfg_e >= K[7:0]) ? K[4:0] : cfg_e[4:0];
                        bcnt <= 0; burst_req <= 1;
                        state <= S_GRD;
                    end
                end else begin
                    goff <= goff + {1'd0, glen};
                    qs_wr <= 0;
                    qs_addr <= emb_base + {9'd0, win[gi], 5'd0}
                               + {17'd0, goff} + {18'd0, glen};
                    qs_len <= ({2'd0, cfg_e[5:0]} - {2'd0, goff} - {3'd0, glen}
                               >= K[7:0])
                              ? K[6:0]
                              : cfg_e[6:0] - {1'd0, goff} - {2'd0, glen};
                    glen   <= ({2'd0, cfg_e[5:0]} - {2'd0, goff} - {3'd0, glen}
                               >= K[7:0])
                              ? K[4:0]
                              : cfg_e[4:0] - goff[4:0] - glen;
                    bcnt <= 0; burst_req <= 1;
                    state <= S_GRD;
                end
            end

            S_LSET: begin
                case (lay)
                2'd0: begin
                    n_cur <= cfg_n1; m_cur <= cfg_h; x_src <= x_base;
                    w_ptr <= w1_base; xw_ptr <= h1_base;
                    sh_cur <= cfg_sh1; relu_cur <= 1; head_cur <= 0;
                end
                2'd1: begin
                    n_cur <= cfg_h; m_cur <= cfg_h; x_src <= h1_base;
                    w_ptr <= w2_base; xw_ptr <= h2_base;
                    sh_cur <= cfg_sh2; relu_cur <= 1; head_cur <= 0;
                end
                default: begin
                    n_cur <= cfg_h; m_cur <= cfg_v; x_src <= h2_base;
                    w_ptr <= w3_base; xw_ptr <= 0;
                    sh_cur <= cfg_sh3; relu_cur <= 0; head_cur <= 1;
                end
                endcase
                ck <= 0; bk <= 0;
                for (i = 0; i < CH; i = i + 1) acc[i] <= 0;
                state <= S_XRD;
                // burst issued on the next cycle via the S_XRD entry below
                burst_req <= 0; bph <= 0; qs_start <= 0;
                bcnt <= 0;
                qs_wr <= 0; qs_len <= K[6:0];
                qs_addr <= (lay == 2'd0) ? x_base :
                           (lay == 2'd1) ? h1_base : h2_base;
                burst_req <= 1;
            end

            S_XRD: if (burst_done) begin
                qs_wr <= 0; qs_addr <= w_ptr; qs_len <= 7'd64;
                kidx <= 0; cidx <= 0;
                burst_req <= 1;
                state <= S_WRD;
            end

            S_WRD: if (burst_done) begin
                w_ptr <= w_ptr + 23'd64;
                if (ck + 6'd1 == ck_total) begin
                    c <= 0;
                    qs_wr <= 0; qs_addr <= bm_ptr; qs_len <= 7'd8;
                    bcnt <= 0; burst_req <= 1;
                    state <= S_BMRD;
                end else begin
                    ck <= ck + 6'd1;
                    qs_wr <= 0; qs_len <= K[6:0];
                    qs_addr <= x_src + {13'd0, ck + 6'd1, 4'd0};
                    bcnt <= 0; burst_req <= 1;
                    state <= S_XRD;
                end
            end

            S_BMRD: if (burst_done) begin
                bm_ptr <= bm_ptr + 23'd8;
                mand   <= {{15{accb[32]}}, accb};
                mplier <= bm_m;
                pb     <= 0;
                fin_rnd <= 1;
                scnt   <= sh_cur;
                state  <= S_MULT;
            end

            S_MULT: begin
                if (mplier == 0) state <= S_FIN;
                else begin
                    if (mplier[0]) pb <= pb + mand;
                    mand   <= mand <<< 1;
                    mplier <= mplier >> 1;
                end
            end

            S_FIN: begin
                if (head_cur) begin
                    if (pb > am_best) begin
                        am_best <= pb;
                        am_idx  <= chan_g;
                    end
                    acc[c] <= 0;
                    if (c == 3'd7) begin
                        if (bk + 8'd1 == bk_total) state <= S_TOK;
                        else begin
                            bk <= bk + 8'd1; ck <= 0; c <= 0;
                            qs_wr <= 0; qs_len <= K[6:0]; qs_addr <= x_src;
                            bcnt <= 0; burst_req <= 1;
                            state <= S_XRD;
                        end
                    end else begin
                        c <= c + 3'd1;
                        qs_wr <= 0; qs_addr <= bm_ptr; qs_len <= 7'd8;
                        bcnt <= 0; burst_req <= 1;
                        state <= S_BMRD;
                    end
                end else if (fin_rnd) begin
                    pb <= pb + rnd;
                    fin_rnd <= 0;
                end else if (scnt != 0) begin
                    pb <= pb >>> 1;
                    scnt <= scnt - 5'd1;
                end else begin
                    xf[{1'b0, c}] <= y8;
                    acc[c] <= 0;
                    if (c == 3'd7) begin
                        qs_wr <= 1; qs_addr <= xw_ptr; qs_len <= 7'd8;
                        widx <= 0; burst_req <= 1;
                        state <= S_YWR;
                    end else begin
                        c <= c + 3'd1;
                        qs_wr <= 0; qs_addr <= bm_ptr; qs_len <= 7'd8;
                        bcnt <= 0; burst_req <= 1;
                        state <= S_BMRD;
                    end
                end
            end

            S_YWR: if (burst_done) begin
                xw_ptr <= xw_ptr + 23'd8;
                if (bk + 8'd1 == bk_total) begin
                    lay <= lay + 2'd1;
                    state <= S_LSET;
                end else begin
                    bk <= bk + 8'd1; ck <= 0;
                    qs_wr <= 0; qs_len <= K[6:0]; qs_addr <= x_src;
                    bcnt <= 0; burst_req <= 1;
                    state <= S_XRD;
                end
            end

            S_TOK: begin
                tok_out   <= am_idx;
                tok_valid <= 1;
                for (i = 0; i < C - 1; i = i + 1) win[i] <= win[i + 1];
                win[C-1] <= am_idx;
                bm_ptr   <= bm_base;
                am_best  <= 48'sh800000000000;
                xw_ptr   <= x_base;
                gi <= 0; goff <= 0;
                tok_cnt  <= tok_cnt + 16'd1;
                state    <= S_TOKW;
            end

            S_TOKW: if (free_run || (ack && !ack_d)) begin
                if (!free_run) tok_valid <= 0;
                if (tok_cnt == cfg_g) begin
                    tok_valid <= 0;
                    state <= S_DONE;
                end else begin
                    qs_wr <= 0;
                    qs_addr <= emb_base + {9'd0, win[0], 5'd0};
                    qs_len <= (cfg_e >= K[7:0]) ? K[6:0] : cfg_e[6:0];
                    glen   <= (cfg_e >= K[7:0]) ? K[4:0] : cfg_e[4:0];
                    bcnt <= 0; burst_req <= 1;
                    state <= S_GRD;
                end
            end

            S_DONE: ;

            default: state <= S_BOOT;
            endcase

            // write-byte index advance (shared by S_GWR / S_YWR)
            if (wnext) widx <= widx + 4'd1;
        end
    end
endmodule
