#!/usr/bin/env python3
"""Closed-loop control handshake for indore-nr-12cell.cc (apply AFTER patch_power_schedule.py).
Default behaviour unchanged: nothing happens unless --ctrlDir=DIR is given.
At the end of every 5 s window ns-3 writes  DIR/req_KKKK_cell.csv  (per cell: power, UEs, packets, loss, mean SINR/RSRP, handovers)
and DIR/req_KKKK_ue.csv (per vehicle), then DIR/req_KKKK.ready, and WAITS for DIR/resp_KKKK.csv (header + lines 'site,tx_dbm').
The new powers are applied immediately (they govern the next window). At the end it writes DIR/end.flag.
usage: python3 patch_ctrl_loop.py scratch/indore-nr-12cell.cc     (writes a .bak4 copy first)"""
import shutil, sys
p = sys.argv[1]; s = open(p).read()
if "ctrlDir" in s: sys.exit("already patched (ctrlDir present); nothing to do")
if "powerSchedule" not in s: sys.exit("apply patch_power_schedule.py first")
def rep(old, new):
    global s
    if s.count(old) != 1: sys.exit(f"anchor not found exactly once:\n{old[:100]}\n(send me the file)")
    s = s.replace(old, new)

rep("#include <vector>\n", "#include <vector>\n#include <cstdio>\n#include <thread>\n")

# globals BEFORE the power-schedule block (SetSiteTx uses g_curTx)
rep("/* ---------------------------------------------------------------- per-window TX power schedule (policy actuation) */",
"""/* ---------------------------------------------------------------- closed-loop control state */
static std::string g_ctrlDir;         // empty = no closed loop
static double g_ctrlTimeout = 7200.0; // s of wall time to wait for a controller reply
static uint32_t g_ctrlK = 0;          // window counter
static std::vector<double> g_curTx;   // current TxPower per site (dBm)

/* ---------------------------------------------------------------- per-window TX power schedule (policy actuation) */""")

rep("        g_powApplied++;\n", "        g_powApplied++;\n        if (site < g_curTx.size())\n        {\n            g_curTx[site] = dbm;\n        }\n")

# handshake functions, after the power block (they call SetSiteTx)
rep("/* ------------------------------------------------------------- radio logging */",
"""/* ---------------------------------------------------------------- closed-loop handshake */
static std::string
CtrlPath(const std::string& name)
{
    return g_ctrlDir + "/" + name;
}

/* waits for resp_<k>.csv (header, then 'site,tx_dbm' lines) and applies it; returns #sites changed, -1 on timeout */
static int
CtrlWaitApply(uint32_t k)
{
    char buf[64];
    std::snprintf(buf, sizeof(buf), "resp_%04u.csv", k);
    const std::string f = CtrlPath(buf);
    const auto t0 = std::chrono::steady_clock::now();
    while (true)
    {
        std::ifstream in(f);
        if (in.is_open())
        {
            std::string line;
            std::getline(in, line); // header
            int n = 0;
            while (std::getline(in, line))
            {
                std::stringstream ss(line);
                std::string site, dbm;
                std::getline(ss, site, ',');
                std::getline(ss, dbm, ',');
                if (site.empty() || dbm.empty())
                {
                    continue;
                }
                const uint32_t st = (uint32_t)std::stoul(site);
                const double v = std::stod(dbm);
                if (st < g_curTx.size() && std::fabs(v - g_curTx[st]) > 1e-9)
                {
                    SetSiteTx(st, v);
                    n++;
                }
            }
            return n;
        }
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > g_ctrlTimeout)
        {
            std::cerr << "WARNING: no controller reply " << f << " within " << g_ctrlTimeout << " s; keeping power\\n";
            return -1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

/* ------------------------------------------------------------- radio logging */""")

# per-cell radio aggregates
rep("""    struct Cell
    {
        uint32_t ues = 0;
        uint64_t txB = 0, rxB = 0;
        uint32_t txP = 0, rxP = 0;
    };
""", """    struct Cell
    {
        uint32_t ues = 0;
        uint64_t txB = 0, rxB = 0;
        uint32_t txP = 0, rxP = 0;
        double sinrSum = 0.0, rsrpSum = 0.0; // dB averages over attached UEs
        uint32_t sinrN = 0, rsrpN = 0, ho = 0;
    };
    std::ostringstream ueReq; // per-vehicle lines for the controller
""")
rep("        w = UeWin();\n", """        const uint32_t hoCnt = w.ho;
        if (!g_ctrlDir.empty())
        {
            ueReq << g_ctx.ids[i] << "," << site << "," << txP << "," << rxP << "," << loss << "," << mdly << "," << rsrp << ","
                  << sinr << "," << pos.x << "," << pos.y << "," << hoCnt << "\\n";
        }
        w = UeWin();
""")
rep("""            cells[site].rxP += rxP;
        }
""", """            cells[site].rxP += rxP;
            if (std::isfinite(sinr))
            {
                cells[site].sinrSum += sinr;
                cells[site].sinrN++;
            }
            if (std::isfinite(rsrp))
            {
                cells[site].rsrpSum += rsrp;
                cells[site].rsrpN++;
            }
            cells[site].ho += hoCnt;
        }
""")
# request + wait, after the window's output is flushed
rep("""    g_ctx.ue.flush(); // partial results survive a crash or Ctrl-C
    g_ctx.cell.flush();
""", """    g_ctx.ue.flush(); // partial results survive a crash or Ctrl-C
    g_ctx.cell.flush();
    if (!g_ctrlDir.empty())
    {
        const uint32_t k = g_ctrlK++;
        char base[64];
        std::snprintf(base, sizeof(base), "req_%04u", k);
        const std::string b(base);
        {
            std::ofstream cf(CtrlPath(b + "_cell.tmp"));
            cf << std::fixed << std::setprecision(4)
               << "k,t_start,t_end,cell,tx_dbm,ues,tx_packets,rx_packets,loss_pct,thr_mbps,mean_sinr_db,mean_rsrp_dbm,handovers,warmup\\n";
            for (uint32_t c = 0; c < N_SITES; ++c)
            {
                const Cell& q = cells[c];
                cf << k << "," << ws + g_tOffset << "," << we + g_tOffset << "," << c << "," << (c < g_curTx.size() ? g_curTx[c] : 0.0)
                   << "," << q.ues << "," << q.txP << "," << q.rxP << ","
                   << (q.txP ? 100.0 * (q.txP > q.rxP ? q.txP - q.rxP : 0) / q.txP : 0.0) << "," << q.rxB * 8.0 / (we - ws) / 1e6 << ","
                   << (q.sinrN ? q.sinrSum / q.sinrN : std::nan("")) << "," << (q.rsrpN ? q.rsrpSum / q.rsrpN : std::nan("")) << ","
                   << q.ho << "," << (we <= g_warmup ? 1 : 0) << "\\n";
            }
        }
        {
            std::ofstream uf(CtrlPath(b + "_ue.tmp"));
            uf << "vehicle_id,cell,tx_packets,rx_packets,loss_pct,mean_delay_ms,rsrp_dbm,sinr_db,x,y,handovers\\n" << ueReq.str();
        }
        std::rename(CtrlPath(b + "_cell.tmp").c_str(), CtrlPath(b + "_cell.csv").c_str());
        std::rename(CtrlPath(b + "_ue.tmp").c_str(), CtrlPath(b + "_ue.csv").c_str());
        {
            std::ofstream rf(CtrlPath(b + ".ready"));
            rf << we + g_tOffset << "\\n";
        }
        const int nch = CtrlWaitApply(k);
        std::cout << "ctrl k=" << k << " t=" << we + g_tOffset << " changed " << nch << " site(s)" << std::endl;
    }
""")
# options
rep("    std::string powerSchedule; // empty = fixed TxPower (default behaviour)\n",
    "    std::string powerSchedule; // empty = fixed TxPower (default behaviour)\n    std::string ctrlDir;       // empty = no closed loop\n    double ctrlTimeout = 7200.0;\n")
rep('    cmd.AddValue("tStart",',
    '    cmd.AddValue("ctrlDir", "closed loop: folder for req_/resp_ files (empty = off)", ctrlDir);\n'
    '    cmd.AddValue("ctrlTimeout", "closed loop: max wall seconds to wait for a reply", ctrlTimeout);\n    cmd.AddValue("tStart",')
rep("""        g_gnbPhy.push_back(nrHelper->GetGnbPhy(gnbDevs.Get(k), 0)); // [NR-API]
    }
""", """        g_gnbPhy.push_back(nrHelper->GetGnbPhy(gnbDevs.Get(k), 0)); // [NR-API]
    }
    g_curTx.clear();
    for (uint32_t k = 0; k < N_SITES; ++k)
    {
        g_curTx.push_back(k == 0 ? txMacro : txRsu);
    }
    if (!ctrlDir.empty())
    {
        g_ctrlDir = ctrlDir;
        g_ctrlTimeout = ctrlTimeout;
        std::cout << "closed-loop control ON via " << ctrlDir << "\\n";
    }
""")
rep('    std::cout << "Wrote " << outPrefix',
    '    if (!g_ctrlDir.empty())\n    {\n        std::ofstream ef(CtrlPath("end.flag"));\n        ef << "done\\n";\n    }\n    std::cout << "Wrote " << outPrefix')
shutil.copy(p, p + ".bak4"); open(p, "w").write(s); print("patched", p, "(backup:", p + ".bak4)")
