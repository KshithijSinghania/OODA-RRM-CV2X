/*
 * indore-nr-12cell.cc   --   C-V2X / 5G-LENA / SUMO, 12 cells, handover, logging
 *
 * Built on indore-nr-sumo-s1-corrected.cc (same channel / band / FlowMonitor
 * window logic), changed as follows:
 *   - 12 gNBs at the pipeline's CELL_SITES (site 0 = macro, 1..11 = RSU/small)
 *   - mobility read from traj.csv made from SUMO fcd.xml (fcd_to_traj.py)
 *   - X2 + NrA3RsrpHandoverAlgorithm so vehicles really change serving cell
 *   - per (vehicle x 5 s window) row now also has serving_cell, handovers,
 *     rsrp_dbm, sinr_db, x, y
 *   - second CSV: one row per (cell x window) with attached UEs and traffic
 *
 * NOT COMPILED BY THE AUTHOR.  Every line that depends on a 5G-LENA API
 * I could not check is tagged  [NR-API].  If one fails to compile, fix that
 * line (or set OODA_ENABLE_HANDOVER / OODA_ENABLE_RADIO_TRACES to 0) and send
 * me the compiler output.
 *
 * Run (from the ns-3 root):
 *   ./ns3 run "indore-nr-12cell --trajFile=traj_S3.csv --situation=S3"
 */

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/nr-module.h"
#include "ns3/point-to-point-module.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <cstdio>
#include <thread>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("IndoreNr12Cell");

#define OODA_ENABLE_HANDOVER 1       // X2 + A3 handover            [NR-API]
#define OODA_ENABLE_RADIO_TRACES 1   // RSRP/SINR + HO-end traces   [NR-API]

static constexpr double WINDOW_SIZE = 5.0;

/* Same coordinates / order as config.CELL_SITES in the Python pipeline. */
static const double SITES[12][2] = {
    {1824.7, 395.1}, {1492.5, 394.0}, {1725.5, 722.1}, {2196.9, 702.4},
    {1882.2, 608.9}, {1404.6, 666.8}, {353.4, 767.4},  {1026.1, 687.0},
    {1092.5, 323.6}, {687.7, 283.3},  {234.0, 259.0},  {2006.1, 1299.7}};
static uint32_t N_SITES = 12; // --nSites (first N of SITES), for speed tests

/* ------------------------------------------------------------------ mobility */
struct Pt
{
    double t, x, y, speed, accel;
};
using Traj = std::vector<Pt>;
static std::map<std::string, Traj> g_traj; // lives for the whole run
static double g_tOffset = 0.0; // chunk start (s) added to output window times
static double g_warmup = 0.0;  // output windows ending before this are flagged warmup=1

static Vector
PosAt(const Traj& tr, double t)
{
    if (t <= tr.front().t)
    {
        return Vector(tr.front().x, tr.front().y, 1.5);
    }
    if (t >= tr.back().t)
    {
        return Vector(tr.back().x, tr.back().y, 1.5);
    }
    auto it = std::upper_bound(tr.begin(), tr.end(), t, [](double v, const Pt& p) { return v < p.t; });
    const Pt& b = *it;
    const Pt& a = *(it - 1);
    const double f = (t - a.t) / (b.t - a.t);
    return Vector(a.x + f * (b.x - a.x), a.y + f * (b.y - a.y), 1.5);
}

/* velocity of the SUMO segment containing t (m/s, x/y); zero outside the trace */
static Vector
VelAt(const Traj& tr, double t)
{
    if (t < tr.front().t || t >= tr.back().t)
    {
        return Vector(0, 0, 0);
    }
    auto it = std::upper_bound(tr.begin(), tr.end(), t, [](double v, const Pt& p) { return v < p.t; });
    const Pt& b = *it;
    const Pt& a = *(it - 1);
    const double dt = b.t - a.t;
    return Vector((b.x - a.x) / dt, (b.y - a.y) / dt, 0.0);
}

/* linearly interpolated SUMO speed (m/s) or acceleration (m/s^2) at t */
static double
ScalarAt(const Traj& tr, double t, bool accel)
{
    auto get = [accel](const Pt& p) { return accel ? p.accel : p.speed; };
    if (t <= tr.front().t)
    {
        return get(tr.front());
    }
    if (t >= tr.back().t)
    {
        return get(tr.back());
    }
    auto it = std::upper_bound(tr.begin(), tr.end(), t, [](double v, const Pt& p) { return v < p.t; });
    const Pt& b = *it;
    const Pt& a = *(it - 1);
    return get(a) + (t - a.t) / (b.t - a.t) * (get(b) - get(a));
}

static void
MoveUe(Ptr<ConstantVelocityMobilityModel> m, const Traj* tr, Time period)
{
    const double now = Simulator::Now().GetSeconds();
    m->SetPosition(PosAt(*tr, now));
    m->SetVelocity(VelAt(*tr, now)); // gives the 3GPP channel a real Doppler
    if (Simulator::Now().GetSeconds() < tr->back().t)
    {
        Simulator::Schedule(period, &MoveUe, m, tr, period);
    }
}

static bool
LoadTraj(const std::string& file, uint32_t maxVehicles, double tStart, double tEnd)
{
    std::ifstream in(file);
    if (!in.is_open())
    {
        std::cerr << "ERROR: cannot open " << file << std::endl;
        return false;
    }
    std::string line;
    std::getline(in, line); // header: vehicle_id,time,x,y
    while (std::getline(in, line))
    {
        std::stringstream ss(line);
        std::string id, t, x, y, sp, an, ac;
        std::getline(ss, id, ',');
        std::getline(ss, t, ',');
        std::getline(ss, x, ',');
        std::getline(ss, y, ',');
        std::getline(ss, sp, ',');
        std::getline(ss, an, ',');
        std::getline(ss, ac, ',');
        if (id.empty() || y.empty())
        {
            continue;
        }
        if (maxVehicles && !g_traj.count(id) && g_traj.size() >= maxVehicles)
        {
            continue;
        }
        const double tt = std::stod(t);
        if (tt < tStart || (tEnd > 0.0 && tt > tEnd))
        {
            continue; // outside this chunk
        }
        g_traj[id].push_back({tt - tStart, std::stod(x), std::stod(y),
                              sp.empty() ? 0.0 : std::stod(sp), ac.empty() ? 0.0 : std::stod(ac)});
    }
    for (auto& kv : g_traj)
    {
        std::sort(kv.second.begin(), kv.second.end(), [](const Pt& a, const Pt& b) { return a.t < b.t; });
    }
    return !g_traj.empty();
}

/* ---------------------------------------------------------------- ambulance alert traffic (reconstructed from S4 data) */
struct CliRef
{
    Ptr<UdpClient> app;
    const Traj* tr;
    bool isAmb;
};
static std::vector<CliRef> g_clis;
static const Traj* g_amb = nullptr;
static double g_alertR = 300.0, g_baseRate = 100.0, g_alertRate = 150.0, g_ambRate = 200.0, g_ambEnd = 0.0;

static void
UpdateRates(Time period)
{
    const double now = Simulator::Now().GetSeconds();
    const bool ambOn = g_amb && now >= g_amb->front().t && now < g_amb->back().t;
    Vector ap(0, 0, 0);
    if (ambOn)
    {
        ap = PosAt(*g_amb, now);
    }
    for (const auto& c : g_clis)
    {
        double rate = g_baseRate;
        if (ambOn)
        {
            if (c.isAmb)
            {
                rate = g_ambRate;
            }
            else if (now >= c.tr->front().t && now < c.tr->back().t)
            {
                const Vector pp = PosAt(*c.tr, now);
                if (std::hypot(pp.x - ap.x, pp.y - ap.y) <= g_alertR)
                {
                    rate = g_alertRate;
                }
            }
        }
        c.app->SetAttribute("Interval", TimeValue(Seconds(1.0 / rate)));
    }
    if (now < g_ambEnd)
    {
        Simulator::Schedule(period, &UpdateRates, period);
    }
}

/* ---------------------------------------------------------------- closed-loop control state */
static std::string g_ctrlDir;         // empty = no closed loop
static double g_ctrlTimeout = 7200.0; // s of wall time to wait for a controller reply
static uint32_t g_ctrlK = 0;          // window counter
static std::vector<double> g_curTx;   // current TxPower per site (dBm)

/* ---------------------------------------------------------------- per-window TX power schedule (policy actuation) */
static std::vector<Ptr<NrGnbPhy>> g_gnbPhy;
static uint32_t g_powApplied = 0;

static void
SetSiteTx(uint32_t site, double dbm)
{
    if (site < g_gnbPhy.size())
    {
        g_gnbPhy[site]->SetAttribute("TxPower", DoubleValue(dbm)); // [NR-API] runtime change, read at the next slot
        g_powApplied++;
        if (site < g_curTx.size())
        {
            g_curTx[site] = dbm;
        }
    }
}

/* returns number of scheduled changes, or -1 if the file cannot be opened */
static int
LoadPowerSchedule(const std::string& file, double tStart)
{
    std::ifstream in(file);
    if (!in.is_open())
    {
        std::cerr << "ERROR: cannot open power schedule " << file << std::endl;
        return -1;
    }
    std::string line;
    std::getline(in, line); // header
    int n = 0;
    while (std::getline(in, line))
    {
        std::stringstream ss(line);
        std::string t, site, dbm;
        std::getline(ss, t, ',');
        std::getline(ss, site, ',');
        std::getline(ss, dbm, ',');
        if (t.empty() || site.empty() || dbm.empty())
        {
            continue;
        }
        const double when = std::max(0.0, std::stod(t) - tStart);
        Simulator::Schedule(Seconds(when), &SetSiteTx, (uint32_t)std::stoul(site), std::stod(dbm));
        n++;
    }
    return n;
}

/* ---------------------------------------------------------------- closed-loop handshake */
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
            std::cerr << "WARNING: no controller reply " << f << " within " << g_ctrlTimeout << " s; keeping power\n";
            return -1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

/* ------------------------------------------------------------- radio logging */
struct UeWin
{
    double rsrpSum = 0.0; // linear mW
    uint32_t rn = 0;
    double sinrSum = 0.0; // linear
    uint32_t n = 0;
    uint32_t ho = 0;
};
static std::vector<UeWin> g_ueWin;

static void
OnSinr(uint32_t idx, uint16_t cellId, uint16_t rnti, double sinr, uint16_t bwpId)
{
    // DlCtrlSinr: (cellId, rnti, sinr [linear], bwpId)
    if (sinr > 0.0 && std::isfinite(sinr))
    {
        UeWin& w = g_ueWin[idx];
        w.sinrSum += sinr;
        w.n++;
    }
}

static void
OnUeMeas(uint32_t idx, uint16_t rnti, uint16_t cellId, double rsrp, double rsrq, bool isServing, uint8_t ccId)
{
    // ReportUeMeasurements: RSRP in dBm, serving cell only
    if (isServing && std::isfinite(rsrp))
    {
        UeWin& w = g_ueWin[idx];
        w.rsrpSum += std::pow(10.0, rsrp / 10.0);
        w.rn++;
    }
}

static void
OnHandoverEnd(uint32_t idx, uint64_t imsi, uint16_t cellId, uint16_t rnti)
{
    g_ueWin[idx].ho++;
}

/* -------------------------------------------------------------- window dump */
struct Prev
{
    uint32_t txP = 0, rxP = 0;
    uint64_t txB = 0, rxB = 0;
    double delay = 0.0, jitter = 0.0;
};

struct Ctx
{
    Ptr<FlowMonitor> monitor;
    Ptr<Ipv4FlowClassifier> classifier;
    Ipv4Address remoteHost;
    std::vector<std::string> ids;
    std::vector<Ipv4Address> ueIp;
    NetDeviceContainer ueDevs;
    std::map<uint16_t, uint32_t> cellToSite; // nr cellId -> site index
    std::map<FlowId, Prev> prev;
    std::ofstream ue, cell;
};
static Ctx g_ctx;

static uint32_t
ServingSite(uint32_t i)
{
    Ptr<NrUeNetDevice> d = DynamicCast<NrUeNetDevice>(g_ctx.ueDevs.Get(i));
    const uint16_t cid = d->GetRrc()->GetCellId(); // [NR-API]
    auto it = g_ctx.cellToSite.find(cid);
    return it == g_ctx.cellToSite.end() ? 999 : it->second;
}

static void
DumpWindow(double ws, double we)
{
    static const auto wall0 = std::chrono::steady_clock::now();
    std::cout << "sim t=" << we << " s   wall="
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count()
              << " s" << std::endl;
    g_ctx.monitor->CheckForLostPackets();
    std::map<Ipv4Address, uint32_t> ipToUe;
    for (uint32_t i = 0; i < g_ctx.ueIp.size(); ++i)
    {
        ipToUe[g_ctx.ueIp[i]] = i;
    }
    struct Cell
    {
        uint32_t ues = 0;
        uint64_t txB = 0, rxB = 0;
        uint32_t txP = 0, rxP = 0;
        double sinrSum = 0.0, rsrpSum = 0.0; // dB averages over attached UEs
        uint32_t sinrN = 0, rsrpN = 0, ho = 0;
    };
    std::ostringstream ueReq; // per-vehicle lines for the controller
    std::vector<Cell> cells(N_SITES);

    for (const auto& kv : g_ctx.monitor->GetFlowStats())
    {
        Ipv4FlowClassifier::FiveTuple t = g_ctx.classifier->FindFlow(kv.first);
        if (t.sourceAddress != g_ctx.remoteHost || !ipToUe.count(t.destinationAddress))
        {
            continue;
        }
        const uint32_t i = ipToUe[t.destinationAddress];
        const FlowMonitor::FlowStats& s = kv.second;
        Prev& p = g_ctx.prev[kv.first];

        const uint32_t txP = s.txPackets - p.txP;
        const uint32_t rxP = s.rxPackets - p.rxP;
        const uint64_t txB = s.txBytes - p.txB;
        const uint64_t rxB = s.rxBytes - p.rxB;
        const double dly = s.delaySum.GetSeconds() - p.delay;
        const double jit = s.jitterSum.GetSeconds() - p.jitter;
        p.txP = s.txPackets;
        p.rxP = s.rxPackets;
        p.txB = s.txBytes;
        p.rxB = s.rxBytes;
        p.delay = s.delaySum.GetSeconds();
        p.jitter = s.jitterSum.GetSeconds();

        if (txP == 0)
        {
            continue; // vehicle not transmitting in this window
        }
        const uint32_t lost = txP > rxP ? txP - rxP : 0;
        const double loss = 100.0 * lost / txP;
        const double thr = rxB * 8.0 / (we - ws) / 1e6;
        const double mdly = rxP ? 1000.0 * dly / rxP : 0.0;
        const double mjit = rxP > 1 ? 1000.0 * jit / rxP : 0.0;

        UeWin& w = g_ueWin[i];
        const double rsrp = w.rn ? 10.0 * std::log10(w.rsrpSum / w.rn) : std::nan("");
        const double sinr = w.n ? 10.0 * std::log10(w.sinrSum / w.n) : std::nan("");
        const Vector pos = DynamicCast<MobilityModel>(g_ctx.ueDevs.Get(i)->GetNode()->GetObject<MobilityModel>())->GetPosition();
        const uint32_t site = ServingSite(i);

        g_ctx.ue << g_ctx.ids[i] << "," << i << "," << ws + g_tOffset << "," << we + g_tOffset << "," << txP << "," << rxP
                 << "," << lost << "," << loss << "," << thr << "," << mdly << "," << mjit << ","
                 << txB << "," << rxB << "," << site << "," << w.ho << "," << rsrp << "," << sinr
                 << "," << pos.x << "," << pos.y << ","
                 << ScalarAt(g_traj.at(g_ctx.ids[i]), we, false) << ","
                 << ScalarAt(g_traj.at(g_ctx.ids[i]), we, true) << "," << (we <= g_warmup ? 1 : 0) << "\n";
        const uint32_t hoCnt = w.ho;
        if (!g_ctrlDir.empty())
        {
            ueReq << g_ctx.ids[i] << "," << site << "," << txP << "," << rxP << "," << loss << "," << mdly << "," << rsrp << ","
                  << sinr << "," << pos.x << "," << pos.y << "," << hoCnt << "\n";
        }
        w = UeWin();

        if (site < N_SITES)
        {
            cells[site].ues++;
            cells[site].txB += txB;
            cells[site].rxB += rxB;
            cells[site].txP += txP;
            cells[site].rxP += rxP;
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
    }
    for (uint32_t c = 0; c < N_SITES; ++c)
    {
        const Cell& k = cells[c];
        g_ctx.cell << c << "," << ws + g_tOffset << "," << we + g_tOffset << "," << k.ues << "," << k.txP << "," << k.rxP
                   << "," << (k.txP ? 100.0 * (k.txP > k.rxP ? k.txP - k.rxP : 0) / k.txP : 0.0)
                   << "," << k.rxB * 8.0 / (we - ws) / 1e6 << "," << (we <= g_warmup ? 1 : 0) << "\n";
    }
    g_ctx.ue.flush(); // partial results survive a crash or Ctrl-C
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
               << "k,t_start,t_end,cell,tx_dbm,ues,tx_packets,rx_packets,loss_pct,thr_mbps,mean_sinr_db,mean_rsrp_dbm,handovers,warmup\n";
            for (uint32_t c = 0; c < N_SITES; ++c)
            {
                const Cell& q = cells[c];
                cf << k << "," << ws + g_tOffset << "," << we + g_tOffset << "," << c << "," << (c < g_curTx.size() ? g_curTx[c] : 0.0)
                   << "," << q.ues << "," << q.txP << "," << q.rxP << ","
                   << (q.txP ? 100.0 * (q.txP > q.rxP ? q.txP - q.rxP : 0) / q.txP : 0.0) << "," << q.rxB * 8.0 / (we - ws) / 1e6 << ","
                   << (q.sinrN ? q.sinrSum / q.sinrN : std::nan("")) << "," << (q.rsrpN ? q.rsrpSum / q.rsrpN : std::nan("")) << ","
                   << q.ho << "," << (we <= g_warmup ? 1 : 0) << "\n";
            }
        }
        {
            std::ofstream uf(CtrlPath(b + "_ue.tmp"));
            uf << "vehicle_id,cell,tx_packets,rx_packets,loss_pct,mean_delay_ms,rsrp_dbm,sinr_db,x,y,handovers\n" << ueReq.str();
        }
        std::rename(CtrlPath(b + "_cell.tmp").c_str(), CtrlPath(b + "_cell.csv").c_str());
        std::rename(CtrlPath(b + "_ue.tmp").c_str(), CtrlPath(b + "_ue.csv").c_str());
        {
            std::ofstream rf(CtrlPath(b + ".ready"));
            rf << we + g_tOffset << "\n";
        }
        const int nch = CtrlWaitApply(k);
        std::cout << "ctrl k=" << k << " t=" << we + g_tOffset << " changed " << nch << " site(s)" << std::endl;
    }
}

/* --------------------------------------------------------------------- main */
int
main(int argc, char* argv[])
{
    std::string trajFile = "traj_S3.csv";
    std::string situation = "S3";
    std::string outPrefix;
    double freq = 5.9e9, bw = 20e6;
    double txMacro = 37.0, txRsu = 30.0; // PLACEHOLDERS (ASSUMPTIONS M-section)
    uint32_t packetSize = 512, packetRate = 100, maxVehicles = 0;
    uint32_t gnbRows = 8, gnbCols = 8, ueRows = 2, ueCols = 2;
    double hysteresisDb = 3.0, tttMs = 256.0, mobilityMs = 100.0;
    bool handover = true;
    double stopTime = 0.0;
    bool radioTraces = true;
    uint32_t srsPeriodicity = 0;
    double tStart = 0.0, tEnd = 0.0, warmup = 0.0;
    std::string ambId; // empty = no ambulance behaviour (S1-S3, S5, S6 unchanged)
    double alertRadius = 300.0, alertRate = 150.0, ambRate = 200.0;
    std::string powerSchedule; // empty = fixed TxPower (default behaviour)
    std::string ctrlDir;       // empty = no closed loop
    double ctrlTimeout = 7200.0;

    CommandLine cmd(__FILE__);
    cmd.AddValue("trajFile", "traj.csv from fcd_to_traj.py", trajFile);
    cmd.AddValue("situation", "label used in output file names", situation);
    cmd.AddValue("outPrefix", "output prefix (default indore_<situation>)", outPrefix);
    cmd.AddValue("frequency", "carrier Hz", freq);
    cmd.AddValue("bandwidth", "Hz", bw);
    cmd.AddValue("txMacro", "site 0 TX power dBm", txMacro);
    cmd.AddValue("txRsu", "sites 1..11 TX power dBm", txRsu);
    cmd.AddValue("packetSize", "bytes", packetSize);
    cmd.AddValue("packetRate", "packets/s per UE", packetRate);
    cmd.AddValue("maxVehicles", "0 = all; else first N (for quick tests)", maxVehicles);
    cmd.AddValue("gnbRows", "gNB antenna rows", gnbRows);
    cmd.AddValue("gnbCols", "gNB antenna cols", gnbCols);
    cmd.AddValue("stopTime", "stop early (s) for speed tests; 0 = full run", stopTime);
    cmd.AddValue("nSites", "use only the first N sites (1..12)", N_SITES);
    cmd.AddValue("radioTraces", "log RSRP/SINR (SINR trace fires every slot)", radioTraces);
    cmd.AddValue("srsPeriodicity", "SRS period in slots (0 = 5G-LENA default 80); try 160/320 if SRS-state crash", srsPeriodicity);
    cmd.AddValue("ambulanceId", "vehicle_id of the ambulance; enables the S4 alert-traffic rule", ambId);
    cmd.AddValue("alertRadius", "m: vehicles this close to the ambulance send at alertRate", alertRadius);
    cmd.AddValue("alertRate", "packets/s inside the alert zone", alertRate);
    cmd.AddValue("ambRate", "packets/s sent by the ambulance itself", ambRate);
    cmd.AddValue("powerSchedule", "csv t_start,site,tx_dbm: per-site TX power changes at window starts (SUMO time)", powerSchedule);
    cmd.AddValue("ctrlDir", "closed loop: folder for req_/resp_ files (empty = off)", ctrlDir);
    cmd.AddValue("ctrlTimeout", "closed loop: max wall seconds to wait for a reply", ctrlTimeout);
    cmd.AddValue("tStart", "chunk start in SUMO time (s); only vehicles present in [tStart,tEnd] are simulated", tStart);
    cmd.AddValue("tEnd", "chunk end in SUMO time (s); 0 = until the trace ends", tEnd);
    cmd.AddValue("warmup", "seconds at chunk start flagged warmup=1 (attach transient)", warmup);
    cmd.AddValue("handover", "enable X2 + A3 handover", handover);
    cmd.AddValue("hysteresisDb", "A3 hysteresis", hysteresisDb);
    cmd.AddValue("tttMs", "A3 time-to-trigger ms", tttMs);
    cmd.AddValue("mobilityMs", "UE position update period ms", mobilityMs);
    cmd.Parse(argc, argv);
    N_SITES = std::max(1u, std::min(N_SITES, 12u));
    if (srsPeriodicity > 0)
    {
        Config::SetDefault("ns3::NrMacSchedulerSrsDefault::StartingPeriodicity", UintegerValue(srsPeriodicity)); // [NR-API]
    }
    if (outPrefix.empty())
    {
        outPrefix = "indore_" + situation;
    }

    g_tOffset = tStart;
    g_warmup = warmup;
    if (!LoadTraj(trajFile, maxVehicles, tStart, tEnd))
    {
        return 1;
    }
    std::vector<std::string> ids;
    double endT = 0.0;
    for (const auto& kv : g_traj)
    {
        ids.push_back(kv.first);
        endT = std::max(endT, kv.second.back().t);
    }
    const uint32_t nUe = ids.size();
    double simEnd = endT + 1.0;
    if (stopTime > 0.0)
    {
        simEnd = std::min(simEnd, stopTime);
    }
    std::cout << "Vehicles " << nUe << ", sites " << N_SITES << ", sim end " << simEnd << " s\n";

    NodeContainer gnbNodes, ueNodes;
    gnbNodes.Create(N_SITES);
    ueNodes.Create(nUe);

    Ptr<NrHelper> nrHelper = CreateObject<NrHelper>();
    Ptr<NrPointToPointEpcHelper> epcHelper = CreateObject<NrPointToPointEpcHelper>();
    nrHelper->SetEpcHelper(epcHelper);

    /* remote host + internet (as in the S1 script) */
    Ptr<Node> pgw = epcHelper->GetPgwNode();
    NodeContainer rhc;
    rhc.Create(1);
    Ptr<Node> remoteHost = rhc.Get(0);
    InternetStackHelper internet;
    internet.Install(remoteHost);
    internet.Install(ueNodes);
    PointToPointHelper p2ph;
    p2ph.SetDeviceAttribute("DataRate", DataRateValue(DataRate("100Gb/s")));
    p2ph.SetChannelAttribute("Delay", TimeValue(MilliSeconds(1)));
    NetDeviceContainer inetDevs = p2ph.Install(pgw, remoteHost);
    Ipv4AddressHelper ipv4h;
    ipv4h.SetBase("1.0.0.0", "255.0.0.0");
    Ipv4InterfaceContainer inetIf = ipv4h.Assign(inetDevs);
    Ipv4StaticRoutingHelper rh;
    rh.GetStaticRouting(remoteHost->GetObject<Ipv4>())
        ->AddNetworkRouteTo(Ipv4Address("7.0.0.0"), Ipv4Mask("255.0.0.0"), 1);

    /* mobility */
    Ptr<ListPositionAllocator> gpos = CreateObject<ListPositionAllocator>();
    for (uint32_t k = 0; k < N_SITES; ++k)
    {
        gpos->Add(Vector(SITES[k][0], SITES[k][1], 25.0));
    }
    MobilityHelper gm;
    gm.SetPositionAllocator(gpos);
    gm.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    gm.Install(gnbNodes);

    MobilityHelper um;
    um.SetMobilityModel("ns3::ConstantVelocityMobilityModel");
    um.Install(ueNodes);
    for (uint32_t i = 0; i < nUe; ++i)
    {
        Ptr<ConstantVelocityMobilityModel> m = ueNodes.Get(i)->GetObject<ConstantVelocityMobilityModel>();
        const Traj* tr = &g_traj.at(ids[i]);
        m->SetPosition(PosAt(*tr, 0.0));
        m->SetVelocity(VelAt(*tr, 0.0));
        Simulator::Schedule(Seconds(0.0), &MoveUe, m, tr, MilliSeconds(mobilityMs));
    }

    /* band + channel (unchanged from S1 script) */
    CcBwpCreator ccBwp;
    CcBwpCreator::SimpleOperationBandConf bandConf(freq, bw, 1);
    OperationBandInfo band = ccBwp.CreateOperationBandContiguousCc(bandConf);
    Ptr<NrChannelHelper> ch = CreateObject<NrChannelHelper>();
    ch->ConfigureFactories("UMi", "Default", "ThreeGpp");
    ch->SetChannelConditionModelAttribute("UpdatePeriod", TimeValue(MilliSeconds(0)));
    ch->SetPathlossAttribute("ShadowingEnabled", BooleanValue(false));
    ch->AssignChannelsToBands({band});

    nrHelper->SetUeAntennaAttribute("NumRows", UintegerValue(ueRows));
    nrHelper->SetUeAntennaAttribute("NumColumns", UintegerValue(ueCols));
    nrHelper->SetGnbAntennaAttribute("NumRows", UintegerValue(gnbRows));
    nrHelper->SetGnbAntennaAttribute("NumColumns", UintegerValue(gnbCols));
    nrHelper->SetGnbPhyAttribute("Numerology", UintegerValue(0));
    nrHelper->SetGnbPhyAttribute("TxPower", DoubleValue(txRsu));

#if OODA_ENABLE_HANDOVER
    if (handover)
    {
        nrHelper->SetHandoverAlgorithmType("ns3::NrA3RsrpHandoverAlgorithm");              // [NR-API]
        nrHelper->SetHandoverAlgorithmAttribute("Hysteresis", DoubleValue(hysteresisDb));   // [NR-API]
        nrHelper->SetHandoverAlgorithmAttribute("TimeToTrigger", TimeValue(MilliSeconds(tttMs))); // [NR-API]
    }
#endif

    NetDeviceContainer gnbDevs = nrHelper->InstallGnbDevice(gnbNodes, band.GetBwps());
    NetDeviceContainer ueDevs = nrHelper->InstallUeDevice(ueNodes, band.GetBwps());

    /* per-site TX power (site 0 = macro) and cellId -> site map */
    for (uint32_t k = 0; k < N_SITES; ++k)
    {
        Ptr<NrGnbNetDevice> g = DynamicCast<NrGnbNetDevice>(gnbDevs.Get(k));
        nrHelper->GetGnbPhy(gnbDevs.Get(k), 0)->SetAttribute("TxPower", DoubleValue(k == 0 ? txMacro : txRsu)); // [NR-API]
        g_ctx.cellToSite[g->GetCellId()] = k;
    }
    g_gnbPhy.clear();
    for (uint32_t k = 0; k < N_SITES; ++k)
    {
        g_gnbPhy.push_back(nrHelper->GetGnbPhy(gnbDevs.Get(k), 0)); // [NR-API]
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
        std::cout << "closed-loop control ON via " << ctrlDir << "\n";
    }
    if (!powerSchedule.empty())
    {
        const int nch = LoadPowerSchedule(powerSchedule, tStart);
        if (nch < 0)
        {
            return 1;
        }
        std::cout << "power schedule ON: " << nch << " changes from " << powerSchedule << "\n";
    }
    int64_t stream = 1;
    stream += nrHelper->AssignStreams(gnbDevs, stream);
    stream += nrHelper->AssignStreams(ueDevs, stream);

    Ipv4InterfaceContainer ueIf = epcHelper->AssignUeIpv4Address(NetDeviceContainer(ueDevs));
    for (uint32_t i = 0; i < nUe; ++i)
    {
        Ptr<Ipv4StaticRouting> r = rh.GetStaticRouting(ueNodes.Get(i)->GetObject<Ipv4>());
        r->SetDefaultRoute(epcHelper->GetUeDefaultGatewayAddress(), 1);
    }

#if OODA_ENABLE_HANDOVER
    if (handover)
    {
        for (uint32_t a = 0; a < N_SITES; ++a)
        {
            for (uint32_t b = a + 1; b < N_SITES; ++b)
            {
                epcHelper->AddX2Interface(gnbNodes.Get(a), gnbNodes.Get(b)); // [NR-API]
            }
        }
    }
#endif

    nrHelper->AttachToClosestGnb(ueDevs, gnbDevs);

    g_ueWin.assign(nUe, UeWin());
#if OODA_ENABLE_RADIO_TRACES
    for (uint32_t i = 0; radioTraces && i < nUe; ++i)
    {
        Ptr<NrUeNetDevice> d = DynamicCast<NrUeNetDevice>(ueDevs.Get(i));
        d->GetPhy(0)->TraceConnectWithoutContext("DlCtrlSinr", MakeBoundCallback(&OnSinr, i));
        d->GetPhy(0)->TraceConnectWithoutContext("ReportUeMeasurements", MakeBoundCallback(&OnUeMeas, i)); // [NR-API]
        d->GetRrc()->TraceConnectWithoutContext("HandoverEndOk",
                                                 MakeBoundCallback(&OnHandoverEnd, i)); // [NR-API]
    }
#endif

    /* traffic: UDP downlink, only while the vehicle exists in SUMO */
    const double interval = 1.0 / packetRate;
    for (uint32_t i = 0; i < nUe; ++i)
    {
        const Traj& tr = g_traj.at(ids[i]);
        const uint16_t port = 9000 + i;
        UdpServerHelper server(port);
        ApplicationContainer sa = server.Install(ueNodes.Get(i));
        sa.Start(Seconds(0.5));
        sa.Stop(Seconds(simEnd));
        UdpClientHelper client(ueIf.GetAddress(i), port);
        client.SetAttribute("MaxPackets", UintegerValue(std::numeric_limits<uint32_t>::max()));
        client.SetAttribute("Interval", TimeValue(Seconds(interval)));
        client.SetAttribute("PacketSize", UintegerValue(packetSize));
        const double caStart = std::max(1.0, tr.front().t) + i * 0.001;
        if (tr.back().t > caStart) // else the vehicle left before its traffic could start: no sender (was: Stop<Start => ghost)
        {
            ApplicationContainer ca = client.Install(remoteHost);
            ca.Start(Seconds(caStart));
            ca.Stop(Seconds(tr.back().t));
            if (!ambId.empty())
            {
                g_clis.push_back({DynamicCast<UdpClient>(ca.Get(0)), &tr, ids[i] == ambId});
            }
        }
    }

    /* FlowMonitor + outputs */
    FlowMonitorHelper fmh;
    g_ctx.monitor = fmh.InstallAll();
    g_ctx.classifier = DynamicCast<Ipv4FlowClassifier>(fmh.GetClassifier());
    g_ctx.remoteHost = inetIf.GetAddress(1);
    g_ctx.ids = ids;
    g_ctx.ueDevs = ueDevs;
    for (uint32_t i = 0; i < nUe; ++i)
    {
        g_ctx.ueIp.push_back(ueIf.GetAddress(i));
    }
    g_ctx.ue.open(outPrefix + "_ns3_5s.csv");
    g_ctx.cell.open(outPrefix + "_cells_5s.csv");
    g_ctx.ue << std::fixed << std::setprecision(6)
             << "vehicle_id,ue_index,window_start,window_end,tx_packets,rx_packets,lost_packets,"
                "packet_loss_percent,throughput_mbps,mean_delay_ms,mean_jitter_ms,tx_bytes,rx_bytes,"
                "serving_cell,handovers,rsrp_dbm,sinr_db,x,y,speed_mps,accel_mps2,warmup\n";
    g_ctx.cell << std::fixed << std::setprecision(6)
               << "cell,window_start,window_end,attached_active_ues,tx_packets,rx_packets,"
                  "packet_loss_percent,dl_throughput_mbps,warmup\n";

    for (double ws = 0.0; ws < simEnd - 1e-9; ws += WINDOW_SIZE)
    {
        const double we = std::min(ws + WINDOW_SIZE, simEnd);
        Simulator::Schedule(Seconds(we), &DumpWindow, ws, we);
    }

    if (!ambId.empty() && g_traj.count(ambId))
    {
        g_amb = &g_traj.at(ambId);
        g_alertR = alertRadius;
        g_baseRate = packetRate;
        g_alertRate = alertRate;
        g_ambRate = ambRate;
        g_ambEnd = simEnd;
        Simulator::Schedule(Seconds(1.0), &UpdateRates, MilliSeconds(100));
        std::cout << "ambulance alert rule ON for " << ambId << " (radius " << alertRadius << " m, " << alertRate << "/" << ambRate << " pkt/s)\n";
    }
    Simulator::Stop(Seconds(simEnd));
    Simulator::Run();
    g_ctx.ue.close();
    g_ctx.cell.close();
    Simulator::Destroy();
    if (!powerSchedule.empty())
    {
        std::cout << "power changes applied: " << g_powApplied << "\n";
    }
    if (!g_ctrlDir.empty())
    {
        std::ofstream ef(CtrlPath("end.flag"));
        ef << "done\n";
    }
    std::cout << "Wrote " << outPrefix << "_ns3_5s.csv and " << outPrefix << "_cells_5s.csv\n";
    return 0;
}
