/*
 * C-V2X / 5G-LENA / SUMO
 *
 * indore-nr-sumo-s1.cc
 *
 * SITUATION 1
 *
 * Configuration:
 *   - Indore-01 telemetry
 *   - ALL vehicles present in telemetry
 *   - 1 gNB
 *   - 5.9 GHz
 *   - 20 MHz
 *   - gNB Tx power = 38.5 dBm
 *   - gNB height = 25 m
 *   - UE antenna = 2x2
 *   - gNB antenna = 8x8
 *   - UDP downlink
 *   - 512-byte packets
 *   - 100 packets/s per UE
 *
 * Network measurements:
 *   - 5-second temporal windows
 *   - tx packets
 *   - rx packets
 *   - lost packets
 *   - packet loss percentage
 *   - throughput
 *   - mean delay
 *   - mean jitter
 *
 * Output:
 *
 *   indore_01_ns3_5s.csv
 *
 * Each row corresponds to:
 *
 *   one vehicle × one 5-second window
 *
 * The context CSV is NOT read here.
 * It will be merged later using:
 *
 *   window_start
 *   window_end
 *
 * This keeps the Situation-1 context CSV as the canonical
 * context/OODA layer.
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
#include <cctype>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("IndoreSituation1");

/* =========================================================================
 * Constants
 * ========================================================================= */

static constexpr double WINDOW_SIZE = 5.0;

/* =========================================================================
 * Telemetry structures
 * ========================================================================= */

struct TelemetryPoint
{
    double time;
    double x;
    double y;
};

using Trajectory = std::vector<TelemetryPoint>;
using TrajectoryMap = std::map<std::string, Trajectory>;

/* =========================================================================
 * CSV parser
 * ========================================================================= */

static std::vector<std::string>
SplitCsvLine(const std::string& line)
{
    std::vector<std::string> fields;

    std::string current;

    bool quoted = false;

    for (char c : line)
    {
        if (c == '"')
        {
            quoted = !quoted;
        }
        else if (c == ',' && !quoted)
        {
            fields.push_back(current);
            current.clear();
        }
        else
        {
            current += c;
        }
    }

    fields.push_back(current);

    return fields;
}

/* =========================================================================
 * Locate CSV column
 * ========================================================================= */

static int
FindColumn(const std::vector<std::string>& header,
           const std::vector<std::string>& candidates)
{
    for (std::size_t i = 0; i < header.size(); ++i)
    {
        std::string name = header[i];

        std::transform(
            name.begin(),
            name.end(),
            name.begin(),
            [](unsigned char c)
            {
                return static_cast<char>(
                    std::tolower(c));
            });

        for (const auto& candidate : candidates)
        {
            if (name == candidate)
            {
                return static_cast<int>(i);
            }
        }
    }

    return -1;
}

/* =========================================================================
 * Load SUMO telemetry
 * ========================================================================= */

static bool
LoadTelemetry(
    const std::string& filename,
    TrajectoryMap& trajectories,
    double& minX,
    double& maxX,
    double& minY,
    double& maxY,
    double& endTime)
{
    std::ifstream input(filename);

    if (!input.is_open())
    {
        std::cerr
            << "ERROR: Cannot open telemetry file: "
            << filename
            << std::endl;

        return false;
    }

    std::string line;

    if (!std::getline(input, line))
    {
        std::cerr
            << "ERROR: Telemetry file is empty."
            << std::endl;

        return false;
    }

    std::vector<std::string> header =
        SplitCsvLine(line);

    int idColumn =
        FindColumn(
            header,
            {
                "vehicle_id",
                "vehicle",
                "id",
                "veh_id",
                "vehicleid"
            });

    int timeColumn =
        FindColumn(
            header,
            {
                "time",
                "timestamp",
                "sim_time",
                "simulation_time"
            });

    int xColumn =
        FindColumn(
            header,
            {
                "x",
                "pos_x",
                "position_x"
            });

    int yColumn =
        FindColumn(
            header,
            {
                "y",
                "pos_y",
                "position_y"
            });

    if (idColumn < 0 ||
        timeColumn < 0 ||
        xColumn < 0 ||
        yColumn < 0)
    {
        std::cerr
            << "ERROR: Could not identify required telemetry columns."
            << std::endl;

        std::cerr
            << "Columns found:"
            << std::endl;

        for (const auto& h : header)
        {
            std::cerr
                << "  "
                << h
                << std::endl;
        }

        return false;
    }

    minX =
        std::numeric_limits<double>::max();

    maxX =
        std::numeric_limits<double>::lowest();

    minY =
        std::numeric_limits<double>::max();

    maxY =
        std::numeric_limits<double>::lowest();

    endTime = 0.0;

    uint64_t rows = 0;

    while (std::getline(input, line))
    {
        if (line.empty())
        {
            continue;
        }

        std::vector<std::string> fields =
            SplitCsvLine(line);

        int maxColumn =
            std::max(
                {
                    idColumn,
                    timeColumn,
                    xColumn,
                    yColumn
                });

        if (fields.size() <
            static_cast<std::size_t>(
                maxColumn + 1))
        {
            continue;
        }

        try
        {
            const std::string vehicleId =
                fields[idColumn];

            const double time =
                std::stod(
                    fields[timeColumn]);

            const double x =
                std::stod(
                    fields[xColumn]);

            const double y =
                std::stod(
                    fields[yColumn]);

            trajectories[
                vehicleId
            ].push_back(
                {
                    time,
                    x,
                    y
                });

            minX =
                std::min(
                    minX,
                    x);

            maxX =
                std::max(
                    maxX,
                    x);

            minY =
                std::min(
                    minY,
                    y);

            maxY =
                std::max(
                    maxY,
                    y);

            endTime =
                std::max(
                    endTime,
                    time);

            ++rows;
        }
        catch (...)
        {
            continue;
        }
    }

    input.close();

    for (auto& item : trajectories)
    {
        std::sort(
            item.second.begin(),
            item.second.end(),
            [](const TelemetryPoint& a,
               const TelemetryPoint& b)
            {
                return a.time < b.time;
            });
    }

    std::cout
        << "\nTelemetry loaded:"
        << std::endl;

    std::cout
        << "  File: "
        << filename
        << std::endl;

    std::cout
        << "  Rows: "
        << rows
        << std::endl;

    std::cout
        << "  Unique vehicles: "
        << trajectories.size()
        << std::endl;

    std::cout
        << "  Simulation end: "
        << endTime
        << " s"
        << std::endl;

    std::cout
        << "  X range: "
        << minX
        << " to "
        << maxX
        << std::endl;

    std::cout
        << "  Y range: "
        << minY
        << " to "
        << maxY
        << std::endl;

    return !trajectories.empty();
}

/* =========================================================================
 * Interpolate telemetry position
 * ========================================================================= */

static Vector
GetPosition(
    const Trajectory& trajectory,
    double time)
{
    if (trajectory.empty())
    {
        return Vector(
            0.0,
            0.0,
            1.5);
    }

    if (time <=
        trajectory.front().time)
    {
        return Vector(
            trajectory.front().x,
            trajectory.front().y,
            1.5);
    }

    if (time >=
        trajectory.back().time)
    {
        return Vector(
            trajectory.back().x,
            trajectory.back().y,
            1.5);
    }

    auto upper =
        std::lower_bound(
            trajectory.begin(),
            trajectory.end(),
            time,
            [](const TelemetryPoint& point,
               double t)
            {
                return point.time < t;
            });

    if (upper ==
        trajectory.begin())
    {
        return Vector(
            upper->x,
            upper->y,
            1.5);
    }

    auto lower =
        upper - 1;

    const double dt =
        upper->time -
        lower->time;

    if (dt <= 0.0)
    {
        return Vector(
            lower->x,
            lower->y,
            1.5);
    }

    const double alpha =
        (time -
         lower->time) /
        dt;

    const double x =
        lower->x +
        alpha *
        (upper->x -
         lower->x);

    const double y =
        lower->y +
        alpha *
        (upper->y -
         lower->y);

    return Vector(
        x,
        y,
        1.5);
}

/* =========================================================================
 * SUMO mobility update
 * ========================================================================= */

static void
UpdateUePosition(
    Ptr<MobilityModel> mobility,
    Trajectory trajectory,
    double simulationTime,
    Time updatePeriod)
{
    if (mobility == nullptr ||
        trajectory.empty())
    {
        return;
    }

    Vector position =
        GetPosition(
            trajectory,
            simulationTime);

    mobility->SetPosition(
        position);

    const double nextTime =
        simulationTime +
        updatePeriod.GetSeconds();

    if (nextTime <=
        trajectory.back().time)
    {
        Simulator::Schedule(
            updatePeriod,
            &UpdateUePosition,
            mobility,
            trajectory,
            nextTime,
            updatePeriod);
    }
    else
    {
        /*
         * Keep vehicle at its final SUMO
         * position until simulation termination.
         */
        Simulator::Schedule(
            updatePeriod,
            [mobility, trajectory]()
            {
                mobility->SetPosition(
                    GetPosition(
                        trajectory,
                        trajectory.back().time));
            });
    }
}

/* =========================================================================
 * Previous FlowMonitor statistics
 * ========================================================================= */

struct PreviousFlowStats
{
    uint64_t txPackets = 0;
    uint64_t rxPackets = 0;

    uint64_t txBytes = 0;
    uint64_t rxBytes = 0;

    double delaySeconds = 0.0;
    double jitterSeconds = 0.0;

    bool initialized = false;
};

using PreviousStatsMap =
    std::map<
        FlowId,
        PreviousFlowStats
    >;

/* =========================================================================
 * Window statistics
 * ========================================================================= */

struct WindowStats
{
    uint64_t txPackets = 0;
    uint64_t rxPackets = 0;

    uint64_t txBytes = 0;
    uint64_t rxBytes = 0;

    double delaySeconds = 0.0;
    double jitterSeconds = 0.0;

    uint64_t jitterSamples = 0;

    bool found = false;
};

/* =========================================================================
 * Write one 5-second FlowMonitor window
 * ========================================================================= */

static void
WriteFlowMonitorWindow(
    Ptr<FlowMonitor> monitor,
    Ptr<Ipv4FlowClassifier> classifier,
    const Ipv4InterfaceContainer& ueIpIfaces,
    const std::vector<std::string>& vehicleIds,
    double windowStart,
    double windowEnd,
    std::ofstream& output,
    PreviousStatsMap& previousStats)
{
    /*
     * Get current cumulative FlowMonitor statistics.
     */
    auto stats =
        monitor->GetFlowStats();

    const uint32_t numberOfVehicles =
        static_cast<uint32_t>(
            vehicleIds.size());

    std::vector<WindowStats> windowStats(
        numberOfVehicles);

    /*
     * Process every FlowMonitor flow.
     */
    for (const auto& flowPair : stats)
    {
        const FlowId flowId =
            flowPair.first;

        const FlowMonitor::FlowStats& flow =
            flowPair.second;

        Ipv4FlowClassifier::FiveTuple tuple =
            classifier->FindFlow(flowId);

        /*
         * UDP only.
         */
        if (tuple.protocol != 17)
        {
            continue;
        }

        /*
         * Match destination address
         * with a UE.
         */
        int matchedUe = -1;

        for (uint32_t i = 0;
             i < numberOfVehicles;
             ++i)
        {
            if (tuple.destinationAddress ==
                ueIpIfaces.GetAddress(i))
            {
                matchedUe =
                    static_cast<int>(i);

                break;
            }
        }

        if (matchedUe < 0)
        {
            continue;
        }

        /*
         * Current cumulative values.
         */
        const uint64_t currentTxPackets =
            flow.txPackets;

        const uint64_t currentRxPackets =
            flow.rxPackets;

        const uint64_t currentTxBytes =
            flow.txBytes;

        const uint64_t currentRxBytes =
            flow.rxBytes;

        const double currentDelay =
            flow.delaySum.GetSeconds();

        const double currentJitter =
            flow.jitterSum.GetSeconds();

        /*
         * Previous cumulative values.
         */
        PreviousFlowStats& previous =
            previousStats[flowId];

        uint64_t deltaTxPackets = 0;
        uint64_t deltaRxPackets = 0;

        uint64_t deltaTxBytes = 0;
        uint64_t deltaRxBytes = 0;

        double deltaDelay = 0.0;
        double deltaJitter = 0.0;

        if (!previous.initialized)
        {
            /*
             * First snapshot.
             */
            deltaTxPackets =
                currentTxPackets;

            deltaRxPackets =
                currentRxPackets;

            deltaTxBytes =
                currentTxBytes;

            deltaRxBytes =
                currentRxBytes;

            deltaDelay =
                currentDelay;

            deltaJitter =
                currentJitter;

            previous.initialized = true;
        }
        else
        {
            if (currentTxPackets >=
                previous.txPackets)
            {
                deltaTxPackets =
                    currentTxPackets -
                    previous.txPackets;
            }

            if (currentRxPackets >=
                previous.rxPackets)
            {
                deltaRxPackets =
                    currentRxPackets -
                    previous.rxPackets;
            }

            if (currentTxBytes >=
                previous.txBytes)
            {
                deltaTxBytes =
                    currentTxBytes -
                    previous.txBytes;
            }

            if (currentRxBytes >=
                previous.rxBytes)
            {
                deltaRxBytes =
                    currentRxBytes -
                    previous.rxBytes;
            }

            deltaDelay =
                std::max(
                    0.0,
                    currentDelay -
                    previous.delaySeconds);

            deltaJitter =
                std::max(
                    0.0,
                    currentJitter -
                    previous.jitterSeconds);
        }

        /*
         * Save current cumulative state.
         */
        previous.txPackets =
            currentTxPackets;

        previous.rxPackets =
            currentRxPackets;

        previous.txBytes =
            currentTxBytes;

        previous.rxBytes =
            currentRxBytes;

        previous.delaySeconds =
            currentDelay;

        previous.jitterSeconds =
            currentJitter;

        /*
         * Add this flow's delta to the
         * corresponding vehicle.
         */
        WindowStats& ws =
            windowStats[
                matchedUe];

        ws.txPackets +=
            deltaTxPackets;

        ws.rxPackets +=
            deltaRxPackets;

        ws.txBytes +=
            deltaTxBytes;

        ws.rxBytes +=
            deltaRxBytes;

        ws.delaySeconds +=
            deltaDelay;

        ws.jitterSeconds +=
            deltaJitter;

        /*
         * FlowMonitor jitter is accumulated
         * between received packets.
         */
        if (deltaRxPackets > 1)
        {
            ws.jitterSamples += deltaRxPackets - 1;
        }

        ws.found = true;
    }

    const double windowDuration =
        windowEnd -
        windowStart;

    /*
     * Write one row for EVERY vehicle.
     */
    for (uint32_t i = 0;
         i < numberOfVehicles;
         ++i)
    {
        const WindowStats& ws =
            windowStats[i];

        /*
         * Packet loss.
         */
        uint64_t lostPackets = 0;

        if (ws.txPackets >=
            ws.rxPackets)
        {
            lostPackets =
                ws.txPackets -
                ws.rxPackets;
        }

        double packetLossPercent = 0.0;

        if (ws.txPackets > 0)
        {
            packetLossPercent =
                100.0 *
                static_cast<double>(
                    lostPackets) /
                static_cast<double>(
                    ws.txPackets);
        }

        /*
         * Throughput.
         *
         * Always normalized by the actual
         * duration of this window.
         */
        double throughputMbps = 0.0;

        if (windowDuration > 0.0)
        {
            throughputMbps =
                static_cast<double>(
                    ws.rxBytes) *
                8.0 /
                windowDuration /
                1e6;
        }

        /*
         * Mean delay.
         */
        double meanDelayMs = 0.0;

        if (ws.rxPackets > 0)
        {
            meanDelayMs =
                ws.delaySeconds /
                static_cast<double>(
                    ws.rxPackets) *
                1000.0;
        }

        /*
         * Mean jitter.
         */
        double meanJitterMs = 0.0;

        if (ws.jitterSamples > 0)
        {
            meanJitterMs =
                ws.jitterSeconds /
                static_cast<double>(
                    ws.jitterSamples) *
                1000.0;
        }

        /*
         * Write row.
         */
        output
            << vehicleIds[i]
            << ","
            << i
            << ","
            << windowStart
            << ","
            << windowEnd
            << ","
            << ws.txPackets
            << ","
            << ws.rxPackets
            << ","
            << lostPackets
            << ","
            << packetLossPercent
            << ","
            << throughputMbps
            << ","
            << meanDelayMs
            << ","
            << meanJitterMs
            << ","
            << ws.txBytes
            << ","
            << ws.rxBytes
            << std::endl;
    }

    std::cout
        << "  Window "
        << std::fixed
        << std::setprecision(1)
        << windowStart
        << " - "
        << windowEnd
        << " s written"
        << std::endl;
}

/* =========================================================================
 * Main
 * ========================================================================= */

int
main(int argc, char* argv[])
{
    /* ---------------------------------------------------------------------
     * Situation 1 parameters
     * --------------------------------------------------------------------- */

    std::string telemetryFile =
        "/mnt/c/C-V2X-OODA/datasets/observe_ds/"
        "indore_01_telemetry_v5.csv";

    std::string outputFile =
        "indore_01_ns3_5s.csv";

    double centralFrequency =
        5.9e9;

    double bandwidth =
        20e6;

    double totalTxPower =
        38.5;

    uint32_t packetSize =
        512;

    uint32_t packetRate =
        100;

    /* ---------------------------------------------------------------------
     * Command line
     * --------------------------------------------------------------------- */

    CommandLine cmd(__FILE__);

    cmd.AddValue(
        "telemetryFile",
        "Situation-1 SUMO telemetry CSV",
        telemetryFile);

    cmd.AddValue(
        "outputFile",
        "Situation-1 5-second NS-3 output CSV",
        outputFile);

    cmd.AddValue(
        "frequency",
        "NR carrier frequency in Hz",
        centralFrequency);

    cmd.AddValue(
        "bandwidth",
        "NR bandwidth in Hz",
        bandwidth);

    cmd.AddValue(
        "txPower",
        "gNB total transmit power in dBm",
        totalTxPower);

    cmd.AddValue(
        "packetSize",
        "UDP packet size in bytes",
        packetSize);

    cmd.AddValue(
        "packetRate",
        "UDP packet rate per second",
        packetRate);

    cmd.Parse(argc, argv);

    /* ---------------------------------------------------------------------
     * Validate
     * --------------------------------------------------------------------- */

    if (packetRate == 0)
    {
        std::cerr
            << "ERROR: packetRate must be greater than zero."
            << std::endl;

        return 1;
    }

    /* ---------------------------------------------------------------------
     * Load ALL Situation-1 telemetry.
     * --------------------------------------------------------------------- */

    TrajectoryMap trajectories;

    double minX;
    double maxX;

    double minY;
    double maxY;

    double sumoEndTime;

    if (!LoadTelemetry(
            telemetryFile,
            trajectories,
            minX,
            maxX,
            minY,
            maxY,
            sumoEndTime))
    {
        return 1;
    }

    /*
     * IMPORTANT:
     *
     * Unlike Situation 2, there is NO:
     *
     *   10 sets × 15 vehicles
     *
     * selection here.
     *
     * Every vehicle found in the Situation-1
     * telemetry is simulated.
     */
    const uint32_t numberOfVehicles =
        static_cast<uint32_t>(
            trajectories.size());

    if (numberOfVehicles == 0)
    {
        std::cerr
            << "ERROR: No vehicles found."
            << std::endl;

        return 1;
    }

    /* ---------------------------------------------------------------------
     * Vehicle IDs
     * --------------------------------------------------------------------- */

    std::vector<std::string> vehicleIds;

    vehicleIds.reserve(
        numberOfVehicles);

    for (const auto& item :
         trajectories)
    {
        vehicleIds.push_back(
            item.first);
    }

    /* ---------------------------------------------------------------------
     * Print configuration.
     * --------------------------------------------------------------------- */

    std::cout
        << "\n========================================"
        << std::endl;

    std::cout
        << "Indore Situation 1"
        << std::endl;

    std::cout
        << "C-V2X / 5G-LENA / SUMO"
        << std::endl;

    std::cout
        << "5-second temporal NS-3 dataset"
        << std::endl;

    std::cout
        << "========================================"
        << std::endl;

    std::cout
        << "Telemetry: "
        << telemetryFile
        << std::endl;

    std::cout
        << "Vehicles: "
        << numberOfVehicles
        << std::endl;

    std::cout
        << "SUMO end time: "
        << sumoEndTime
        << " s"
        << std::endl;

    std::cout
        << "Frequency: "
        << centralFrequency / 1e9
        << " GHz"
        << std::endl;

    std::cout
        << "Bandwidth: "
        << bandwidth / 1e6
        << " MHz"
        << std::endl;

    std::cout
        << "gNB Tx power: "
        << totalTxPower
        << " dBm"
        << std::endl;

    std::cout
        << "UE antenna: 2x2"
        << std::endl;

    std::cout
        << "gNB antenna: 8x8"
        << std::endl;

    std::cout
        << "Packet size: "
        << packetSize
        << " bytes"
        << std::endl;

    std::cout
        << "Packet rate: "
        << packetRate
        << " packets/s/UE"
        << std::endl;

    std::cout
        << "Measurement window: "
        << WINDOW_SIZE
        << " s"
        << std::endl;

    std::cout
        << "\nSelected vehicles:"
        << std::endl;

    for (uint32_t i = 0;
         i < numberOfVehicles;
         ++i)
    {
        std::cout
            << "  UE "
            << i
            << " -> "
            << vehicleIds[i]
            << std::endl;
    }

    /* ---------------------------------------------------------------------
     * Simulation duration.
     *
     * Add 1 second after the final SUMO sample.
     * --------------------------------------------------------------------- */

    const double simulationEnd =
        sumoEndTime + 1.0;

    /* ---------------------------------------------------------------------
     * Nodes
     * --------------------------------------------------------------------- */

    NodeContainer gnbNodes;

    gnbNodes.Create(1);

    NodeContainer ueNodes;

    ueNodes.Create(
        numberOfVehicles);

    /* ---------------------------------------------------------------------
     * NR helper + EPC
     * --------------------------------------------------------------------- */

    Ptr<NrHelper> nrHelper =
        CreateObject<NrHelper>();

    Ptr<NrPointToPointEpcHelper> epcHelper =
        CreateObject<
            NrPointToPointEpcHelper>();

    nrHelper->SetEpcHelper(
        epcHelper);

    /* ---------------------------------------------------------------------
     * Remote host
     * --------------------------------------------------------------------- */

    Ptr<Node> pgw =
        epcHelper->GetPgwNode();

    NodeContainer remoteHostContainer;

    remoteHostContainer.Create(1);

    Ptr<Node> remoteHost =
        remoteHostContainer.Get(0);

    /* ---------------------------------------------------------------------
     * Internet
     * --------------------------------------------------------------------- */

    InternetStackHelper internet;

    internet.Install(
        remoteHost);

    internet.Install(
        ueNodes);

    /* ---------------------------------------------------------------------
     * PGW <-> Remote host
     * --------------------------------------------------------------------- */

    PointToPointHelper p2ph;

    p2ph.SetDeviceAttribute(
        "DataRate",
        DataRateValue(
            DataRate("100Gb/s")));

    p2ph.SetChannelAttribute(
        "Delay",
        TimeValue(
            MilliSeconds(1)));

    NetDeviceContainer internetDevices =
        p2ph.Install(
            pgw,
            remoteHost);

    Ipv4AddressHelper ipv4h;

    ipv4h.SetBase(
        "1.0.0.0",
        "255.0.0.0");

    Ipv4InterfaceContainer internetIpIfaces =
        ipv4h.Assign(
            internetDevices);

    Ipv4Address remoteHostAddr =
        internetIpIfaces.GetAddress(1);

    /* ---------------------------------------------------------------------
     * Remote-host route to UE subnet
     * --------------------------------------------------------------------- */

    Ipv4StaticRoutingHelper
        ipv4RoutingHelper;

    Ptr<Ipv4StaticRouting>
        remoteHostStaticRouting =
            ipv4RoutingHelper.GetStaticRouting(
                remoteHost
                    ->GetObject<Ipv4>());

    remoteHostStaticRouting
        ->AddNetworkRouteTo(
            Ipv4Address("7.0.0.0"),
            Ipv4Mask("255.0.0.0"),
            1);

    /* ---------------------------------------------------------------------
     * gNB mobility
     *
     * Fixed at the center of the
     * Situation-1 telemetry bounding box.
     * --------------------------------------------------------------------- */

    const double gnbX =
        (minX + maxX) / 2.0;

    const double gnbY =
        (minY + maxY) / 2.0;

    Ptr<ListPositionAllocator>
        gnbPosition =
            CreateObject<
                ListPositionAllocator>();

    gnbPosition->Add(
        Vector(
            gnbX,
            gnbY,
            25.0));

    MobilityHelper gnbMobility;

    gnbMobility.SetPositionAllocator(
        gnbPosition);

    gnbMobility.SetMobilityModel(
        "ns3::ConstantPositionMobilityModel");

    gnbMobility.Install(
        gnbNodes);

    /* ---------------------------------------------------------------------
     * UE mobility
     * --------------------------------------------------------------------- */

    MobilityHelper ueMobility;

    ueMobility.SetMobilityModel(
        "ns3::ConstantPositionMobilityModel");

    ueMobility.Install(
        ueNodes);

    const Time mobilityUpdatePeriod =
        MilliSeconds(10);

    for (uint32_t i = 0;
         i < numberOfVehicles;
         ++i)
    {
        const std::string& vehicleId =
            vehicleIds[i];

        const Trajectory& trajectory =
            trajectories.at(
                vehicleId);

        Ptr<MobilityModel> mobility =
            ueNodes.Get(i)
                ->GetObject<
                    MobilityModel>();

        if (mobility == nullptr)
        {
            std::cerr
                << "ERROR: UE "
                << i
                << " has no MobilityModel."
                << std::endl;

            Simulator::Destroy();

            return 1;
        }

        Vector initialPosition =
            GetPosition(
                trajectory,
                0.0);

        mobility->SetPosition(
            initialPosition);

        Simulator::Schedule(
            Seconds(0.0),
            &UpdateUePosition,
            mobility,
            trajectory,
            0.0,
            mobilityUpdatePeriod);
    }

    /* ---------------------------------------------------------------------
     * NR operation band
     * --------------------------------------------------------------------- */

    CcBwpCreator ccBwpCreator;

    const uint8_t numCcPerBand = 1;

    CcBwpCreator::SimpleOperationBandConf
        bandConf(
            centralFrequency,
            bandwidth,
            numCcPerBand);

    OperationBandInfo band =
        ccBwpCreator
            .CreateOperationBandContiguousCc(
                bandConf);

    /* ---------------------------------------------------------------------
     * NR channel
     * --------------------------------------------------------------------- */

    Ptr<NrChannelHelper>
        channelHelper =
            CreateObject<NrChannelHelper>();

    channelHelper->ConfigureFactories(
        "UMi",
        "Default",
        "ThreeGpp");

    channelHelper
        ->SetChannelConditionModelAttribute(
            "UpdatePeriod",
            TimeValue(
                MilliSeconds(0)));

    channelHelper->SetPathlossAttribute(
        "ShadowingEnabled",
        BooleanValue(false));

    channelHelper->AssignChannelsToBands(
        {band});

    /* ---------------------------------------------------------------------
     * UE antennas
     * --------------------------------------------------------------------- */

    nrHelper->SetUeAntennaAttribute(
        "NumRows",
        UintegerValue(2));

    nrHelper->SetUeAntennaAttribute(
        "NumColumns",
        UintegerValue(2));

    /* ---------------------------------------------------------------------
     * gNB antennas
     * --------------------------------------------------------------------- */

    nrHelper->SetGnbAntennaAttribute(
        "NumRows",
        UintegerValue(8));

    nrHelper->SetGnbAntennaAttribute(
        "NumColumns",
        UintegerValue(8));

    /* ---------------------------------------------------------------------
     * gNB PHY
     *
     * Numerology 0 = 15 kHz SCS.
     *
     * Do not set Numerology on NrUePhy.
     * --------------------------------------------------------------------- */

    nrHelper->SetGnbPhyAttribute(
        "Numerology",
        UintegerValue(0));

    nrHelper->SetGnbPhyAttribute(
        "TxPower",
        DoubleValue(
            totalTxPower));

    /* ---------------------------------------------------------------------
     * Install gNB
     * --------------------------------------------------------------------- */

    NetDeviceContainer gnbDevices =
        nrHelper->InstallGnbDevice(
            gnbNodes,
            band.GetBwps());

    /* ---------------------------------------------------------------------
     * Install ALL UE devices
     * --------------------------------------------------------------------- */

    NetDeviceContainer ueDevices =
        nrHelper->InstallUeDevice(
            ueNodes,
            band.GetBwps());

    /* ---------------------------------------------------------------------
     * UE IPv4 addresses
     * --------------------------------------------------------------------- */

    Ipv4InterfaceContainer ueIpIfaces =
        epcHelper->AssignUeIpv4Address(
            NetDeviceContainer(
                ueDevices));

    /* ---------------------------------------------------------------------
     * Attach all UEs
     * --------------------------------------------------------------------- */

    nrHelper->AttachToClosestGnb(
        ueDevices,
        gnbDevices);

    /*
     * IMPORTANT:
     *
     * No dedicated QoS bearers are created.
     *
     * Every UE uses the default QoS flow
     * created by InstallUeDevice().
     */

    /* ---------------------------------------------------------------------
     * UDP downlink
     * --------------------------------------------------------------------- */

    const uint16_t basePort = 9000;

    const double packetInterval =
        1.0 /
        static_cast<double>(
            packetRate);

    for (uint32_t i = 0;
         i < numberOfVehicles;
         ++i)
    {
        const uint16_t port =
            basePort + i;

        /* -------------------------------------------------------------
         * UDP server on UE
         * ------------------------------------------------------------- */

        UdpServerHelper server(
            port);

        ApplicationContainer serverApps =
            server.Install(
                ueNodes.Get(i));

        serverApps.Start(
            Seconds(0.5));

        serverApps.Stop(
            Seconds(sumoEndTime));

        /* -------------------------------------------------------------
         * UDP client on remote host
         * ------------------------------------------------------------- */

        UdpClientHelper client(
            ueIpIfaces.GetAddress(i),
            port);

        client.SetAttribute(
            "MaxPackets",
            UintegerValue(
                std::numeric_limits<
                    uint32_t>::max()));

        client.SetAttribute(
            "Interval",
            TimeValue(
                Seconds(
                    packetInterval)));

        client.SetAttribute(
            "PacketSize",
            UintegerValue(
                packetSize));

        ApplicationContainer clientApps =
            client.Install(
                remoteHost);

        /*
         * Stagger each UE by 1 ms.
         */
        clientApps.Start(
            Seconds(
                1.0 +
                i * 0.001));

        clientApps.Stop(
            Seconds(sumoEndTime));
    }

    /* ---------------------------------------------------------------------
     * FlowMonitor
     * --------------------------------------------------------------------- */

    FlowMonitorHelper flowmonHelper;

    Ptr<FlowMonitor> monitor =
        flowmonHelper.InstallAll();

    Ptr<Ipv4FlowClassifier>
        classifier =
            DynamicCast<
                Ipv4FlowClassifier>(
                flowmonHelper.GetClassifier());

    if (classifier == nullptr)
    {
        std::cerr
            << "ERROR: Could not obtain "
            << "Ipv4FlowClassifier."
            << std::endl;

        Simulator::Destroy();

        return 1;
    }

    /* ---------------------------------------------------------------------
     * Output file
     * --------------------------------------------------------------------- */

    std::ofstream output(
        outputFile,
        std::ios::out |
        std::ios::trunc);

    if (!output.is_open())
    {
        std::cerr
            << "ERROR: Could not open output file: "
            << outputFile
            << std::endl;

        Simulator::Destroy();

        return 1;
    }

    output
        << std::fixed
        << std::setprecision(6);

    /* ---------------------------------------------------------------------
     * Output header
     * --------------------------------------------------------------------- */

    output
        << "vehicle_id,"
        << "ue_index,"
        << "window_start,"
        << "window_end,"
        << "tx_packets,"
        << "rx_packets,"
        << "lost_packets,"
        << "packet_loss_percent,"
        << "throughput_mbps,"
        << "mean_delay_ms,"
        << "mean_jitter_ms,"
        << "tx_bytes,"
        << "rx_bytes"
        << std::endl;

    /* ---------------------------------------------------------------------
     * Previous cumulative FlowMonitor statistics
     * --------------------------------------------------------------------- */

    PreviousStatsMap previousStats;

    /* ---------------------------------------------------------------------
     * Schedule 5-second windows at ABSOLUTE simulation times.
     *
     * FlowMonitor counters are cumulative. WriteFlowMonitorWindow()
     * converts them into per-window deltas using previousStats.
     * --------------------------------------------------------------------- */

    for (double windowStart = 0.0;
         windowStart < simulationEnd - 1e-9;
         windowStart += WINDOW_SIZE)
    {
        const double windowEnd =
            std::min(windowStart + WINDOW_SIZE, simulationEnd);

        Simulator::Schedule(
            Seconds(windowEnd),
            [&, windowStart, windowEnd]()
            {
                WriteFlowMonitorWindow(
                    monitor,
                    classifier,
                    ueIpIfaces,
                    vehicleIds,
                    windowStart,
                    windowEnd,
                    output,
                    previousStats);
            });
    }

    /* ---------------------------------------------------------------------
     * Run simulation
     * --------------------------------------------------------------------- */

    std::cout
        << "\nStarting Situation-1 simulation..."
        << std::endl;

    std::cout
        << "Vehicles: "
        << numberOfVehicles
        << std::endl;

    std::cout
        << "Simulation time: "
        << simulationEnd
        << " seconds"
        << std::endl;

    std::cout
        << "Measurement window: "
        << WINDOW_SIZE
        << " seconds"
        << std::endl;

    std::cout
        << "Output: "
        << outputFile
        << std::endl;

    Simulator::Stop(
        Seconds(simulationEnd));

    Simulator::Run();

    /* ---------------------------------------------------------------------
     * Final FlowMonitor check
     *
     * Used only for final FlowMonitor validation.
     *
     * The actual CSV has already been generated
     * using the 5-second snapshots.
     * --------------------------------------------------------------------- */

    monitor->CheckForLostPackets();

    output.close();

    /* ---------------------------------------------------------------------
     * Final summary
     * --------------------------------------------------------------------- */

    std::cout
        << "\n========================================"
        << std::endl;

    std::cout
        << "Situation-1 simulation complete"
        << std::endl;

    std::cout
        << "========================================"
        << std::endl;

    std::cout
        << "Vehicles simulated: "
        << numberOfVehicles
        << std::endl;

    std::cout
        << "Telemetry duration: "
        << sumoEndTime
        << " s"
        << std::endl;

    std::cout
        << "Temporal window: "
        << WINDOW_SIZE
        << " s"
        << std::endl;

    std::cout
        << "Output: "
        << outputFile
        << std::endl;

    std::cout
        << "========================================"
        << std::endl;

    Simulator::Destroy();

    return 0;
}