/*
 * ============================================================
 *
 * INDORE SUMO-DRIVEN 5G NR SIMULATION
 *
 * ns-3.48 + NR module
 *
 * Features:
 *
 * - Reads SUMO telemetry CSV
 * - Creates one UE per SUMO vehicle
 * - Maps SUMO trajectories to ns-3 WaypointMobilityModel
 * - 5G NR at 28 GHz
 * - 100 MHz bandwidth
 * - One gNB
 * - EPC core network
 * - Remote host
 * - UDP downlink traffic
 * - FlowMonitor statistics
 * - Per-vehicle output CSV
 *
 * ============================================================
 */

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/nr-module.h"
#include "ns3/point-to-point-module.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("IndoreNrSumoSimulation");

/*
 * ============================================================
 * SUMO TELEMETRY STRUCTURES
 * ============================================================
 */

struct TelemetryRecord
{
    double time;
    std::string vehicleId;

    double x;
    double y;

    double speed;
    double acceleration;
    double angle;

    std::string edgeId;
    std::string laneId;
};

struct VehicleTrajectory
{
    std::string vehicleId;

    std::vector<TelemetryRecord> records;
};

/*
 * ============================================================
 * STRING UTILITIES
 * ============================================================
 */

std::string
Trim(const std::string& value)
{
    std::size_t first = value.find_first_not_of(
        " \t\r\n");

    if (first == std::string::npos)
    {
        return "";
    }

    std::size_t last = value.find_last_not_of(
        " \t\r\n");

    return value.substr(
        first,
        last - first + 1);
}

std::vector<std::string>
SplitCsvLine(const std::string& line)
{
    std::vector<std::string> values;

    std::string current;

    bool insideQuotes = false;

    for (std::size_t i = 0;
         i < line.size();
         ++i)
    {
        char c = line[i];

        if (c == '"')
        {
            insideQuotes = !insideQuotes;
        }
        else if (c == ',' &&
                 !insideQuotes)
        {
            values.push_back(
                Trim(current));

            current.clear();
        }
        else
        {
            current += c;
        }
    }

    values.push_back(
        Trim(current));

    return values;
}

double
ParseDouble(
    const std::string& value,
    double defaultValue = 0.0)
{
    try
    {
        if (value.empty())
        {
            return defaultValue;
        }

        return std::stod(value);
    }
    catch (...)
    {
        return defaultValue;
    }
}

/*
 * ============================================================
 * CSV COLUMN LOOKUP
 * ============================================================
 */

int
GetColumnIndex(
    const std::map<std::string, int>& columns,
    const std::string& name)
{
    auto iterator =
        columns.find(name);

    if (iterator == columns.end())
    {
        return -1;
    }

    return iterator->second;
}

/*
 * ============================================================
 * LOAD SUMO TELEMETRY
 * ============================================================
 */

bool
LoadSumoTelemetry(
    const std::string& filename,
    std::map<std::string, VehicleTrajectory>& trajectories,
    double& simulationEndTime,
    double& minX,
    double& maxX,
    double& minY,
    double& maxY)
{
    std::ifstream file(filename);

    if (!file.is_open())
    {
        std::cerr
            << std::endl
            << "ERROR: Unable to open telemetry file:"
            << std::endl
            << filename
            << std::endl;

        return false;
    }

    std::string line;

    if (!std::getline(
            file,
            line))
    {
        std::cerr
            << "ERROR: Empty telemetry file."
            << std::endl;

        return false;
    }

    std::vector<std::string> header =
        SplitCsvLine(line);

    std::map<std::string, int> columns;

    for (uint32_t i = 0;
         i < header.size();
         ++i)
    {
        columns[
            Trim(header[i])] =
                static_cast<int>(i);
    }

    int timeColumn =
        GetColumnIndex(
            columns,
            "time");

    int vehicleIdColumn =
        GetColumnIndex(
            columns,
            "vehicle_id");

    int xColumn =
        GetColumnIndex(
            columns,
            "x");

    int yColumn =
        GetColumnIndex(
            columns,
            "y");

    int speedColumn =
        GetColumnIndex(
            columns,
            "speed");

    int accelerationColumn =
        GetColumnIndex(
            columns,
            "acceleration");

    int angleColumn =
        GetColumnIndex(
            columns,
            "angle");

    int edgeColumn =
        GetColumnIndex(
            columns,
            "edge_id");

    int laneColumn =
        GetColumnIndex(
            columns,
            "lane_id");

    if (timeColumn < 0 ||
        vehicleIdColumn < 0 ||
        xColumn < 0 ||
        yColumn < 0)
    {
        std::cerr
            << std::endl
            << "ERROR: Required CSV columns missing."
            << std::endl;

        std::cerr
            << "Required columns:"
            << std::endl;

        std::cerr
            << "time, vehicle_id, x, y"
            << std::endl;

        return false;
    }

    simulationEndTime = 0.0;

    minX =
        std::numeric_limits<double>::max();

    maxX =
        std::numeric_limits<double>::lowest();

    minY =
        std::numeric_limits<double>::max();

    maxY =
        std::numeric_limits<double>::lowest();

    uint64_t recordCount = 0;

    while (std::getline(
        file,
        line))
    {
        if (line.empty())
        {
            continue;
        }

        std::vector<std::string> values =
            SplitCsvLine(line);

        int requiredMaximum =
            std::max(
                std::max(
                    timeColumn,
                    vehicleIdColumn),
                std::max(
                    xColumn,
                    yColumn));

        if (static_cast<int>(
                values.size()) <=
            requiredMaximum)
        {
            continue;
        }

        TelemetryRecord record;

        record.time =
            ParseDouble(
                values[timeColumn]);

        record.vehicleId =
            values[vehicleIdColumn];

        record.x =
            ParseDouble(
                values[xColumn]);

        record.y =
            ParseDouble(
                values[yColumn]);

        record.speed =
            speedColumn >= 0 &&
                    speedColumn <
                        static_cast<int>(
                            values.size())
                ? ParseDouble(
                      values[speedColumn])
                : 0.0;

        record.acceleration =
            accelerationColumn >= 0 &&
                    accelerationColumn <
                        static_cast<int>(
                            values.size())
                ? ParseDouble(
                      values[accelerationColumn])
                : 0.0;

        record.angle =
            angleColumn >= 0 &&
                    angleColumn <
                        static_cast<int>(
                            values.size())
                ? ParseDouble(
                      values[angleColumn])
                : 0.0;

        record.edgeId =
            edgeColumn >= 0 &&
                    edgeColumn <
                        static_cast<int>(
                            values.size())
                ? values[edgeColumn]
                : "";

        record.laneId =
            laneColumn >= 0 &&
                    laneColumn <
                        static_cast<int>(
                            values.size())
                ? values[laneColumn]
                : "";

        if (record.vehicleId.empty())
        {
            continue;
        }

        VehicleTrajectory& trajectory =
            trajectories[
                record.vehicleId];

        trajectory.vehicleId =
            record.vehicleId;

        trajectory.records.push_back(
            record);

        simulationEndTime =
            std::max(
                simulationEndTime,
                record.time);

        minX =
            std::min(
                minX,
                record.x);

        maxX =
            std::max(
                maxX,
                record.x);

        minY =
            std::min(
                minY,
                record.y);

        maxY =
            std::max(
                maxY,
                record.y);

        ++recordCount;
    }

    /*
     * Sort trajectories by time.
     */

    for (auto& item :
         trajectories)
    {
        std::sort(
            item.second.records.begin(),
            item.second.records.end(),
            [](
                const TelemetryRecord& a,
                const TelemetryRecord& b)
            {
                return a.time < b.time;
            });
    }

    std::cout
        << std::endl
        << "===================================================="
        << std::endl;

    std::cout
        << "SUMO TELEMETRY LOADED"
        << std::endl;

    std::cout
        << "===================================================="
        << std::endl;

    std::cout
        << "Records: "
        << recordCount
        << std::endl;

    std::cout
        << "Vehicles: "
        << trajectories.size()
        << std::endl;

    std::cout
        << "Simulation end time: "
        << simulationEndTime
        << " seconds"
        << std::endl;

    std::cout
        << "SUMO X range: "
        << minX
        << " to "
        << maxX
        << std::endl;

    std::cout
        << "SUMO Y range: "
        << minY
        << " to "
        << maxY
        << std::endl;

    std::cout
        << "===================================================="
        << std::endl;

    return true;
}

/*
 * ============================================================
 * MAIN
 * ============================================================
 */

int
main(
    int argc,
    char* argv[])
{
    /*
     * ========================================================
     * DEFAULT PARAMETERS
     * ========================================================
     */

    std::string telemetryFile =
        "indore_01_telemetry_v5.csv";

    std::string outputFile =
        "indore-nr-sumo-results.csv";

    double centralFrequency =
        28e9;

    double bandwidth =
        100e6;

    double totalTxPower =
        30.0;

    uint32_t packetSize =
        512;

    uint32_t packetRate =
        100;

    /*
     * ========================================================
     * COMMAND LINE
     * ========================================================
     */

    CommandLine cmd(
        __FILE__);

    cmd.AddValue(
        "telemetryFile",
        "SUMO telemetry CSV file",
        telemetryFile);

    cmd.AddValue(
        "outputFile",
        "Output CSV file",
        outputFile);

    cmd.AddValue(
        "frequency",
        "Carrier frequency in Hz",
        centralFrequency);

    cmd.AddValue(
        "bandwidth",
        "Bandwidth in Hz",
        bandwidth);

    cmd.AddValue(
        "txPower",
        "gNB transmission power in dBm",
        totalTxPower);

    cmd.AddValue(
        "packetSize",
        "UDP packet size in bytes",
        packetSize);

    cmd.AddValue(
        "packetRate",
        "UDP packets per second per vehicle",
        packetRate);

    cmd.Parse(
        argc,
        argv);

    /*
     * ========================================================
     * LOAD SUMO TELEMETRY
     * ========================================================
     */

    std::map<
        std::string,
        VehicleTrajectory>
        trajectories;

    double sumoEndTime =
        0.0;

    double minX =
        0.0;

    double maxX =
        0.0;

    double minY =
        0.0;

    double maxY =
        0.0;

    bool loaded =
        LoadSumoTelemetry(
            telemetryFile,
            trajectories,
            sumoEndTime,
            minX,
            maxX,
            minY,
            maxY);

    if (!loaded)
    {
        return 1;
    }

    if (trajectories.empty())
    {
        std::cerr
            << "ERROR: No vehicle trajectories found."
            << std::endl;

        return 1;
    }

    /*
     * Add one second after SUMO finishes.
     */

    double simulationDuration =
        sumoEndTime + 1.0;

    Time simTime =
        Seconds(
            simulationDuration);

    uint32_t numVehicles =
        trajectories.size();

    /*
     * ========================================================
     * PRINT CONFIGURATION
     * ========================================================
     */

    std::cout
        << std::endl
        << "===================================================="
        << std::endl;

    std::cout
        << "INDORE SUMO-DRIVEN 5G NR SIMULATION"
        << std::endl;

    std::cout
        << "===================================================="
        << std::endl;

    std::cout
        << "Vehicles from SUMO: "
        << numVehicles
        << std::endl;

    std::cout
        << "SUMO duration: "
        << sumoEndTime
        << " seconds"
        << std::endl;

    std::cout
        << "ns-3 duration: "
        << simulationDuration
        << " seconds"
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

    /*
     * ========================================================
     * CREATE NR AND EPC HELPERS
     * ========================================================
     */

    Ptr<NrHelper> nrHelper =
        CreateObject<NrHelper>();

    Ptr<IdealBeamformingHelper>
        idealBeamformingHelper =
            CreateObject<
                IdealBeamformingHelper>();

    nrHelper->SetBeamformingHelper(
        idealBeamformingHelper);

    Ptr<NrPointToPointEpcHelper>
        nrEpcHelper =
            CreateObject<
                NrPointToPointEpcHelper>();

    nrHelper->SetEpcHelper(
        nrEpcHelper);

    /*
     * ========================================================
     * CREATE NODES
     * ========================================================
     */

    NodeContainer gnbNodes;

    gnbNodes.Create(
        1);

    NodeContainer vehicleNodes;

    vehicleNodes.Create(
        numVehicles);

    Ptr<Node> remoteHost =
        CreateObject<Node>();

    NodeContainer remoteHostContainer;

    remoteHostContainer.Add(
        remoteHost);

    /*
     * ========================================================
     * SCENARIO SIZE
     * ========================================================
     */

    double scenarioWidth =
        maxX - minX;

    double scenarioHeight =
        maxY - minY;

    double gnbX =
        (minX + maxX) / 2.0;

    double gnbY =
        (minY + maxY) / 2.0;

    std::cout
        << std::endl;

    std::cout
        << "Normalized scenario size: "
        << scenarioWidth
        << " m x "
        << scenarioHeight
        << " m"
        << std::endl;

    std::cout
        << "gNB position: ("
        << gnbX
        << ", "
        << gnbY
        << ", 10)"
        << std::endl;

    /*
     * ========================================================
     * gNB MOBILITY
     * ========================================================
     */

    MobilityHelper gnbMobility;

    Ptr<ListPositionAllocator>
        gnbPositionAllocator =
            CreateObject<
                ListPositionAllocator>();

    gnbPositionAllocator->Add(
        Vector(
            gnbX,
            gnbY,
            10.0));

    gnbMobility.SetPositionAllocator(
        gnbPositionAllocator);

    gnbMobility.SetMobilityModel(
        "ns3::ConstantPositionMobilityModel");

    gnbMobility.Install(
        gnbNodes);

    /*
     * ========================================================
     * VEHICLE MOBILITY
     * ========================================================
     */

    MobilityHelper vehicleMobility;

    vehicleMobility.SetMobilityModel(
        "ns3::WaypointMobilityModel");

    vehicleMobility.Install(
        vehicleNodes);

    /*
     * Mapping between SUMO vehicle IDs
     * and ns-3 UE indexes.
     */

    std::vector<std::string>
        vehicleIdList;

    std::map<
        std::string,
        uint32_t>
        vehicleToUeIndex;

    uint32_t ueIndex =
        0;

    for (const auto& item :
         trajectories)
    {
        vehicleIdList.push_back(
            item.first);

        vehicleToUeIndex[
            item.first] =
                ueIndex;

        ++ueIndex;
    }

    /*
     * Add SUMO waypoints.
     */

    uint32_t printedMappings =
        0;

    for (const auto& item :
         trajectories)
    {
        const std::string& vehicleId =
            item.first;

        const VehicleTrajectory&
            trajectory =
                item.second;

        uint32_t index =
            vehicleToUeIndex[
                vehicleId];

        Ptr<WaypointMobilityModel>
            mobility =
                vehicleNodes
                    .Get(index)
                    ->GetObject<
                        WaypointMobilityModel>();

        if (!mobility)
        {
            std::cerr
                << "ERROR: WaypointMobilityModel not found."
                << std::endl;

            return 1;
        }

        bool firstWaypoint =
            true;

        uint32_t waypointCount =
            0;

        for (const TelemetryRecord&
                 record :
             trajectory.records)
        {
            double waypointTime =
                record.time;

            if (waypointTime < 0.0)
            {
                continue;
            }

            /*
             * Waypoint times must be strictly increasing.
             */

            if (!firstWaypoint &&
                waypointTime <=
                    trajectory
                        .records[
                            waypointCount - 1]
                        .time)
            {
                continue;
            }

            mobility->AddWaypoint(
                Waypoint(
                    Seconds(
                        waypointTime),
                    Vector(
                        record.x,
                        record.y,
                        1.5)));

            firstWaypoint =
                false;

            ++waypointCount;
        }

        if (trajectory.records.empty())
        {
            continue;
        }

        /*
         * Ensure initial position exists.
         */

        const TelemetryRecord&
            firstRecord =
                trajectory.records.front();

        mobility->SetPosition(
            Vector(
                firstRecord.x,
                firstRecord.y,
                1.5));

        if (printedMappings < 10)
        {
            std::cout
                << "Mapped SUMO vehicle "
                << vehicleId
                << " -> ns-3 UE "
                << index
                << " | "
                << waypointCount
                << " waypoints"
                << std::endl;

            ++printedMappings;
        }
    }

    if (numVehicles > 10)
    {
        std::cout
            << "... "
            << numVehicles - 10
            << " additional vehicles mapped."
            << std::endl;
    }

    /*
     * ========================================================
     * CREATE NR OPERATION BAND
     * ========================================================
     */

    CcBwpCreator
        ccBwpCreator;

    const uint8_t
        numCcPerBand =
            1;

    CcBwpCreator::
        SimpleOperationBandConf
            bandConf(
                centralFrequency,
                bandwidth,
                numCcPerBand);

    OperationBandInfo band =
        ccBwpCreator
            .CreateOperationBandContiguousCc(
                bandConf);

    /*
     * ========================================================
     * CHANNEL CONFIGURATION
     * ========================================================
     */

    Ptr<NrChannelHelper>
        channelHelper =
            CreateObject<
                NrChannelHelper>();

    channelHelper->ConfigureFactories(
        "UMi",
        "Default",
        "ThreeGpp");

    channelHelper
        ->SetChannelConditionModelAttribute(
            "UpdatePeriod",
            TimeValue(
                MilliSeconds(
                    0)));

    channelHelper
        ->SetPathlossAttribute(
            "ShadowingEnabled",
            BooleanValue(
                false));

    channelHelper
        ->AssignChannelsToBands(
            {band});

    BandwidthPartInfoPtrVector
        allBwps =
            CcBwpCreator::
                GetAllBwps(
                    {band});

    /*
     * ========================================================
     * ANTENNA CONFIGURATION
     * ========================================================
     */

    nrHelper->SetUeAntennaAttribute(
        "NumRows",
        UintegerValue(
            2));

    nrHelper->SetUeAntennaAttribute(
        "NumColumns",
        UintegerValue(
            2));

    nrHelper->SetGnbAntennaAttribute(
        "NumRows",
        UintegerValue(
            4));

    nrHelper->SetGnbAntennaAttribute(
        "NumColumns",
        UintegerValue(
            4));

    /*
     * ========================================================
     * INSTALL NR DEVICES
     * ========================================================
     */

    NetDeviceContainer
        gnbDevices =
            nrHelper
                ->InstallGnbDevice(
                    gnbNodes,
                    allBwps);

    NetDeviceContainer
        vehicleDevices =
            nrHelper
                ->InstallUeDevice(
                    vehicleNodes,
                    allBwps);

    /*
     * ========================================================
     * PHY CONFIGURATION
     *
     * IMPORTANT:
     *
     * We do NOT set the "Numerology" attribute
     * directly on NrUePhy.
     *
     * That attribute does not exist in your ns-3.48
     * implementation and caused the previous crash.
     *
     * ========================================================
     */

    NrHelper::GetGnbPhy(
        gnbDevices.Get(
            0),
        0)
        ->SetAttribute(
            "TxPower",
            DoubleValue(
                totalTxPower));

    /*
     * ========================================================
     * ASSIGN STREAMS
     * ========================================================
     */

    nrHelper->AssignStreams(
        {
            .scenario = nullptr,
            .gnbDevs = gnbDevices,
            .ueDevs = vehicleDevices
        });

    /*
     * ========================================================
     * INSTALL INTERNET STACK
     * ========================================================
     */

    InternetStackHelper
        internet;

    internet.Install(
        remoteHostContainer);

    internet.Install(
        vehicleNodes);

    /*
     * ========================================================
     * CONNECT REMOTE HOST TO EPC
     * ========================================================
     */

    Ptr<Node>
        pgw =
            nrEpcHelper
                ->GetPgwNode();

    PointToPointHelper
        p2ph;

    p2ph.SetDeviceAttribute(
        "DataRate",
        StringValue(
            "100Gb/s"));

    p2ph.SetChannelAttribute(
        "Delay",
        StringValue(
            "1ms"));

    NetDeviceContainer
        internetDevices =
            p2ph.Install(
                pgw,
                remoteHost);

    Ipv4AddressHelper
        ipv4h;

    ipv4h.SetBase(
        "1.0.0.0",
        "255.0.0.0");

    Ipv4InterfaceContainer
        internetIpInterfaces =
            ipv4h.Assign(
                internetDevices);

    Ipv4Address
        remoteHostAddress =
            internetIpInterfaces.GetAddress(
                1);

    /*
     * ========================================================
     * UE IP ADDRESSES
     * ========================================================
     */

    Ipv4InterfaceContainer
        vehicleInterfaces =
            nrEpcHelper
                ->AssignUeIpv4Address(
                    NetDeviceContainer(
                        vehicleDevices));

    /*
     * ========================================================
     * ROUTING
     * ========================================================
     */

    Ipv4StaticRoutingHelper
        ipv4RoutingHelper;

    Ptr<Ipv4StaticRouting>
        remoteHostStaticRouting =
            ipv4RoutingHelper
                .GetStaticRouting(
                    remoteHost
                        ->GetObject<
                            Ipv4>());

    remoteHostStaticRouting
        ->AddNetworkRouteTo(
            Ipv4Address(
                "7.0.0.0"),
            Ipv4Mask(
                "255.0.0.0"),
            1);

    /*
     * Default route for every UE.
     */

    for (uint32_t i = 0;
         i < vehicleNodes.GetN();
         ++i)
    {
        Ptr<Ipv4StaticRouting>
            ueStaticRouting =
                ipv4RoutingHelper
                    .GetStaticRouting(
                        vehicleNodes
                            .Get(i)
                            ->GetObject<
                                Ipv4>());

        ueStaticRouting
            ->SetDefaultRoute(
                nrEpcHelper
                    ->GetUeDefaultGatewayAddress(),
                1);
    }

    /*
     * ========================================================
     * ATTACH UEs TO CLOSEST gNB
     * ========================================================
     */

    nrHelper
        ->AttachToClosestGnb(
            vehicleDevices,
            gnbDevices);

    /*
     * ========================================================
     * UDP APPLICATIONS
     * ========================================================
     */

    uint16_t basePort =
        5000;

    ApplicationContainer
        serverApps;

    ApplicationContainer
        clientApps;

    for (uint32_t i = 0;
         i < numVehicles;
         ++i)
    {
        uint16_t port =
            basePort + i;

        UdpServerHelper
            server(
                port);

        ApplicationContainer
            serverApp =
                server.Install(
                    vehicleNodes.Get(
                        i));

        serverApp.Start(
            Seconds(
                0.01));

        serverApp.Stop(
            simTime);

        serverApps.Add(
            serverApp);

        UdpClientHelper
            client(
                vehicleInterfaces
                    .GetAddress(
                        i),
                port);

        client.SetAttribute(
            "MaxPackets",
            UintegerValue(
                0xffffffff));

        client.SetAttribute(
            "Interval",
            TimeValue(
                Seconds(
                    1.0 /
                    static_cast<double>(
                        packetRate))));

        client.SetAttribute(
            "PacketSize",
            UintegerValue(
                packetSize));

        ApplicationContainer
            clientApp =
                client.Install(
                    remoteHost);

        /*
         * Stagger applications slightly.
         */

        double startTime =
            1.0 +
            static_cast<double>(
                i) *
                0.001;

        clientApp.Start(
            Seconds(
                startTime));

        clientApp.Stop(
            Seconds(
                sumoEndTime));

        clientApps.Add(
            clientApp);
    }

    /*
     * ========================================================
     * FLOW MONITOR
     * ========================================================
     */

    FlowMonitorHelper
        flowmonHelper;

    Ptr<FlowMonitor>
        monitor =
            flowmonHelper.InstallAll();

    /*
     * ========================================================
     * OUTPUT FILE
     * ========================================================
     */

    std::ofstream
        resultsFile(
            outputFile);

    if (!resultsFile.is_open())
    {
        std::cerr
            << "ERROR: Cannot create output file:"
            << std::endl;

        std::cerr
            << outputFile
            << std::endl;

        return 1;
    }

    resultsFile
        << "vehicle_id,"
        << "ue_index,"
        << "source_address,"
        << "destination_address,"
        << "tx_packets,"
        << "rx_packets,"
        << "lost_packets,"
        << "packet_loss_ratio,"
        << "throughput_mbps,"
        << "mean_delay_ms,"
        << "mean_jitter_ms"
        << std::endl;

    /*
     * ========================================================
     * RUN SIMULATION
     * ========================================================
     */

    std::cout
        << std::endl;

    std::cout
        << "Starting simulation..."
        << std::endl;

    Simulator::Stop(
        simTime);

    Simulator::Run();

    std::cout
        << "Simulation completed."
        << std::endl;

    /*
     * ========================================================
     * FLOW RESULTS
     * ========================================================
     */

    monitor
        ->CheckForLostPackets();

    Ptr<Ipv4FlowClassifier>
        classifier =
            DynamicCast<
                Ipv4FlowClassifier>(
                flowmonHelper
                    .GetClassifier());

    FlowMonitor::
        FlowStatsContainer
            stats =
                monitor
                    ->GetFlowStats();

    std::cout
        << std::endl;

    std::cout
        << "===================================================="
        << std::endl;

    std::cout
        << "FLOW RESULTS"
        << std::endl;

    std::cout
        << "===================================================="
        << std::endl;

    uint32_t validFlows =
        0;

    double totalThroughput =
        0.0;

    double totalDelay =
        0.0;

    /*
     * Mapping destination IP -> UE.
     */

    std::map<
        Ipv4Address,
        uint32_t>
        destinationToUe;

    for (uint32_t i = 0;
         i < numVehicles;
         ++i)
    {
        destinationToUe[
            vehicleInterfaces
                .GetAddress(
                    i)] =
                        i;
    }

    for (const auto& flow :
         stats)
    {
        FlowId flowId =
            flow.first;

        FlowMonitor::
            FlowStats
                flowStats =
                    flow.second;

        Ipv4FlowClassifier::
            FiveTuple
                tuple =
                    classifier
                        ->FindFlow(
                            flowId);

        /*
         * UDP only.
         */

        if (tuple.protocol != 17)
        {
            continue;
        }

        /*
         * Only remote host -> vehicle flows.
         */

        if (tuple.sourceAddress !=
            remoteHostAddress)
        {
            continue;
        }

        auto destinationIterator =
            destinationToUe.find(
                tuple.destinationAddress);

        if (destinationIterator ==
            destinationToUe.end())
        {
            continue;
        }

        uint32_t ue =
            destinationIterator
                ->second;

        std::string vehicleId =
            ue <
                vehicleIdList.size()
                ? vehicleIdList[ue]
                : "unknown";

        uint64_t txPackets =
            flowStats.txPackets;

        uint64_t rxPackets =
            flowStats.rxPackets;

        uint64_t lostPackets =
            txPackets >= rxPackets
                ? txPackets - rxPackets
                : 0;

        double packetLossRatio =
            txPackets > 0
                ? (
                    static_cast<double>(
                        lostPackets) /
                    static_cast<double>(
                        txPackets))
                    * 100.0
                : 0.0;

        double throughput =
            0.0;

        if (flowStats.rxPackets > 0)
        {
            double duration =
                flowStats.timeLastRxPacket
                    .GetSeconds() -
                flowStats.timeFirstTxPacket
                    .GetSeconds();

            if (duration > 0.0)
            {
                throughput =
                    (
                        flowStats.rxBytes *
                        8.0) /
                    duration /
                    1e6;
            }
        }

        double meanDelay =
            flowStats.rxPackets > 0
                ? (
                    flowStats.delaySum
                        .GetSeconds() /
                    flowStats.rxPackets)
                    * 1000.0
                : 0.0;

        double meanJitter =
            flowStats.rxPackets > 1
                ? (
                    flowStats.jitterSum
                        .GetSeconds() /
                    (
                        flowStats.rxPackets -
                        1))
                    * 1000.0
                : 0.0;

        std::cout
            << "Vehicle "
            << vehicleId
            << " | UE "
            << ue
            << std::endl;

        std::cout
            << "  Tx Packets: "
            << txPackets
            << std::endl;

        std::cout
            << "  Rx Packets: "
            << rxPackets
            << std::endl;

        std::cout
            << "  Lost Packets: "
            << lostPackets
            << std::endl;

        std::cout
            << std::fixed
            << std::setprecision(
                   4);

        std::cout
            << "  Packet Loss: "
            << packetLossRatio
            << " %"
            << std::endl;

        std::cout
            << "  Throughput: "
            << throughput
            << " Mbps"
            << std::endl;

        std::cout
            << "  Mean Delay: "
            << meanDelay
            << " ms"
            << std::endl;

        std::cout
            << "  Mean Jitter: "
            << meanJitter
            << " ms"
            << std::endl;

        resultsFile
            << vehicleId
            << ","
            << ue
            << ","
            << tuple.sourceAddress
            << ","
            << tuple.destinationAddress
            << ","
            << txPackets
            << ","
            << rxPackets
            << ","
            << lostPackets
            << ","
            << packetLossRatio
            << ","
            << throughput
            << ","
            << meanDelay
            << ","
            << meanJitter
            << std::endl;

        if (rxPackets > 0)
        {
            ++validFlows;

            totalThroughput +=
                throughput;

            totalDelay +=
                meanDelay;
        }
    }

    resultsFile.close();

    /*
     * ========================================================
     * SUMMARY
     * ========================================================
     */

    double averageThroughput =
        validFlows > 0
            ? totalThroughput /
                  validFlows
            : 0.0;

    double averageDelay =
        validFlows > 0
            ? totalDelay /
                  validFlows
            : 0.0;

    std::cout
        << std::endl;

    std::cout
        << "===================================================="
        << std::endl;

    std::cout
        << "SIMULATION SUMMARY"
        << std::endl;

    std::cout
        << "===================================================="
        << std::endl;

    std::cout
        << "Vehicles: "
        << numVehicles
        << std::endl;

    std::cout
        << "Valid flows: "
        << validFlows
        << std::endl;

    std::cout
        << "Average throughput: "
        << averageThroughput
        << " Mbps"
        << std::endl;

    std::cout
        << "Average delay: "
        << averageDelay
        << " ms"
        << std::endl;

    std::cout
        << "Results saved to: "
        << outputFile
        << std::endl;

    std::cout
        << "===================================================="
        << std::endl;

    /*
     * ========================================================
     * CLEANUP
     * ========================================================
     */

    Simulator::Destroy();

    return 0;
}
