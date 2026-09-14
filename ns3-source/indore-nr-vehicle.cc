/*
 * Indore NR Vehicle Simulation
 *
 * Basic 5G NR simulation for C-V2X/OODA dataset generation.
 *
 * This version is compatible with:
 * ns-3.48 + 5G-LENA NR module
 *
 * It creates:
 * - Multiple vehicle-like UEs
 * - One or more gNBs
 * - A 5G NR network
 * - UDP traffic flows
 * - Flow-level communication metrics
 *
 * The mobility values can later be replaced with SUMO-derived
 * vehicle trajectories.
 */

#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/nr-module.h"
#include "ns3/point-to-point-module.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("IndoreNrVehicle");

int
main(int argc, char* argv[])
{
    /*
     * ============================================================
     * CONFIGURATION PARAMETERS
     * ============================================================
     */

    uint32_t numVehicles = 20;
    uint32_t numGnb = 1;

    double simTimeSeconds = 10.0;

    double centralFrequency = 28e9;
    double bandwidth = 100e6;

    uint16_t numerology = 2;
    double totalTxPower = 30.0;

    uint32_t packetSize = 512;
    uint32_t packetRate = 100;

    double vehicleSpacing = 5.0;

    std::string outputFile =
        "indore-nr-results.csv";

    CommandLine cmd(__FILE__);

    cmd.AddValue(
        "numVehicles",
        "Number of vehicle UEs",
        numVehicles);

    cmd.AddValue(
        "numGnb",
        "Number of gNBs",
        numGnb);

    cmd.AddValue(
        "simTime",
        "Simulation duration in seconds",
        simTimeSeconds);

    cmd.AddValue(
        "frequency",
        "NR carrier frequency in Hz",
        centralFrequency);

    cmd.AddValue(
        "bandwidth",
        "NR bandwidth in Hz",
        bandwidth);

    cmd.AddValue(
        "numerology",
        "NR numerology",
        numerology);

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

    cmd.AddValue(
        "outputFile",
        "CSV output file",
        outputFile);

    cmd.Parse(argc, argv);

    Time simTime = Seconds(simTimeSeconds);

    std::cout << std::endl;
    std::cout << "===================================================="
              << std::endl;
    std::cout << "INDORE 5G NR VEHICLE SIMULATION"
              << std::endl;
    std::cout << "===================================================="
              << std::endl;

    std::cout << "Vehicles: " << numVehicles << std::endl;
    std::cout << "gNBs: " << numGnb << std::endl;
    std::cout << "Simulation Time: "
              << simTimeSeconds
              << " seconds"
              << std::endl;

    std::cout << "Frequency: "
              << centralFrequency / 1e9
              << " GHz"
              << std::endl;

    std::cout << "Bandwidth: "
              << bandwidth / 1e6
              << " MHz"
              << std::endl;

    /*
     * ============================================================
     * CREATE NR AND EPC HELPERS
     * ============================================================
     */

    Ptr<NrHelper> nrHelper =
        CreateObject<NrHelper>();

    Ptr<IdealBeamformingHelper> idealBeamformingHelper =
        CreateObject<IdealBeamformingHelper>();

    nrHelper->SetBeamformingHelper(
        idealBeamformingHelper);

    Ptr<NrPointToPointEpcHelper> nrEpcHelper =
        CreateObject<NrPointToPointEpcHelper>();

    nrHelper->SetEpcHelper(
        nrEpcHelper);

    /*
     * ============================================================
     * CREATE NODES
     * ============================================================
     */

    NodeContainer gnbNodes;
    gnbNodes.Create(numGnb);

    NodeContainer vehicleNodes;
    vehicleNodes.Create(numVehicles);

    Ptr<Node> remoteHost =
        CreateObject<Node>();

    NodeContainer remoteHostContainer;
    remoteHostContainer.Add(remoteHost);

    /*
     * ============================================================
     * MOBILITY
     * ============================================================
     */

    /*
     * gNB positions
     */

    MobilityHelper gnbMobility;

    Ptr<ListPositionAllocator> gnbPositions =
        CreateObject<ListPositionAllocator>();

    for (uint32_t i = 0; i < numGnb; ++i)
    {
        double x =
            static_cast<double>(i) * 500.0;

        gnbPositions->Add(
            Vector(x, 0.0, 10.0));
    }

    gnbMobility.SetPositionAllocator(
        gnbPositions);

    gnbMobility.SetMobilityModel(
        "ns3::ConstantPositionMobilityModel");

    gnbMobility.Install(
        gnbNodes);

    /*
     * Vehicle positions
     *
     * These currently use a simple linear road.
     *
     * Later we will replace these coordinates
     * with SUMO mobility traces.
     */

    MobilityHelper vehicleMobility;

    Ptr<ListPositionAllocator> vehiclePositions =
        CreateObject<ListPositionAllocator>();

    for (uint32_t i = 0;
         i < numVehicles;
         ++i)
    {
        double x =
            static_cast<double>(i) *
            vehicleSpacing;

        double y = 20.0;

        vehiclePositions->Add(
            Vector(x, y, 1.5));
    }

    vehicleMobility.SetPositionAllocator(
        vehiclePositions);

    vehicleMobility.SetMobilityModel(
        "ns3::ConstantVelocityMobilityModel");

    vehicleMobility.Install(
        vehicleNodes);

    /*
     * Give vehicles velocity.
     */

    for (uint32_t i = 0;
         i < numVehicles;
         ++i)
    {
        Ptr<ConstantVelocityMobilityModel>
            mobility =
                vehicleNodes.Get(i)
                    ->GetObject<
                        ConstantVelocityMobilityModel>();

        if (mobility)
        {
            mobility->SetVelocity(
                Vector(
                    10.0,
                    0.0,
                    0.0));
        }
    }

    /*
     * ============================================================
     * CREATE 5G NR SPECTRUM CONFIGURATION
     * ============================================================
     */

    CcBwpCreator ccBwpCreator;

    const uint8_t numCcPerBand = 1;

    CcBwpCreator::SimpleOperationBandConf bandConf(
        centralFrequency,
        bandwidth,
        numCcPerBand);

    OperationBandInfo band =
        ccBwpCreator
            .CreateOperationBandContiguousCc(
                bandConf);

    /*
     * ============================================================
     * CHANNEL CONFIGURATION
     * ============================================================
     */

    Ptr<NrChannelHelper> channelHelper =
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

    channelHelper
        ->SetPathlossAttribute(
            "ShadowingEnabled",
            BooleanValue(false));

    channelHelper
        ->AssignChannelsToBands(
            {band});

    BandwidthPartInfoPtrVector allBwps =
        CcBwpCreator::GetAllBwps(
            {band});

    /*
     * ============================================================
     * ANTENNA CONFIGURATION
     * ============================================================
     */

    nrHelper->SetUeAntennaAttribute(
        "NumRows",
        UintegerValue(2));

    nrHelper->SetUeAntennaAttribute(
        "NumColumns",
        UintegerValue(2));

    nrHelper->SetGnbAntennaAttribute(
        "NumRows",
        UintegerValue(4));

    nrHelper->SetGnbAntennaAttribute(
        "NumColumns",
        UintegerValue(4));

    /*
     * ============================================================
     * INSTALL NR DEVICES
     * ============================================================
     */

    NetDeviceContainer gnbDevices =
        nrHelper->InstallGnbDevice(
            gnbNodes,
            allBwps);

    NetDeviceContainer vehicleDevices =
        nrHelper->InstallUeDevice(
            vehicleNodes,
            allBwps);

    /*
     * ============================================================
     * CONFIGURE PHY PARAMETERS
     * ============================================================
     */

    for (uint32_t i = 0;
         i < gnbDevices.GetN();
         ++i)
    {
        NrHelper::GetGnbPhy(
            gnbDevices.Get(i),
            0)
            ->SetAttribute(
                "Numerology",
                UintegerValue(
                    numerology));

        NrHelper::GetGnbPhy(
            gnbDevices.Get(i),
            0)
            ->SetAttribute(
                "TxPower",
                DoubleValue(
                    totalTxPower));
    }

    /*
     * ============================================================
     * EPC / INTERNET CONFIGURATION
     * ============================================================
     */

    Ptr<Node> pgw =
        nrEpcHelper->GetPgwNode();

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

    InternetStackHelper internet;

    internet.Install(
        remoteHostContainer);

    Ipv4AddressHelper ipv4h;

    ipv4h.SetBase(
        "1.0.0.0",
        "255.0.0.0");

    Ipv4InterfaceContainer internetInterfaces =
        ipv4h.Assign(
            internetDevices);

    Ipv4Address remoteHostAddress =
        internetInterfaces.GetAddress(1);

    Ipv4StaticRoutingHelper ipv4RoutingHelper;

    Ptr<Ipv4StaticRouting>
        remoteHostStaticRouting =
            ipv4RoutingHelper
                .GetStaticRouting(
                    remoteHost
                        ->GetObject<Ipv4>());

    remoteHostStaticRouting
        ->AddNetworkRouteTo(
            Ipv4Address("7.0.0.0"),
            Ipv4Mask("255.0.0.0"),
            1);

    /*
     * Install IP stack on vehicles.
     */

    internet.Install(
        vehicleNodes);

    Ipv4InterfaceContainer vehicleInterfaces =
        nrEpcHelper->AssignUeIpv4Address(
            NetDeviceContainer(
                vehicleDevices));

    for (uint32_t i = 0;
         i < numVehicles;
         ++i)
    {
        Ptr<Ipv4StaticRouting>
            vehicleStaticRouting =
                ipv4RoutingHelper
                    .GetStaticRouting(
                        vehicleNodes.Get(i)
                            ->GetObject<Ipv4>());

        vehicleStaticRouting
            ->SetDefaultRoute(
                nrEpcHelper
                    ->GetUeDefaultGatewayAddress(),
                1);
    }

    /*
     * ============================================================
     * ATTACH VEHICLES TO CLOSEST GNB
     * ============================================================
     */

    nrHelper->AttachToClosestGnb(
        vehicleDevices,
        gnbDevices);

    /*
     * ============================================================
     * UDP APPLICATIONS
     * ============================================================
     */

    uint16_t port = 5000;

    ApplicationContainer serverApps;
    ApplicationContainer clientApps;

    for (uint32_t i = 0;
         i < numVehicles;
         ++i)
    {
        UdpServerHelper server(
            port + i);

        ApplicationContainer serverApp =
            server.Install(
                vehicleNodes.Get(i));

        serverApps.Add(
            serverApp);

        UdpClientHelper client(
            vehicleInterfaces.GetAddress(i),
            port + i);

        client.SetAttribute(
            "MaxPackets",
            UintegerValue(
                packetRate *
                simTimeSeconds));

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

        ApplicationContainer clientApp =
            client.Install(
                remoteHost);

        clientApps.Add(
            clientApp);
    }

    serverApps.Start(
        Seconds(0.1));

    clientApps.Start(
        Seconds(0.2));

    serverApps.Stop(
        simTime);

    clientApps.Stop(
        simTime);

    /*
     * ============================================================
     * FLOW MONITOR
     * ============================================================
     */

    FlowMonitorHelper flowmonHelper;

    NodeContainer endpointNodes;

    endpointNodes.Add(
        remoteHost);

    endpointNodes.Add(
        vehicleNodes);

    Ptr<FlowMonitor> monitor =
        flowmonHelper.Install(
            endpointNodes);

    /*
     * ============================================================
     * RUN SIMULATION
     * ============================================================
     */

    std::cout << std::endl;

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
     * ============================================================
     * PROCESS RESULTS
     * ============================================================
     */

    monitor->CheckForLostPackets();

    Ptr<Ipv4FlowClassifier> classifier =
        DynamicCast<Ipv4FlowClassifier>(
            flowmonHelper
                .GetClassifier());

    FlowMonitor::FlowStatsContainer stats =
        monitor->GetFlowStats();

    std::ofstream csvFile(
        outputFile);

    csvFile
        << "flow_id,"
        << "source,"
        << "destination,"
        << "tx_packets,"
        << "rx_packets,"
        << "lost_packets,"
        << "throughput_mbps,"
        << "mean_delay_ms,"
        << "mean_jitter_ms"
        << std::endl;

    double totalThroughput = 0.0;
    double totalDelay = 0.0;

    uint32_t validFlows = 0;

    std::cout << std::endl;

    std::cout
        << "===================================================="
        << std::endl;

    std::cout
        << "FLOW RESULTS"
        << std::endl;

    std::cout
        << "===================================================="
        << std::endl;

    for (auto& flow : stats)
    {
        uint32_t flowId =
            flow.first;

        FlowMonitor::FlowStats flowStats =
            flow.second;

        Ipv4FlowClassifier::FiveTuple tuple =
            classifier->FindFlow(
                flowId);

        if (flowStats.txPackets == 0)
        {
            continue;
        }

        double throughputMbps = 0.0;

        double meanDelayMs = 0.0;

        double meanJitterMs = 0.0;

        uint32_t lostPackets =
            flowStats.txPackets -
            flowStats.rxPackets;

        if (flowStats.rxPackets > 0)
        {
            double duration =
                flowStats.timeLastRxPacket
                    .GetSeconds() -
                flowStats.timeFirstTxPacket
                    .GetSeconds();

            if (duration > 0.0)
            {
                throughputMbps =
                    (flowStats.rxBytes * 8.0) /
                    duration /
                    1e6;
            }

            meanDelayMs =
                flowStats.delaySum
                    .GetSeconds() *
                1000.0 /
                flowStats.rxPackets;

            meanJitterMs =
                flowStats.jitterSum
                    .GetSeconds() *
                1000.0 /
                flowStats.rxPackets;

            totalThroughput +=
                throughputMbps;

            totalDelay +=
                meanDelayMs;

            validFlows++;
        }

        std::cout
            << "Flow "
            << flowId
            << ": "
            << tuple.sourceAddress
            << " -> "
            << tuple.destinationAddress
            << std::endl;

        std::cout
            << "  Tx Packets: "
            << flowStats.txPackets
            << std::endl;

        std::cout
            << "  Rx Packets: "
            << flowStats.rxPackets
            << std::endl;

        std::cout
            << "  Lost Packets: "
            << lostPackets
            << std::endl;

        std::cout
            << "  Throughput: "
            << std::fixed
            << std::setprecision(4)
            << throughputMbps
            << " Mbps"
            << std::endl;

        std::cout
            << "  Mean Delay: "
            << meanDelayMs
            << " ms"
            << std::endl;

        std::cout
            << "  Mean Jitter: "
            << meanJitterMs
            << " ms"
            << std::endl;

        csvFile
            << flowId
            << ","
            << tuple.sourceAddress
            << ","
            << tuple.destinationAddress
            << ","
            << flowStats.txPackets
            << ","
            << flowStats.rxPackets
            << ","
            << lostPackets
            << ","
            << throughputMbps
            << ","
            << meanDelayMs
            << ","
            << meanJitterMs
            << std::endl;
    }

    csvFile.close();

    std::cout << std::endl;

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
        << "Valid flows: "
        << validFlows
        << std::endl;

    if (validFlows > 0)
    {
        std::cout
            << "Average throughput: "
            << totalThroughput /
               validFlows
            << " Mbps"
            << std::endl;

        std::cout
            << "Average delay: "
            << totalDelay /
               validFlows
            << " ms"
            << std::endl;
    }

    std::cout
        << "Results saved to: "
        << outputFile
        << std::endl;

    std::cout
        << "===================================================="
        << std::endl;

    Simulator::Destroy();

    return 0;
}
