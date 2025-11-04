/*
 *
 * This example demonstrates the use of Ns2MobilityHelper class to work with mobility.
 *
 * Modified to include a V2I DDoS attack scenario with:
 * - Mobile vehicle nodes (from trace)
 * - Static RSU (router) nodes, randomly placed
 * - 802.11p Wi-Fi in infrastructure mode (V2I)
 * - Random assignment of Attacker and Normal nodes
 * - UDP broadcast flood from Attackers to all non-attacker nodes
 * - Normal background traffic between Normal nodes
 * - NetAnim coloring for visualization
 */

#include "ns3/core-module.h"
#include "ns3/mobility-module.h"
#include "ns3/ns2-mobility-helper.h"
#include "ns3/netanim-module.h"

// --- Headers required for the network simulation ---
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/wifi-module.h"
#include "ns3/applications-module.h"
#include "ns3/ipv4-global-routing-helper.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <random>
#include <numeric>
#include <string> // --- Added for std::to_string ---

using namespace ns3;

// Prints actual position and velocity when a course change event occurs
static void
CourseChange(std::ostream* os, std::string foo, Ptr<const MobilityModel> mobility)
{
    Vector pos = mobility->GetPosition(); // Get position
    Vector vel = mobility->GetVelocity(); // Get velocity

    // Prints position and velocities
    *os << Simulator::Now() << " POS: x=" << pos.x << ", y=" << pos.y << ", z=" << pos.z
        << "; VEL:" << vel.x << ", y=" << vel.y << ", z=" << vel.z << std::endl;
}

// Example to use ns2 traces file in ns3
int
main(int argc, char* argv[])
{
    std::string traceFile;
    std::string logFile;

    int nodeNum;
    double duration;

    // --- New parameters for the DDoS scenario ---
    int numRsus = 5;       // Number of routers (RSUs)
    int numAttackers = 5;  // Number of attacker vehicles
    // --- REMOVED numVictims ---
    uint16_t attackPort = 9; // Port for UDP flood
    uint16_t normalPort = 90; // Port for normal traffic

    // --- New parameters for RSU grid placement ---
    double gridMinX = 0.0, gridMaxX = 1000.0;
    double gridMinY = 0.0, gridMaxY = 1000.0;

    // Enable logging from the ns2 helper
    LogComponentEnable("Ns2MobilityHelper", LOG_LEVEL_DEBUG);

    // Parse command line attribute
    CommandLine cmd(__FILE__);
    cmd.AddValue("traceFile", "Ns2 movement trace file", traceFile);
    cmd.AddValue("nodeNum", "Number of nodes in trace file", nodeNum);
    cmd.AddValue("duration", "Duration of Simulation", duration);
    cmd.AddValue("logFile", "Log file", logFile);
    cmd.AddValue("numRsus", "Number of RSUs (routers)", numRsus);
    cmd.AddValue("numAttackers", "Number of attacker nodes", numAttackers);
    // --- REMOVED cmd.AddValue("numVictims", ...) ---
    
    // --- Add new grid arguments ---
    cmd.AddValue("gridMinX", "Min X coordinate for RSU placement", gridMinX);
    cmd.AddValue("gridMaxX", "Max X coordinate for RSU placement", gridMaxX);
    cmd.AddValue("gridMinY", "Min Y coordinate for RSU placement", gridMinY);
    cmd.AddValue("gridMaxY", "Max Y coordinate for RSU placement", gridMaxY);
    cmd.Parse(argc, argv);

    // Check command line arguments
    if (traceFile.empty() || nodeNum <= 0 || duration <= 0 || logFile.empty())
    {
        std::cout << "Usage of " << argv[0]
                  << " :\n\n"
                     "./ns3 run \"<script_name>"
                     " --traceFile=src/mobility/examples/default.ns_movements"
                     " --nodeNum=2 --duration=100.0 --logFile=ns2-mob.log"
                     " --numRsus=5 --numAttackers=1\" \n\n" // --- Simplified usage string ---
                     "NOTE: ns2-traces-file could be an absolute or relative path.\n\n"
                     "NOTE 2: Number of nodes present in the trace file must match --nodeNum.\n\n"
                     "NOTE 3: Ensure nodeNum > numAttackers.\n\n" // --- Simplified note ---
                     "NOTE 4: Set --gridMinX/--gridMaxX/--gridMinY/--gridMaxY to match your trace file's boundaries.\n"
                     "        This ensures RSUs are placed in the same area as the vehicles.\n\n";

        return 0;
    }

    // --- MODIFIED Check: nodeNum must just be greater than numAttackers ---
    if (nodeNum <= numAttackers)
    {
        std::cerr << "Error: Not enough nodes for attackers and normal nodes." << std::endl;
        std::cerr << "nodeNum must be greater than numAttackers" << std::endl;
        return 1;
    }


    // --- 1. Create Nodes ---
    // Create mobile vehicle nodes (from trace)
    NodeContainer vehicleNodes;
    vehicleNodes.Create(nodeNum);

    // Create static RSU (router) nodes
    NodeContainer rsuNodes;
    rsuNodes.Create(numRsus);

    // --- 2. Set up Mobility ---
    // A. Apply NS2 trace to vehicle nodes
    Ns2MobilityHelper ns2 = Ns2MobilityHelper(traceFile);
    ns2.Install();

    // B. Set up static, random positions for RSUs
    MobilityHelper rsuMobility;
    rsuMobility.SetPositionAllocator(
        "ns3::RandomRectanglePositionAllocator",
        "X",
        StringValue("ns3::UniformRandomVariable[Min=" + std::to_string(gridMinX) + "|Max=" + std::to_string(gridMaxX) + "]"),
        "Y",
        StringValue("ns3::UniformRandomVariable[Min=" + std::to_string(gridMinY) + "|Max=" + std::to_string(gridMaxY) + "]"));
    
    // --- Set model back to ConstantPosition ---
    rsuMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    rsuMobility.Install(rsuNodes);

    // --- 3. Set up Wi-Fi (V2I) Network ---
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211p); // Vehicular standard
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                 "DataMode",
                                 StringValue("OfdmRate6Mbps"),
                                 "ControlMode",
                                 StringValue("OfdmRate6Mbps"));

    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    YansWifiPhyHelper phy;
    phy.SetChannel(channel.Create());

    // --- ADDED: Set TxPower for longer range ---
    phy.Set("TxPowerStart", DoubleValue(33.0));
    phy.Set("TxPowerEnd", DoubleValue(33.0));

    // Configure STA (vehicles) MAC in infrastructure mode
    WifiMacHelper staMac;
    staMac.SetType("ns3::StaWifiMac");

    // Configure AP (RSU) MAC
    WifiMacHelper apMac;
    apMac.SetType("ns3::ApWifiMac");

    // Install Wi-Fi devices
    NetDeviceContainer staDevices = wifi.Install(phy, staMac, vehicleNodes);
    NetDeviceContainer apDevices = wifi.Install(phy, apMac, rsuNodes);

    // --- 4. Install Internet Stack & IP Addresses ---
    InternetStackHelper stack;
    stack.Install(vehicleNodes);
    stack.Install(rsuNodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer staInterfaces = address.Assign(staDevices);
    Ipv4InterfaceContainer apInterfaces = address.Assign(apDevices);

    // Set up global routing
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    // --- 5. Role Assignment & NetAnim Coloring ---
    AnimationInterface anim("sumoDDOS.xml");

    // A. Color RSUs (Green)
    for (uint32_t i = 0; i < rsuNodes.GetN(); ++i)
    {
        anim.UpdateNodeColor(rsuNodes.Get(i), 0, 255, 0); // Green
        anim.UpdateNodeDescription(rsuNodes.Get(i), "RSU");
        // --- ADDED BACK: Set RSU positions for NetAnim ---
        Ptr<MobilityModel> mob = rsuNodes.Get(i)->GetObject<MobilityModel>();
        Vector pos = mob->GetPosition();
        anim.SetConstantPosition(rsuNodes.Get(i), pos.x, pos.y);
    }

    // B. Select Attackers and Normal nodes randomly
    std::vector<int> nodeIndices(nodeNum);
    std::iota(nodeIndices.begin(), nodeIndices.end(), 0); // Fill with 0, 1, ..., nodeNum-1
    
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(nodeIndices.begin(), nodeIndices.end(), g); // Shuffle indices

    NodeContainer attackerNodes;
    // --- REMOVED victimNodes ---
    NodeContainer normalNodes;
    Ptr<Node> normalServerNode; // One normal node will be a server

    std::cout << "--- Node Roles (by Original Node ID) ---" << std::endl;

    // Assign Attackers (Red)
    std::cout << "Attacker Nodes:" << std::endl;
    for (int i = 0; i < numAttackers; ++i)
    {
        Ptr<Node> attacker = vehicleNodes.Get(nodeIndices[i]);
        attackerNodes.Add(attacker);
        anim.UpdateNodeColor(attacker, 255, 0, 0); // Red
        anim.UpdateNodeDescription(attacker, "ATTACKER");
        std::cout << "  Node " << attacker->GetId() << std::endl;
    }

    // --- REMOVED Victim Assignment Block ---


    // Assign Normal Nodes (Blue)
    // --- MODIFIED: Start index is now numAttackers ---
    std::cout << "Normal Nodes (Targets):" << std::endl;
    // First normal node is the server
    normalServerNode = vehicleNodes.Get(nodeIndices[numAttackers]);
    anim.UpdateNodeColor(normalServerNode, 0, 0, 255); // Blue
    anim.UpdateNodeDescription(normalServerNode, "NORMAL_SERVER");
    std::cout << "  Node " << normalServerNode->GetId() << " (Server)" << std::endl;

    // The rest are normal clients
    for (int i = numAttackers + 1; i < nodeNum; ++i)
    {
        Ptr<Node> normal = vehicleNodes.Get(nodeIndices[i]);
        normalNodes.Add(normal);
        anim.UpdateNodeColor(normal, 0, 0, 255); // Blue
        anim.UpdateNodeDescription(normal, "NORMAL_CLIENT");
        std::cout << "  Node " << normal->GetId() << " (Client)" << std::endl;
    }
    std::cout << "----------------------------------------" << std::endl;


    // --- 6. Set up Applications ---

    // A. DDoS Attack Application (UDP Flood)
    
    // --- MODIFIED: Install Attack Sinks on ALL normal nodes ---
    PacketSinkHelper attackSinkHelper("ns3::UdpSocketFactory",
                                    InetSocketAddress(Ipv4Address::GetAny(), attackPort));
    ApplicationContainer attackSinks;
    attackSinks.Add(attackSinkHelper.Install(normalServerNode));
    attackSinks.Add(attackSinkHelper.Install(normalNodes));
    attackSinks.Start(Seconds(1.0));
    attackSinks.Stop(Seconds(duration - 1.0));


// --- MODIFIED: Install UDP Broadcast Flooders ---
    std::cout << "--- Attack Assignment (Broadcast) ---" << std::endl;
    
    // All attackers will send to the subnet broadcast address
    Ipv4Address broadcastAddress("10.1.1.255");

    // 1. Create the OnOffHelper, pointing to the broadcast address
    OnOffHelper ddosClient("ns3::UdpSocketFactory",
                           InetSocketAddress(broadcastAddress, attackPort));
    
    // 2. Configure the OnOff application for a constant flood
    ddosClient.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
    ddosClient.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
    ddosClient.SetAttribute("DataRate", DataRateValue(DataRate("50Mbps"))); 
    ddosClient.SetAttribute("PacketSize", UintegerValue(1024));

    // 5. Install the application
    ApplicationContainer attackerApps = ddosClient.Install(attackerNodes);
    attackerApps.Start(Seconds(2.0));
    attackerApps.Stop(Seconds(duration - 1.0));
    
    std::cout << "  All " << numAttackers << " attackers flooding " << broadcastAddress << " at 50Mbps" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    // B. Normal Traffic Application (OnOff)
    // (This section remains unchanged)
    PacketSinkHelper normalSinkHelper("ns3::UdpSocketFactory",
                                     InetSocketAddress(Ipv4Address::GetAny(), normalPort));
    ApplicationContainer serverSink = normalSinkHelper.Install(normalServerNode);
    serverSink.Start(Seconds(1.0));
    serverSink.Stop(Seconds(duration - 1.0));

    Ipv4Address serverAddress = normalServerNode->GetObject<Ipv4>()->GetAddress(1, 0).GetLocal();
    OnOffHelper normalClient("ns3::UdpSocketFactory",
                             InetSocketAddress(serverAddress, normalPort));
    normalClient.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
    normalClient.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
    normalClient.SetAttribute("DataRate", DataRateValue(DataRate("50kbps")));
    normalClient.SetAttribute("PacketSize", UintegerValue(512));

    ApplicationContainer normalApps = normalClient.Install(normalNodes);
    normalApps.Start(Seconds(3.0));
    normalApps.Stop(Seconds(duration - 1.0));


    // --- 7. Simulation Run ---
    // open log file for output
    std::ofstream os;
    os.open(logFile);

    // Configure callback for logging
    Config::Connect("/NodeList/*/$ns3::MobilityModel/CourseChange",
                    MakeBoundCallback(&CourseChange, &os));

    Simulator::Stop(Seconds(duration));
    Simulator::Run();
    Simulator::Destroy();

    os.close(); // close log file
    return 0;
}
