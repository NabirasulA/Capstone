/* vanet_blackhole_sumo.cc
   NS-3 Blackhole attack demo with SUMO mobility integration.
   Description:
   A sender tries to send TCP traffic to a victim. A legitimate path exists via a router.
   However, a malicious blackhole node is also connected to the sender. We manually inject
   a static route into the sender's routing table, tricking it into sending all its traffic
   to the blackhole node, which then drops the packets.
   Requirements:
   - SUMO installed and available on PATH
   - A SUMO config file (e.g., "sumo_config.sumocfg")
   - Vehicles in SUMO with IDs: sender_car, victim_car, legit_router_car, blackhole_car
*/

#include <ns3/core-module.h>
#include <ns3/network-module.h>
#include <ns3/internet-module.h>
#include <ns3/point-to-point-module.h>
#include <ns3/applications-module.h>
#include <ns3/mobility-module.h>
#include <ns3/ipv4-global-routing-helper.h>
#include <ns3/netanim-module.h>
#include <ns3/ipv4-static-routing-helper.h>
#include <ns3/ipv4-list-routing-helper.h>

#include <fstream>
#include <string>
#include <unordered_map>
#include <regex>
#include <cstdlib>

using namespace ns3;

#define TCP_SINK_PORT 9000
#define MAX_SIMULATION_TIME 20.0
#define MAX_BULK_BYTES 1000000

NS_LOG_COMPONENT_DEFINE("BlackholeAttackWithSUMO");

// --- SUMO Integration ---
static const std::string SUMO_CONFIG = "sumo_config.sumocfg";
static const std::string SUMO_BIN = "sumo"; // or "sumo-gui"
static const std::string SUMO_FCD_OUTPUT = "sumo_fcd_output.xml";
static const double sumoStep = 0.1; // seconds

static NodeContainer g_allNodes;
static std::vector<std::string> g_vehicleIds;

static void ParseSumoFcd(const std::string &filename, std::unordered_map<std::string, std::pair<double, double>> &outPositions)
{
    outPositions.clear();
    std::ifstream ifs(filename);
    if (!ifs.is_open()) return;

    std::string line;
    std::regex vehRegex("<vehicle[^>]\\bid\\s*=\\s*\"([^\"]+)\"[^>]\\bx\\s*=\\s*\"([^\"]+)\"[^>]\\by\\s*=\\s*\"([^\"]+)\"[^>]*>");
    std::smatch match;
    while (std::getline(ifs, line))
    {
        if (std::regex_search(line, match, vehRegex) && match.size() >= 4)
        {
            try
            {
                outPositions[match[1].str()] = {std::stod(match[2].str()), std::stod(match[3].str())};
            }
            catch (...) {}
        }
    }
}

static void UpdatePositionsFromSUMO()
{
    std::unordered_map<std::string, std::pair<double, double>> positions;
    ParseSumoFcd(SUMO_FCD_OUTPUT, positions);

    for (uint32_t i = 0; i < g_allNodes.GetN(); ++i)
    {
        std::string vid = g_vehicleIds[i];
        auto it = positions.find(vid);
        if (it != positions.end())
        {
            Ptr<MobilityModel> mm = g_allNodes.Get(i)->GetObject<MobilityModel>();
            if (mm)
            {
                mm->SetPosition(Vector(it->second.first, it->second.second, 0.0));
            }
        }
    }
    Simulator::Schedule(Seconds(sumoStep), &UpdatePositionsFromSUMO);
}
// --- End SUMO Integration ---


int main(int argc, char *argv[])
{
    CommandLine cmd;
    cmd.Parse(argc, argv);

    Time::SetResolution(Time::NS);
    LogComponentEnable("PacketSink", LOG_LEVEL_INFO);

    // --- 1. Create Nodes ---
    // Nodes: Sender, Victim, a legitimate Router, and a malicious Blackhole node
    NodeContainer nodes;
    nodes.Create(4);
    Ptr<Node> senderNode = nodes.Get(0);
    Ptr<Node> victimNode = nodes.Get(1);
    Ptr<Node> legitRouterNode = nodes.Get(2);
    Ptr<Node> blackholeNode = nodes.Get(3);
    g_allNodes = nodes;

    // --- 2. Create Links ---
    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue("100Mbps"));
    // Legitimate path has higher delay
    p2p.SetChannelAttribute("Delay", StringValue("10ms"));
    NetDeviceContainer d_sender_router = p2p.Install(senderNode, legitRouterNode);
    NetDeviceContainer d_router_victim = p2p.Install(legitRouterNode, victimNode);

    // Malicious path has lower delay to seem more attractive
    p2p.SetChannelAttribute("Delay", StringValue("1ms"));
    NetDeviceContainer d_sender_blackhole = p2p.Install(senderNode, blackholeNode);

    // --- 3. Install Internet Stack & IP Addresses ---
    // Use a list router to prioritize our fake static route
    Ipv4ListRoutingHelper listRouting;
    Ipv4StaticRoutingHelper staticRoutingHelper;
    listRouting.Add(staticRoutingHelper, 10); // High priority for static routes

    InternetStackHelper stack;
    stack.SetRoutingHelper(listRouting);
    stack.Install(nodes);

    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.1.1.0", "255.255.255.0");
    ipv4.Assign(d_sender_router);
    ipv4.SetBase("10.1.2.0", "255.255.255.0");
    Ipv4InterfaceContainer i_router_victim = ipv4.Assign(d_router_victim);
    ipv4.SetBase("10.1.3.0", "255.255.255.0");
    Ipv4InterfaceContainer i_sender_blackhole = ipv4.Assign(d_sender_blackhole);

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    // --- 4. Blackhole Attack Configuration ---
    NS_LOG_INFO("Configuring blackhole attack...");
    // The blackhole node does not forward any packets it receives
    blackholeNode->GetObject<Ipv4>()->SetAttribute("IpForward", BooleanValue(false));

    // Inject a malicious static route into the SENDER's routing table
    Ptr<Ipv4> ipv4Sender = senderNode->GetObject<Ipv4>();
    Ptr<Ipv4StaticRouting> staticRouting = Ipv4RoutingHelper::GetRouting<Ipv4StaticRouting>(ipv4Sender->GetRoutingProtocol());
    
    Ipv4Address victimIp = i_router_victim.GetAddress(1);
    Ipv4Address blackholeIp = i_sender_blackhole.GetAddress(1);
    uint32_t senderIfToBlackhole = ipv4Sender->GetInterfaceForDevice(d_sender_blackhole.Get(0));

    // This route tells the sender: "To get to the victim, send packets to the blackhole node"
    staticRouting->AddHostRouteTo(victimIp, blackholeIp, senderIfToBlackhole);
    
    // --- 5. Application Setup ---
    // Legitimate TCP sender trying to reach the victim
    BulkSendHelper bulkSend("ns3::TcpSocketFactory", InetSocketAddress(victimIp, TCP_SINK_PORT));
    bulkSend.SetAttribute("MaxBytes", UintegerValue(MAX_BULK_BYTES));
    ApplicationContainer senderApp = bulkSend.Install(senderNode);
    senderApp.Start(Seconds(1.0));
    senderApp.Stop(Seconds(MAX_SIMULATION_TIME - 1.0));

    // Victim's TCP sink application
    PacketSinkHelper tcpSink("ns3::TcpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), TCP_SINK_PORT));
    ApplicationContainer sinkApp = tcpSink.Install(victimNode);
    sinkApp.Start(Seconds(0.0));
    sinkApp.Stop(Seconds(MAX_SIMULATION_TIME));

    // --- 6. Mobility and SUMO Launch ---
    MobilityHelper mobility;
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(nodes);
    
    // This gives NetAnim a good initial layout before SUMO takes over.
    Ptr<MobilityModel> senderMobility = senderNode->GetObject<MobilityModel>();
    senderMobility->SetPosition(Vector(0.0, 20.0, 0.0));

    Ptr<MobilityModel> victimMobility = victimNode->GetObject<MobilityModel>();
    victimMobility->SetPosition(Vector(40.0, 20.0, 0.0));

    Ptr<MobilityModel> routerMobility = legitRouterNode->GetObject<MobilityModel>();
    routerMobility->SetPosition(Vector(20.0, 30.0, 0.0));

    Ptr<MobilityModel> blackholeMobility = blackholeNode->GetObject<MobilityModel>();
    blackholeMobility->SetPosition(Vector(20.0, 10.0, 0.0));
    
    g_vehicleIds = {"sender_car", "victim_car", "legit_router_car", "blackhole_car"};

    std::ostringstream sumoCmd;
    sumoCmd << SUMO_BIN << " -c " << SUMO_CONFIG << " --step-length " << sumoStep
            << " --fcd-output " << SUMO_FCD_OUTPUT << " > sumo_stdout.log 2> sumo_stderr.log &";
    NS_LOG_UNCOND("Launching SUMO: " << sumoCmd.str());
    if (std::system(sumoCmd.str().c_str()) != 0) {
        NS_LOG_WARN("Failed to launch SUMO. Please start it manually.");
    }
    Simulator::Schedule(Seconds(sumoStep + 0.01), &UpdatePositionsFromSUMO);

    // --- 7. NetAnim Visualization ---
    AnimationInterface anim("BlackholeSim_SUMO.xml");
    anim.UpdateNodeDescription(senderNode, "Sender");
    anim.UpdateNodeDescription(victimNode, "Victim");
    anim.UpdateNodeDescription(legitRouterNode, "Legit Router");
    anim.UpdateNodeDescription(blackholeNode, "Blackhole");
    
    anim.UpdateNodeColor(senderNode, 0, 255, 0);       // Green
    anim.UpdateNodeColor(victimNode, 255, 0, 0);       // Red
    anim.UpdateNodeColor(legitRouterNode, 0, 0, 255);  // Blue
    anim.UpdateNodeColor(blackholeNode, 255, 165, 0);  // Orange

    // --- 8. Run Simulation ---
    Simulator::Stop(Seconds(MAX_SIMULATION_TIME));
    Simulator::Run();

    // Check results: the victim should have received 0 bytes
    Ptr<PacketSink> sink = DynamicCast<PacketSink>(sinkApp.Get(0));
    NS_LOG_UNCOND("Attack finished. Total bytes received by victim: " << sink->GetTotalRx());
    
    Simulator::Destroy();
    return 0;
}