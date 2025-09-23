/* vanet_ddos_sumo.cc
   NS-3 DDoS demo with SUMO mobility integration using SUMO FCD output parsing.
   Author: ChatGPT (example)
   Requirements:
    - SUMO installed and available on PATH
    - A SUMO config file named "sumo_config.sumocfg" (or change path below)
    - Vehicles in SUMO with IDs: sender_car, router_car, victim_car, attacker_0 ... attacker_(NUMBER_OF_BOTS-1)
*/

#include <ns3/core-module.h>
#include <ns3/network-module.h>
#include <ns3/internet-module.h>
#include <ns3/point-to-point-module.h>
#include <ns3/applications-module.h>
#include <ns3/mobility-module.h>
#include <ns3/ipv4-global-routing-helper.h>
#include <ns3/netanim-module.h>

#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <regex>
#include <cstdlib> // for system()

using namespace ns3;

#define TCP_SINK_PORT 9000
#define UDP_SINK_PORT 9001

// experimental parameters
#define MAX_BULK_BYTES 100000
#define DDOS_RATE "20480kb/s"
#define MAX_SIMULATION_TIME 10.0

// Number of Bots for DDoS
#define NUMBER_OF_BOTS 10

NS_LOG_COMPONENT_DEFINE("DDoSAttackWithSUMO");

// Path to SUMO config and output file (change as needed)
static const std::string SUMO_CONFIG = "sumo_config.sumocfg";
static const std::string SUMO_BIN = "sumo"; // or "sumo-gui"
static const std::string SUMO_FCD_OUTPUT = "sumo_fcd_output.xml";

// How often (ns) ns-3 reads SUMO output and updates positions (should match SUMO step-length)
static const double sumoStep = 0.1; // seconds

// Simple helper: parse SUMO FCD (xml) and fill mapping vehicleId -> (x,y)
static void ParseSumoFcd(const std::string &filename, std::unordered_map<std::string, std::pair<double, double>> &outPositions)
{
    outPositions.clear();
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        // file may not be ready yet
        return;
    }

    std::string line;
    // We'll parse lines like:
    // <vehicle id="sender_car" x="0.00" y="12.34" angle="0.00" speed="0.00" />
    // This simple parse uses regex to extract id/x/y attributes.
    std::regex vehRegex(
    "<vehicle[^>]\\bid\\s=\\s*\"([^\"]+)\"[^>]\\bx\\s=\\s*\"([^\"]+)\"[^>]\\by\\s=\\s*\"([^\"]+)\"[^>]*>",
    std::regex::icase
);
    std::smatch match;
    while (std::getline(ifs, line)) {
        std::string l = line;
        // quick search for "<vehicle"
        auto pos = l.find("<vehicle");
        if (pos == std::string::npos) continue;
        if (std::regex_search(l, match, vehRegex)) {
            if (match.size() >= 4) {
                std::string id = match[1].str();
                std::string xs = match[2].str();
                std::string ys = match[3].str();
                try {
                    double x = std::stod(xs);
                    double y = std::stod(ys);
                    outPositions[id] = std::make_pair(x, y);
                } catch (...) {
                    continue;
                }
            }
        } else {
            // handle single-line vehicle self-closing tags or different order by scanning attributes
            // naive attribute extraction:
            size_t idpos = l.find("id=\"");
            size_t xpos = l.find(" x=\"");
            size_t ypos = l.find(" y=\"");
            if (idpos != std::string::npos && xpos != std::string::npos && ypos != std::string::npos) {
                size_t idstart = idpos + 4;
                size_t idend = l.find("\"", idstart);
                size_t xstart = xpos + 4;
                size_t xend = l.find("\"", xstart);
                size_t ystart = ypos + 4;
                size_t yend = l.find("\"", ystart);
                if (idend != std::string::npos && xend != std::string::npos && yend != std::string::npos) {
                    std::string id = l.substr(idstart, idend - idstart);
                    std::string xs = l.substr(xstart, xend - xstart);
                    std::string ys = l.substr(ystart, yend - ystart);
                    try {
                        double x = std::stod(xs);
                        double y = std::stod(ys);
                        outPositions[id] = std::make_pair(x, y);
                    } catch (...) { }
                }
            }
        }
    }
}

// Global references for scheduled update
static NodeContainer g_nodes;
static NodeContainer g_botNodes;
static std::vector<std::string> g_vehicleIds; // mapping index -> vehicleId for node/bot
static Ptr<MobilityModel> GetNodeMobilityPtr(Ptr<Node> node)
{
    Ptr<MobilityModel> mm = node->GetObject<MobilityModel>();
    return mm;
}

// Update positions in ns-3 according to parsed SUMO positions
static void UpdatePositionsFromSUMO()
{
    std::unordered_map<std::string, std::pair<double, double>> positions;
    ParseSumoFcd(SUMO_FCD_OUTPUT, positions);

    // For each mapping, if SUMO provided coords, set ns-3 node position (z=0)
    for (uint32_t i = 0; i < g_nodes.GetN(); ++i) {
        std::string vid = g_vehicleIds[i];
        auto it = positions.find(vid);
        if (it != positions.end()) {
            double x = it->second.first;
            double y = it->second.second;
            Ptr<MobilityModel> mm = GetNodeMobilityPtr(g_nodes.Get(i));
            if (mm) {
                mm->SetPosition(Vector(x, y, 0.0));
            }
        }
    }
    // bots next
    for (uint32_t j = 0; j < g_botNodes.GetN(); ++j) {
        std::string vid = g_vehicleIds[g_nodes.GetN() + j];
        auto it = positions.find(vid);
        if (it != positions.end()) {
            double x = it->second.first;
            double y = it->second.second;
            Ptr<MobilityModel> mm = GetNodeMobilityPtr(g_botNodes.Get(j));
            if (mm) {
                mm->SetPosition(Vector(x, y, 0.0));
            }
        }
    }

    // schedule next update (keep in sync with SUMO step length)
    Simulator::Schedule(Seconds(sumoStep), &UpdatePositionsFromSUMO);
}

int main(int argc, char *argv[])
{
    CommandLine cmd;
    cmd.Parse(argc, argv);

    Time::SetResolution(Time::NS);
    LogComponentEnable("UdpEchoClientApplication", LOG_LEVEL_INFO);
    LogComponentEnable("UdpEchoServerApplication", LOG_LEVEL_INFO);

    // Legitimate connection nodes
    NodeContainer nodes;
    nodes.Create(3);

    // Nodes for attack bots
    NodeContainer botNodes;
    botNodes.Create(NUMBER_OF_BOTS);

    g_nodes = nodes;
    g_botNodes = botNodes;

    // Define the Point-To-Point Links
    PointToPointHelper pp1, pp2;
    pp1.SetDeviceAttribute("DataRate", StringValue("100Mbps"));
    pp1.SetChannelAttribute("Delay", StringValue("1ms"));

    pp2.SetDeviceAttribute("DataRate", StringValue("100Mbps"));
    pp2.SetChannelAttribute("Delay", StringValue("1ms"));

    // Install Point-To-Point Connections
    NetDeviceContainer d02, d12;
    std::vector<NetDeviceContainer> botDeviceContainer(NUMBER_OF_BOTS);
    d02 = pp1.Install(nodes.Get(0), nodes.Get(1));
    d12 = pp1.Install(nodes.Get(1), nodes.Get(2));

    for (int i = 0; i < NUMBER_OF_BOTS; ++i) {
        botDeviceContainer[i] = pp2.Install(botNodes.Get(i), nodes.Get(1));
    }

    // Install Internet stack
    InternetStackHelper stack;
    stack.Install(nodes);
    stack.Install(botNodes);

    // Assign IP addresses
    Ipv4AddressHelper ipv4_n;
    ipv4_n.SetBase("10.0.0.0", "255.255.255.252");
    Ipv4AddressHelper a02, a12;
    a02.SetBase("10.1.1.0", "255.255.255.0");
    a12.SetBase("10.1.2.0", "255.255.255.0");

    for (int j = 0; j < NUMBER_OF_BOTS; ++j) {
        ipv4_n.Assign(botDeviceContainer[j]);
        ipv4_n.NewNetwork();
    }

    Ipv4InterfaceContainer i02, i12;
    i02 = a02.Assign(d02);
    i12 = a12.Assign(d12);

    // DDoS Application for bots
    OnOffHelper onoff("ns3::UdpSocketFactory", Address(InetSocketAddress(i12.GetAddress(1), UDP_SINK_PORT)));
    onoff.SetConstantRate(DataRate(DDOS_RATE));
    onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=30]"));
    onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
    ApplicationContainer onOffApp[NUMBER_OF_BOTS];

    for (int k = 0; k < NUMBER_OF_BOTS; ++k) {
        onOffApp[k] = onoff.Install(botNodes.Get(k));
        onOffApp[k].Start(Seconds(0.0));
        onOffApp[k].Stop(Seconds(MAX_SIMULATION_TIME));
    }

    // Sender Application (TCP BulkSend)
    BulkSendHelper bulkSend("ns3::TcpSocketFactory", InetSocketAddress(i12.GetAddress(1), TCP_SINK_PORT));
    bulkSend.SetAttribute("MaxBytes", UintegerValue(MAX_BULK_BYTES));
    ApplicationContainer bulkSendApp = bulkSend.Install(nodes.Get(0));
    bulkSendApp.Start(Seconds(0.0));
    bulkSendApp.Stop(Seconds(MAX_SIMULATION_TIME - 1)); // stop a bit earlier

    // UDP Sink on victim
    PacketSinkHelper UDPsink("ns3::UdpSocketFactory",
                             Address(InetSocketAddress(Ipv4Address::GetAny(), UDP_SINK_PORT)));
    ApplicationContainer UDPSinkApp = UDPsink.Install(nodes.Get(2));
    UDPSinkApp.Start(Seconds(0.0));
    UDPSinkApp.Stop(Seconds(MAX_SIMULATION_TIME));

    // TCP Sink on victim
    PacketSinkHelper TCPsink("ns3::TcpSocketFactory",
                             InetSocketAddress(Ipv4Address::GetAny(), TCP_SINK_PORT));
    ApplicationContainer TCPSinkApp = TCPsink.Install(nodes.Get(2));
    TCPSinkApp.Start(Seconds(0.0));
    TCPSinkApp.Stop(Seconds(MAX_SIMULATION_TIME));

    // Populate routing
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    // Mobility: install ConstantPositionMobilityModel on all nodes (we'll update positions from SUMO)
    MobilityHelper mobility;
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(nodes);
    mobility.Install(botNodes);

    // Set initial positions to keep NetAnim readable before SUMO outputs
    Ptr<ConstantPositionMobilityModel> m0 = nodes.Get(0)->GetObject<ConstantPositionMobilityModel>();
    if (m0) m0->SetPosition(Vector(0, 0, 0));
    Ptr<ConstantPositionMobilityModel> m1 = nodes.Get(1)->GetObject<ConstantPositionMobilityModel>();
    if (m1) m1->SetPosition(Vector(10, 10, 0));
    Ptr<ConstantPositionMobilityModel> m2 = nodes.Get(2)->GetObject<ConstantPositionMobilityModel>();
    if (m2) m2->SetPosition(Vector(20, 10, 0));
    for (uint32_t b = 0; b < botNodes.GetN(); ++b) {
        Ptr<ConstantPositionMobilityModel> mb = botNodes.Get(b)->GetObject<ConstantPositionMobilityModel>();
        if (mb) mb->SetPosition(Vector(b, 30, 0));
    }

    // NetAnim
    AnimationInterface anim("DDoSim_SUMO.xml");

    // Label legitimate nodes
    anim.UpdateNodeDescription(nodes.Get(0), "Sender");
    anim.UpdateNodeDescription(nodes.Get(1), "Router");
    anim.UpdateNodeDescription(nodes.Get(2), "Victim");

    anim.UpdateNodeColor(nodes.Get(0), 0, 255, 0);       // Sender - Green
    anim.UpdateNodeColor(nodes.Get(1), 0, 0, 255);       // Router - Blue
    anim.UpdateNodeColor(nodes.Get(2), 255, 0, 0);       // Victim - Red

    // Label bot nodes (attackers)
    for (int i = 0; i < NUMBER_OF_BOTS; ++i) {
        anim.UpdateNodeDescription(botNodes.Get(i), "Attacker " + std::to_string(i + 1));
        anim.UpdateNodeColor(botNodes.Get(i), 255, 0, 255);  // Attackers - Magenta
    }

    // Prepare vehicleId mapping:
    // First three: sender_car, router_car, victim_car
    g_vehicleIds.clear();
    g_vehicleIds.push_back("sender_car");
    g_vehicleIds.push_back("router_car");
    g_vehicleIds.push_back("victim_car");
    for (int i = 0; i < NUMBER_OF_BOTS; ++i) {
        g_vehicleIds.push_back(std::string("attacker_") + std::to_string(i));
    }

    // Try to launch SUMO in background with fcd output.
    // NOTE: This is platform-dependent and simple. For robust usage, launch SUMO externally and ensure it writes SUMO_FCD_OUTPUT.
    std::ostringstream sumoCmd;
    // Use --step-length equal to sumoStep for consistent updates.
    sumoCmd << SUMO_BIN
            << " -c " << SUMO_CONFIG
            << " --step-length " << sumoStep
            << " --fcd-output " << SUMO_FCD_OUTPUT
            << " > sumo_stdout.log 2> sumo_stderr.log &";

    NS_LOG_UNCOND("Launching SUMO with command: " << sumoCmd.str());
    int ret = std::system(sumoCmd.str().c_str());
    if (ret == -1) {
        NS_LOG_UNCOND("Warning: failed to launch SUMO via system() call. You may start SUMO manually:");
        NS_LOG_UNCOND("  sumo -c " << SUMO_CONFIG << " --step-length " << sumoStep << " --fcd-output " << SUMO_FCD_OUTPUT);
    } else {
        NS_LOG_UNCOND("SUMO launched (system() returned " << ret << ").");
    }

    // Schedule first update after a small delay to let SUMO create the file
    Simulator::Schedule(Seconds(sumoStep + 0.01), &UpdatePositionsFromSUMO);

    // Run simulation
    Simulator::Stop(Seconds(MAX_SIMULATION_TIME));
    Simulator::Run();
    Simulator::Destroy();
    return 0;
}