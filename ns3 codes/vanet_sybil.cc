#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/csma-module.h" // Using CSMA for a robust LAN environment
#include "ns3/applications-module.h"
#include "ns3/mobility-module.h"
#include "ns3/netanim-module.h"

using namespace ns3;

// --- Simulation Parameters ---
#define TCP_SINK_PORT 9000
#define UDP_SINK_PORT 9001
#define MAX_BULK_BYTES 1000000
#define SYBIL_RATE "2048kb/s" // Data rate for each Sybil identity
#define MAX_SIMULATION_TIME 12.0

// --- Attacker Configuration ---
#define NUMBER_OF_SYBIL_IDENTITIES 10

// --- Network Configuration ---
#define NUMBER_OF_LEGITIMATE_SENDERS 3

NS_LOG_COMPONENT_DEFINE("SybilAttackExplained");

int main(int argc, char* argv[])
{
    // --- Boilerplate Setup ---
    CommandLine cmd;
    cmd.Parse(argc, argv);
    Time::SetResolution(Time::NS);
    LogComponentEnable("SybilAttackExplained", LOG_LEVEL_INFO);

    // --- 1. Network Topology Setup ---
    // We create a LAN (CSMA network) for all senders and the attacker.
    // This LAN connects to a router, which then connects to the victim via a P2P link.
    // This is a robust topology that ensures routing works correctly.

    NS_LOG_INFO("Creating network nodes...");
    NodeContainer legitimateSenders;
    legitimateSenders.Create(NUMBER_OF_LEGITIMATE_SENDERS);

    NodeContainer sybilAttackerNode;
    sybilAttackerNode.Create(1);
    Ptr<Node> attacker = sybilAttackerNode.Get(0);

    NodeContainer routerNode;
    routerNode.Create(1);
    Ptr<Node> router = routerNode.Get(0);

    NodeContainer victimNode;
    victimNode.Create(1);
    Ptr<Node> victim = victimNode.Get(0);

    // Group all nodes that will be on the LAN (senders, attacker, and one port of the router)
    NodeContainer lanNodes;
    lanNodes.Add(legitimateSenders);
    lanNodes.Add(sybilAttackerNode);
    lanNodes.Add(router); // The router is also part of the LAN


    // --- 2. Configure Links and Devices ---
    NS_LOG_INFO("Configuring CSMA (LAN) and Point-to-Point links...");

    // Setup the LAN
    CsmaHelper csma;
    csma.SetChannelAttribute("DataRate", StringValue("100Mbps"));
    csma.SetChannelAttribute("Delay", TimeValue(NanoSeconds(6560)));
    NetDeviceContainer lanDevices = csma.Install(lanNodes);

    // Setup the separate link from the router to the victim
    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue("100Mbps"));
    p2p.SetChannelAttribute("Delay", StringValue("2ms"));
    NetDeviceContainer routerVictimDev = p2p.Install(router, victim);


    // --- 3. Install Internet Stack and Assign IPs ---
    NS_LOG_INFO("Installing internet stack and assigning IP addresses...");
    InternetStackHelper stack;
    stack.Install(legitimateSenders);
    stack.Install(sybilAttackerNode);
    stack.Install(routerNode);
    stack.Install(victimNode);

    Ipv4AddressHelper ipv4;

    // Assign IP addresses to the LAN
    ipv4.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer lanInterfaces = ipv4.Assign(lanDevices);

    // Assign IP addresses to the router-victim link
    ipv4.SetBase("10.1.2.0", "255.255.255.0");
    Ipv4InterfaceContainer victimInterface = ipv4.Assign(routerVictimDev);
    Ipv4Address victimIp = victimInterface.GetAddress(1);


    // --- 4. Setup Applications ---

    // a) Legitimate Traffic: Senders to Victim (TCP BulkSend)
    NS_LOG_INFO("Setting up legitimate TCP traffic...");
    for (uint32_t i = 0; i < NUMBER_OF_LEGITIMATE_SENDERS; ++i)
    {
        BulkSendHelper bulkSend("ns3::TcpSocketFactory", InetSocketAddress(victimIp, TCP_SINK_PORT));
        bulkSend.SetAttribute("MaxBytes", UintegerValue(MAX_BULK_BYTES));
        ApplicationContainer bulkSendApp = bulkSend.Install(legitimateSenders.Get(i));
        bulkSendApp.Start(Seconds(i * 0.5 + 1.0));
        bulkSendApp.Stop(Seconds(MAX_SIMULATION_TIME - 1.0));
    }

    // b) Victim Sinks
    NS_LOG_INFO("Setting up packet sinks on the victim node...");
    PacketSinkHelper udpSink("ns3::UdpSocketFactory", Address(InetSocketAddress(Ipv4Address::GetAny(), UDP_SINK_PORT)));
    ApplicationContainer udpSinkApp = udpSink.Install(victim);
    udpSinkApp.Start(Seconds(0.0));
    udpSinkApp.Stop(Seconds(MAX_SIMULATION_TIME));

    PacketSinkHelper tcpSink("ns3::TcpSocketFactory", Address(InetSocketAddress(Ipv4Address::GetAny(), TCP_SINK_PORT)));
    ApplicationContainer tcpSinkApp = tcpSink.Install(victim);
    tcpSinkApp.Start(Seconds(0.0));
    tcpSinkApp.Stop(Seconds(MAX_SIMULATION_TIME));


    // c) Sybil Attack Traffic
    NS_LOG_INFO("Setting up the Sybil Attack...");
    OnOffHelper onoff("ns3::UdpSocketFactory", Address(InetSocketAddress(victimIp, UDP_SINK_PORT)));
    onoff.SetConstantRate(DataRate(SYBIL_RATE));
    onoff.SetAttribute("PacketSize", UintegerValue(1024));
    onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
    onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));

    for (int i = 0; i < NUMBER_OF_SYBIL_IDENTITIES; ++i)
    {
        ApplicationContainer sybilApp = onoff.Install(attacker);
        sybilApp.Start(Seconds(1.0 + 0.1 * i));
        sybilApp.Stop(Seconds(MAX_SIMULATION_TIME));
    }


    // --- 5. Routing and Simulation Execution ---
    NS_LOG_INFO("Populating routing tables and running simulation...");
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();


    // --- 6. NetAnim Visualization Setup ---
    NS_LOG_INFO("Configuring NetAnim visualization...");
    // NOTE on Visualization: Because we use a CSMA network (a LAN), you will see
    // packets from one sender travel to all other nodes on the LAN. This is normal!
    // It represents the broadcast nature of a shared network medium. Other nodes
    // will simply ignore packets not addressed to them. The logical flow is still
    // correct: Sender -> Router -> Victim.
    AnimationInterface anim("SybilAttackExplained.xml");

    for (uint32_t i = 0; i < NUMBER_OF_LEGITIMATE_SENDERS; ++i)
    {
        std::ostringstream senderName;
        senderName << "Sender " << i + 1;
        anim.UpdateNodeDescription(legitimateSenders.Get(i), senderName.str());
        anim.UpdateNodeColor(legitimateSenders.Get(i), 0, 255, 0); // Green
        anim.SetConstantPosition(legitimateSenders.Get(i), 10, (i + 1) * 15);
    }

    anim.UpdateNodeDescription(router, "Router");
    anim.UpdateNodeColor(router, 100, 100, 255); // Blue
    anim.SetConstantPosition(router, 30, 40);

    anim.UpdateNodeDescription(victim, "Victim");
    anim.UpdateNodeColor(victim, 255, 0, 0); // Red
    anim.SetConstantPosition(victim, 50, 40);

    anim.UpdateNodeDescription(attacker, "Sybil Attacker");
    anim.UpdateNodeColor(attacker, 255, 0, 255); // Magenta
    anim.SetConstantPosition(attacker, 10, 65);

    Simulator::Run();
    Simulator::Destroy();

    NS_LOG_INFO("Simulation finished.");
    return 0;
}

