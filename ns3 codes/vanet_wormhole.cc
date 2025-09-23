#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/mobility-module.h"
#include "ns3/netanim-module.h"
#include "ns3/ipv4-global-routing-helper.h"

#define UDP_SINK_PORT 9001
#define MAX_SIMULATION_TIME 20.0

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("WormholeAttackWithRouters");

int main(int argc, char *argv[])
{
    CommandLine cmd;
    cmd.Parse(argc, argv);

    // --- 1. Create Nodes ---
    // We will create a sender, a victim, three routers for the long path,
    // and two wormhole nodes for the shortcut.
    NodeContainer allNodes;
    allNodes.Create(7);

    // For clarity, let's assign nodes to named pointers
    Ptr<Node> sender    = allNodes.Get(0);
    Ptr<Node> victim    = allNodes.Get(1);
    Ptr<Node> router1   = allNodes.Get(2);
    Ptr<Node> router2   = allNodes.Get(3);
    Ptr<Node> router3   = allNodes.Get(4);
    Ptr<Node> wormhole1 = allNodes.Get(5);
    Ptr<Node> wormhole2 = allNodes.Get(6);

    // --- 2. Create Links ---
    // Helper for normal, slower links
    PointToPointHelper normalLink;
    normalLink.SetDeviceAttribute("DataRate", StringValue("10Mbps"));
    normalLink.SetChannelAttribute("Delay", StringValue("5ms"));

    // Helper for the fast, low-delay wormhole tunnel
    PointToPointHelper wormholeLink;
    wormholeLink.SetDeviceAttribute("DataRate", StringValue("100Mbps"));
    wormholeLink.SetChannelAttribute("Delay", StringValue("10us"));

    // Install devices for the legitimate (long) path: Sender -> R1 -> R2 -> R3 -> Victim
    NetDeviceContainer dev_s_r1 = normalLink.Install(sender, router1);
    NetDeviceContainer dev_r1_r2 = normalLink.Install(router1, router2);
    NetDeviceContainer dev_r2_r3 = normalLink.Install(router2, router3);
    NetDeviceContainer dev_r3_v = normalLink.Install(router3, victim);

    // Install devices for the wormhole (short) path: Sender -> W1 -> W2 -> Victim
    NetDeviceContainer dev_s_w1 = normalLink.Install(sender, wormhole1);
    NetDeviceContainer dev_w1_w2 = wormholeLink.Install(wormhole1, wormhole2);
    NetDeviceContainer dev_w2_v = normalLink.Install(wormhole2, victim);

    // --- 3. Install Internet Stack ---
    InternetStackHelper stack;
    stack.Install(allNodes);

    // --- 4. Assign IP Addresses ---
    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.1.1.0", "255.255.255.0");
    ipv4.Assign(dev_s_r1);

    ipv4.SetBase("10.1.2.0", "255.255.255.0");
    ipv4.Assign(dev_r1_r2);
    
    ipv4.SetBase("10.1.3.0", "255.255.255.0");
    ipv4.Assign(dev_r2_r3);

    ipv4.SetBase("10.1.4.0", "255.255.255.0");
    ipv4.Assign(dev_r3_v);

    ipv4.SetBase("10.1.5.0", "255.255.255.0");
    ipv4.Assign(dev_s_w1);

    ipv4.SetBase("10.1.6.0", "255.255.255.0");
    ipv4.Assign(dev_w1_w2);

    ipv4.SetBase("10.1.7.0", "255.255.255.0");
    Ipv4InterfaceContainer if_w2_v = ipv4.Assign(dev_w2_v);

    // --- 5. Configure Routing ---
    // This is the key change. Instead of manual static routes, we let the
    // Global Routing Helper calculate the best path for all nodes.
    // It will see that the cumulative delay through the wormhole is much lower.
    NS_LOG_INFO("Populating routing tables...");
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    // --- 6. Setup Applications ---
    Ipv4Address victimAddr = if_w2_v.GetAddress(1);

    OnOffHelper client("ns3::UdpSocketFactory",
                       Address(InetSocketAddress(victimAddr, UDP_SINK_PORT)));
    client.SetConstantRate(DataRate("1Mbps"));
    ApplicationContainer app = client.Install(sender);
    app.Start(Seconds(1.0));
    app.Stop(Seconds(MAX_SIMULATION_TIME));

    PacketSinkHelper sink("ns3::UdpSocketFactory",
                          Address(InetSocketAddress(Ipv4Address::GetAny(), UDP_SINK_PORT)));
    ApplicationContainer sinkApp = sink.Install(victim);
    sinkApp.Start(Seconds(0.0));
    sinkApp.Stop(Seconds(MAX_SIMULATION_TIME));

    // --- 7. Setup Animation ---
    MobilityHelper mobility;
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(allNodes);

    AnimationInterface anim("WormholeMultiRouter.xml");
    
    // Set descriptive names for NetAnim
    anim.UpdateNodeDescription(sender, "Sender");
    anim.UpdateNodeDescription(victim, "Victim");
    anim.UpdateNodeDescription(router1, "R1");
    anim.UpdateNodeDescription(router2, "R2");
    anim.UpdateNodeDescription(router3, "R3");
    anim.UpdateNodeDescription(wormhole1, "W1");
    anim.UpdateNodeDescription(wormhole2, "W2");

    // Set colors for different node types
    anim.UpdateNodeColor(sender, 0, 255, 0);   // Green
    anim.UpdateNodeColor(victim, 255, 0, 0);   // Red
    anim.UpdateNodeColor(router1, 0, 0, 255);  // Blue
    anim.UpdateNodeColor(router2, 0, 0, 255);
    anim.UpdateNodeColor(router3, 0, 0, 255);
    anim.UpdateNodeColor(wormhole1, 255, 165, 0); // Orange
    anim.UpdateNodeColor(wormhole2, 255, 165, 0);

    // Position nodes to visualize the two paths
    // The routers form a long arc below, while the wormhole is a direct link above.
    AnimationInterface::SetConstantPosition(sender, 0, 0);
    AnimationInterface::SetConstantPosition(victim, 100, 0);
    
    AnimationInterface::SetConstantPosition(router1, 25, -20);
    AnimationInterface::SetConstantPosition(router2, 50, -25);
    AnimationInterface::SetConstantPosition(router3, 75, -20);

    AnimationInterface::SetConstantPosition(wormhole1, 33, 20);
    AnimationInterface::SetConstantPosition(wormhole2, 66, 20);

    // --- 8. Run Simulation ---
    Simulator::Stop(Seconds(MAX_SIMULATION_TIME));
    Simulator::Run();
    Simulator::Destroy();
    return 0;
}