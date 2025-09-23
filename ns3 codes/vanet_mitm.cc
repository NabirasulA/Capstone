#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/mobility-module.h"
#include "ns3/netanim-module.h"

using namespace ns3;

#define UDP_SINK_PORT 9001
#define MAX_SIMULATION_TIME 10.0

NS_LOG_COMPONENT_DEFINE("EnhancedMITMAttackWithRouter");

// MITM Application: can log, modify, or drop packets
class MITMApp : public Application
{
public:
    MITMApp() : m_dropRate(0.3) {} // drop 30% packets randomly
    virtual ~MITMApp() {}
    void Setup(Ptr<Socket> recvSocket, Ptr<Socket> sendSocket, Ipv4Address finalVictimAddr, uint16_t port)
    {
        m_recvSocket = recvSocket;
        m_sendSocket = sendSocket;
        m_finalVictimAddr = finalVictimAddr;
        m_port = port;
    }

private:
    virtual void StartApplication() override
    {
        if (m_recvSocket)
            m_recvSocket->SetRecvCallback(MakeCallback(&MITMApp::HandleRead, this));
    }

    virtual void StopApplication() override
    {
        if (m_recvSocket)
            m_recvSocket->Close();
        if (m_sendSocket)
            m_sendSocket->Close();
    }

    void HandleRead(Ptr<Socket> socket)
    {
        Ptr<Packet> packet;
        Address from;
        while ((packet = socket->RecvFrom(from)))
        {
            // Randomly drop some packets to simulate packet loss
            double randVal = (double)rand() / RAND_MAX;
            if (randVal < m_dropRate)
            {
                NS_LOG_INFO("MITM DROPPED a packet of size " << packet->GetSize() << " from " << InetSocketAddress::ConvertFrom(from).GetIpv4());
                continue;
            }

            // Modify packet (example: add 20 bytes of padding)
            Ptr<Packet> modifiedPacket = packet->Copy();
            modifiedPacket->AddPaddingAtEnd(20);

            NS_LOG_INFO("MITM forwarding modified packet of size " << modifiedPacket->GetSize() << " to victim " << m_finalVictimAddr);
            // Forward the packet to the actual intended victim
            m_sendSocket->SendTo(modifiedPacket, 0, InetSocketAddress(m_finalVictimAddr, m_port));
        }
    }

    Ptr<Socket> m_recvSocket;
    Ptr<Socket> m_sendSocket;
    Ipv4Address m_finalVictimAddr;
    uint16_t m_port;
    double m_dropRate;
};

int main(int argc, char *argv[])
{
    CommandLine cmd;
    cmd.Parse(argc, argv);

    Time::SetResolution(Time::NS);
    LogComponentEnable("EnhancedMITMAttackWithRouter", LOG_LEVEL_INFO);

    // --- Topology Setup ---
    // Nodes: Senders, MITM, Router, Victim
    // The senders and MITM are on one side of the router, the victim is on the other.
    NodeContainer senderNodes;
    senderNodes.Create(2); 
    NodeContainer mitmNode;
    mitmNode.Create(1);
    NodeContainer routerNode;
    routerNode.Create(1);
    NodeContainer victimNode;
    victimNode.Create(1);

    // Consolidate all nodes for easier stack installation
    NodeContainer allNodes;
    allNodes.Add(senderNodes);
    allNodes.Add(mitmNode);
    allNodes.Add(routerNode);
    allNodes.Add(victimNode);

    // Point-to-Point Links
    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue("100Mbps"));
    p2p.SetChannelAttribute("Delay", StringValue("1ms"));

    // Create links connecting nodes to the router
    NetDeviceContainer devSenderRouter[2];
    devSenderRouter[0] = p2p.Install(senderNodes.Get(0), routerNode.Get(0));
    devSenderRouter[1] = p2p.Install(senderNodes.Get(1), routerNode.Get(0));

    NetDeviceContainer devMITMRouter = p2p.Install(mitmNode.Get(0), routerNode.Get(0));
    NetDeviceContainer devRouterVictim = p2p.Install(routerNode.Get(0), victimNode.Get(0));

    // Install Internet stack on all nodes
    InternetStackHelper stack;
    stack.Install(allNodes);

    // --- IP Addressing ---
    Ipv4AddressHelper ipv4;
    Ipv4InterfaceContainer ifSenderRouter[2], ifMITMRouter, ifRouterVictim;

    // Assign IP addresses to sender-router links
    ipv4.SetBase("10.1.1.0", "255.255.255.0");
    ifSenderRouter[0] = ipv4.Assign(devSenderRouter[0]);

    ipv4.SetBase("10.1.2.0", "255.255.255.0");
    ifSenderRouter[1] = ipv4.Assign(devSenderRouter[1]);

    // Assign IP address to mitm-router link
    ipv4.SetBase("10.1.3.0", "255.255.255.0");
    ifMITMRouter = ipv4.Assign(devMITMRouter);

    // Assign IP address to router-victim link
    ipv4.SetBase("10.1.4.0", "255.255.255.0");
    ifRouterVictim = ipv4.Assign(devRouterVictim);

    // --- Application Setup ---
    Ipv4Address mitmAddr = ifMITMRouter.GetAddress(0);
    Ipv4Address victimAddr = ifRouterVictim.GetAddress(1);

    // Sender Applications (UDP traffic)
    // Senders are tricked into sending packets to the MITM node instead of the real victim
    for (uint32_t i = 0; i < senderNodes.GetN(); ++i)
    {
        OnOffHelper udpSend("ns3::UdpSocketFactory", Address(InetSocketAddress(mitmAddr, UDP_SINK_PORT)));
        udpSend.SetConstantRate(DataRate("5Mbps"));
        udpSend.SetAttribute("PacketSize", UintegerValue(1024));
        ApplicationContainer senderApp = udpSend.Install(senderNodes.Get(i));
        senderApp.Start(Seconds(1.0));
        senderApp.Stop(Seconds(MAX_SIMULATION_TIME));
    }

    // Victim UDP Sink
    PacketSinkHelper udpSink("ns3::UdpSocketFactory", Address(InetSocketAddress(Ipv4Address::GetAny(), UDP_SINK_PORT)));
    ApplicationContainer sinkApp = udpSink.Install(victimNode.Get(0));
    sinkApp.Start(Seconds(0.0));
    sinkApp.Stop(Seconds(MAX_SIMULATION_TIME + 1.0));

    // MITM Application Setup
    Ptr<Socket> mitmRecvSocket = Socket::CreateSocket(mitmNode.Get(0), UdpSocketFactory::GetTypeId());
    mitmRecvSocket->Bind(InetSocketAddress(Ipv4Address::GetAny(), UDP_SINK_PORT));

    Ptr<Socket> mitmSendSocket = Socket::CreateSocket(mitmNode.Get(0), UdpSocketFactory::GetTypeId());

    Ptr<MITMApp> mitmApp = CreateObject<MITMApp>();
    // The MITM app knows the real victim's address to forward packets to.
    mitmApp->Setup(mitmRecvSocket, mitmSendSocket, victimAddr, UDP_SINK_PORT);
    mitmNode.Get(0)->AddApplication(mitmApp);
    mitmApp->SetStartTime(Seconds(0.5));
    mitmApp->SetStopTime(Seconds(MAX_SIMULATION_TIME));

    // Populate routing tables for all nodes
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    // --- NetAnim Visualization ---
    AnimationInterface anim("MITM_With_Router.xml");

    anim.UpdateNodeDescription(senderNodes.Get(0), "Sender 1");
    anim.UpdateNodeDescription(senderNodes.Get(1), "Sender 2");
    anim.UpdateNodeDescription(mitmNode.Get(0), "MITM");
    anim.UpdateNodeDescription(routerNode.Get(0), "Router");
    anim.UpdateNodeDescription(victimNode.Get(0), "Victim");

    anim.UpdateNodeColor(senderNodes.Get(0), 0, 255, 0);   // Green
    anim.UpdateNodeColor(senderNodes.Get(1), 0, 255, 0);   // Green
    anim.UpdateNodeColor(mitmNode.Get(0), 255, 0, 255);    // Magenta
    anim.UpdateNodeColor(routerNode.Get(0), 100, 100, 100); // Grey
    anim.UpdateNodeColor(victimNode.Get(0), 255, 0, 0);    // Red

    // Set positions for a clear star-like topology around the router
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();
    positionAlloc->Add(Vector(0, 20, 0));  // Sender 1
    positionAlloc->Add(Vector(0, 40, 0));  // Sender 2
    positionAlloc->Add(Vector(20, 0, 0));  // MITM
    positionAlloc->Add(Vector(20, 30, 0)); // Router
    positionAlloc->Add(Vector(40, 30, 0)); // Victim
    mobility.SetPositionAllocator(positionAlloc);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(allNodes);

    Simulator::Run();
    Simulator::Destroy();
    return 0;
}
