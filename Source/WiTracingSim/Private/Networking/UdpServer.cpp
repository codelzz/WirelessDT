#include "Networking/UdpServer.h"
#include "UdpUtils.h"

FUdpServer::FUdpServer(const FString Host, const uint16 Port)
	: SocketSubSystem(ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM))
	, Delegate(nullptr)
{
	FIPv4Address IPAddress;
	FIPv4Address::Parse(Host, IPAddress);
	Endpoint = FIPv4Endpoint(IPAddress, Port);
}


FUdpServer::~FUdpServer()
{
	if (Sender != nullptr)
	{
		if (Sender.IsValid()) { Sender.Reset(); }
		Sender = nullptr;
	}
	if (Receiver != nullptr)
	{
		if (Receiver.IsValid()) { Receiver.Reset(); }
		Receiver = nullptr;
	}
	if (Socket != nullptr)
	{
		SocketSubSystem->DestroySocket(Socket);
		Socket = nullptr;
	}
}

void FUdpServer::Listen()
{
	Socket = FUdpSocketBuilder("UdpSocket").AsReusable();
	TSharedPtr<FInternetAddr, ESPMode::ThreadSafe> InternetAddr = SocketSubSystem->CreateInternetAddr();
	InternetAddr->SetIp(Endpoint.Address.Value);
	InternetAddr->SetPort(Endpoint.Port);
	Socket->Bind(*InternetAddr);
	Socket->Listen(1);
	{
		Sender = MakeShared<FUdpSocketSender, ESPMode::ThreadSafe>(Socket, TEXT("UdpSocketSender"));
		Receiver = MakeShared<FUdpSocketReceiver, ESPMode::ThreadSafe>(Socket, FTimespan::FromMilliseconds(0.0f), TEXT("UdpSocketReceiver"));
		if (Receiver.IsValid())
		{
			Receiver->OnDataReceived().BindRaw(this, &FUdpServer::OnDataRecv);
			Receiver->Start();
		}
	}

}

void FUdpServer::Close()
{
	Socket->Close();
}

void FUdpServer::OnDataRecv(const FArrayReaderPtr& ReaderPtr, const FIPv4Endpoint& InEndpoint)
{
	FString RecvData = *FUdpUtils::ArrayReaderPtrToString(ReaderPtr);
	int32 Index;
	if (RecvData.FindChar(TEXT('}'), Index))
	{
		if (Delegate)
		{
			RecvData = RecvData.Mid(0, Index + 1);
			FString RespData;
			Delegate->OnUdpServerDataRecv(RecvData, RespData);
			if (!RespData.IsEmpty()) {
				UE_LOG(LogTemp, Warning, TEXT("%s"), *RespData);
				Send(RespData, InEndpoint);
			}
		}		
	}
}

bool FUdpServer::Send(const FString& Data, const FIPv4Endpoint& InEndpoint)
{
	return Send(FUdpUtils::StringToBytesSharedPtr(Data).ToSharedRef(), InEndpoint);
}

bool FUdpServer::Send(const TSharedRef<TArray<uint8>, ESPMode::ThreadSafe>& Data, const FIPv4Endpoint& InEndpoint)
{
	if (Sender.IsValid()) {
		Sender->Send(Data, InEndpoint);
		return true;
	}
	return false;
}