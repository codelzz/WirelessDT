#include "UdpSocketServer.h"

FUdpSocketServer::FUdpSocketServer(const FIPv4Endpoint& InServerEndpoint, const FIPv4Endpoint& InClientEndpoint)
	: ClientEndpoint(InClientEndpoint)
	, ServerEndpoint(InServerEndpoint)
	, SocketSubSystem(ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM))
{
}

FUdpSocketServer::~FUdpSocketServer()
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

FIPv4Endpoint FUdpSocketServer::StringToEndpoint(const FString& InIP, uint16 InPort)
{
	FIPv4Address IPAddress;
	FIPv4Address::Parse(InIP, IPAddress);
	FIPv4Endpoint Endpoint(IPAddress, InPort);
	return Endpoint;
}

FString FUdpSocketServer::ArrayReaderPtrToString(const FArrayReaderPtr& Reader)
{
	return FString(UTF8_TO_TCHAR(reinterpret_cast<const char*>(Reader->GetData())));
}

FString FUdpSocketServer::BytesSharedPtrToString(const TSharedPtr<TArray<uint8>, ESPMode::ThreadSafe>& BytesPtr)
{
	if (BytesPtr.IsValid())
	{
		return FString(UTF8_TO_TCHAR(reinterpret_cast<const char*>(BytesPtr->GetData())));
	}
	return TEXT("");
}

FString FUdpSocketServer::BytesToString(const TArray<uint8>& Bytes)
{
	return FString(UTF8_TO_TCHAR(reinterpret_cast<const char*>(Bytes.GetData())));
}

TSharedPtr<TArray<uint8>, ESPMode::ThreadSafe> FUdpSocketServer::StringToByteArray(const FString& String)
{
	TSharedPtr<TArray<uint8>, ESPMode::ThreadSafe> Bytes = MakeShared<TArray<uint8>, ESPMode::ThreadSafe>();
	if (Bytes.IsValid())
	{
		Bytes->SetNum(String.Len());
		memcpy(Bytes->GetData(), TCHAR_TO_UTF8(*String), String.Len());
	}
	return Bytes;
}

void FUdpSocketServer::Listen()
{
	Socket = FUdpSocketBuilder(GetName()).AsReusable();
	TSharedPtr<FInternetAddr, ESPMode::ThreadSafe> InternetAddr = SocketSubSystem->CreateInternetAddr();
	InternetAddr->SetIp(ServerEndpoint.Address.Value);
	InternetAddr->SetPort(ServerEndpoint.Port);
	Socket->Bind(*InternetAddr);
	// [ISSUE] what does it mean when listen return false, and why the connection still work even listen false?
	Socket->Listen(1);
	{
		Sender = MakeShared<FUdpSocketSender, ESPMode::ThreadSafe>(Socket, TEXT("UdpSocketSender"));
		Receiver = MakeShared<FUdpSocketReceiver, ESPMode::ThreadSafe>(Socket, FTimespan::FromMilliseconds(1.0f), TEXT("UdpSocketReceiver"));
		if (Receiver.IsValid())
		{
			Receiver->OnDataReceived().BindRaw(this, &FUdpSocketServer::OnDataReceived);
			Receiver->Start();
		}
	}

}

void FUdpSocketServer::Close()
{
	Socket->Close();
}

void FUdpSocketServer::OnDataReceived(const FArrayReaderPtr& ReaderPtr, const FIPv4Endpoint& Endpoint)
{
	// if data is send from client endpoint then do thing
	// if (Endpoint == ClientEndpoint)
	UE_LOG(LogTemp, Warning, TEXT("[%s] OnDataReceived -> %s"), *GetName(), *ArrayReaderPtrToString(ReaderPtr));
}

bool FUdpSocketServer::Send(const TSharedRef<TArray<uint8>, ESPMode::ThreadSafe>& Data)
{
	if (Sender.IsValid()) { 
		Sender->Send(Data, ClientEndpoint);
		return true;
	}
	return false;
}
