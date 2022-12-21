#include "Networking/UdpClientComponent.h"
#include "UdpUtils.h"

//--------------------------------- UdpClient --------------------------------------------------

FUdpClient::FUdpClient()
{
	Socket = FUdpSocketBuilder(FString("UdpSocket")).AsReusable();
	{
		Sender = MakeShared<FUdpSocketSender, ESPMode::ThreadSafe>(Socket, TEXT("UdpSocketSender"));
	}
}

FUdpClient::~FUdpClient()
{
	if (Socket != nullptr)
	{
		Socket = nullptr;
	}
}

bool FUdpClient::Send(const FString& Data, const FString Host, const uint16 Port)
{
	return Send(FUdpUtils::StringToBytesSharedPtr(Data).ToSharedRef(), FUdpUtils::ParseEndpoint(Host, Port));
}
bool FUdpClient::Send(const TSharedRef<TArray<uint8>, ESPMode::ThreadSafe>& Data, const FIPv4Endpoint& Endpoint)
{
	if (Sender.IsValid()) {
		Sender->Send(Data, Endpoint);
		return true;
	}
	return false;
}
void FUdpClient::Close()
{
	Socket->Close();
}

//--------------------------------- UUdpClientComponent -----------------------------------------------
UUdpClientComponent::UUdpClientComponent(const FObjectInitializer& ObjectInitializer)
	: Super(ObjectInitializer)
{
	UdpClient = MakeShared<FUdpClient>();
}

void UUdpClientComponent::BeginPlay()
{
	Super::BeginPlay();
}

void UUdpClientComponent::EndPlay(EEndPlayReason::Type EndPlayReason)
{
	switch (EndPlayReason)
	{
	case EEndPlayReason::Destroyed:
	{
		Close();
		break;
	}
	}
}

void UUdpClientComponent::Send(const FString& Data, const FString Host, const int64 Port)
{
	if (IsValid())
	{
		UdpClient->Send(Data, Host, Port);
	}
}

bool UUdpClientComponent::IsValid()
{
	return UdpClient != nullptr && UdpClient.IsValid();
}

void UUdpClientComponent::Close()
{
	if (IsValid())
	{
		UdpClient->Close();
		UdpClient.Reset();
		UdpClient = nullptr;
	}
}