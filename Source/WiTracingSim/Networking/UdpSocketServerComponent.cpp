#include "UdpSocketServerComponent.h"

UUdpSocketServerComponent::UUdpSocketServerComponent(const FObjectInitializer& ObjectInitializer)
	: Super(ObjectInitializer)
{
}

void UUdpSocketServerComponent::BeginPlay()
{
	Super::BeginPlay();
	Listen();
}

void UUdpSocketServerComponent::EndPlay(EEndPlayReason::Type EndPlayReason)
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

void UUdpSocketServerComponent::Send(const FString& Data)
{
	if (IsValid())
	{
		UdpSocketServer->Send(FUdpSocketServer::StringToByteArray(Data).ToSharedRef());
	}
}

void UUdpSocketServerComponent::Listen()
{
	UdpSocketServer = MakeShared<FUdpSocketServer>(
		FUdpSocketServer::StringToEndpoint(ServerIP, ServerPort), 
		FUdpSocketServer::StringToEndpoint(ClientIP, ClientPort));
	if (IsValid())
	{
		UdpSocketServer->Listen();
	}
}

void UUdpSocketServerComponent::Close()
{
	if (IsValid())
	{
		UdpSocketServer->Close();
		UdpSocketServer.Reset();
		UdpSocketServer = nullptr;
	}
}

bool UUdpSocketServerComponent::IsValid()
{
	return UdpSocketServer != nullptr && UdpSocketServer.IsValid();
}