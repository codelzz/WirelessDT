#include "Networking/UdpServerComponent.h"

UUdpServerComponent::UUdpServerComponent(const FObjectInitializer& ObjectInitializer)
	: Super(ObjectInitializer)
	, Delegate(nullptr)
{
}

void UUdpServerComponent::BeginPlay()
{
	Super::BeginPlay();
	Listen();
}

void UUdpServerComponent::EndPlay(EEndPlayReason::Type EndPlayReason)
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

void UUdpServerComponent::Listen()
{
	UdpServer = MakeShared<FUdpServer>(Host, Port);
	UdpServer->Delegate = this;
	if (IsValid())
	{
		UdpServer->Listen();
	}
}

void UUdpServerComponent::Close()
{
	if (IsValid())
	{
		UdpServer->Close();
		UdpServer.Reset();
		UdpServer = nullptr;
	}
}

bool UUdpServerComponent::IsValid()
{
	return UdpServer != nullptr && UdpServer.IsValid();
}

void UUdpServerComponent::OnUdpServerDataRecv(FString RecvData, FString& RespData)
{
	// deliver the callback to higher layer
	if (Delegate)
	{
		Delegate->OnUdpServerComponentDataRecv(RecvData, RespData);
	}
}