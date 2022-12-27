#pragma once

#include "CoreMinimal.h"
#include "UObject/ObjectMacros.h"
#include "Engine/EngineTypes.h"
#include "Components/SceneComponent.h"
#include "Sockets.h"
#include "Networking.h"
#include "Interfaces/IPv4/IPv4Endpoint.h"
#include "UdpServerComponent.generated.h"

//--- CLASS: FUdpServerDelegate -----------------------------------------------------------------------
class FUdpServerDelegate {
public:
	virtual void OnUdpServerDataRecv(FString RecvData, FString& RespData) = 0;
};

class FUdpServerRLAgentDelegate {
public:
	virtual void OnUdpServerRLAgentDataRecv(FString RecvData, FString& RespData) = 0;
};

//--- CLASS: FUdpServer -----------------------------------------------------------------------
class FUdpServer {
public:
	FUdpServer(const FString Host, const uint16 Port);
	~FUdpServer();

	void Listen();
	void Close();

	bool Send(const TSharedRef<TArray<uint8>, ESPMode::ThreadSafe>& Data, const FIPv4Endpoint& Endpoint);
	bool Send(const FString& Data, const FIPv4Endpoint& Endpoint);
	void OnDataRecv(const FArrayReaderPtr&, const FIPv4Endpoint&);

protected:
	FIPv4Endpoint Endpoint;

	FSocket* Socket = nullptr;
	ISocketSubsystem* SocketSubSystem;

	TSharedPtr<FUdpSocketSender, ESPMode::ThreadSafe> Sender = nullptr;
	TSharedPtr<FUdpSocketReceiver, ESPMode::ThreadSafe> Receiver = nullptr;

public:
	FUdpServerDelegate* Delegate = nullptr;
	FUdpServerRLAgentDelegate* RLDelegate = nullptr;
};

//--- CLASS: FUdpServerComponentDelegate ---------------------------------------------------------------

class FUdpServerComponentDelegate {
public:
	virtual void OnUdpServerComponentDataRecv(FString RecvData, FString& RespData) = 0;
};

//--- CLASS: FUdpServerComponentRLAgentDelegate ---------------------------------------------------------------

class FUdpServerComponentRLAgentDelegate {
public:
	virtual void OnUdpServerComponentRLAgentDataRecv(FString RecvData, FString& RespData) = 0;
};

//--- CLASS: UUdpServerComponent -----------------------------------------------------------------------

UCLASS(Blueprintable, meta = (BlueprintSpawnableComponent))
class UUdpServerComponent : public USceneComponent, public FUdpServerDelegate, public FUdpServerRLAgentDelegate
{
	GENERATED_UCLASS_BODY()

protected:
	virtual void BeginPlay() override;
	virtual void EndPlay(EEndPlayReason::Type EndPlayReason) override;

public:
	void Listen();
	void Close();
	bool IsValid();

public:
	UPROPERTY(EditAnywhere, category = "Endpint")
		FString Host = TEXT("127.0.0.1");
	UPROPERTY(EditAnywhere, category = "Endpint")
		uint16 Port = 9000;

public:
	/** Returns WirelessTransmitterComponent subobject */
	TSharedPtr<class FUdpServer> GetUdpServer() const { return UdpServer; }

public:
	virtual void OnUdpServerDataRecv(FString RecvData, FString& RespData) override;
	virtual void OnUdpServerRLAgentDataRecv(FString RecvData, FString& RespData) override;

private:
	TSharedPtr<class FUdpServer> UdpServer;

public:
	FUdpServerComponentDelegate* Delegate = nullptr;
	FUdpServerComponentRLAgentDelegate* RLDelegate = nullptr;
};