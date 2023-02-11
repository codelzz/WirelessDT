#pragma once

#include "CoreMinimal.h"
#include "Sockets.h"
#include "Networking.h"
#include "Interfaces/IPv4/IPv4Endpoint.h"
#include "UObject/ObjectMacros.h"
#include "Engine/EngineTypes.h"
#include "Components/SceneComponent.h"
#include "TcpClientComponent.generated.h"

//---CLASS: FTcpClient ----------------------------------------------------------------------------------

class FTcpClient {
public:
	FTcpClient();
	~FTcpClient();

	bool Connect(const FString Host, const uint16 Port);
	bool Send(const FString& Data);
	bool Close();

	//bool Send(const FString& Data, const FString Host, const uint16 Port);
	//bool Send(const TSharedRef<TArray<uint8>, ESPMode::ThreadSafe>& Data, const FIPv4Endpoint& Endpoint);
	//void Close();

	// socket getter and setter
	void SetSocket(FSocket* Socket) { _Socket = Socket; }
	FSocket* GetSocket() { return _Socket; }

protected:
	FSocket* _Socket = nullptr;
	//TSharedPtr<FUdpSocketSender, ESPMode::ThreadSafe> Sender = nullptr;
};

//---CLASS: UUdpClientComponent -----------------------------------------------------------------------

UCLASS(Blueprintable, meta = (BlueprintSpawnableComponent))
class UTcpClientComponent : public USceneComponent
{
	GENERATED_UCLASS_BODY()
protected:
	virtual void BeginPlay() override;
	virtual void EndPlay(EEndPlayReason::Type EndPlayReason) override;

public:
	UFUNCTION(BlueprintCallable, Category = "Networking")
		void Connect(const FString Host, const int64 Port);

	UFUNCTION(BlueprintCallable, Category = "Networking")
		void Send(const FString& Data);

	UFUNCTION(BlueprintCallable, Category = "Networking")
		void Close();

	bool IsValid();

	TSharedPtr<class FTcpClient> GetTcpClient() const { return TcpClient; }

private:
	TSharedPtr<class FTcpClient> TcpClient;
};