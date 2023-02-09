#pragma once

#include "CoreMinimal.h"
#include "Sockets.h"
#include "Networking.h"
#include "Interfaces/IPv4/IPv4Endpoint.h"
#include "UObject/ObjectMacros.h"
#include "Engine/EngineTypes.h"
#include "Components/SceneComponent.h"
#include "UdpClientComponent.generated.h"

//---CLASS: FUdpClient ----------------------------------------------------------------------------------

class FUdpClient {
public:
	FUdpClient();
	~FUdpClient();

	bool Send(const FString& Data, const FString Host, const uint16 Port);
	bool Send(const TSharedRef<TArray<uint8>, ESPMode::ThreadSafe>& Data, const FIPv4Endpoint& Endpoint);
	void Close();

protected:
	FSocket* Socket = nullptr;
	TSharedPtr<FUdpSocketSender, ESPMode::ThreadSafe> Sender = nullptr;
};

//---CLASS: UUdpClientComponent -----------------------------------------------------------------------

UCLASS(Blueprintable, meta = (BlueprintSpawnableComponent))
class UUdpClientComponent : public USceneComponent
{
	GENERATED_UCLASS_BODY()
protected:
	virtual void BeginPlay() override;
	virtual void EndPlay(EEndPlayReason::Type EndPlayReason) override;

public:
	UFUNCTION(BlueprintCallable, Category = "Networking")
		void Send(const FString& Data, const FString Host, const int64 Port);

	UFUNCTION(BlueprintCallable, Category = "Networking")
		void Close();

	bool IsValid();

	TSharedPtr<class FUdpClient> GetUdpClient() const { return UdpClient; }

private:
	TSharedPtr<class FUdpClient> UdpClient;
};