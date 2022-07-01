#pragma once

#include "CoreMinimal.h"
#include "UObject/ObjectMacros.h"
#include "Engine/EngineTypes.h"
#include "Components/SceneComponent.h"
#include "UdpSocketServer.h"
#include "UdpSocketServerComponent.generated.h"

UCLASS(Blueprintable, meta = (BlueprintSpawnableComponent))
class UUdpSocketServerComponent : public USceneComponent
{
	GENERATED_UCLASS_BODY()

protected:
	virtual void BeginPlay() override;
	virtual void EndPlay(EEndPlayReason::Type EndPlayReason) override;

public:
	void Send(const FString& Data);
	void Listen();
	void Close();
	bool IsValid();

public:
	UFUNCTION(BlueprintCallable, Category = "Udp Socket Server")
	FString GetServerEndpoint()
	{
		return FString::Printf(TEXT("%s:%d"), *ServerIP, ServerPort);
	}

	UFUNCTION(BlueprintCallable, Category = "Udp Socket Server")
	FString GetClientEndpoint()
	{
		return FString::Printf(TEXT("%s:%d"), *ClientIP, ClientPort);
	}

public:
	UPROPERTY(EditAnywhere, category = "Server")
		FString ServerIP = TEXT("127.0.0.1");
	UPROPERTY(EditAnywhere, category = "Server")
		uint16 ServerPort = 8888;
	UPROPERTY(EditAnywhere, category = "Client")
		FString ClientIP = TEXT("127.0.0.1");
	UPROPERTY(EditAnywhere, category = "Client")
		uint16 ClientPort = 8080;

public:
	/** Returns WirelessTransmitterComponent subobject */
	TSharedPtr<class FUdpSocketServer> GetUdpSocketServer() const { return UdpSocketServer; }

private:
	TSharedPtr<class FUdpSocketServer> UdpSocketServer;
};