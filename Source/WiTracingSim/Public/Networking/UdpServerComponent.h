#pragma once

#include "CoreMinimal.h"
#include "UObject/ObjectMacros.h"
#include "Engine/EngineTypes.h"
#include "Components/SceneComponent.h"
#include "UdpServer.h"
#include "UdpServerComponent.generated.h"

class FUdpServerComponentDelegate {
public:
	virtual void OnUdpServerComponentDataRecv(FString RecvData, FString& RespData) = 0;
};

UCLASS(Blueprintable, meta = (BlueprintSpawnableComponent))
class UUdpServerComponent : public USceneComponent, public FUdpServerDelegate
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

private:
	TSharedPtr<class FUdpServer> UdpServer;

public:
	FUdpServerComponentDelegate* Delegate = nullptr;
};