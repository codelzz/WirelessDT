#pragma once

#include "CoreMinimal.h"
#include "UObject/ObjectMacros.h"
#include "Networking/UdpSocketServerComponent.h"
#include "WiTracingDigitalTwin.generated.h"

USTRUCT(BlueprintType)
struct FDigitalTwinData
{
	GENERATED_BODY()

public:
	UPROPERTY(BlueprintReadWrite, Category = "Location")
		float x = 0.0f;
	UPROPERTY(BlueprintReadWrite, Category = "Location")
		float y = 0.0f;
	UPROPERTY(BlueprintReadWrite, Category = "Location")
		float z = 0.0f;
	UPROPERTY(BlueprintReadWrite, Category = "Rotation")
		float Pitch = 0.0f;
	UPROPERTY(BlueprintReadWrite, Category = "Rotation")
		float Yaw = 0.0f;
	UPROPERTY(BlueprintReadWrite, Category = "Rotation")
		float Roll = 0.0f;
	UPROPERTY(BlueprintReadWrite, Category = "Signal")
		float rssi = 0.0f;
	UPROPERTY(BlueprintReadWrite, Category = "Signal")
		FString address = "";

	FVector GetLocation()
	{
		return FVector(x * 100.0f + BaseLocation.X, y * 100.0f + BaseLocation.Y, z * 100.0f + +BaseLocation.Z) ;
	}

	FRotator GetRotator()
	{
		return FRotator(Pitch, Yaw, Roll);
	}

	FVector BaseLocation = FVector(0,0,0);
};

UCLASS()
class AWiTracingDigitalTwin : public AActor, public FUdpSocketServerComponentDelegate
{
	GENERATED_BODY()

public:
	AWiTracingDigitalTwin();

	virtual void BeginPlay() override;
	virtual void Tick(float DeltaScends) override;

public:
	UPROPERTY(BlueprintReadOnly, EditAnywhere, Category = "Wi Tracing", meta = (AllowPrivateAccess = "true"))
		TObjectPtr<class UUdpSocketServerComponent> UdpSocketServerComponent;

	UPROPERTY(BlueprintReadOnly, EditAnywhere, Category = "Wi Tracing", meta = (AllowPrivateAccess = "true"))
		FDigitalTwinData Data;

public:
	UPROPERTY()
		USceneComponent* Root;

private:
	// Is this really necessary?
	class UUdpSocketServerComponent* GetUdpSocketServerComponent() const { return UdpSocketServerComponent; }
	// UdpSocketServerComponent Callback
	virtual void OnUdpSocketServerComponentDataRecv(FString) override;

	
	bool bNeedSync = false;
	// FVector BaseLocation = FVector(0, 0, 0);
};