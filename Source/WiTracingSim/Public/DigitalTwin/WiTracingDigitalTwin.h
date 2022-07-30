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
		float x;
	UPROPERTY(BlueprintReadWrite, Category = "Location")
		float y;
	UPROPERTY(BlueprintReadWrite, Category = "Location")
		float z;
	UPROPERTY(BlueprintReadWrite, Category = "Rotation")
		float Pitch;
	UPROPERTY(BlueprintReadWrite, Category = "Rotation")
		float Yaw;
	UPROPERTY(BlueprintReadWrite, Category = "Rotation")
		float Roll;
	UPROPERTY(BlueprintReadWrite, Category = "Signal")
		float rssi;
	UPROPERTY(BlueprintReadWrite, Category = "Signal")
		FString address;

	FVector GetLocation()
	{
		return FVector(x * 100.0f, y * 100.0f, z * 100.0f) ;
	}

	FRotator GetRotator()
	{
		return FRotator(Pitch, Yaw, Roll);
	}

	void SetLocation(FVector Location)
	{
		x = Location.X;
		y = Location.Y;
		z = Location.Z;
	}
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

public:
	UPROPERTY()
		USceneComponent* Root;

private:
	// Is this really necessary?
	class UUdpSocketServerComponent* GetUdpSocketServerComponent() const { return UdpSocketServerComponent; }
	// UdpSocketServerComponent Callback
	virtual void OnUdpSocketServerComponentDataRecv(FString) override;

	FDigitalTwinData DigitalTwinData;
	bool bNeedSync = false;
};