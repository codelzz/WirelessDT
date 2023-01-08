#pragma once

#include "CoreMinimal.h"
#include "UObject/ObjectMacros.h"
#include "Networking/UdpServerComponent.h"
#include "SensorSync.generated.h"


USTRUCT(BlueprintType)
struct FSensorData
{
	GENERATED_BODY()

public:
	UPROPERTY(BlueprintReadWrite, Category = "Position")
		float x = 0.0f;
	UPROPERTY(BlueprintReadWrite, Category = "Position")
		float y = 0.0f;
	UPROPERTY(BlueprintReadWrite, Category = "Position")
		float z = 0.0f;

	UPROPERTY(BlueprintReadWrite, Category = "Orientation")
		float pitch = 0.0f;
	UPROPERTY(BlueprintReadWrite, Category = "Orientation")
		float yaw = 0.0f;
	UPROPERTY(BlueprintReadWrite, Category = "Orientation")
		float roll = 0.0f;

	UPROPERTY(BlueprintReadWrite, Category = "Signal")
		float rssi = -255.0f;
	UPROPERTY(BlueprintReadWrite, Category = "Signal")
		FString tx = "";

	FVector GetLocation()
	{
		return FVector(x * 100.0f + BaseLocation.X, y * 100.0f + BaseLocation.Y, z * 100.0f + BaseLocation.Z);
	}

	FRotator GetRotator()
	{
		return FRotator(pitch, yaw, roll);
	}

	FVector BaseLocation = FVector(0, 0, 0);
};

UCLASS()
class ASensorSync : public AActor, public FUdpServerComponentDelegate
{
	GENERATED_BODY()

public:
	ASensorSync();

	virtual void BeginPlay() override;
	virtual void Tick(float DeltaScends) override;

public:
	// UPROPERTY Object can be seen in blueprint detial tab
	// USceneComponent define the transform information including location, scale and rotation
	UPROPERTY()
		USceneComponent* Root;

	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "Network", meta = (AllowPrivateAccess = "true"))
		TObjectPtr<class UUdpServerComponent> UdpServerComponent;

	UPROPERTY(BlueprintReadOnly, EditAnywhere, Category = "Sensor", meta = (AllowPrivateAccess = "true"))
		FSensorData SensorData;

private:
	// Is this really necessary?
	class UUdpServerComponent* GetUdpServerComponent() const { return UdpServerComponent; }

	// UdpServerComponent Callback
	virtual void OnUdpServerComponentDataRecv(FString RecvData, FString& RespData) override;

	bool bNeedSync = false;
};
