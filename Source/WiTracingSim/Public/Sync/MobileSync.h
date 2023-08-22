#pragma once

#include "CoreMinimal.h"
#include "UObject/ObjectMacros.h"
#include "Networking/UdpServerComponent.h"
#include "MobileSync.generated.h"


USTRUCT(BlueprintType)
struct FMobileData
{
	GENERATED_BODY()

public:
	// Position
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

	// BLE
	UPROPERTY(BlueprintReadWrite, Category = "Training")
		TArray<FString> rxname;
	UPROPERTY(BlueprintReadWrite, Category = "Training")
		TArray<int64> rxrssi;

	FVector GetLocation()
	{
		return FVector(x + BaseLocation.X, y + BaseLocation.Y, z + BaseLocation.Z);
	}
	
	FRotator GetRotator()
	{
		return FRotator(pitch, yaw, roll);
	}

	TArray<FString> GetRXNames() 
	{
		return rxname;
	}

	TArray<int64> GetRXRssis()
	{
		return rxrssi;
	}



	FVector BaseLocation = FVector(0, 0, 0);
};

UCLASS()
class AMobileSync : public AActor, public FUdpServerComponentDelegate
{
	GENERATED_BODY()

public:
	AMobileSync();

	virtual void BeginPlay() override;
	virtual void Tick(float DeltaScends) override;

public:
	// UPROPERTY Object can be seen in blueprint detial tab
	// USceneComponent define the transform information including location, scale and rotation
	UPROPERTY()
		USceneComponent* Root;

	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "Network", meta = (AllowPrivateAccess = "true"))
		TObjectPtr<class UUdpServerComponent> UdpServerComponent;

	UPROPERTY(BlueprintReadOnly, EditAnywhere, Category = "Mobile", meta = (AllowPrivateAccess = "true"))
		FMobileData MobileData;
	UPROPERTY(BlueprintReadOnly, EditAnywhere, Category = "Mobile", meta = (AllowPrivateAccess = "true"))
		bool BeaconReceived;

public:
	UFUNCTION(BlueprintCallable, Category = "Mobile")
		TArray<FString> GetRXName();
	UFUNCTION(BlueprintCallable, Category = "Mobile")
		TArray<int64> GetRXRssi();

private:
	// Is this really necessary?
	class UUdpServerComponent* GetUdpServerComponent() const { return UdpServerComponent; }

	// UdpServerComponent Callback
	virtual void OnUdpServerComponentDataRecv(FString RecvData, FString& RespData) override;

	bool bNeedSync = false;
};
