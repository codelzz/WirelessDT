#pragma once

#include "CoreMinimal.h"
#include "UObject/ObjectMacros.h"
#include "Networking/UdpSocketServerComponent.h"
#include "DeepLearningAgent.generated.h"


USTRUCT(BlueprintType)
struct FPredictionData
{
	GENERATED_BODY()

public:
	UPROPERTY(BlueprintReadWrite, Category = "Location")
		float x = 0.0f;
	UPROPERTY(BlueprintReadWrite, Category = "Location")
		float y = 0.0f;
	UPROPERTY(BlueprintReadWrite, Category = "Location")
		float z = 0.0f;

	FVector GetLocation()
	{
		return FVector(x, y, z);
	}
};

UCLASS()
class ADeepLearningAgent : public AActor, public FUdpSocketServerComponentDelegate
{
	GENERATED_BODY()

public:
	ADeepLearningAgent();

	virtual void BeginPlay() override;
	virtual void Tick(float DeltaScends) override;

public:
	// UPROPERTY Object can be seen in blueprint detial tab
	// USceneComponent define the transform information including location, scale and rotation
	UPROPERTY()
		USceneComponent* Root;

	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "Network", meta = (AllowPrivateAccess = "true"))
		TObjectPtr<class UUdpSocketServerComponent> UdpSocketServerComponent;

	UPROPERTY(BlueprintReadOnly, EditAnywhere, Category = "Wi Tracing", meta = (AllowPrivateAccess = "true"))
		FPredictionData Data;

private:
	// Is this really necessary?
	class UUdpSocketServerComponent* GetUdpSocketServerComponent() const { return UdpSocketServerComponent; }

	// UdpSocketServerComponent Callback
	virtual void OnUdpSocketServerComponentDataRecv(FString) override;

	bool bNeedSync = false;
};
