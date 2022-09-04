#pragma once

#include "CoreMinimal.h"
#include "UObject/ObjectMacros.h"
#include "Engine/TextureRenderTarget2D.h"
#include "WiTracing/Devices/WirelessTransmitter.h"
#include "Networking/UdpSocketServerComponent.h"
#include "WiTracingAgent.generated.h"


USTRUCT()
struct FWiTracingResult
{
	GENERATED_USTRUCT_BODY()

	// Here we use lowercase for attribute naming 
	// because it will affect the key when convert to json string

	UPROPERTY()
		FString tag;


	UPROPERTY()
		TArray<float> coordinates;

	UPROPERTY()
		TArray<int64> rssipdf;


	UPROPERTY()
		int64 timestamp;

	FWiTracingResult()
	{
		coordinates.Empty();
		rssipdf.Empty();
	}

	FWiTracingResult(FString InTag, FVector InCoordinates, TArray<int64> InRSSIPdf, int64 InTimestamp)
	{
		tag = InTag;
		timestamp = InTimestamp;
		float TempCoordinates[] = { InCoordinates.X, InCoordinates.Y, InCoordinates.Z };
		coordinates.Empty();
		coordinates.Append(TempCoordinates, UE_ARRAY_COUNT(TempCoordinates));
		rssipdf.Empty();
		rssipdf = InRSSIPdf;
	}
};

/**
 *  AWiTracingAgent is a management agent to control the wi tracing process in the scene.
 *  After BeginPlay(), it will iterative do wi tracing for each transmitter in the scene.
 *  The render result will be hold in texture render target for preview or debugging.
 */
UCLASS()
class AWiTracingAgent: public AActor
{
	GENERATED_BODY()

public:
	AWiTracingAgent();

	virtual void BeginPlay() override;
	virtual void Tick(float DeltaScends) override;

public:
	// UPROPERTY Object can be seen in blueprint detial tab
	// USceneComponent define the transform information including location, scale and rotation
	UPROPERTY()
	USceneComponent* Root;
	UPROPERTY(EditAnywhere, Category="Wi Tracing")
	UTextureRenderTarget2D* TextureRenderTarget;		// Texture holding data required for calculation
	UPROPERTY(EditAnywhere, Category="Wi Tracing")
	UTextureRenderTarget2D* TextureRenderTargetTemp;	// Texture for temporal used

	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "Wi Tracing", meta = (AllowPrivateAccess = "true"))
	TObjectPtr<class UUdpSocketServerComponent> UdpSocketServerComponent;

	/* [ISSUE] Performance Issue
	 * As we need to isolate the impact from different TX in the scene, at each iteration we only do WiTracing
	 * for one TX. This might signficantly affect the scalability of size of environment and limit the number
	 * of TXs in the scene. This need to be solved in future version
	 */
	UFUNCTION(BlueprintCallable, Category = "Wi Tracing")
	void IterativeWiTracing(FTransform Transform, TArray<float>& RSSIPdf, bool bVisualized = true);
	
	UFUNCTION(BlueprintCallable, Category = "Wi Tracing")
	void GlobalWiTracing(FTransform Transform, TArray<float>& RSSIPdf, bool bVisualized=true);

	/**
	 * The RSSI sampling simulate the physical layer rssi sampling process to generate RSSI sample
	 * Instead of average over n signal period, we do n time sampling then find the maximum value
	 * as our sample.
	 */
	/*UFUNCTION(BlueprintCallable, Category = "Wi Tracing")
		int64 RSSISampling(TArray<float>& RSSIPdf, int64 n);*/
	UFUNCTION(BlueprintCallable, Category = "Wi Tracing")
		int64 RSSISampling(TArray<float> RSSIPdf);

	UFUNCTION(BlueprintCallable, Category = "Wi Tracing")
		int64 RSSIMultiSampling(TArray<float> RSSIPdf, int64 n=8);

	UFUNCTION(BlueprintCallable, Category = "Wi Tracing")
		int64 RSSIMaxSampling(TArray<float> RSSIPdf);
	/**
	 * Get the TX will be traced in next iterative witracing 
	 */
	UFUNCTION(BlueprintCallable, Category = "Wi Tracing")
		AWirelessTransmitter* GetTX(); 

private:
	// Is this really necessary?
	class UUdpSocketServerComponent* GetUdpSocketServerComponent() const { return UdpSocketServerComponent; }

private:
	APlayerController* PlayerController;

private:
	void InitRenderTargets();
	void CachePlayerController();
	void CacheTXs();

	TArray<AWirelessTransmitter*> TXs;
	int32 TXIndex = 0;
	const int64 RSSI_MIN = -255;
};
