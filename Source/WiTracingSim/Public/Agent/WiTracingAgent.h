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
	UPROPERTY(EditAnywhere, Category = "Wi Tracing")
	bool bEnableBackgroundDenoising = false;
	UPROPERTY(EditAnywhere, Category = "Wi Tracing")
	int32 BackgroundNoiseSNR = 20;

	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "Wi Tracing", meta = (AllowPrivateAccess = "true"))
	TObjectPtr<class UUdpSocketServerComponent> UdpSocketServerComponent;

	/* [ISSUE] Performance Issue
	 * As we need to isolate the impact from different TX in the scene, at each iteration we only do WiTracing
	 * for one TX. This might signficantly affect the scalability of size of environment and limit the number
	 * of TXs in the scene. This need to be solved in future version
	 */
	UFUNCTION(BlueprintCallable, Category = "Wi Tracing")
	void IterativeWiTracing(FTransform Transform, TArray<int64>& RSSIPdf, bool bVisualized = true);
	
	UFUNCTION(BlueprintCallable, Category = "Wi Tracing")
	void GlobalWiTracing(FTransform Transform, TArray<int64>& RSSIPdf, bool bVisualized=true);
	/**
	 * Get the TX will be traced in next iterative witracing 
	 */
	UFUNCTION(BlueprintCallable, Category = "Wi Tracing")
		AWirelessTransmitter* GetTX(); 

	/*
	 * Background Denoising (BackgroundNoiseSNR Interface)
	 */ 
	UFUNCTION(BlueprintCallable, Category = "Wi Tracing|Denoising")
		bool GetBackgroundDenoising() { return bEnableBackgroundDenoising; }
	UFUNCTION(BlueprintCallable, Category = "Wi Tracing|Denoising")
		void SetBackgroundDenoising(bool EnableDenoising) { bEnableBackgroundDenoising = EnableDenoising;  }
	UFUNCTION(BlueprintCallable, Category = "Wi Tracing|Denoising")
		void SwitchBackgroundDenoising() { bEnableBackgroundDenoising = !bEnableBackgroundDenoising; }

private:
	// Is this really necessary?
	class UUdpSocketServerComponent* GetUdpSocketServerComponent() const { return UdpSocketServerComponent; }

private:
	APlayerController* PlayerController;

private:
	void InitRenderTargets();
	void CachePlayerController();
	void CacheTXs();
	void RemoveBackgroundNoise(TArray<int64>& RSSIPdf);

	TArray<AWirelessTransmitter*> TXs;
	int32 TXIndex = 0;
};
