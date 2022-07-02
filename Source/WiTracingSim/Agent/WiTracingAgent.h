#pragma once

#include "CoreMinimal.h"
#include "UObject/ObjectMacros.h"
#include "Engine/TextureRenderTarget2D.h"
#include "WiTracing/Devices/WirelessTransmitter.h"
#include "WiTracingAgent.generated.h"


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

	/* [ISSUE] Performance Issue
	 * As we need to isolate the impact from different TX in the scene, at each iteration we only do WiTracing
	 * for one TX. This might signficantly affect the scalability of size of environment and limit the number
	 * of TXs in the scene. This need to be solved in future version
	 */
	UFUNCTION(BlueprintCallable, Category = "Wi Tracing")
	void IterativeWiTracing(TArray<int64>& RSSIPdf, bool bVisualized = true);
	
	UFUNCTION(BlueprintCallable, Category = "Wi Tracing")
	void GlobalWiTracing(TArray<int64>& RSSIPdf, bool bVisualized=true);

	/**
	 * Get the TX will be traced in next iterative witracing 
	 */
	UFUNCTION(BlueprintCallable, Category = "Wi Tracing")
		AWirelessTransmitter* GetTX() const;

	// for background denoising
	UFUNCTION(BlueprintCallable, Category = "Wi Tracing|Denoising")
		bool GetBackgroundDenoising() { return bEnableBackgroundDenoising; }
	UFUNCTION(BlueprintCallable, Category = "Wi Tracing|Denoising")
		void SetBackgroundDenoising(bool EnableDenoising) { bEnableBackgroundDenoising = EnableDenoising;  }
	UFUNCTION(BlueprintCallable, Category = "Wi Tracing|Denoising")
		void SwitchBackgroundDenoising() { bEnableBackgroundDenoising = !bEnableBackgroundDenoising; }

private:
	APlayerController* PlayerController;

private:
	void InitRenderTargets();
	void CachePlayerController();
	void CacheTXs();
	void RemoveBackgroundNoise(TArray<int64>& RSSIPdf);

	TArray<AWirelessTransmitter*> TXs;
	int32 TXIndex = 0;
	// int32 BackgroundNoiseThreshold = 20;
};
