#pragma once

#include "CoreMinimal.h"
#include "UObject/ObjectMacros.h"
#include "Engine/TextureRenderTarget2D.h"
#include "WiTracing/Devices/WirelessTX.h"
#include "WiTracing/Devices/WirelessRX.h"
#include "WiTracing/WiTracingRendererBlueprintLibrary.h"
#include "Networking/UdpClientComponent.h"
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

	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "Wi Tracing", meta = (AllowPrivateAccess = "true"))
	TObjectPtr<class UUdpClientComponent> UdpClientComponent;

public:
	UPROPERTY(EditAnywhere, category = "Endpint")
		FString Host = TEXT("127.0.0.1");
	UPROPERTY(EditAnywhere, category = "Endpint")
		uint16 Port = 8888;

	/* [ISSUE] Performance Issue
	 * As we need to isolate the impact from different TX in the scene, at each iteration we only do WiTracing
	 * for one TX. This might signficantly affect the scalability of size of environment and limit the number
	 * of TXs in the scene. This need to be solved in future version
	 */
	//UFUNCTION(BlueprintCallable, Category = "Wi Tracing")
	//void IterativeWiTracing(FTransform Transform, TArray<float>& RSSIPdf, bool bVisualized = true);
	
	//UFUNCTION(BlueprintCallable, Category = "Wi Tracing")
	//void GlobalWiTracing(FTransform Transform, TArray<float>& RSSIPdf, bool bVisualized=true);

	UFUNCTION(BlueprintCallable, Category = "Wi Tracing")
	void WiTracing(AWirelessTX* WirelessTX, AWirelessRX* WirelessRX, FWiTracingResult& Result, bool OctahedralProjection = true, bool bDenoised = false, bool bVisualized = false);

	UFUNCTION(BlueprintCallable, Category = "Wi Tracing")
	void MultiWiTracing(TArray<AWirelessTX*> WirelessTXs, AWirelessRX* WirelessRX, TArray<FWiTracingResult>& Results, bool OctahedralProjection = true, bool bDenoised = false, bool bVisualized = false);

	UFUNCTION(BlueprintCallable, Category = "Wi Tracing")
	void PreviewWiTracing(TArray<AWirelessTX*> WirelessTXs, AWirelessRX* WirelessRX, bool OctahedralProjection = true, bool bDenoised = false);

	// Send Result to UDP Client
	UFUNCTION(BlueprintCallable, Category = "Wi Tracing")
	void UDPSendWiTracingResult(FWiTracingResult Result);

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
		AWirelessTX* GetNextTX(); 
	
	/// Get TXs
	/// =======================
	/// Get all transmitters
	///	- returns: all transmitter in current world
	UFUNCTION(BlueprintCallable, Category = "Wi Tracing")
		TArray<AWirelessTX*> GetTXs() { return TXs; };

	/// Get TXs within range
	/// =======================
	/// Get all transmitters within range
	///	- returns: transmitter in range
	UFUNCTION(BlueprintCallable, Category = "Wi Tracing")
		TArray<AWirelessTX*> GetTXsInRange(FVector Origin, float Radius);

	/// Get TXs out of range
	/// =======================
	/// Get all transmitters within range
	///	- returns: transmitter outside the range
	UFUNCTION(BlueprintCallable, Category = "Wi Tracing")
		TArray<AWirelessTX*> GetTXsOutRange(FVector Origin, float Radius);

private:
	// Is this really necessary?
	class UUdpClientComponent* GetUdpClientComponent() const { return UdpClientComponent; }

private:
	APlayerController* PlayerController;

private:
	void InitRenderTargets();
	void CachePlayerController();
	void CacheTXs();

	TArray<AWirelessTX*> TXs;
	int32 IterativeTXIndex = 0;
	const int64 RSSI_MIN = -255;
};