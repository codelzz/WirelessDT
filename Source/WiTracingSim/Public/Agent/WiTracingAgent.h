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
 *  AWiTracingAgent is a management agent to control the WiTracing process in the scene.
 *  After BeginPlay(), it will iterative do wi tracing for each transmitter in the scene.
 *  The render result will be hold in texture render target for preview or debugging.
 */

/** A agent to manage WiTracing process */
UCLASS()
class AWiTracingAgent: public AActor
{
	GENERATED_BODY()

public:
	AWiTracingAgent();
	virtual void BeginPlay() override;
	virtual void Tick(float DeltaScends) override;

public:
	// AWiTracingAgent Properties ---

	UPROPERTY()
	USceneComponent* Root;

	/** Udp client for communication */
	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "Wi Tracing", meta = (AllowPrivateAccess = "true"))
		TObjectPtr<class UUdpClientComponent> UdpClientComponent;

	/** buffer holding data for WiTracing Raw data, compare to the one without Vis, it won't be rendered to the scene for visualization */
	UPROPERTY(EditAnywhere, Category="WiTracing")
	UTextureRenderTarget2D* TextureRenderTarget;

	/** buffer holding data for WiTracing Raw data, intermediate WiTracing result will be stored here if result is use for visualization */
	UPROPERTY(EditAnywhere, Category="WiTracing")
	UTextureRenderTarget2D* TextureRenderTargetVis;

	/** The host of target server indicate where the data is sent to */
	UPROPERTY(EditAnywhere, category = "Endpint")
		FString Host = TEXT("127.0.0.1");

	/** The port of target server */
	UPROPERTY(EditAnywhere, category = "Endpint")
		uint16 Port = 8888;

public:
	// AWiTracingAgent Blueprint Functions ---

	/**
	 * Perform WiTracing for single TX-RX pair.
	 * @param WirelessTX - the wireless transmitter
	 * @param WirelessRX - the wireless receiver
	 * @param OctahedralProjection - control the projection approach (default: true, enable octahedral projection, otherwise use normal projection)
	 * @param bDenoised - control whether use the built-in DNN denoiser (default: true)
	 * @param bVisualized - let function known whether the intermediate result (store in TextureRenderTarget) will be used for visualziation.
	 * @return Result - the result of WiTracing
	 */
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