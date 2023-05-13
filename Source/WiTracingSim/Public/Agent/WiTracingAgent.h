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
	 * @return Result - the result of WiTracing
	 */
	UFUNCTION(BlueprintCallable, Category = "WiTracing")
	void WiTracing(AWirelessTX* WirelessTX, AWirelessRX* WirelessRX, FWiTracingResult& Result, bool OctahedralProjection = true, bool bDenoised = false, bool bVisualized = false);

	/**
	 * Perform WiTracing for multiple TX-RX pairs.
	 * @param WirelessTX - the wireless transmitters
	 * @param WirelessRX - the wireless receiver
	 * @param OctahedralProjection - control the projection approach (default: true, enable octahedral projection, otherwise use normal projection)
	 * @param bDenoised - control whether use the built-in DNN denoiser (default: true)
	 * @param bVisualized - let function known whether the intermediate result (store in TextureRenderTarget) will be used for visualziation.
	 * @return Result - the result of WiTracing
	 */
	UFUNCTION(BlueprintCallable, Category = "WiTracing")
	void MultiTXWiTracing(TArray<AWirelessTX*> WirelessTXs, AWirelessRX* WirelessRX, TArray<FWiTracingResult>& Results, bool OctahedralProjection = true, bool bDenoised = false, bool bVisualized = false);

	/**
	 * Perform WiTracing for multiple TX-RX pairs.
	 * @param WirelessTX - the wireless transmitters
	 * @param WirelessRX - the wireless receivers
	 * @param OctahedralProjection - control the projection approach (default: true, enable octahedral projection, otherwise use normal projection)
	 * @param bDenoised - control whether use the built-in DNN denoiser (default: true)
	 * @return Result - the result of WiTracing
	 */
	UFUNCTION(BlueprintCallable, Category = "WiTracing")
	void MultiWiTracing(TArray<AWirelessRX*> WirelessRXs, TArray<AWirelessTX*> WirelessTXs,  TArray<FWiTracingResult>& Results, bool OctahedralProjection = true, bool bDenoised = false);

	/**
	 * Perform WiTracing for multiple TX-RX pair for visualization purpose.
	 * @param WirelessTX - the wireless transmitters
	 * @param WirelessRX - the wireless receiver
	 * @param OctahedralProjection - control the projection approach (default: true, enable octahedral projection, otherwise use normal projection)
	 * @param bDenoised - control whether use the built-in DNN denoiser (default: true)
	 */
	UFUNCTION(BlueprintCallable, Category = "WiTracing")
	void PreviewWiTracing(TArray<AWirelessTX*> WirelessTXs, AWirelessRX* WirelessRX, bool OctahedralProjection = true, bool bDenoised = false);

	/**
	 * Send Result via UDP Client
	 * @param Result - the WiTracing result
	 */
	UFUNCTION(BlueprintCallable, Category = "WiTracing")
	void UDPSendWiTracingResult(FWiTracingResult Result);

	/**
	 * get all transmitter in current world
	 * @return - all transmitter in current world
	 */
	UFUNCTION(BlueprintCallable, Category = "WiTracing")
	const TArray<AWirelessTX*> GetTXs() { return TXs; };
	
	/**
	 * get all receivers in current world
	 * @return - all transmitter in current world
	 */
	UFUNCTION(BlueprintCallable, Category = "WiTracing")
	const TArray<AWirelessRX*> GetRXs() { return RXs; };

	/**
	 * get all transmitter in current world within given range
	 * @param Origin - the center of range
	 * @param Radius - the radius of the range
	 * @return - all transmitter in current world within range
	 */
	UFUNCTION(BlueprintCallable, Category = "WiTracing")
	const TArray<AWirelessTX*> GetTXsInRange(FVector Origin, float Radius);

	/**
	 * get all transmitter in current world outside given range
	 * @param Origin - the center of range
	 * @param Radius - the radius of the range
	 * @return - all transmitter in current world outside the range
	 */
	UFUNCTION(BlueprintCallable, Category = "WiTracing")
	const TArray<AWirelessTX*> GetTXsOutRange(FVector Origin, float Radius);

	UFUNCTION(BlueprintCallable, Category = "WiTracing")
	void CacheTXs();

private:
	/** initialize render target for saving WiTracing result */
	void InitRenderTargets();

	/** cache all the transmitters in the world */
	// void CacheTXs();

	/** cache all the transmitters in the world */
	void CacheRXs();

	/** get render target for caching WiTracing result */
	UTextureRenderTarget2D* GetRenderTarget(bool bVisualized = false);

	/**
	 * get udp client
	 * @return the udp client component
	 */
	class UUdpClientComponent* GetUdpClientComponent() const { return UdpClientComponent; }

private:
	/** transmitters in current world */
	TArray<AWirelessTX*> TXs;

	/** receivers in current world */
	TArray<AWirelessRX*> RXs;
};

// 	APlayerController* PlayerController;
// 	void CachePlayerController();
// 	int32 IterativeTXIndex = 0;

/**
 * Get the TX will be traced in next iterative witracing
 */
//UFUNCTION(BlueprintCallable, Category = "Wi Tracing")
//AWirelessTX* GetNextTX();

/**
 * The RSSI sampling simulate the physical layer rssi sampling process to generate RSSI sample
 * Instead of average over n signal period, we do n time sampling then find the maximum value
 * as our sample.
 */
 /*UFUNCTION(BlueprintCallable, Category = "Wi Tracing")
	 int64 RSSISampling(TArray<float>& RSSIPdf, int64 n);*/
	 //UFUNCTION(BlueprintCallable, Category = "Wi Tracing")
	 //	int64 RSSISampling(TArray<float> RSSIPdf);

	 //UFUNCTION(BlueprintCallable, Category = "Wi Tracing")
	 //	int64 RSSIMultiSampling(TArray<float> RSSIPdf, int64 n=8);

	 //UFUNCTION(BlueprintCallable, Category = "Wi Tracing")
	 //	int64 RSSIMaxSampling(TArray<float> RSSIPdf);