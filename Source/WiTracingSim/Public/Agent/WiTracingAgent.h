#pragma once

#include "CoreMinimal.h"
#include "UObject/ObjectMacros.h"
#include "Engine/TextureRenderTarget2D.h"
#include "WiTracing/Devices/WirelessTX.h"
#include "WiTracing/Devices/WirelessRX.h"
#include "WiTracing/WiTracingRendererBlueprintLibrary.h"
//#include "Networking/UdpClientComponent.h"
//#include "Networking/TcpClientComponent.h"
//#include "Networking/WebSocketComponent.h"
#include "WebSocketsModule.h"
#include "IWebSocket.h"
#include "WiTracingAgent.generated.h"


/**
 *  AWiTracingAgent is a management agent to control the WiTracing process in the scene.
 *  After BeginPlay(), it will iterative do wi tracing for each transmitter in the scene.
 *  The render result will be hold in texture render target for preview or debugging.
 */

/** Camera Tracking Result */
USTRUCT(BlueprintType)
struct FCamTrackingResult
{
	GENERATED_BODY()

public:
	UPROPERTY(BlueprintReadWrite, Category = "detection")
		FString pedid;
	UPROPERTY(BlueprintReadWrite, Category = "detection")
		float camx = 0.0f;
	UPROPERTY(BlueprintReadWrite, Category = "detection")
		float camy = 0.0f;
	UPROPERTY(BlueprintReadWrite, Category = "detection")
		float worldx = 0.0f;
	UPROPERTY(BlueprintReadWrite, Category = "detection")
		float worldy = 0.0f;
	UPROPERTY(BlueprintReadWrite, Category = "detection")
		bool los = false;
	UPROPERTY(BlueprintReadWrite, Category = "detection")
		int64 timestamp = 0;

};

/** IMU Result */
USTRUCT(BlueprintType)
struct FIMUResult
{
	GENERATED_BODY()

public:
	UPROPERTY(BlueprintReadWrite, Category = "imu")
		FString imuid;
	UPROPERTY(BlueprintReadWrite, Category = "imu")
		float accx = 0.0f;
	UPROPERTY(BlueprintReadWrite, Category = "imu")
		float accy = 0.0f;
	UPROPERTY(BlueprintReadWrite, Category = "imu")
		float accz = 0.0f;
	UPROPERTY(BlueprintReadWrite, Category = "imu")
		float orix = 0.0f;
	UPROPERTY(BlueprintReadWrite, Category = "imu")
		float oriy = 0.0f;
	UPROPERTY(BlueprintReadWrite, Category = "imu")
		float oriz = 0.0f;
	UPROPERTY(BlueprintReadWrite, Category = "imu")
		bool isTuring = false;
	UPROPERTY(BlueprintReadWrite, Category = "imu")
		int64 timestamp = 0;

};

/** A agent to manage WiTracing process */
UCLASS()
class AWiTracingAgent: public AActor
{
	GENERATED_BODY()

public:
	AWiTracingAgent();
	virtual void BeginPlay() override;
	virtual void EndPlay(EEndPlayReason::Type EndPlayReason) override;
	virtual void Tick(float DeltaScends) override;

public:
	// AWiTracingAgent Properties ---

	UPROPERTY()
	USceneComponent* Root;

	/** buffer holding data for WiTracing Raw data, compare to the one without Vis, it won't be rendered to the scene for visualization */
	UPROPERTY(EditAnywhere, Category="WiTracing")
	UTextureRenderTarget2D* TextureRenderTarget;

	/** buffer holding data for WiTracing Raw data, intermediate WiTracing result will be stored here if result is use for visualization */
	UPROPERTY(EditAnywhere, Category="WiTracing")
	UTextureRenderTarget2D* TextureRenderTargetVis;

	/** The host of target server indicate where the data is sent to */
	UPROPERTY(EditAnywhere, category = "Networking")
		FString Host = TEXT("127.0.0.1");

	/** The port of target server */
	UPROPERTY(EditAnywhere, category = "Networking")
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
	 * Send Result via WebSocket Client
	 * @param Results - the Array of WiTracing result
	 */
	UFUNCTION(BlueprintCallable, Category = "WiTracing")
	void WebSocketSend(TArray<FWiTracingResult> Results);

	/**
	 * Send Result via WebSocket Client
	 * @param Results - the Array of WiTracing result
	 */
	UFUNCTION(BlueprintCallable, Category = "WiTracing")
	void WebSocketSend2(TArray<FWiTracingResult> Results, TArray<FCamTrackingResult> CamResults, TArray<FIMUResult> IMUResults);
	
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

	/**
	 * get current timestamp in millisecond
	 * @return - current timestamp in millisecond
	 */
	UFUNCTION(BlueprintCallable, Category = "WiTracing")
	int64 GetCurrentMillisecondTimestamp();

private:
	//--WebSocket
	void InitWebSocket();
	//--WebSocket End

	/** initialize render target for saving WiTracing result */
	void InitRenderTargets();

	/** cache all the transmitters in the world */
	void CacheTXs();

	/** cache all the transmitters in the world */
	void CacheRXs();

	/** get render target for caching WiTracing result */
	UTextureRenderTarget2D* GetRenderTarget(bool bVisualized = false);

private:
	/** transmitters in current world */
	TArray<AWirelessTX*> TXs;

	/** receivers in current world */
	TArray<AWirelessRX*> RXs;

	/** Web Socket */
	TSharedPtr<IWebSocket> WebSocket;
};