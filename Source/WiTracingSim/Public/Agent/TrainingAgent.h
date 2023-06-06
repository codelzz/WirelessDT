#pragma once

#include "CoreMinimal.h"
#include "UObject/ObjectMacros.h"
#include "WiTracing/Devices/WirelessTX.h"
#include "WiTracing/Devices/WirelessRX.h"
#include "WiTracing/WiTracingRendererBlueprintLibrary.h"

#include "IWebSocket.h"
#include "TrainingAgent.generated.h"

//UENUM(BlueprintType)
//enum class EStageEnum : int64
//{
//	UNKNOWN = 0 UMETA(DisplayName = "Unknown"),
//	TRAINING = 1 UMETA(DisplayName = "Training"),
//	EVALUATION = 2 UMETA(DisplayName = "Evaluation"),
//};

USTRUCT(BlueprintType)
struct FTrainingAgentRecvData
{
	GENERATED_BODY()

public:
	UPROPERTY(BlueprintReadWrite, Category = "Training")
	TArray<int64> rxi;
	UPROPERTY(BlueprintReadWrite, Category = "Training")
	TArray<float> rxx;
	UPROPERTY(BlueprintReadWrite, Category = "Training")
	TArray<float> rxy;
	UPROPERTY(BlueprintReadWrite, Category = "Training")
	TArray<float> rxz;
	UPROPERTY(BlueprintReadWrite, Category = "Training")
	TArray<float> rxrssi;
	UPROPERTY(BlueprintReadWrite, Category = "Training")
	TArray<int64> txi;
	UPROPERTY(BlueprintReadWrite, Category = "Training")
	TArray<float> txx;
	UPROPERTY(BlueprintReadWrite, Category = "Training")
	TArray<float> txy;
	UPROPERTY(BlueprintReadWrite, Category = "Training")
	TArray<float> txz;
	UPROPERTY(BlueprintReadWrite, Category = "Training")
	TArray<float> txpower;

	// WiTracing Engine
	UPROPERTY(BlueprintReadWrite, Category = "Training")
		float penetrationcoef;
	UPROPERTY(BlueprintReadWrite, Category = "Training")
		float reflectioncoef;
	
	// Fitting
	UPROPERTY(BlueprintReadWrite, Category = "Training")
		float x;
	UPROPERTY(BlueprintReadWrite, Category = "Training")
		float y;
	UPROPERTY(BlueprintReadWrite, Category = "Training")
		float z;
	// Payload
	UPROPERTY(BlueprintReadWrite, Category = "Training")
		FString type = "";
	//	int64 stage;
	void Empty() {
		rxi.Empty();
		rxx.Empty();
		rxy.Empty();
		rxz.Empty();
		rxrssi.Empty();

		txi.Empty();
		txx.Empty();
		txy.Empty();
		txz.Empty();
		txpower.Empty();

		x = 0;
		y = 0;
		z = 0;
		type = "";
	}
};

USTRUCT(BlueprintType)
struct FTrainingAgentRecvMetaData
{
	GENERATED_BODY()

public:
	UPROPERTY(BlueprintReadWrite, Category = "Training")
		FString type = "";
};

USTRUCT(BlueprintType)
struct FTrainingAgentRecvCost
{
	GENERATED_BODY()

public:
	UPROPERTY(BlueprintReadWrite, Category = "Training")
		float x;
	UPROPERTY(BlueprintReadWrite, Category = "Training")
		float y;
	UPROPERTY(BlueprintReadWrite, Category = "Training")
		float z;
	UPROPERTY(BlueprintReadWrite, Category = "Training")
		float cost = 0;
};

USTRUCT(BlueprintType)
struct FTrainingAgentSentData
{
	GENERATED_BODY()

public:
	UPROPERTY(BlueprintReadWrite, Category = "Training")
		TArray<float> txi;
	UPROPERTY(BlueprintReadWrite, Category = "Training")
		TArray<float> rxi;
	UPROPERTY(BlueprintReadWrite, Category = "Training")
		TArray<float> rxrssi;
	UPROPERTY(BlueprintReadWrite, Category = "Training")
		FString sender = "UnrealEngine";
	UPROPERTY(BlueprintReadWrite, Category = "Training")
		FString recipient = "Jupyter";
	// Stage: Training 1, Evaluation 2
	//UPROPERTY(BlueprintReadWrite, Category = "Training")
	//	int64 stage;

};

/** A agent to manage WiTracing process */
UCLASS()
class ATrainingAgent : public AActor
{
	GENERATED_BODY()

public:
	ATrainingAgent();
	virtual void BeginPlay() override;
	virtual void EndPlay(EEndPlayReason::Type EndPlayReason) override;
	virtual void Tick(float DeltaScends) override;

public:
	UPROPERTY()
		USceneComponent* Root;

	/** The host of target server indicate where the data is sent to */
	UPROPERTY(EditAnywhere, category = "Networking")
		FString Host = TEXT("127.0.0.1");

	/** The port of target server */
	UPROPERTY(EditAnywhere, category = "Networking")
		uint16 Port = 8000;


	/**
	 * Send Result via WebSocket Client
	 * @param Results - the Array of WiTracing result
	 */
	UFUNCTION(BlueprintCallable, Category = "WiTracing")
		void WebSocketSend();

private:
	//--WebSocket
	void InitWebSocketClient();

	/** Handle Received Message */
	void WebSocketOnMessageHandler(const FString& Message);
	//--WebSocket End

	//--Training
	void ConfigEngine(const FTrainingAgentRecvData& Data);

	void SpawnRXs(const FTrainingAgentRecvData& Data);
	void SpawnTXs(const FTrainingAgentRecvData& Data);

	void DestroyRXs();
	void DestroyTXs();

	void PrepareTrainStep(const FTrainingAgentRecvData& Data);
	void WiTracingInference();
	void SendInferenceResult(TArray<FWiTracingResult> Results);
	void FinalizeTrainStep();
	//--Training End

private:
	/** Web Socket */
	TSharedPtr<IWebSocket> WebSocket;

	/** For TX Estimation */
	FTrainingAgentRecvData RecvData;
	TArray<AWirelessRX*> RXs;
	TArray<AWirelessTX*> TXs;
	TMap<FString, int32> Labels;
	TMap<FString, int64> RXIndices;
	TMap<FString, int64> TXIndices;
};