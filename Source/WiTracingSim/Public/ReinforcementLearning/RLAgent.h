// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Networking/UdpServerComponent.h"
#include "Networking/UdpClientComponent.h"
#include "RLAgent.generated.h"

// DECLARE_MULTICAST_DELEGATE(FInputAction)

USTRUCT(BlueprintType)
struct FAgentData
{
	GENERATED_BODY()

public:
	UPROPERTY(BlueprintReadWrite, Category = "action")
		bool move_forward = false;
	UPROPERTY(BlueprintReadWrite, Category = "action")
		bool turn_left = false;
	UPROPERTY(BlueprintReadWrite, Category = "action")
		bool turn_right = false;
	UPROPERTY(BlueprintReadWrite, Category = "action")
		bool reset = false;

};

USTRUCT(BlueprintType)
struct FAgentReward
{
	GENERATED_BODY()

public:
	UPROPERTY(BlueprintReadWrite, Category = "reward")
		float position_reward;

};

UCLASS()
class ARLAgent : public AActor, public FUdpServerComponentRLAgentDelegate
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	ARLAgent();

	virtual void BeginPlay() override;
	virtual void Tick(float DeltaScends) override;

public:
	// UPROPERTY Object can be seen in blueprint detial tab
	// USceneComponent define the transform information including location, scale and rotation
	UPROPERTY()
		USceneComponent* Root;

	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "Network", meta = (AllowPrivateAccess = "true"))
		TObjectPtr<class UUdpServerComponent> UdpServerComponent;
	/** Udp client for communication */
	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "Network", meta = (AllowPrivateAccess = "true"))
		TObjectPtr<class UUdpClientComponent> UdpClientComponent;
	UPROPERTY(BlueprintReadOnly, EditAnywhere, Category = "Agent", meta = (AllowPrivateAccess = "true"))
		FAgentData AgentData;
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "Agent", meta = (AllowPrivateAccess = "true"))
		FAgentReward AgentReward;
	UPROPERTY(BlueprintReadWrite, Category = "Network")
		FString Message = "";
	UPROPERTY(BlueprintReadWrite, Category = "Action")
		bool Forward = false;
	UPROPERTY(BlueprintReadWrite, Category = "Action")
		bool Forwardkeep = false;
	UPROPERTY(BlueprintReadWrite, Category = "Action")
		bool Turnleft = false;
	UPROPERTY(BlueprintReadWrite, Category = "Action")
		bool Turnright = false;
	UPROPERTY(BlueprintReadWrite, Category = "Action")
		bool Reset = false;

	/** The host of target server indicate where the data is sent to */
	UPROPERTY(EditAnywhere, category = "Endpint")
		FString Host = TEXT("127.0.0.1");

	/** The port of target server */
	UPROPERTY(EditAnywhere, category = "Endpint")
		uint16 Port = 9002;
public:
	/**
	 * Send Result via UDP Client
	 * @param Result - the WiTracing result
	 */
	UFUNCTION(BlueprintCallable, Category = "WiTracing")
		void UDPSendRLReward(FAgentReward Reward);
	UFUNCTION(BlueprintImplementableEvent)
		void GetEnvStatus();

private:
	// Is this really necessary?
	class UUdpServerComponent* GetUdpServerComponent() const { return UdpServerComponent; }

	// UdpServerComponent Callback
	virtual void OnUdpServerComponentRLAgentDataRecv(FString RecvData, FString& RespData) override;
	// bool Forward = true;
	// bool Forwardkeep = true;
	// bool Turnleft = true;
	// bool Turnright = false;
	class UUdpClientComponent* GetUdpClientComponent() const { return UdpClientComponent; }

	bool bActionReceived = false;

};
