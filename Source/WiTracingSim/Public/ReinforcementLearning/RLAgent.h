// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Networking/UdpServerComponent.h"
#include "RLAgent.generated.h"


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
	UPROPERTY(BlueprintReadOnly, EditAnywhere, Category = "Agent", meta = (AllowPrivateAccess = "true"))
		FAgentData AgentData;
	UPROPERTY(BlueprintReadWrite, Category = "Network")
		FString Message = "";
	UPROPERTY(BlueprintReadWrite, Category = "Network")
		bool Forward = false;
	UPROPERTY(BlueprintReadWrite, Category = "Network")
		bool Forwardkeep = false;
	UPROPERTY(BlueprintReadWrite, Category = "Network")
		bool Turnleft = false;
	UPROPERTY(BlueprintReadWrite, Category = "Network")
		bool Turnright = false;

private:
	// Is this really necessary?
	class UUdpServerComponent* GetUdpServerComponent() const { return UdpServerComponent; }

	// UdpServerComponent Callback
	virtual void OnUdpServerComponentRLAgentDataRecv(FString RecvData, FString& RespData) override;
	// bool Forward = true;
	// bool Forwardkeep = true;
	// bool Turnleft = true;
	// bool Turnright = false;

};
