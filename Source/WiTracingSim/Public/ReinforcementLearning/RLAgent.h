// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Networking/UdpServerComponent.h"
#include "RLAgent.generated.h"

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

private:
	// Is this really necessary?
	class UUdpServerComponent* GetUdpServerComponent() const { return UdpServerComponent; }

	// UdpServerComponent Callback
	virtual void OnUdpServerComponentRLAgentDataRecv(FString RecvData, FString& RespData) override;


};
