// Fill out your copyright notice in the Description page of Project Settings.


#include "ReinforcementLearning/RLAgent.h"
#include "JsonObjectConverter.h"
#include "DrawDebugHelpers.h"

// Sets default values
ARLAgent::ARLAgent()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;
	Root = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
	RootComponent = Root;

	UdpServerComponent = CreateDefaultSubobject<UUdpServerComponent>(TEXT("UdpServerComponent0"));
	UdpServerComponent = CastChecked<UUdpServerComponent>(GetUdpServerComponent());
	UdpServerComponent->SetupAttachment(Root);

	UdpServerComponent->Host = "127.0.0.1";
	UdpServerComponent->Port = 9001;
	UdpServerComponent->RLDelegate = this;

}

// Called when the game starts or when spawned
void ARLAgent::BeginPlay()
{
	Super::BeginPlay();
	
}

// Called every frame
void ARLAgent::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}

void ARLAgent::OnUdpServerComponentRLAgentDataRecv(FString RecvData, FString& RespData)
{
	RespData = FString("Receiving RLAgent message");
}

