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
	// AgentData.BaseLocation = this->GetActorLocation();
	// AgentData.x = 1.0;
}

// Called every frame
void ARLAgent::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
	// AgentData.BaseLocation = this->GetActorLocation();
	if (AgentData.move_forward && !Forwardkeep)
	{
		Forward = true;
		Turnright = false;
		Turnleft = false;
		Forwardkeep = true;
	}
	else
	{
		Forward = false;
	}

	if (AgentData.turn_left)
	{
		Turnleft = true;
		Turnright = false;
		Forwardkeep = false;

	}

	if (AgentData.turn_right)
	{
		Turnleft = false;
		Turnright = true;
		Forwardkeep = false;

	}

	// Turnleft = AgentData.turn_left;
	// Turnright = AgentData.turn_right;

	if (!AgentData.move_forward && !AgentData.turn_left && !AgentData.turn_right)
	{
		Turnleft = false;
		Turnright = false;
		Forwardkeep = false;
	}
	
}

void ARLAgent::OnUdpServerComponentRLAgentDataRecv(FString RecvData, FString& RespData)
{
	// RespData = FString("Receiving RLAgent message");
	// RespData = RecvData;
	// Message = RecvData;
	// Parse data to struct
	TSharedPtr<FJsonObject, ESPMode::ThreadSafe> JsonObject;
	TSharedRef< TJsonReader<> > Reader = TJsonReaderFactory<>::Create(RecvData);
	if (FJsonSerializer::Deserialize(Reader, JsonObject))
	{
		FJsonObjectConverter::JsonObjectStringToUStruct(*RecvData, &AgentData, 0, 0);
		RespData = "Action received.";
	}
	else
	{
		RespData = FString("{\"state\":\"[ERR] Decode json string failed.\"}");
	}
}

