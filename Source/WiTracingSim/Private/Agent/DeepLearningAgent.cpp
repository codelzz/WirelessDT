#include "Agent/DeepLearningAgent.h"
#include "JsonObjectConverter.h"
#include "DrawDebugHelpers.h"

ADeepLearningAgent::ADeepLearningAgent()
{
	PrimaryActorTick.bCanEverTick = true;
	Root = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
	RootComponent = Root;

	UdpSocketServerComponent = CreateDefaultSubobject<UUdpSocketServerComponent>(TEXT("UdpSocketServerComponent0"));
	UdpSocketServerComponent = CastChecked<UUdpSocketServerComponent>(GetUdpSocketServerComponent());
	UdpSocketServerComponent->SetupAttachment(Root);

	UdpSocketServerComponent->ServerPort = 7000;
	UdpSocketServerComponent->ClientPort = 7001;
	UdpSocketServerComponent->Delegate = this;
}

void ADeepLearningAgent::BeginPlay()
{
	Super::BeginPlay();
}


void ADeepLearningAgent::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
	if (bNeedSync)
	{
		UE_LOG(LogTemp, Warning, TEXT("%s"), *Data.GetLocation().ToString());
		bNeedSync = false;

		DrawDebugBox(GetWorld(), Data.GetLocation(), FVector(5, 5, 5), FColor::Red, true, 30.f,  (uint8)0U, 5.0f);
	}
}

void ADeepLearningAgent::OnUdpSocketServerComponentDataRecv(FString InData)
{
	// Parse data to struct
	TSharedPtr<FJsonObject, ESPMode::ThreadSafe> JsonObject;
	TSharedRef< TJsonReader<> > Reader = TJsonReaderFactory<>::Create(InData);
	if (FJsonSerializer::Deserialize(Reader, JsonObject))
	{
		FJsonObjectConverter::JsonObjectStringToUStruct(*InData, &Data, 0, 0);
		bNeedSync = true;
	}
}