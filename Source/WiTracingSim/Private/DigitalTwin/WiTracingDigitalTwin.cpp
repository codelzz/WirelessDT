#include "DigitalTwin/WiTracingDigitalTwin.h"
#include "JsonObjectConverter.h"

AWiTracingDigitalTwin::AWiTracingDigitalTwin()
{
	Root = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
	RootComponent = Root;
	// UdpSocketServer 
	UdpSocketServerComponent = CreateDefaultSubobject<UUdpSocketServerComponent>(TEXT("UdpSocketServerComponent0"));
	UdpSocketServerComponent = CastChecked<UUdpSocketServerComponent>(GetUdpSocketServerComponent());
	UdpSocketServerComponent->SetupAttachment(Root);
	UdpSocketServerComponent->ServerPort = 8800;
	UdpSocketServerComponent->ClientPort = 8000;
	UdpSocketServerComponent->Delegate = this;
	// Data Struct
	DigitalTwinData.SetLocation(this->GetActorLocation());
	// Tick
	PrimaryActorTick.bCanEverTick = true;
	this->SetActorTickInterval(0.01);
}

void AWiTracingDigitalTwin::BeginPlay()
{
	Super::BeginPlay();
}


void AWiTracingDigitalTwin::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
	if (bNeedSync)
	{
		SetActorLocationAndRotation(DigitalTwinData.GetLocation(), DigitalTwinData.GetRotator(), false, 0, ETeleportType::ResetPhysics);
		bNeedSync = false;
	}
}

void AWiTracingDigitalTwin::OnUdpSocketServerComponentDataRecv(FString Data)
{
	// Parse data to struct
	TSharedPtr<FJsonObject, ESPMode::ThreadSafe> JsonObject;
	TSharedRef< TJsonReader<> > Reader = TJsonReaderFactory<>::Create(Data);
	if (FJsonSerializer::Deserialize(Reader, JsonObject))
	{
		FJsonObjectConverter::JsonObjectStringToUStruct(*Data, &DigitalTwinData, 0, 0);
		bNeedSync = true;
		// UE_LOG(LogTemp, Warning, TEXT("UDP -> %s"), *DigitalTwinData.GetLocation().ToString());
		// we can't modify the location here, it only can be used in game thread
	}
}
