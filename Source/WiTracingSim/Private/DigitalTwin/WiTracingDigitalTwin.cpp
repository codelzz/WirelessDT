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
	// Tick
	PrimaryActorTick.bCanEverTick = true;
	this->SetActorTickInterval(0.01);
}

void AWiTracingDigitalTwin::BeginPlay()
{
	Super::BeginPlay();
	// Set the BaseLocation
	Data.BaseLocation = this->GetActorLocation();
}


void AWiTracingDigitalTwin::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
	if (bNeedSync)
	{
		// if synchorization is reqired then change the position
		SetActorLocationAndRotation(Data.GetLocation(), Data.GetRotator(), false, 0, ETeleportType::ResetPhysics);
		bNeedSync = false;
	}
}

void AWiTracingDigitalTwin::OnUdpSocketServerComponentDataRecv(FString InData)
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
