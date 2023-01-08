#include "Sync/SensorSync.h"
#include "JsonObjectConverter.h"
#include "DrawDebugHelpers.h"

ASensorSync::ASensorSync()
{
	PrimaryActorTick.bCanEverTick = true;
	Root = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
	RootComponent = Root;

	UdpServerComponent = CreateDefaultSubobject<UUdpServerComponent>(TEXT("UdpServerComponent0"));
	UdpServerComponent = CastChecked<UUdpServerComponent>(GetUdpServerComponent());
	UdpServerComponent->SetupAttachment(Root);

	UdpServerComponent->Host = "127.0.0.1";
	UdpServerComponent->Port = 9000;
	UdpServerComponent->Delegate = this;
}

void ASensorSync::BeginPlay()
{
	Super::BeginPlay();
	SensorData.BaseLocation = this->GetActorLocation();
}

void ASensorSync::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
	if (bNeedSync)
	{
		UE_LOG(LogTemp, Warning, TEXT("%s, %s"), *SensorData.GetLocation().ToString(), *SensorData.GetRotator().ToString());
		bNeedSync = false;
		SetActorLocationAndRotation(SensorData.GetLocation(), SensorData.GetRotator(), false, 0, ETeleportType::ResetPhysics);
	}
}

void ASensorSync::OnUdpServerComponentDataRecv(FString RecvData, FString& RespData)
{
	// Parse data to struct
	TSharedPtr<FJsonObject, ESPMode::ThreadSafe> JsonObject;
	TSharedRef< TJsonReader<> > Reader = TJsonReaderFactory<>::Create(RecvData);
	if (FJsonSerializer::Deserialize(Reader, JsonObject))
	{
		FJsonObjectConverter::JsonObjectStringToUStruct(*RecvData, &SensorData, 0, 0);
		bNeedSync = true;
	}
	else
	{
		RespData = FString("{\"state\":\"[ERR] Decode json string failed.\"}");
	}
}