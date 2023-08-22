#include "Sync/MobileSync.h"
#include "JsonObjectConverter.h"
#include "DrawDebugHelpers.h"

AMobileSync::AMobileSync()
{
	PrimaryActorTick.bCanEverTick = true;
	Root = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
	RootComponent = Root;

	UdpServerComponent = CreateDefaultSubobject<UUdpServerComponent>(TEXT("UdpServerComponent0"));
	UdpServerComponent = CastChecked<UUdpServerComponent>(GetUdpServerComponent());
	UdpServerComponent->SetupAttachment(Root);

	UdpServerComponent->Host = "192.168.0.55";
	UdpServerComponent->Port = 9000;
	UdpServerComponent->Delegate = this;
}

void AMobileSync::BeginPlay()
{
	Super::BeginPlay();
	MobileData.BaseLocation = this->GetActorLocation();
	BeaconReceived = false;
}


void AMobileSync::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
	if (bNeedSync)
	{
		GEngine->AddOnScreenDebugMessage(100, 15.0f, FColor::Green, FString::Printf(TEXT(" Location | %s, %s"), *MobileData.GetLocation().ToString(), *MobileData.GetRotator().ToString()));
		bNeedSync = false;
		SetActorLocationAndRotation(MobileData.GetLocation(), MobileData.GetRotator(), false, 0, ETeleportType::ResetPhysics);
	}
}

TArray<FString> AMobileSync::GetRXName() 
{
	return MobileData.GetRXNames();
}

TArray<int64> AMobileSync::GetRXRssi()
{
	return MobileData.GetRXRssis();
}

void AMobileSync::OnUdpServerComponentDataRecv(FString RecvData, FString& RespData)
{
	// Parse data to struct
	TSharedPtr<FJsonObject, ESPMode::ThreadSafe> JsonObject;
	TSharedRef< TJsonReader<> > Reader = TJsonReaderFactory<>::Create(RecvData);
	if (FJsonSerializer::Deserialize(Reader, JsonObject))
	{
		FJsonObjectConverter::JsonObjectStringToUStruct(*RecvData, &MobileData, 0, 0);
		if (MobileData.rxname.Num() > 0) {
			for (int Index = 0; Index < MobileData.rxname.Num(); Index++) {
				GEngine->AddOnScreenDebugMessage(Index, 15.0f, FColor::Green, FString::Printf(TEXT("Mobile data received! | %s %d"), *MobileData.rxname[Index], MobileData.rxrssi[Index]));
				BeaconReceived = true;
			}
		}
		else {
			GEngine->AddOnScreenDebugMessage(0, 15.0f, FColor::Green, FString::Printf(TEXT("Mobile data received! | No Beacon Detected!")));
			BeaconReceived = false;
		}
		bNeedSync = true;
	} 
	else
	{
		RespData = FString("{\"state\":\"[ERR] Decode json string failed.\"}");
	}
}