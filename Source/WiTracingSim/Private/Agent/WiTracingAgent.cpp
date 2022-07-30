#include "Agent/WiTracingAgent.h"
#include "Kismet/GameplayStatics.h"
#include "Kismet/KismetRenderingLibrary.h"
#include "JsonObjectConverter.h"
#include "WiTracing/WiTracingRendererBlueprintLibrary.h"

AWiTracingAgent::AWiTracingAgent()
{
	// PrimaryActorTick.bCanEverTick = true;

	Root = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
	RootComponent = Root;

	UdpSocketServerComponent = CreateDefaultSubobject<UUdpSocketServerComponent>(TEXT("UdpSocketServerComponent0"));
	UdpSocketServerComponent = CastChecked<UUdpSocketServerComponent>(GetUdpSocketServerComponent());

	UdpSocketServerComponent->SetupAttachment(Root);
}

void AWiTracingAgent::BeginPlay()
{
	Super::BeginPlay();

	InitRenderTargets();
	CachePlayerController();
	CacheTXs();
}

void AWiTracingAgent::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	// UE_LOG(LogTemp, Warning, TEXT("WiTracing Tick"));

}

void AWiTracingAgent::IterativeWiTracing(TArray<int64>& RSSIPdf, bool bVisualized)
{
	// Need to be optimized by reducing redundant code compared to GetTX()
	const int32 TXNum = TXs.Num();
	if (TXNum > 0)
	{
		// If there is TXs in the scene do ...
		TXIndex = TXIndex % (TXNum + 1);
		if (PlayerController)
		{
			AWirelessTransmitter* TX = nullptr;
			FTransform Transform = PlayerController->PlayerCameraManager->GetActorTransform();
			if (TXIndex < TXNum)
			{
				TX = TXs[TXIndex];
				UWiTracingRendererBlueprintLibrary::RenderWiTracingByTransmitter(GetWorld(), Transform, bVisualized ? TextureRenderTarget : TextureRenderTargetTemp, TX, RSSIPdf);
			}
			else
			{
				UWiTracingRendererBlueprintLibrary::RenderWiTracing(GetWorld(), Transform, bVisualized ? TextureRenderTarget : TextureRenderTargetTemp, RSSIPdf);
			}

			// remove background noise if required
			if (bEnableBackgroundDenoising)
			{
				RemoveBackgroundNoise(RSSIPdf);
			}

			if (UdpSocketServerComponent)
			{
				FWiTracingResult StructData(
					TX ? TX->GetName() : FString("Total"),
					Transform.GetLocation(), 
					RSSIPdf, 
					FDateTime::UtcNow().ToUnixTimestamp() * 1000 + FDateTime::UtcNow().GetMillisecond());
				FString JsonData;
				if (FJsonObjectConverter::UStructToJsonObjectString(StructData, JsonData, 0, 0))
				{
					UdpSocketServerComponent->Send(JsonData);
				}
			}
		}
		TXIndex++;
	}
}

void AWiTracingAgent::RemoveBackgroundNoise(TArray<int64>& RSSIPdf)
{
	int32 MaxRSSIIndex = -1;
	int32 Index = 0;
	for (auto& RSSI : RSSIPdf)
	{
		if (RSSI != 0)
		{
			// we only set max index when found it at the first time
			if (MaxRSSIIndex < 0)
			{
				MaxRSSIIndex = Index;
			}
			
			if (Index > MaxRSSIIndex + BackgroundNoiseSNR)
			{
				// set all RSSI value out of valid boundary to 0
				RSSI = 0;
			}
		}
		Index++;
	}
}

AWirelessTransmitter* AWiTracingAgent::GetTX()
{
	AWirelessTransmitter* TX = nullptr;
	const int32 TXNum = TXs.Num();
	if (TXNum > 0)
	{
		const int32 Index = TXIndex % (TXNum + 1);
		if (Index < TXNum)
		{
			TX = TXs[Index];
		}
	}
	return TX;
}

void AWiTracingAgent::GlobalWiTracing(TArray<int64>& RSSIPdf, bool bVisualized)
{
	if (PlayerController)
	{
		FTransform Transform = PlayerController->PlayerCameraManager->GetActorTransform();
		UWiTracingRendererBlueprintLibrary::RenderWiTracing(GetWorld(), Transform, bVisualized ? TextureRenderTarget : TextureRenderTargetTemp, RSSIPdf);
	}
}


void AWiTracingAgent::CachePlayerController()
{
	PlayerController = UGameplayStatics::GetPlayerController(GetWorld(), 0);
}

void AWiTracingAgent::InitRenderTargets()
{
	// Clean up texture
	if (TextureRenderTarget)
	{
		UKismetRenderingLibrary::ClearRenderTarget2D(GetWorld(), TextureRenderTarget, FLinearColor::Transparent);
	}
	if (TextureRenderTargetTemp)
	{
		UKismetRenderingLibrary::ClearRenderTarget2D(GetWorld(), TextureRenderTargetTemp, FLinearColor::Transparent);
	}
}

void AWiTracingAgent::CacheTXs()
{
	UWorld* World = GetWorld();
	if (World)
	{
		TArray<AActor*> Actors;
		UGameplayStatics::GetAllActorsOfClass(World, AWirelessTransmitter::StaticClass(), Actors);
		TXs.Empty();
		for (AActor* Actor : Actors)
		{
			TXs.Add(static_cast<AWirelessTransmitter*>(Actor));
		}
	}
}