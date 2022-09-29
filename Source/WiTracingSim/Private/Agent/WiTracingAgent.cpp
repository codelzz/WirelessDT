#include "Agent/WiTracingAgent.h"
#include "Kismet/GameplayStatics.h"
#include "Kismet/KismetRenderingLibrary.h"
#include "Kismet/KismetMathLibrary.h"
#include "JsonObjectConverter.h"
#include "WiTracing/WiTracingRendererBlueprintLibrary.h"

TArray<float> CalcPdf(TArray<int64> CountArray)
{
	TArray<float> Pdf;
	int64 Sum = 0;
	for (auto Count : CountArray)
	{
		Sum += Count;
	}
	if (Sum > 0)
	{
		Pdf.Empty(CountArray.Num());
		for (auto Count : CountArray)
		{
			Pdf.Emplace(Count * 1.0f / Sum);
		}
	}
	return Pdf;
}



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

void AWiTracingAgent::WiTracing(AWirelessTX* WirelessTX, AWirelessRX* WirelessRX)
{
	if (WirelessTX && WirelessRX)
	{
		UWiTracingRendererBlueprintLibrary::WiTracing(GetWorld(), TextureRenderTarget, WirelessTX, WirelessRX);
	}
}

void AWiTracingAgent::MultiWiTracing(TArray<AWirelessTX*> WirelessTXs, AWirelessRX* WirelessRX)
{
	if (WirelessTXs.Num() > 0 && WirelessRX)
	{
		UWiTracingRendererBlueprintLibrary::MultiWiTracing(GetWorld(), TextureRenderTarget, WirelessTXs, WirelessRX);
	}
}

void AWiTracingAgent::IterativeWiTracing(FTransform Transform, TArray<float>& RSSIPdf, bool bVisualized)
{
	// Need to be optimized by reducing redundant code compared to GetTX()
	
	const int32 TXNum = TXs.Num();
	if (TXNum > 0)
	{
		// If there is TXs in the scene do ...
		TXIndex = TXIndex % (TXNum + 1);
		// if (PlayerController)
		{
			TArray<int64> RSSICount;
			AWirelessTransmitter* TX = nullptr;
			// FTransform Transform = PlayerController->PlayerCameraManager->GetActorTransform();
			if (TXIndex < TXNum)
			{
				TX = TXs[TXIndex];
				UWiTracingRendererBlueprintLibrary::RenderWiTracingByTransmitter(GetWorld(), Transform, bVisualized ? TextureRenderTarget : TextureRenderTargetTemp, TX, RSSICount);
			}
			else
			{
				UWiTracingRendererBlueprintLibrary::RenderWiTracing(GetWorld(), Transform, bVisualized ? TextureRenderTarget : TextureRenderTargetTemp, RSSICount);
			}

			// Calculate PDF
			RSSIPdf = CalcPdf(RSSICount);

			// not denoise required anymore
			// remove background noise if required
			//if (bEnableBackgroundDenoising)
			//{
			//	RemoveBackgroundNoise(RSSIPdf);
			//}

			// send result
			if (UdpSocketServerComponent)
			{
				// todo: send pdf instead of count
				FWiTracingResult StructData(
					TX ? TX->GetName() : FString("Total"),
					Transform.GetLocation(), 
					RSSICount,
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

int64 AWiTracingAgent::RSSISampling(TArray<float> RSSIPdf)
{
	// We based the fact that stronger signal has better reception rate
	// hence it is more likely to be detected by sensor
	// So, we apply a pdf according this fact to the origin RSSIPdf
	// According to the related work experiment this effect will 
	// act as a grey area. However this grey area only happen on around -87 dBm
	// with as small window which already be consider in wi tracing shadering process

	// The array store the not zero value index from RSSIPdf 
	TArray<int64> RSSIIndices;
	TArray<float> RSSIIndicesPdf;
	for (int Index = 0; Index < RSSIPdf.Num(); Index++)
	{
		const float Pdf = RSSIPdf[Index];
		if (Pdf > 0)
		{
			RSSIIndices.Emplace(Index);
			RSSIIndicesPdf.Emplace(Pdf);
		}
	}
	// Step 1: Sample on non-zero probability indices
	if (RSSIIndices.Num() > 0)
	{
		// construct Indices CDF
		TArray<float> RSSIIndicesCDF;
		RSSIIndicesCDF.Empty(RSSIIndices.Num());
		float PdfSum = 0.0f;
		// Calculate CDF
		for (auto Pdf : RSSIIndicesPdf)
		{
			PdfSum += Pdf;
			RSSIIndicesCDF.Emplace(PdfSum);
		}

		// Step 2: Binary Search for Index of RSSIIndices
		int Index = 0;
		for (int Count = RSSIIndicesCDF.Num(); Count > 0;)
		{
			int Step = Count / 2;
			int Iterator = Index + Step;
			float RandNum = FMath::RandRange(0.0f, 1.0f);
			if (RandNum < RSSIIndicesCDF[Iterator])
			{
				Count = Step;
			}
			else
			{
				Index = Iterator + 1;
				Count -= Step + 1;
			}
		}

		// Step 3: Convert to RSSI reading
		return -RSSIIndices[Index];
	} 
	return RSSI_MIN;
}

int64 AWiTracingAgent::RSSIMultiSampling(TArray<float> RSSIPdf, int64 n)
{
	TArray<int32> Samples;
	Samples.Empty(n);
	for (int Index = 0; Index < n; Index++)
	{
		int32 Sample = RSSISampling(RSSIPdf);
		if (Sample > RSSI_MIN)
		{
			Samples.Emplace(Sample);
		}
	}
	if (Samples.Num() > 0)
	{
		int32 Max;
		int32 Index;
		UKismetMathLibrary::MaxOfIntArray(Samples, Index, Max);
		return Max;
	}
	return 0;
}

int64 AWiTracingAgent::RSSIMaxSampling(TArray<float> RSSIPdf)
{	
	int64 RSSI = RSSI_MIN;
	for (int32 Index = 0; Index < RSSIPdf.Num(); Index++)
	{
		if (RSSIPdf[Index] > 0.0f)
		{
			RSSI = -Index;
			break;
		}
	}
	return RSSI;
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

void AWiTracingAgent::GlobalWiTracing(FTransform Transform, TArray<float>& RSSIPdf, bool bVisualized)
{
	// if (PlayerController)
	{
		// todo: change player controller transform to receiver transform
		// FTransform Transform = PlayerController->PlayerCameraManager->GetActorTransform();
		TArray<int64> RSSICount;
		UWiTracingRendererBlueprintLibrary::RenderWiTracing(GetWorld(), Transform, bVisualized ? TextureRenderTarget : TextureRenderTargetTemp, RSSICount);
		RSSIPdf = CalcPdf(RSSICount);
	}
}

void AWiTracingAgent::CachePlayerController()
{
	PlayerController = UGameplayStatics::GetPlayerController(GetWorld(), 0);
}

void AWiTracingAgent::InitRenderTargets()
{
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