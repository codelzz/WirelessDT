#include "Agent/WiTracingAgent.h"
#include "Kismet/GameplayStatics.h"
#include "Kismet/KismetRenderingLibrary.h"
#include "Kismet/KismetMathLibrary.h"
#include "JsonObjectConverter.h"
#include "WebSocketsModule.h"
#include "WiTracing/WiTracingRendererBlueprintLibrary.h"


AWiTracingAgent::AWiTracingAgent()
{
	Root = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
	RootComponent = Root;
}

void AWiTracingAgent::BeginPlay()
{
	Super::BeginPlay();

	InitWebSocket();
	InitRenderTargets();
	CacheTXs();
	CacheRXs();
}

void AWiTracingAgent::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
}

void AWiTracingAgent::EndPlay(EEndPlayReason::Type EndPlayReason)
{
	if (WebSocket->IsConnected())
	{
		WebSocket->Close();
	}
	Super::EndPlay(EndPlayReason);
}

void AWiTracingAgent::WiTracing(AWirelessTX* WirelessTX, AWirelessRX* WirelessRX, FWiTracingResult& Result, bool OctahedralProjection, bool bDenoised, bool bVisualized)
{
	if (WirelessTX == nullptr || WirelessRX == nullptr) {
		// skip if Wireless Transmitter or Wireless Receiver not exist
		return;
	}
	UWiTracingRendererBlueprintLibrary::WiTracing(GetWorld(), GetRenderTarget(bVisualized), WirelessTX, WirelessRX, Result, OctahedralProjection, bDenoised);
}

void AWiTracingAgent::MultiTXWiTracing(TArray<AWirelessTX*> WirelessTXs, AWirelessRX* WirelessRX, TArray<FWiTracingResult>& Results, bool OctahedralProjection, bool bDenoised, bool bVisualized)
{
	if (WirelessTXs.Num() < 1 || WirelessRX == nullptr) {
		// skip if Wireless Transmitter or Wireless Receiver not exist
		return;
	}
	UWiTracingRendererBlueprintLibrary::MultiTXWiTracing(GetWorld(), GetRenderTarget(bVisualized), WirelessTXs, WirelessRX, Results, OctahedralProjection, bDenoised);
}

void AWiTracingAgent::MultiWiTracing(TArray<AWirelessRX*> WirelessRXs, TArray<AWirelessTX*> WirelessTXs,  TArray<FWiTracingResult>& Results, bool OctahedralProjection, bool bDenoised)
{
	if (WirelessRXs.Num() < 1 || WirelessTXs.Num() < 1) {
		// skip if Wireless Transmitter or Wireless Receiver not exist
		return;
	}
	UWiTracingRendererBlueprintLibrary::MultiWiTracing(GetWorld(), GetRenderTarget(false), WirelessTXs, WirelessRXs, Results, OctahedralProjection, bDenoised);
}

void AWiTracingAgent::PreviewWiTracing(TArray<AWirelessTX*> WirelessTXs, AWirelessRX* WirelessRX, bool OctahedralProjection, bool bDenoised) {

	if (WirelessTXs.Num() < 1 || WirelessRX == nullptr) {
		// skip if Wireless Transmitter or Wireless Receiver not exist
		return;
	}
	UWiTracingRendererBlueprintLibrary::PreviewWiTracing(GetWorld(), TextureRenderTargetVis, WirelessTXs, WirelessRX, OctahedralProjection, bDenoised);
}

void AWiTracingAgent::WebSocketSend(TArray<FWiTracingResult> Results)
{
	if (!WebSocket->IsConnected())
	{
		WebSocket->Connect();
	}
	
	// Send Data
	FString Data = "";
	for (auto Result : Results) {
		FString JsonData;
		if (FJsonObjectConverter::UStructToJsonObjectString(Result, JsonData, 0, 0))
		{
			Data += JsonData;
		}
	}
	if (Data != "" && WebSocket->IsConnected()) {
		WebSocket->Send(Data);
	}
}

UTextureRenderTarget2D* AWiTracingAgent::GetRenderTarget(bool bVisualized) {
	UTextureRenderTarget2D* RenderTarget = TextureRenderTarget;
	if (bVisualized) {
		RenderTarget = TextureRenderTargetVis;
	}
	return RenderTarget;
}

const TArray<AWirelessTX*> AWiTracingAgent::GetTXsInRange(FVector Origin, float Radius) {
	TArray<AWirelessTX*> InRangeTXs;
	InRangeTXs.Empty();
	for (AWirelessTX* TX : TXs)
	{
		if (FVector::Dist(Origin, TX->GetActorLocation()) <= Radius) {
			InRangeTXs.Add(TX);
		}
	}
	return InRangeTXs;
}

const TArray<AWirelessTX*> AWiTracingAgent::GetTXsOutRange(FVector Origin, float Radius) {
	TArray<AWirelessTX*> InRangeTXs;
	InRangeTXs.Empty();
	for (AWirelessTX* TX : TXs)
	{
		if (FVector::Dist(Origin, TX->GetActorLocation()) > Radius) {
			InRangeTXs.Add(TX);
		}
	}
	return InRangeTXs;
}

//--- WEBSOCKET
void AWiTracingAgent::InitWebSocket()
{
	if (!FModuleManager::Get().IsModuleLoaded("WebSockets")) 
	{
		FModuleManager::Get().LoadModule("WebSockets");
	}

	WebSocket = FWebSocketsModule::Get().CreateWebSocket(FString::Printf(TEXT("ws://%s:%d"), *Host, Port));
	
	WebSocket->OnConnected().AddLambda([]()
		{
			GEngine->AddOnScreenDebugMessage(-1, 15.0f, FColor::Green, "Successful connected!");
		});

	WebSocket->OnConnectionError().AddLambda([](const FString& Error)
		{
			GEngine->AddOnScreenDebugMessage(-1, 15.0f, FColor::Red, Error);
		});

	WebSocket->OnClosed().AddLambda([](int32 StatusCode, const FString& Reason, bool bWasClean)
		{
			GEngine->AddOnScreenDebugMessage(-1, 15.0f, bWasClean ? FColor::Green : FColor::Red, "Connection closed " + Reason);
		});

	WebSocket->OnMessage().AddLambda([](const FString& Message)
		{
			GEngine->AddOnScreenDebugMessage(-1, 15.0f, FColor::Cyan, "[RECV] " + Message);
		});

	WebSocket->OnMessageSent().AddLambda([](const FString& Message)
		{
			GEngine->AddOnScreenDebugMessage(-1, 15.0f, FColor::Yellow, "[SENT] ");
		});

	WebSocket->Connect();
}
//--- WEBSOCKET


void AWiTracingAgent::InitRenderTargets()
{
	if (TextureRenderTarget)
	{
		UKismetRenderingLibrary::ClearRenderTarget2D(GetWorld(), TextureRenderTarget, FLinearColor::Transparent);
	}
	if (TextureRenderTargetVis)
	{
		UKismetRenderingLibrary::ClearRenderTarget2D(GetWorld(), TextureRenderTargetVis, FLinearColor::Transparent);
	}
}

void AWiTracingAgent::CacheTXs()
{
	UWorld* World = GetWorld();
	if (World)
	{
		TArray<AActor*> Actors;
		UGameplayStatics::GetAllActorsOfClass(World, AWirelessTX::StaticClass(), Actors);
		TXs.Empty();
		for (AActor* Actor : Actors)
		{
			TXs.Add(static_cast<AWirelessTX*>(Actor));
		}
	}
}

void AWiTracingAgent::CacheRXs()
{
	UWorld* World = GetWorld();
	if (World)
	{
		TArray<AActor*> Actors;
		UGameplayStatics::GetAllActorsOfClass(World, AWirelessRX::StaticClass(), Actors);
		RXs.Empty();
		for (AActor* Actor : Actors)
		{
			RXs.Add(static_cast<AWirelessRX*>(Actor));
		}
	}
}