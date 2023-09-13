#include "Agent/TrainingAgent.h"
#include "Kismet/GameplayStatics.h"
#include "Kismet/KismetMathLibrary.h"
#include "JsonObjectConverter.h"
#include "WebSocketsModule.h"
#include "DrawDebugHelpers.h"
#include "Engine/TextureRenderTarget2D.h"


ATrainingAgent::ATrainingAgent()
{
	Root = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
	RootComponent = Root;
}

void ATrainingAgent::BeginPlay()
{
	Super::BeginPlay();
	InitWebSocketClient();
}

void ATrainingAgent::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
}

void ATrainingAgent::EndPlay(EEndPlayReason::Type EndPlayReason)
{
	if (WebSocket->IsConnected())
	{
		WebSocket->Close();
	}
	Super::EndPlay(EndPlayReason);
}

//--- WEBSOCKET
void ATrainingAgent::WebSocketSend()
{
	if (!WebSocket->IsConnected())
	{
		WebSocket->Connect();
	}

	// Send Data
	FString Data = "{\"Data\":\"Data From Fitting Agent\"}";
	/*for (auto Result : Results) {
		FString JsonData;
		if (FJsonObjectConverter::UStructToJsonObjectString(Result, JsonData, 0, 0))
		{
			Data += JsonData;
		}
	}*/
	if (Data != "" && WebSocket->IsConnected()) {
		WebSocket->Send(Data);
	}
}

void ATrainingAgent::InitWebSocketClient()
{
	if (!FModuleManager::Get().IsModuleLoaded("WebSockets"))
	{
		FModuleManager::Get().LoadModule("WebSockets");
	}

	WebSocket = FWebSocketsModule::Get().CreateWebSocket(FString::Printf(TEXT("ws://%s:%d"), *Host, Port));

	WebSocket->OnConnected().AddLambda([this]()
		{
			this->WebSocket->Send("{\"sender\":\"UnrealEngine\"}");
			GEngine->AddOnScreenDebugMessage(-1, 15.0f, FColor::Green, "Successful connected!");
		});

	WebSocket->OnConnectionError().AddLambda([](const FString& Error)
		{
			GEngine->AddOnScreenDebugMessage(-1, 15.0f, FColor::Red, "OnConnectionError:" + Error);
		});

	WebSocket->OnClosed().AddLambda([](int32 StatusCode, const FString& Reason, bool bWasClean)
		{
			GEngine->AddOnScreenDebugMessage(-1, 15.0f, bWasClean ? FColor::Green : FColor::Red, "Connection closed:" + Reason);
		});


	WebSocket->OnMessage().AddLambda([this](const FString& Message)
		{
			this->WebSocketOnMessageHandler(Message);
		});

	WebSocket->OnMessageSent().AddLambda([](const FString& Message)
		{
			GEngine->AddOnScreenDebugMessage(-1, 15.0f, FColor::Yellow, "[SENT] " + Message.Left(50));
		});

	WebSocket->Connect();
}

void ATrainingAgent::WebSocketOnMessageHandler(const FString& Message)
{
	TSharedPtr<FJsonObject, ESPMode::ThreadSafe> JsonObject;
	TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Message);
	if (FJsonSerializer::Deserialize(Reader, JsonObject))
	{
		FTrainingAgentRecvMetaData MetaData;
		if (FJsonObjectConverter::JsonObjectStringToUStruct(*Message, &MetaData, 0, 0)) {
			if (MetaData.type == "data") {
				if (FJsonObjectConverter::JsonObjectStringToUStruct(*Message, &RecvData, 0, 0)) {
					GEngine->AddOnScreenDebugMessage(0, 15.0f, FColor::Yellow, "[RECV]" + Message);

					if (RecvData.type == "data") {
						this->PrepareTrainStep(RecvData);
						this->WiTracingInference();
						this->FinalizeTrainStep();
					}
				}
			}
			else if (MetaData.type == "loss")
			{
				// handle loss
				FTrainingAgentRecvCost RecvCost;
				if (FJsonObjectConverter::JsonObjectStringToUStruct(*Message, &RecvCost, 0, 0)) {
					GEngine->AddOnScreenDebugMessage(0, 15.0f, FColor::Yellow, "[RECV] Cost");

					FVector Center(RecvCost.x, RecvCost.y, RecvCost.cost / 2);
					FVector Extent(50.f, 50.f, RecvCost.cost / 2);
					FColor Color(255, 0, 0);
					//DrawDebugSolidBox(GetWorld(), Center, Extent, Color, true, -1.f, 0);
					DrawDebugBox(GetWorld(), Center, Extent, Color, true, -1.f, 0, 10.f);


				}
			}
		}
	}
}
//--- WEBSOCKET
void ATrainingAgent::WiTracingInference()
{
	if (TXs.Num() <= 0 || RXs.Num() <= 0) {return;}

	{
		TArray<FWiTracingResult> Results;
		UTextureRenderTarget2D* RenderTarget = NewObject<UTextureRenderTarget2D>(GetTransientPackage(), FName("RenderTarget"));
		RenderTarget->InitAutoFormat(32, 32);
		RenderTarget->ClearColor = FLinearColor(0.0f, 0.0f, 0.0f, 1.0f);

		FTimer Timer;
		float CurrentTime = Timer.GetCurrentTime();
		UWiTracingRendererBlueprintLibrary::MultiWiTracing(GetWorld(), RenderTarget, TXs, RXs, Results, true, false);
		float ElapsedTime = Timer.GetCurrentTime() - CurrentTime;

		TMap<FString, int32> RXIndice;
		RXIndice.Empty(RXs.Num());
		for (int32 Index = 0; Index < RXs.Num(); Index++)
		{
			RXIndice.Add(RXs[Index]->GetActorLabel(), Index);
		}

		this->SendInferenceResult(Results);
	}
}

void ATrainingAgent::SendInferenceResult(TArray<FWiTracingResult> Results)
{
	if (!WebSocket->IsConnected())
	{
		WebSocket->Connect();
	}

	FTrainingAgentSentData SentData;
	SentData.rxi.Empty(Results.Num());
	SentData.rxrssi.Empty(Results.Num());

	for (auto Result : Results)
	{
		FString RXName = Result.rxname;
		FString TXName = Result.txname;
		int32 RXIndex = RXIndices[RXName];
		int32 TXIndex = TXIndices[TXName];
		int32 RSSI = Result.rssi;

		SentData.txi.Emplace(TXIndex);
		SentData.rxi.Emplace(RXIndex);
		SentData.rxrssi.Emplace(RSSI);

		//GEngine->AddOnScreenDebugMessage(*RXIndice.Find(RXName), 15.0f, FColor::Cyan,
		//	FString::Printf(TEXT("[Inference] Result %5s | %5s@%5d:rssi=%5d, diff=%5d"),
		//		*Result.txname, *RXName, RXIndex, RSSI, FMath::Abs(RSSILabel - RSSI)));
	}

	FString JsonString;
	if (FJsonObjectConverter::UStructToJsonObjectString(SentData, JsonString, 0, 0))
	{
		if (JsonString != "" && WebSocket->IsConnected()) {
			WebSocket->Send(JsonString);
			// GEngine->AddOnScreenDebugMessage(Results.Num() + 1, 15.0f, FColor::Yellow, "[SENT]" + JsonString);
		}
	}
}

void ATrainingAgent::PrepareTrainStep(const FTrainingAgentRecvData& Data)
{
	this->ConfigEngine(RecvData);
	this->SpawnRXs(RecvData);
	this->SpawnTXs(RecvData);

	DrawDebugPoint(GetWorld(), FVector(RecvData.x, RecvData.y, RecvData.z), 10, FColor::Yellow, false, 0.2f, 0U);
}

void ATrainingAgent::FinalizeTrainStep()
{
	this->DestroyRXs();
	this->DestroyTXs();
	Labels.Empty();
	RecvData.Empty();
}

void ATrainingAgent::SpawnRXs(const FTrainingAgentRecvData& Data)
{
	int64 Size = RecvData.rxrssi.Num();
	RXs.Empty(Size);
	RXIndices.Empty(Size);
	for (int32 Index = 0; Index < Size; Index++)
	{
		int64 i = RecvData.rxi[Index];
		float x = RecvData.rxx[Index];
		float y = RecvData.rxy[Index];
		float z = RecvData.rxz[Index];
		float rssi = RecvData.rxrssi[Index];
		FVector Location = FVector(x, y, z);

		FActorSpawnParameters SpawnInfo;
		SpawnInfo.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
		auto Actor = GetWorld()->SpawnActor<AWirelessRX>(AWirelessRX::StaticClass(), Location, FRotator(0), SpawnInfo);
		AWirelessRX* RX = Cast<AWirelessRX>(Actor);
		FString RXName = FString::Printf(TEXT("RX%d"), i);
		RX->SetActorLabel(RXName);
		RXs.Emplace(RX);
		Labels.Add(RXName, int32(rssi));
		RXIndices.Add(RXName, i);
		DrawDebugPoint(GetWorld(), RX->GetActorLocation(), 5, FColor::Red, false, 0.5f, 0U);
	}
}

void ATrainingAgent::SpawnTXs(const FTrainingAgentRecvData& Data)
{
	int64 Size = RecvData.txpower.Num();
	TXs.Empty(Size);
	TXIndices.Empty(Size);
	for (int32 Index = 0; Index < Size; Index++)
	{
		int64 i = RecvData.txi[Index];
		float x = RecvData.txx[Index];
		float y = RecvData.txy[Index];
		float z = RecvData.txz[Index];
		float power = RecvData.txpower[Index];
		FVector Location = FVector(x, y, z);

		FActorSpawnParameters SpawnInfo;
		SpawnInfo.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
		auto Actor = GetWorld()->SpawnActor<AWirelessTX>(AWirelessTX::StaticClass(), Location, FRotator(0), SpawnInfo);
		AWirelessTX* TX = Cast<AWirelessTX>(Actor);
		FString TXName = FString::Printf(TEXT("TX%d"), i);
		TX->GetWirelessTXComponent()->SetPower(power);
		TX->SetActorLabel(TXName);
		TXs.Emplace(TX);
		TXIndices.Add(TXName, i);
		DrawDebugPoint(GetWorld(), TX->GetActorLocation(), 5, FColor::Green, true, 1.f, 0U);
	}
}


void ATrainingAgent::DestroyRXs()
{
	for (auto RX : RXs) {
		RX->Destroy();
	}
	RXs.Empty();
}

void ATrainingAgent::DestroyTXs()
{
	for (auto TX : TXs) {
		TX->Destroy();
	}
	TXs.Empty();
}

void ATrainingAgent::ConfigEngine(const FTrainingAgentRecvData& Data)
{
	IConsoleVariable* ReflectionCoef = IConsoleManager::Get().FindConsoleVariable(TEXT("r.WiTracing.ReflectionCoeficient"));
	if (ReflectionCoef)
	{
		ReflectionCoef->Set(Data.reflectioncoef, EConsoleVariableFlags::ECVF_SetByCode);
	}

	IConsoleVariable* PenetrationCoef = IConsoleManager::Get().FindConsoleVariable(TEXT("r.WiTracing.PenetrationCoeficient"));
	if (PenetrationCoef)
	{
		PenetrationCoef->Set(Data.penetrationcoef, EConsoleVariableFlags::ECVF_SetByCode);
	}
	
	GEngine->AddOnScreenDebugMessage(99, 15.0f, FColor::Cyan, FString::Printf(TEXT("[WiTracing] Coefiction | Penetration: %.4f | Reflection: %.4f"), Data.penetrationcoef, Data.reflectioncoef));
}