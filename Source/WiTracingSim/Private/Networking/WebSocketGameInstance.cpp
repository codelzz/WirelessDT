// Fill out your copyright notice in the Description page of Project Settings.


#include "Networking/WebSocketGameInstance.h"
#include "WebSocketsModule.h"

void UWebSocketGameInstance::Init()
{
	Super::Init();

	if (!FModuleManager::Get().IsModuleLoaded("WebSockets"))
	{
		FModuleManager::Get().LoadModule("WebSockets");
	}

	WebSocket = FWebSocketsModule::Get().CreateWebSocket("ws://localhost:8080");

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

	// for binary
	//WebSocket->OnRawMessage().AddLambda([](const FString& Message)
	//	{
	//		GEngine->AddOnScreenDebugMessage(-1, 15.0f, FColor::Cyan, "[RECV] " + Message);
	//	});

	WebSocket->OnMessageSent().AddLambda([](const FString& Message)
		{
			// GEngine->AddOnScreenDebugMessage(-1, 15.0f, FColor::Yellow, "[SENT] " + Message);
			GEngine->AddOnScreenDebugMessage(-1, 15.0f, FColor::Yellow, "[SENT] ");
		});

	WebSocket->Connect();
}


void UWebSocketGameInstance::Shutdown()
{
	if (WebSocket->IsConnected())
	{
		WebSocket->Close();
	}
	Super::Shutdown();
}