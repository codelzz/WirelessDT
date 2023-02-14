//
// Ref: [How to Use WebSockets in Unreal Engine](https://youtu.be/l9TTmtDBTWY)

#pragma once

#include "CoreMinimal.h"
#include "Engine/GameInstance.h"
#include "IWebSocket.h"
#include "WebSocketGameInstance.generated.h"

/**
 * 
 */
UCLASS()
class WITRACINGSIM_API UWebSocketGameInstance : public UGameInstance
{
	GENERATED_BODY()
public:
	virtual void Init() override;
	virtual void Shutdown() override;

	TSharedPtr<IWebSocket> WebSocket;
};
