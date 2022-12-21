#pragma once

#include "CoreMinimal.h"
#include "Networking.h"

class FUdpUtils {
public:
	static FString ArrayReaderPtrToString(const FArrayReaderPtr& Reader);
	static FString BytesSharedPtrToString(const TSharedPtr<TArray<uint8>, ESPMode::ThreadSafe>& BytesPtr);
	static FString BytesToString(const TArray<uint8>& Bytes);
	static TSharedPtr<TArray<uint8>, ESPMode::ThreadSafe> StringToBytesSharedPtr(const FString& String);;
	static FIPv4Endpoint ParseEndpoint(const FString Host, const uint16 Port);
};