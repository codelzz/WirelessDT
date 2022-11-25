#include "UdpUtils.h"

FString FUdpUtils::ArrayReaderPtrToString(const FArrayReaderPtr& Reader)
{
	return FString(UTF8_TO_TCHAR(reinterpret_cast<const char*>(Reader->GetData())));
}

FString FUdpUtils::BytesSharedPtrToString(const TSharedPtr<TArray<uint8>, ESPMode::ThreadSafe>& BytesPtr)
{
	if (BytesPtr.IsValid())
	{
		return FString(UTF8_TO_TCHAR(reinterpret_cast<const char*>(BytesPtr->GetData())));
	}
	return TEXT("");
}

FString FUdpUtils::BytesToString(const TArray<uint8>& Bytes)
{
	return FString(UTF8_TO_TCHAR(reinterpret_cast<const char*>(Bytes.GetData())));
}

TSharedPtr<TArray<uint8>, ESPMode::ThreadSafe> FUdpUtils::StringToBytesSharedPtr(const FString& String)
{
	TSharedPtr<TArray<uint8>, ESPMode::ThreadSafe> Bytes = MakeShared<TArray<uint8>, ESPMode::ThreadSafe>();
	if (Bytes.IsValid())
	{
		Bytes->SetNum(String.Len());
		memcpy(Bytes->GetData(), TCHAR_TO_UTF8(*String), String.Len());
	}
	return Bytes;
}