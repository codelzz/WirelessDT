#pragma once

#include "CoreMinimal.h"
#include "Sockets.h"
#include "Networking.h"
#include "Interfaces/IPv4/IPv4Endpoint.h"

class FUdpSocketServer: public TSharedFromThis<FUdpSocketServer>
{
public:
	FUdpSocketServer(const FIPv4Endpoint& InServerEndpoint, const FIPv4Endpoint& InClientEndpoint);
	~FUdpSocketServer();

public:
	static FIPv4Endpoint StringToEndpoint(const FString& InIP, uint16 InPort);
	static FString ArrayReaderPtrToString(const FArrayReaderPtr& Reader);
	static FString BytesSharedPtrToString(const TSharedPtr<TArray<uint8>, ESPMode::ThreadSafe>& BytesPtr);
	static FString BytesToString(const TArray<uint8>& Bytes);
	static TSharedPtr<TArray<uint8>, ESPMode::ThreadSafe> StringToByteArray(const FString& String);

public:
	FString GetName() const { return TEXT("UdpSocketServer");}
	bool Listen();
	void Close();

public:
	bool Send(const TSharedRef<TArray<uint8>, ESPMode::ThreadSafe>& Data);
	void OnDataReceived(const FArrayReaderPtr&, const FIPv4Endpoint&);

protected:
	FIPv4Endpoint ClientEndpoint;
	FIPv4Endpoint ServerEndpoint;

	FSocket* Socket = nullptr;
	ISocketSubsystem* SocketSubSystem;

	TSharedPtr<FUdpSocketSender, ESPMode::ThreadSafe> Sender = nullptr;
	TSharedPtr<FUdpSocketReceiver, ESPMode::ThreadSafe> Receiver = nullptr;
};