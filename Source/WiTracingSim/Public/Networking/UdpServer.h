#include "CoreMinimal.h"
#include "Sockets.h"
#include "Networking.h"
#include "Interfaces/IPv4/IPv4Endpoint.h"

class FUdpServerDelegate {
public:
	virtual void OnUdpServerDataRecv(FString RecvData, FString& RespData) = 0;
};

class FUdpServer {
public:
	FUdpServer(const FString Host, const uint16 Port);
	~FUdpServer();

	void Listen();
	void Close();

	bool Send(const TSharedRef<TArray<uint8>, ESPMode::ThreadSafe>& Data, const FIPv4Endpoint& Endpoint);
	bool Send(const FString& Data, const FIPv4Endpoint& Endpoint);
	void OnDataRecv(const FArrayReaderPtr&, const FIPv4Endpoint&);

protected:
	FIPv4Endpoint Endpoint;

	FSocket* Socket = nullptr;
	ISocketSubsystem* SocketSubSystem;

	TSharedPtr<FUdpSocketSender, ESPMode::ThreadSafe> Sender = nullptr;
	TSharedPtr<FUdpSocketReceiver, ESPMode::ThreadSafe> Receiver = nullptr;

public:
	FUdpServerDelegate* Delegate = nullptr;
};