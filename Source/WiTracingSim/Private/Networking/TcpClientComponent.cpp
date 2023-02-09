//
// Reference: https://github.com/How2Compute/Socketer/blob/master/Source/Socketer/Private/SocketerBPLibrary.cpp

#include "Networking/TcpClientComponent.h"
#include "UdpUtils.h"

//--------------------------------- UdpClient --------------------------------------------------

FTcpClient::FTcpClient()
{
	//Socket = FTcpSocketBuilder(FString("TcpSocket")).AsReusable();
	//{
	//	Sender = MakeShared<FUdpSocketSender, ESPMode::ThreadSafe>(Socket, TEXT("TcpSocketSender"));
	//}
}

FTcpClient::~FTcpClient()
{
	Close();
	SetSocket(nullptr);
}

bool FTcpClient::Connect(const FString Host, const uint16 Port)
{
	// Create socket pointer
	FSocket* Socket = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->CreateSocket(EName::Stream, TEXT("TcpSocket"), false);

	// Attemp to parse host name and IP
	FAddressInfoResult AddressInfo = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->GetAddressInfo(*Host, nullptr, EAddressInfoFlags::Default, EName::None);
	if (AddressInfo.ReturnCode != ESocketErrors::SE_NO_ERROR || AddressInfo.Results.Num() < 1)
	{
		UE_LOG(LogTemp, Warning, TEXT("Unable to resolve host \"%s\"!"), *Host);
		return false;
	}

	// Grab the first result and use the passed in port
	TSharedRef<FInternetAddr> SockAddress = AddressInfo.Results[0].Address;
	SockAddress->SetPort(Port);

	// Attempt to connect
	bool Connected = Socket->Connect(*SockAddress);

	// Verify it is connected
	if (!Connected)
	{
		UE_LOG(LogTemp, Warning, TEXT("Unable to connect to TCP \"%s\":\"%d\"!"), *Host, Port);
		return false;
	}

	SetSocket(Socket);
	return true;
}

bool FTcpClient::Send(const FString& Data) 
{
	FSocket* Socket = GetSocket();
	// return if socket invalid
	if (Socket == nullptr) {
		UE_LOG(LogTemp, Warning, TEXT("Invalid Tcp Socket!"));
		return false;
	}

	// Serialize data
	const TCHAR* serializedData = Data.GetCharArray().GetData();

	// Prepare parameters
	int32 Size = FCString::Strlen(serializedData);
	int32 Sent = 0;

	// Send
	bool OK = Socket->Send((uint8*)TCHAR_TO_UTF8(serializedData), Size, Sent);

	// Check if it is successful
	if (!OK) {
		UE_LOG(LogTemp, Warning, TEXT("Error while sending message: \"%s\""), *Data);
		return false;
	}

	return true;
}

bool FTcpClient::Close()
{
	FSocket* Socket = GetSocket();
	if (Socket == nullptr)
	{
		return false;
	}

	return Socket->Close();
}


//--------------------------------- UUdpClientComponent -----------------------------------------------
UTcpClientComponent::UTcpClientComponent(const FObjectInitializer& ObjectInitializer)
	: Super(ObjectInitializer)
{
	TcpClient = MakeShared<FTcpClient>();
}

void UTcpClientComponent::BeginPlay()
{
	Super::BeginPlay();
}

void UTcpClientComponent::EndPlay(EEndPlayReason::Type EndPlayReason)
{
	switch (EndPlayReason)
	{
	case EEndPlayReason::Destroyed:
	{
		Close();
		break;
	}
	}
}

void UTcpClientComponent::Connect(const FString Host, const int64 Port)
{
	TcpClient->Connect(Host, (uint16)Port);
}

void UTcpClientComponent::Send(const FString& Data)
{
	if (IsValid())
	{
		TcpClient->Send(Data);
	}
}

bool UTcpClientComponent::IsValid()
{
	return TcpClient != nullptr && TcpClient.IsValid();
}

void UTcpClientComponent::Close()
{
	if (IsValid())
	{
		TcpClient->Close();
	}
}