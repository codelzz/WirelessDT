// Copyright Epic Games, Inc. All Rights Reserved.

#include "WiTracingSimGameMode.h"
#include "WiTracingSimCharacter.h"
#include "UObject/ConstructorHelpers.h"

AWiTracingSimGameMode::AWiTracingSimGameMode()
{
	// set default pawn class to our Blueprinted character
	static ConstructorHelpers::FClassFinder<APawn> PlayerPawnBPClass(TEXT("/Game/Characters/BP_ThirdPersonCharacter"));
	if (PlayerPawnBPClass.Class != NULL)
	{
		DefaultPawnClass = PlayerPawnBPClass.Class;
	}
}
