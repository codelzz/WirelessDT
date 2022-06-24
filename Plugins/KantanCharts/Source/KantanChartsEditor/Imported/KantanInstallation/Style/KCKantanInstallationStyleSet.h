// Copyright 2018-2020 Cameron Angus. All Rights Reserved.

#pragma once

#include "Styling/SlateStyle.h"
#include "CoreMinimal.h"


namespace KCKantanInstallation
{
	class FKantanInstallationStyleSet
	{
	public:
		static void Initialize(const FString& PluginName);
		static void Shutdown();
		static void ReloadTextures();

		static const class ISlateStyle& Get();

		static FName GetStyleSetName();

	private:
		static TSharedRef< class FSlateStyleSet > Create(const FString& PluginName);

	private:
		static TSharedPtr< class FSlateStyleSet > StyleInstance;
	};
}

