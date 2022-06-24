// Copyright 2018-2020 Cameron Angus. All rights reserved.

#pragma once

#include "CoreMinimal.h"
#include "Misc/ConfigCacheIni.h"


class SWidget;

namespace KCKantanInstallation
{
	class IKantanPluginInstallation
	{
	public:
		virtual ~IKantanPluginInstallation() {}

		enum class EKantanInfoShowReason
		{
			DirectRequest,
			FirstTime,
			VersionUpdate,
		};

		struct FExtensionWidget
		{
		    DECLARE_DELEGATE_RetVal(TSharedPtr<SWidget>, FGetWidgetContent);

			FGetWidgetContent OnGetWidget;
		};

		struct FExtensionWidgetConfig
		{
		    TArray< FExtensionWidget > ExtensionCallbacks;
		};

		virtual bool DisplayEditorPopup(const FText& WindowTitle, TSharedRef< SWidget > Content) const = 0;
		virtual bool DisplayDefaultKantanEditorPopup(EKantanInfoShowReason Reason, bool bHasUserFocusedContent, FExtensionWidgetConfig = {}) const = 0;
		virtual void ShowPluginDocumentation() const = 0;

		struct FConfigCheckOp
		{
			typedef TFunction< bool(FConfigFile&, const FString& /* Section */, const FString& /* Key */) > FPerformCheck;
			typedef TFunction< void(FConfigFile&, const FString& /* Section */, const FString& /* Key */) > FPostOperation;

			enum class EDefaultCheckTypes
			{
				AlwaysPass,
				BoolValueCheckPassIfMissing,
			};

			bool PerformCheck(FConfigFile& Config, const FString& Section, const FString& Key) const
			{
				return !_PerformCheck || _PerformCheck(Config, Section, Key);
			}

			void PostOperation(FConfigFile& Config, const FString& Section, const FString& Key) const
			{
				if (_PostOperation)
				{
					_PostOperation(Config, Section, Key);
				}
			}

			static FConfigCheckOp AlwaysPass()
			{
				return FConfigCheckOp();
			}

			static FConfigCheckOp BoolCheck(const bool bPassIfMissing)
			{
				const auto Check = [=](const FConfigFile& Config, const FString& Section, const FString& Key)
				{
					bool bValue = false;
					bool bReadValue = Config.GetBool(*Section, *Key, bValue);
					return bPassIfMissing ?
						(!bReadValue || !bValue) :
						(bReadValue && !bValue);
				};

				const auto Post = [](FConfigFile& Config, const FString& Section, const FString& Key)
				{
					Config.SetString(*Section, *Key, TEXT("True"));
				};

				return FConfigCheckOp(Check, Post);
			}

			FConfigCheckOp(FPerformCheck Chk = {}, FPostOperation Post = {}) :
				_PerformCheck(Chk), _PostOperation(Post)
			{}

		private:
			FPerformCheck _PerformCheck;
			FPostOperation _PostOperation;
		};

		virtual bool ConditionallyExecuteWithConfigCheck(TFunction< bool() > Func, const FString& CheckIdentifier, FConfigCheckOp& CheckOperator) const = 0;
	};


	IKantanPluginInstallation& InitializeKantanPluginInstallation(
		const FString& PluginName,
		bool bHasUserFocusedContent, 
		IKantanPluginInstallation::FExtensionWidgetConfig = {});
	IKantanPluginInstallation& GetKantanPluginInstallation();
	void ShutdownKantanPluginInstallation();
}

