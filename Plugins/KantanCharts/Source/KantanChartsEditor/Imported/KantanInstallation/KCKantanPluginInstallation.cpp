// Copyright 2018-2020 Cameron Angus. All rights reserved.

#include "Include/KCIKantanPluginInstallation.h"
#include "KCKantanChangelist.h"
#include "Style/KCKantanInstallationStyleSet.h"
#include "InfoAsset/KCKantanInfoAsset.h"

#include "Interfaces/IMainFrameModule.h"
#include "ContentBrowserModule.h"
#include "IContentBrowserSingleton.h"
#include "AssetToolsModule.h"
#include "AssetRegistryModule.h"
#include "Interfaces/IPluginManager.h"
#include "Misc/ConfigCacheIni.h"
#include "Widgets/Layout/SBox.h"
#include "Widgets/Text/STextBlock.h"
#include "Widgets/Input/SButton.h"
#include "Widgets/Images/SImage.h"
#include "Widgets/Layout/SBorder.h"
#include "Widgets/Input/SCheckBox.h"
#include "Widgets/Text/SRichTextBlock.h"
#include "Widgets/Layout/SScrollBox.h"
#include "Widgets/Views/SListView.h"
#include "Framework/Text/SlateHyperlinkRun.h"
#include "Framework/Application/SlateApplication.h"
#include "EditorStyleSet.h"


#define LOCTEXT_NAMESPACE "KantanPluginInstallation"


namespace KCKantanInstallation
{
	class FKantanPluginInstallationImpl : public IKantanPluginInstallation
	{
	public:
		static FKantanPluginInstallationImpl& Initialize(const FString& InPluginName, bool bHasUserFocusedContent, FExtensionWidgetConfig ExtensionConfig);
		static bool IsAvailable();
		static FKantanPluginInstallationImpl& Get();
		static void Shutdown();

	public:
		FKantanPluginInstallationImpl(const FString& InPluginName);
		~FKantanPluginInstallationImpl();

	private:
		TSharedPtr< IPlugin > GetPlugin() const;

		void InternalInitialize(bool bHasUserFocusedContent, FExtensionWidgetConfig ExtensionConfig);
		void InitializeKantanInfoAssetType();
		void DeinitializeKantanInfoAssetType();

		UKCKantanInfoAssetBase* CreateInfoAsset(TSubclassOf< UKCKantanInfoAssetBase > Class, const FString& AssetName) const;
		bool SetupKantanPluginInfoAsset(FExtensionWidgetConfig) const;
		bool SetupKantanPluginDocsAsset() const;
		void RegisterPopup(bool bHasUserFocusedContent, FExtensionWidgetConfig ExtensionConfig) const;

		static TSharedRef< SWidget > CreateMessageWidget(const FText& Message, const FName& TextStyleName = NAME_None, const FMargin& Margin = FMargin(0), float LineHeight = 1.0f);
		static TSharedRef< SWidget > CreateMessageWidgetWithIcon(const FText& Message, const FName& IconName, const FVector2D& IconSize, const TSharedRef< ITextDecorator >& LinkDecorator, const FMargin& Margin = FMargin(0), const float LineHeight = 1.0f);
		static TSharedRef< SWidget > CreateIntroMessageWidget(const FText& ProductName);
		static TSharedRef< SWidget > CreateVersionUpdateMessageWidget(const FText& ProductName);
		static TSharedRef< SWidget > CreateKantanLinksWidget(const FText& ProductName, const FString& DocsUrl);
		static TSharedRef< SWidget > CreatePopupOnVersionUpdateToggle();
		static TSharedRef< SWidget > CreateChangelistWidget();
		static TSharedRef< SWidget > CreateCloseButtons(const FString& NavigationPath);

		static void OnVersionUpdatePopupEnabledChanged(ECheckBoxState NewState, TSharedRef< ECheckBoxState > PersistentState);

		/** IKantanPluginInstallation implementation */
		bool DisplayEditorPopup(const FText& WindowTitle, TSharedRef< SWidget > Content) const override;
		bool DisplayDefaultKantanEditorPopup(EKantanInfoShowReason Reason, bool bHasUserFocusedContent, FExtensionWidgetConfig = {}) const override;
		void ShowPluginDocumentation() const override;
		bool ConditionallyExecuteWithConfigCheck(TFunction< bool() > Func, const FString& CheckIdentifier, FConfigCheckOp& CheckOperator) const override;
		/**/

	private:
		const FString PluginName;
		TSharedPtr< IAssetTypeActions > KantanInfoAssetActions;

		static TUniquePtr< FKantanPluginInstallationImpl > Instance;
		static const FString AutoPopupOnVersionUpdateKey;
		static const FString LastAutoPopupVersionKey;
		static const FString PluginSpecificDevelopmentIniSuffix;
		static const FString InstallationConfigSectionName;

		FString GetPluginSpecificDevelopmentIniPath() const
		{
			const FString PluginSpecificDevelopmentIniFilename = PluginName + PluginSpecificDevelopmentIniSuffix;
			return FPaths::EngineVersionAgnosticUserDir() / TEXT("Kantan") / PluginSpecificDevelopmentIniFilename;
		}
	};

	TUniquePtr< FKantanPluginInstallationImpl > FKantanPluginInstallationImpl::Instance;
	const FString FKantanPluginInstallationImpl::AutoPopupOnVersionUpdateKey = TEXT("AutoPopupOnVersionUpdate");
	const FString FKantanPluginInstallationImpl::LastAutoPopupVersionKey = TEXT("LastAutoPopupVersion");
	const FString FKantanPluginInstallationImpl::PluginSpecificDevelopmentIniSuffix = TEXT("DevUserConfig.ini");
	const FString FKantanPluginInstallationImpl::InstallationConfigSectionName = TEXT("Installation");

	const bool bShowPopupOnVersionByDefault = true;


	FKantanPluginInstallationImpl& FKantanPluginInstallationImpl::Initialize(const FString& InPluginName, bool bHasUserFocusedContent, FExtensionWidgetConfig ExtensionConfig)
	{
		if (Instance.IsValid() == false)
		{
			Instance = MakeUnique< FKantanPluginInstallationImpl >(InPluginName);
			Instance->InternalInitialize(bHasUserFocusedContent, ExtensionConfig);
		}

		return Get();
	}

	bool FKantanPluginInstallationImpl::IsAvailable()
	{
		return Instance.IsValid();
	}

	FKantanPluginInstallationImpl& FKantanPluginInstallationImpl::Get()
	{
		check(IsAvailable());
		return *Instance;
	}

	void FKantanPluginInstallationImpl::Shutdown()
	{
		if (IsAvailable())
		{
			Instance.Reset();
		}
	}

	FKantanPluginInstallationImpl::FKantanPluginInstallationImpl(const FString& InPluginName) : PluginName(InPluginName)
	{
	}

	FKantanPluginInstallationImpl::~FKantanPluginInstallationImpl()
	{
		// @TODO: Should we delete the info assets?
		DeinitializeKantanInfoAssetType();

		FKantanInstallationStyleSet::Shutdown();
	}

	void FKantanPluginInstallationImpl::InternalInitialize(const bool bHasUserFocusedContent, FExtensionWidgetConfig ExtensionConfig)
	{
		FKantanInstallationStyleSet::Initialize(PluginName);

		InitializeKantanInfoAssetType();
		SetupKantanPluginInfoAsset(ExtensionConfig);
		SetupKantanPluginDocsAsset();

		RegisterPopup(bHasUserFocusedContent, ExtensionConfig);
	}

	void FKantanPluginInstallationImpl::InitializeKantanInfoAssetType()
	{
		check(KantanInfoAssetActions.IsValid() == false);

		auto& AssetTools = FAssetToolsModule::GetModule().Get();
		KantanInfoAssetActions = MakeShared< FAssetTypeActions_KantanInfo >();
		AssetTools.RegisterAssetTypeActions(KantanInfoAssetActions.ToSharedRef());
	}

	void FKantanPluginInstallationImpl::DeinitializeKantanInfoAssetType()
	{
		if (KantanInfoAssetActions.IsValid())
		{
			if (FModuleManager::Get().IsModuleLoaded(TEXT("AssetTools")))
			{
				auto& AssetTools = FAssetToolsModule::GetModule().Get();
				AssetTools.UnregisterAssetTypeActions(KantanInfoAssetActions.ToSharedRef());
			}
			KantanInfoAssetActions.Reset();
		}
	}

	UKCKantanInfoAssetBase* FKantanPluginInstallationImpl::CreateInfoAsset(TSubclassOf< UKCKantanInfoAssetBase > Class, const FString& AssetName) const
	{
		const auto Plugin = GetPlugin();
		if (!Plugin.IsValid() || !Plugin->CanContainContent())
		{
			return nullptr;
		}

		const FString PluginContentPath = Plugin->GetMountedAssetPath();
		const FString AssetPackageName = PluginContentPath / AssetName;

		// This is just doing some checks so that in the event the info asset got saved for some reason, we don't try to create another
		// at the same path.
		const auto AssetPackage = CreatePackage(*AssetPackageName);
		AssetPackage->FullyLoad();
		auto Asset = FindObject< UObject >(AssetPackage, *AssetName);
		if (Asset == nullptr)
		{
			Asset = NewObject< UKCKantanInfoAssetBase >(AssetPackage, Class, *AssetName, RF_Standalone | RF_Public);
			FAssetRegistryModule::AssetCreated(Asset);
		}

		return Cast< UKCKantanInfoAssetBase >(Asset);
	}

	bool FKantanPluginInstallationImpl::SetupKantanPluginInfoAsset(FExtensionWidgetConfig ExtConfig) const
	{
		const FString InfoAssetName = PluginName + TEXT("_Info");
		
		if (const auto Info = CreateInfoAsset(UKCKantanInfoAsset::StaticClass(), InfoAssetName))
		{
			Info->DefaultAction.BindLambda([ExtConfig]
			{
				if (FKantanPluginInstallationImpl::IsAvailable())
				{
					FKantanPluginInstallationImpl::Get().DisplayDefaultKantanEditorPopup(
						EKantanInfoShowReason::DirectRequest,
						false,
						ExtConfig);
				}
			});
			return true;
		}

		return false;
	}

	bool FKantanPluginInstallationImpl::SetupKantanPluginDocsAsset() const
	{
		const FString DocsAssetName = PluginName + TEXT("_Docs");

		if (const auto Docs = CreateInfoAsset(UKCKantanDocsAsset::StaticClass(), DocsAssetName))
		{
			Docs->DefaultAction.BindLambda([]
			{
				if (FKantanPluginInstallationImpl::IsAvailable())
				{
					FKantanPluginInstallationImpl::Get().ShowPluginDocumentation();
				}
			});
			return true;
		}

		return false;
	}

	void FKantanPluginInstallationImpl::RegisterPopup(bool bHasUserFocusedContent, FExtensionWidgetConfig ExtensionConfig) const
	{
		// Config check based on last plugin version for which the popup was shown.
		const auto Plugin = GetPlugin();
		if (Plugin.IsValid())
		{
			const int32 CurrentPluginVersion = Plugin->GetDescriptor().Version;
			
			TSharedRef< EKantanInfoShowReason > ShowReason = MakeShared< EKantanInfoShowReason >(EKantanInfoShowReason::DirectRequest);
			const FConfigCheckOp::FPerformCheck DoCheck = [CurrentPluginVersion, ShowReason](FConfigFile& Config, const FString& Section, const FString& Key) -> bool
			{
				bool bWantPopup = true;
				bool bReadWantPopup = Config.GetBool(*Section, *FKantanPluginInstallationImpl::AutoPopupOnVersionUpdateKey, bWantPopup);
				
				if (!bReadWantPopup)
				{
					*ShowReason = EKantanInfoShowReason::FirstTime;
					// Write default
					Config.SetString(*Section, *FKantanPluginInstallationImpl::AutoPopupOnVersionUpdateKey, bShowPopupOnVersionByDefault ? TEXT("True") : TEXT("False"));
					return true;
				}

				if (bWantPopup)
				{
					int32 LastPopupVersion = 0;
					bool bReadLastPopup = Config.GetInt(*Section, *Key, LastPopupVersion);

					if (bReadLastPopup && (CurrentPluginVersion > LastPopupVersion))
					{
						*ShowReason = EKantanInfoShowReason::VersionUpdate;
						return true;
					}
				}
				
				// Don't show popup
				return false;
			};

			const FConfigCheckOp::FPostOperation PostOp = [CurrentPluginVersion](FConfigFile& Config, const FString& Section, const FString& Key)
			{
				Config.SetInt64(*Section, *Key, CurrentPluginVersion);
			};

			auto ConfigCheck = FConfigCheckOp(DoCheck, PostOp);

			const auto ShowPopup = [ShowReason, bHasUserFocusedContent, ExtensionConfig]() -> bool
			{
				if (FKantanPluginInstallationImpl::IsAvailable())
				{
					return FKantanPluginInstallationImpl::Get().DisplayDefaultKantanEditorPopup(*ShowReason, bHasUserFocusedContent, ExtensionConfig);
				}
				return false;
			};
			ConditionallyExecuteWithConfigCheck(ShowPopup, LastAutoPopupVersionKey, ConfigCheck);
		}
	}

	TSharedPtr< IPlugin > FKantanPluginInstallationImpl::GetPlugin() const
	{
		auto& PluginManager = IPluginManager::Get();
		return PluginManager.FindPlugin(PluginName);
	}

	static void OnHyperlinkClicked(const FSlateHyperlinkRun::FMetadata& Metadata)
	{
		if (const auto URL = Metadata.Find(TEXT("href")))
		{
			FPlatformProcess::LaunchURL(**URL, nullptr, nullptr);
		}
	}

	TSharedRef< SWidget > FKantanPluginInstallationImpl::CreateMessageWidget(const FText& Message, const FName& TextStyleName, const FMargin& Margin, const float LineHeight)
	{
		const FName TextStyle = (TextStyleName.IsNone() ? TEXT("Text.Normal") : TextStyleName);
		return SNew(SRichTextBlock)
			.Text(Message)
			.AutoWrapText(true)
			.TextStyle(FKantanInstallationStyleSet::Get(), TextStyle)
			.DecoratorStyleSet(&FKantanInstallationStyleSet::Get())
			.Margin(Margin)
			.LineHeightPercentage(LineHeight)
			;
	}

	TSharedRef< SWidget > FKantanPluginInstallationImpl::CreateMessageWidgetWithIcon(const FText& Message, const FName& IconName, const FVector2D& IconSize, const TSharedRef< ITextDecorator >& LinkDecorator, const FMargin& Margin, const float LineHeight)
	{
		return SNew(SBox)
			.Padding(Margin)
			[
				SNew(SHorizontalBox)
				+ SHorizontalBox::Slot().AutoWidth().VAlign(EVerticalAlignment::VAlign_Center)
				[
					SNew(SBox)
					.WidthOverride(IconSize.X)
					.HeightOverride(IconSize.Y)
					.HAlign(EHorizontalAlignment::HAlign_Fill)
					.VAlign(EVerticalAlignment::VAlign_Fill)
					[
						SNew(SImage)
						.Image(FKantanInstallationStyleSet::Get().GetBrush(IconName))
					]
				]
				+ SHorizontalBox::Slot().AutoWidth().VAlign(EVerticalAlignment::VAlign_Center).Padding(FMargin(5, 0, 0, 0))
				[
					//CreateMessageWidget(Message, FMargin(5), LineHeight)
					SNew(SRichTextBlock)
					.Text(Message)
					.AutoWrapText(true)
					.TextStyle(FKantanInstallationStyleSet::Get(), "Text.Normal")
					.DecoratorStyleSet(&FKantanInstallationStyleSet::Get())
					.Margin(FMargin(5))
					.LineHeightPercentage(LineHeight)
					+ LinkDecorator
				]
			]				
			;
	}

	TSharedRef< SWidget > FKantanPluginInstallationImpl::CreateIntroMessageWidget(const FText& ProductName)
	{
		const auto IntroMessageText = FText::Format(
			LOCTEXT("KantanIntroMsgFmt",
				"Thank you for installing {0}.\n\n"
				"In future, this window will only show when the plugin has received updates (or not at all if you deselect the check box below).\n"
				"It can be brought up on demand via the info asset in the plugin's content folder."
			),
			ProductName
		);

		return CreateMessageWidget(IntroMessageText);
	}

	TSharedRef< SWidget > FKantanPluginInstallationImpl::CreateVersionUpdateMessageWidget(const FText& ProductName)
	{
		const auto VersionUpdateText = FText::Format(
			LOCTEXT("VersionUpdateMsgFmt",
				"The {0} plugin has received updates. Please see the changelist below."
			),
			ProductName
		);

		return CreateMessageWidget(VersionUpdateText);
	}

	TSharedRef< SWidget > FKantanPluginInstallationImpl::CreateKantanLinksWidget(const FText& ProductName, const FString& DocsUrl)
	{
		const auto DocsText = FText::Format(LOCTEXT("DocsFmt", R"({0} <a id="lnk_product_docs" style="Text.Hyperlink" href="{1}">online documentation.</>)")
			, ProductName
			, FText::FromString(DocsUrl)
			);
		const auto DocsLink = SRichTextBlock::HyperlinkDecorator(TEXT("lnk_product_docs"), FSlateHyperlinkRun::FOnClick::CreateStatic(&OnHyperlinkClicked));
		const auto DiscordText = FText::Format(LOCTEXT("DiscordFmt", R"(Please join my <a id="lnk_kantandev_discord" style="Text.Hyperlink" href="{0}">Discord server</> for support and discussion related to any of my plugins.)")
			, FText::FromString(TEXT("https://discord.gg/6BVdzAE"))
			);
		const auto DiscordLink = SRichTextBlock::HyperlinkDecorator(TEXT("lnk_kantandev_discord"), FSlateHyperlinkRun::FOnClick::CreateStatic(&OnHyperlinkClicked));
		const auto WebsiteText = FText::Format(LOCTEXT("WebsiteFmt", R"(<a id="lnk_kantandev_web" style="Text.Hyperlink" href="{0}">My website</> has info on all my plugins (both free and paid), as well as some articles on UE4 C++ coding.)")
			, FText::FromString(TEXT("https://kantandev.com/"))
			);
		const auto WebsiteLink = SRichTextBlock::HyperlinkDecorator(TEXT("lnk_kantandev_web"), FSlateHyperlinkRun::FOnClick::CreateStatic(&OnHyperlinkClicked));
		const auto TwitterText = FText::Format(LOCTEXT("TwitterFmt", R"(Follow my dev account on <a id="lnk_kantandev_twitter" style="Text.Hyperlink" href="{0}">Twitter</> for updates on what I'm working on.)")
			, FText::FromString(TEXT("https://twitter.com/kantandev"))
			);
		const auto TwitterLink = SRichTextBlock::HyperlinkDecorator(TEXT("lnk_kantandev_twitter"), FSlateHyperlinkRun::FOnClick::CreateStatic(&OnHyperlinkClicked));
		const auto MarketplaceText = FText::Format(LOCTEXT("MarketplaceFmt", R"(You can see all my UE4 Marketplace packs <a id="lnk_kantandev_marketplace" style="Text.Hyperlink" href="{0}">here</>.)")
			, FText::FromString(TEXT("https://www.unrealengine.com/marketplace/profile/Kantan%20Dev"))
			);
		const auto MarketplaceLink = SRichTextBlock::HyperlinkDecorator(TEXT("lnk_kantandev_marketplace"), FSlateHyperlinkRun::FOnClick::CreateStatic(&OnHyperlinkClicked));
	
		const FVector2D IconSize(24, 24);
		const FMargin LinkMargin(5, 0);
		return SNew(SBorder)
			.BorderImage(FEditorStyle::GetBrush("ToolPanel.GroupBorder"))
			[
				SNew(SVerticalBox)
				+ SVerticalBox::Slot().AutoHeight()
				[
					CreateMessageWidgetWithIcon(DocsText, TEXT("Icon.Documentation"), IconSize, DocsLink, LinkMargin)
				]
				+ SVerticalBox::Slot().AutoHeight()
				[
					CreateMessageWidgetWithIcon(DiscordText, TEXT("Icon.Discord"), IconSize, DiscordLink, LinkMargin)
				]
				+ SVerticalBox::Slot().AutoHeight()
				[
					CreateMessageWidgetWithIcon(WebsiteText, TEXT("Icon.Website"), IconSize, WebsiteLink, LinkMargin)
				]
				+ SVerticalBox::Slot().AutoHeight()
				[
					CreateMessageWidgetWithIcon(TwitterText, TEXT("Icon.Twitter"), IconSize, TwitterLink, LinkMargin)
				]
				+ SVerticalBox::Slot().AutoHeight()
				[
					CreateMessageWidgetWithIcon(MarketplaceText, TEXT("Icon.Marketplace"), IconSize, MarketplaceLink, LinkMargin)
				]
			]
			;
	}

	TSharedRef< SWidget > FKantanPluginInstallationImpl::CreatePopupOnVersionUpdateToggle()
	{
		// Get initial value
		const auto ConfigFilePath = FKantanPluginInstallationImpl::Get().GetPluginSpecificDevelopmentIniPath();
		FConfigFile DevUserConfig;
		DevUserConfig.Read(ConfigFilePath);
		bool bShouldPopup = bShowPopupOnVersionByDefault;
		DevUserConfig.GetBool(*InstallationConfigSectionName, *AutoPopupOnVersionUpdateKey, bShouldPopup);

		TSharedRef< ECheckBoxState > CheckState = MakeShared< ECheckBoxState >(bShouldPopup ? ECheckBoxState::Checked : ECheckBoxState::Unchecked);
		const auto GetCheckState = [CheckState]() -> ECheckBoxState
		{
			return *CheckState;
		};

		TSharedPtr< SCheckBox > CheckBox;
		TSharedRef< SWidget > Widget = SNew(SHorizontalBox)
			+ SHorizontalBox::Slot().AutoWidth().Padding(0, 0, 5, 0).VAlign(EVerticalAlignment::VAlign_Center)
			[
				CreateMessageWidget(
					LOCTEXT("EnablePopupOnVersionUpdateLabel", "Show this window with changelist whenever this plugin has been updated"),
					TEXT("Text.Small")
				)
			]
			+ SHorizontalBox::Slot().AutoWidth()
			[
				SAssignNew(CheckBox, SCheckBox)
				.IsChecked_Lambda(GetCheckState)
				.OnCheckStateChanged_Static(&FKantanPluginInstallationImpl::OnVersionUpdatePopupEnabledChanged, CheckState)
			]
		;

		return Widget;
	}

	TSharedRef< SWidget > FKantanPluginInstallationImpl::CreateChangelistWidget()
	{
		// @NOTE: For now, assuming always in file called Changelist.xml at the plugin base directory.
		const FString ChangelistFilename = TEXT("Changelist.xml");
		const FString ChangelistDirectory = FKantanPluginInstallationImpl::Get().GetPlugin()->GetBaseDir();

		const FString ChangelistFilepath = FPaths::Combine(ChangelistDirectory, ChangelistFilename);
		const auto Changelist = FKantanChangelist::ReadFromXml(ChangelistFilepath);
		if (Changelist.IsValid() == false)
		{
			return SNullWidget::NullWidget;
		}

		typedef SListView< FKantanVersionChangelistItem > FVersionListWidget;
		typedef SListView< FKantanChangelistEntryItem > FEntryListWidget;

		auto GenerateEntryRow = [](const FKantanChangelistEntryItem& EntryItem, const TSharedRef< class STableViewBase >& OwnerTable)
		{
			const FText EntryText = FText::Format(
				LOCTEXT("ChangelistEntryFmt", "<Text.Changelist.EntryTitle>{0}</>. {1}")
				, FText::FromString(EntryItem->Title)
				, FText::FromString(EntryItem->Text)
			);
			return SNew(STableRow< FKantanVersionChangelistItem >, OwnerTable)
				//.Style(&FEditorStyle::Get().GetWidgetStyle< FTableRowStyle >("TableView.Row"))
				.Padding(FMargin(4.0f, 2.0f))
				[
					SNew(SRichTextBlock)
					.Text(EntryText)
					.AutoWrapText(true)
					.TextStyle(FKantanInstallationStyleSet::Get(), "Text.Changelist.Entry")
					.DecoratorStyleSet(&FKantanInstallationStyleSet::Get())
				]
			;
		};

		auto GenerateVersionRow = [Changelist, GenerateEntryRow](const FKantanVersionChangelistItem& VersionItem, const TSharedRef< class STableViewBase >& OwnerTable)
		{
			const TSharedRef< FEntryListWidget > EntryList = SNew(FEntryListWidget)
				.ListItemsSource(&VersionItem->Entries)
				.OnGenerateRow_Lambda(GenerateEntryRow)
				.SelectionMode(ESelectionMode::None)
				.ScrollbarVisibility(EVisibility::Collapsed)
				;

			const FText VersionText = FText::Format(
				LOCTEXT("ChangelistVersionFmt", "<Text.Changelist.VersionName>{0}</> [{1}]")
				, FText::FromString(VersionItem->VersionName)
				, FText::AsDate(VersionItem->Timestamp, EDateTimeStyle::Default)
			);
			return SNew(STableRow< FKantanVersionChangelistItem >, OwnerTable)
				//.Style(&FEditorStyle::Get().GetWidgetStyle< FTableRowStyle >("TableView.Row"))
				.Padding(FMargin(4.0f, 4.0f))
				[
					SNew(SVerticalBox)
					+ SVerticalBox::Slot().AutoHeight()
					[
						SNew(SRichTextBlock)
						.Text(VersionText)
						.AutoWrapText(true)
						.TextStyle(FKantanInstallationStyleSet::Get(), "Text.Changelist.VersionInfo")
						.DecoratorStyleSet(&FKantanInstallationStyleSet::Get())
					]
					+ SVerticalBox::Slot()
					[
						EntryList
					]
				];
		};

		const TSharedRef< FVersionListWidget > VersionList = SNew(FVersionListWidget)
			.ListItemsSource(&Changelist->Versions)
			.OnGenerateRow_Lambda(GenerateVersionRow)
			.SelectionMode(ESelectionMode::None)
			;

		return SNew(SVerticalBox)
	        + SVerticalBox::Slot().AutoHeight().Padding(0, 0, 0, 4)
		    [
	            SNew(STextBlock)
				.TextStyle(FKantanInstallationStyleSet::Get(), TEXT("Text.Changelist.Title"))
				.Text(LOCTEXT("ChangelistTitle", "Changelist"))
		    ]
	        + SVerticalBox::Slot().FillHeight(1)
		    [
				SNew(SBorder)
			    .BorderImage(FEditorStyle::GetBrush("ToolPanel.GroupBorder"))
			    [
					SNew(SBox)
					.MinDesiredHeight(150)//300.0f)
					[
				        VersionList
					]
			    ]
			];
	}

	TSharedRef< SWidget > FKantanPluginInstallationImpl::CreateCloseButtons(const FString& NavigationPath)
	{
		TSharedRef< SHorizontalBox > Box = SNew(SHorizontalBox)
			+ SHorizontalBox::Slot().FillWidth(1)
			[
				SNullWidget::NullWidget
			];

		auto WkWidget = TWeakPtr< SWidget >(Box);
		auto OnClose = [WkWidget](FString NavPath)
		{
			if (WkWidget.IsValid())
			{
				const auto ContextWidget = WkWidget.Pin().ToSharedRef();
				const auto Wnd = FSlateApplication::Get().FindWidgetWindow(ContextWidget);
				if (Wnd.IsValid())
				{
					Wnd->RequestDestroyWindow();
				}
			}

			if (!NavPath.IsEmpty())
			{
				auto& ContentBrowser = FModuleManager::LoadModuleChecked< FContentBrowserModule >("ContentBrowser").Get();
				ContentBrowser.FocusPrimaryContentBrowser(false);
				ContentBrowser.ForceShowPluginContent(true);

				TArray< FString > Paths = { NavPath };
				ContentBrowser.SetSelectedPaths(Paths, true);
			}
		};

		if (!NavigationPath.IsEmpty())
		{
			Box->AddSlot().AutoWidth()
				[
					SNew(SButton)
					.ContentPadding(FMargin(4, 2))
					.OnClicked_Lambda([OnClose, NavigationPath] { OnClose(NavigationPath); return FReply::Handled(); })
					[
						SNew(STextBlock)
						.Text(FText::FromString(TEXT("Close and show plugin content")))
					]
				];
		}

		Box->AddSlot().AutoWidth()
			[
				SNew(SButton)
				.ContentPadding(FMargin(15, 2))
				.OnClicked_Lambda([OnClose] { OnClose(FString()); return FReply::Handled(); })
				[
					SNew(STextBlock)
					.Text(FText::FromString(TEXT("Close")))
				]
			];

		return Box;
	}

	void FKantanPluginInstallationImpl::OnVersionUpdatePopupEnabledChanged(ECheckBoxState NewState, TSharedRef< ECheckBoxState > PersistentState)
	{
		const bool bShouldPopup = (NewState == ECheckBoxState::Checked);
		const auto ConfigFilePath = FKantanPluginInstallationImpl::Get().GetPluginSpecificDevelopmentIniPath();
		FConfigFile DevUserConfig;
		DevUserConfig.Read(ConfigFilePath);
		DevUserConfig.SetString(*InstallationConfigSectionName, *AutoPopupOnVersionUpdateKey, bShouldPopup ? TEXT("True") : TEXT("False"));
		DevUserConfig.Write(ConfigFilePath);

		*PersistentState = NewState;
	}


	bool FKantanPluginInstallationImpl::DisplayEditorPopup(const FText& WindowTitle, TSharedRef< SWidget > Content) const
	{
		IMainFrameModule& MainFrame = FModuleManager::LoadModuleChecked< IMainFrameModule >("MainFrame");

		auto CreateIntroWindow = [WindowTitle, Content](TSharedPtr< SWindow > RootWnd, bool bNewProject)
		{
			if (bNewProject)
			{
				return;
			}

			ensure(RootWnd.IsValid());

			TSharedPtr< SWindow > Window =
				SNew(SWindow)
				.Title(WindowTitle)
				.MinWidth(400.0f)
				.MaxWidth(900.0f)
				.MinHeight(300.0f)
				.MaxHeight(700.0f)
				.SupportsMaximize(false)
				.SupportsMinimize(false)
				.SizingRule(ESizingRule::Autosized)
			    .AutoCenter(EAutoCenter::PreferredWorkArea)
				;

			Window->SetContent(Content);

			//FSlateApplication::Get().AddModalWindow(Window.ToSharedRef(), RootWnd.ToSharedRef());
			FSlateApplication::Get().AddWindowAsNativeChild(Window.ToSharedRef(), RootWnd.ToSharedRef());
		};

		if (MainFrame.IsWindowInitialized())
		{
			CreateIntroWindow(MainFrame.GetParentWindow(), false);
		}
		else
		{
			MainFrame.OnMainFrameCreationFinished().Add(IMainFrameModule::FMainFrameCreationFinishedEvent::FDelegate::CreateLambda(CreateIntroWindow));
		}

		// @TODO: Ideally, we have a callback instead, which is executed when the window is shown, or perhaps when it's closed.
		return true;
	}

	bool FKantanPluginInstallationImpl::DisplayDefaultKantanEditorPopup(
		const EKantanInfoShowReason Reason, 
		const bool bHasUserFocusedContent,
		FExtensionWidgetConfig ExtensionConfig) const
	{
		const auto Plugin = GetPlugin();
		if (Plugin.IsValid())
		{
			const auto& Descriptor = Plugin->GetDescriptor();
			const FText DisplayName = FText::FromString(Descriptor.FriendlyName);
			const FString DocsUrl = Descriptor.DocsURL;
			const FString ContentPath = Plugin->GetMountedAssetPath();
			const auto PluginIconBrush = FKantanInstallationStyleSet::Get().GetBrush(TEXT("Icon.PluginLogo"));
			const FText PluginNameAndVersion = FText::Format(LOCTEXT("NameAndVersionFmt", "{0}\n<Text.PluginHeader.Version>{1}</>")
				, DisplayName
				, FText::FromString(Descriptor.VersionName)
				);

			TSharedPtr< SWidget > TopMessage = nullptr;
			if (Reason == EKantanInfoShowReason::FirstTime)
			{
				TopMessage = CreateIntroMessageWidget(DisplayName);
			}
			else if (Reason == EKantanInfoShowReason::VersionUpdate)
			{
				TopMessage = CreateVersionUpdateMessageWidget(DisplayName);
			}
			if (TopMessage.IsValid())
			{
			    TopMessage = SNew(SBox).Padding(FMargin(0, 15, 0, 0))
			        [
						SNew(SBorder).Padding(5)
			            .BorderImage(FEditorStyle::GetBrush("ToolPanel.GroupBorder"))
			            [
				    	    TopMessage.ToSharedRef()
				        ]
				    ];
			}
			else
			{
			    TopMessage = SNullWidget::NullWidget;
			}

			TSharedRef< SVerticalBox > ExtensionWidgetsBox = SNew(SVerticalBox);
			for (auto& Ext : ExtensionConfig.ExtensionCallbacks)
			{
			    if (auto const Widget = Ext.OnGetWidget.Execute())
			    {
			        ExtensionWidgetsBox->AddSlot()
					    .AutoHeight()
					    [
							SNew(SBox).Padding(FMargin(0, 2))
							[
								Widget.ToSharedRef()
							]
					    ];
			    }
			}

			const TSharedRef< SWidget > Content = SNew(SVerticalBox)
				+ SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 0, 0, 10))
				[
					SNew(SBox)
					.Padding(FMargin(5))
					[
						SNew(SHorizontalBox)
						+ SHorizontalBox::Slot().AutoWidth().VAlign(EVerticalAlignment::VAlign_Center)
						[
							SNew(SBox)
							.Visibility(PluginIconBrush ? EVisibility::Visible : EVisibility::Collapsed)
							.Padding(FMargin(0, 0, 10, 0))
							[
								SNew(SImage)
								.Image(PluginIconBrush)
							]
						]
						+ SHorizontalBox::Slot().FillWidth(1).VAlign(EVerticalAlignment::VAlign_Fill)
						[
							SNew(SVerticalBox)
							+ SVerticalBox::Slot().AutoHeight().HAlign(EHorizontalAlignment::HAlign_Right)
							[
								SNew(SRichTextBlock)
								.Text(PluginNameAndVersion)
								.TextStyle(FKantanInstallationStyleSet::Get(), TEXT("Text.PluginHeader"))
								.DecoratorStyleSet(&FKantanInstallationStyleSet::Get())
								.Justification(ETextJustify::Right)
							]
							+ SVerticalBox::Slot().FillHeight(1).VAlign(EVerticalAlignment::VAlign_Bottom)
							[
								TopMessage.ToSharedRef()
							]
						]
					]
				]
			    + SVerticalBox::Slot().AutoHeight()
			    [
					ExtensionWidgetsBox
				]
				+ SVerticalBox::Slot().FillHeight(1)
				[
					CreateChangelistWidget()
				]
				+ SVerticalBox::Slot().AutoHeight().HAlign(EHorizontalAlignment::HAlign_Right).Padding(FMargin(0, 1, 0, 5))
				[
					CreatePopupOnVersionUpdateToggle()
				]
				+ SVerticalBox::Slot().AutoHeight()
				[
					CreateKantanLinksWidget(DisplayName, DocsUrl)
				]
				+ SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 4, 0, 0))
				[
					CreateCloseButtons((Reason == EKantanInfoShowReason::FirstTime && bHasUserFocusedContent) ? ContentPath : FString())
				]
				;

			const FText WindowTitle = DisplayName;
			if (DisplayEditorPopup(WindowTitle, Content))
			{
				const auto ConfigFilePath = GetPluginSpecificDevelopmentIniPath();
				FConfigFile DevUserConfig;
				DevUserConfig.Read(ConfigFilePath);
				DevUserConfig.SetInt64(*InstallationConfigSectionName, *LastAutoPopupVersionKey, Descriptor.Version);
				DevUserConfig.Write(ConfigFilePath);
				return true;
			}
		}

		return false;
	}

	void FKantanPluginInstallationImpl::ShowPluginDocumentation() const
	{
		const auto Plugin = GetPlugin();
		if (Plugin.IsValid())
		{
			const auto& Descriptor = Plugin->GetDescriptor();
			const FString DocsUrl = Descriptor.DocsURL;
			if (!DocsUrl.IsEmpty())
			{
				FPlatformProcess::LaunchURL(*DocsUrl, nullptr, nullptr);
			}
		}
	}

	bool FKantanPluginInstallationImpl::ConditionallyExecuteWithConfigCheck(TFunction< bool() > Func, const FString& CheckIdentifier, FConfigCheckOp& CheckOperator) const
	{
		const auto ConfigFilePath = GetPluginSpecificDevelopmentIniPath();
		FConfigFile DevUserConfig;
		DevUserConfig.Read(ConfigFilePath);

		if (CheckOperator.PerformCheck(DevUserConfig, InstallationConfigSectionName, CheckIdentifier))
		{
			if (Func())
			{
				CheckOperator.PostOperation(DevUserConfig, InstallationConfigSectionName, CheckIdentifier);

				DevUserConfig.Write(ConfigFilePath);
				return true;
			}
		}

		return false;
	}


	IKantanPluginInstallation& InitializeKantanPluginInstallation(
		const FString& PluginName, 
		const bool bHasUserFocusedContent,
		IKantanPluginInstallation::FExtensionWidgetConfig ExtensionConfig)
	{
		return FKantanPluginInstallationImpl::Initialize(PluginName, bHasUserFocusedContent, ExtensionConfig);
	}

	IKantanPluginInstallation& GetKantanPluginInstallation()
	{
		return FKantanPluginInstallationImpl::Get();
	}

	void ShutdownKantanPluginInstallation()
	{
		FKantanPluginInstallationImpl::Shutdown();
	}
}


#undef LOCTEXT_NAMESPACE



