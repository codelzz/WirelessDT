// Copyright 2018-2020 Cameron Angus. All rights reserved.

#include "KCKantanInstallationStyleSet.h"
#include "Styling/SlateStyle.h"
#include "Framework/Application/SlateApplication.h"
#include "Styling/SlateStyleRegistry.h"
#include "EditorStyleSet.h"
#include "Interfaces/IPluginManager.h"


namespace KCKantanInstallation
{
	TSharedPtr< FSlateStyleSet > FKantanInstallationStyleSet::StyleInstance = nullptr;


	void FKantanInstallationStyleSet::Initialize(const FString& PluginName)
	{
		if (!StyleInstance.IsValid())
		{
			StyleInstance = Create(PluginName);
			FSlateStyleRegistry::RegisterSlateStyle(*StyleInstance);
		}
	}

	void FKantanInstallationStyleSet::Shutdown()
	{
		FSlateStyleRegistry::UnRegisterSlateStyle(*StyleInstance);
		ensure(StyleInstance.IsUnique());
		StyleInstance.Reset();
	}

	FName FKantanInstallationStyleSet::GetStyleSetName()
	{
		// @NOTE: Intentionally separated out so that the style name will be prefixed when this code is imported.
		static FName StyleSetName(TEXT("KCKantanInstallation" "StyleSet"));
		return StyleSetName;
	}


#define IMAGE_BRUSH( RelativePath, ... ) FSlateImageBrush( Style->RootToContentDir( RelativePath, TEXT(".png") ), __VA_ARGS__ )
#define BOX_BRUSH( RelativePath, ... ) FSlateBoxBrush( Style->RootToContentDir( RelativePath, TEXT(".png") ), __VA_ARGS__ )
#define BORDER_BRUSH( RelativePath, ... ) FSlateBorderBrush( Style->RootToContentDir( RelativePath, TEXT(".png") ), __VA_ARGS__ )
#define COLOR_BRUSH( ... ) FSlateColorBrush( __VA_ARGS__)
#define TTF_FONT( RelativePath, ... ) FSlateFontInfo( Style->RootToContentDir( RelativePath, TEXT(".ttf") ), __VA_ARGS__ )
#define OTF_FONT( RelativePath, ... ) FSlateFontInfo( Style->RootToContentDir( RelativePath, TEXT(".otf") ), __VA_ARGS__ )

#define ENGINE_TTF_FONT( RelativePath, ... ) FSlateFontInfo( (FPaths::EngineContentDir() / RelativePath) + TEXT(".ttf") , __VA_ARGS__ )

	const FVector2D Icon14x14(14.0f, 14.0f);
	const FVector2D Icon16x16(16.0f, 16.0f);
	const FVector2D Icon18x18(18.0f, 18.0f);
	const FVector2D Icon20x20(20.0f, 20.0f);
	const FVector2D Icon32x32(32.0f, 32.0f);
	const FVector2D Icon40x40(40.0f, 40.0f);
	const FVector2D Icon64x64(64.0f, 64.0f);
	const FVector2D Icon128x128(128.0f, 128.0f);


	TSharedRef< FSlateStyleSet > FKantanInstallationStyleSet::Create(const FString& PluginName)
	{
		auto Plugin = IPluginManager::Get().FindPlugin(PluginName);
		check(Plugin.IsValid());
		TSharedRef<FSlateStyleSet> Style = MakeShareable(new FSlateStyleSet(GetStyleSetName()));
		Style->SetContentRoot(Plugin->GetBaseDir() / TEXT("Resources") / TEXT("KInstallation"));
		Style->SetCoreContentRoot(Plugin->GetBaseDir() / TEXT("Resources") / TEXT("KInstallation"));

		const auto NormalFont = ENGINE_TTF_FONT("Slate/Fonts/Roboto-Regular", 9);
		const auto SmallFont = ENGINE_TTF_FONT("Slate/Fonts/Roboto-Regular", 8);

		const auto ChangelistTitleFont = ENGINE_TTF_FONT("Slate/Fonts/Roboto-Italic", 11);
		const auto ChangelistVersionNameFont = ENGINE_TTF_FONT("Slate/Fonts/Roboto-Bold", 14);
		const auto ChangelistVersionInfoFont = ENGINE_TTF_FONT("Slate/Fonts/Roboto-Regular", 12);
		const auto ChangelistEntryTitleFont = ENGINE_TTF_FONT("Slate/Fonts/Roboto-Bold", 9);
		const auto ChangelistEntryFont = ENGINE_TTF_FONT("Slate/Fonts/Roboto-Regular", 8);

		const auto PluginHeaderFont = TTF_FONT("Fonts/SourceCodePro-Regular", 24);
		const auto PluginHeaderVersionFont = TTF_FONT("Fonts/SourceCodePro-Regular", 14);

		Style->Set("Font.PluginHeader", PluginHeaderFont);
		Style->Set("Font.PluginHeader.Version", PluginHeaderVersionFont);

		const auto NormalText = FTextBlockStyle()
			.SetFont(NormalFont)
			.SetColorAndOpacity(FSlateColor::UseForeground())
			;
		Style->Set("Text.Normal", NormalText);

		const auto SmallText = FTextBlockStyle()
			.SetFont(SmallFont)
			.SetColorAndOpacity(FSlateColor::UseForeground())
			;
		Style->Set("Text.Small", SmallText);

		Style->Set("Text.PluginHeader", FTextBlockStyle()
			.SetFont(PluginHeaderFont)
			.SetColorAndOpacity(FSlateColor::UseForeground())
		);
		Style->Set("Text.PluginHeader.Version", FTextBlockStyle()
			.SetFont(PluginHeaderVersionFont)
			.SetColorAndOpacity(FSlateColor::UseForeground())
		);

		const auto HyperlinkText = FTextBlockStyle(NormalText)
			.SetColorAndOpacity(FLinearColor(0.7f, 0.7f, 0.0f))
			;
		FHyperlinkStyle Hyperlink = FHyperlinkStyle(FCoreStyle::Get().GetWidgetStyle< FHyperlinkStyle >(TEXT("Hyperlink")))
			.SetTextStyle(HyperlinkText)
			;
		Style->Set("Text.Hyperlink", Hyperlink);

		Style->Set("Text.Changelist.Title", FTextBlockStyle()
			.SetFont(ChangelistTitleFont)
			.SetColorAndOpacity(FSlateColor::UseForeground())
		);
		Style->Set("Text.Changelist.VersionName", FTextBlockStyle()
			.SetFont(ChangelistVersionNameFont)
			.SetColorAndOpacity(FSlateColor::UseForeground())
		);
		Style->Set("Text.Changelist.VersionInfo", FTextBlockStyle()
			.SetFont(ChangelistVersionInfoFont)
			.SetColorAndOpacity(FSlateColor::UseForeground())
		);
		Style->Set("Text.Changelist.EntryTitle", FTextBlockStyle()
			.SetFont(ChangelistEntryTitleFont)
			.SetColorAndOpacity(FSlateColor::UseForeground())
		);
		Style->Set("Text.Changelist.Entry", FTextBlockStyle()
			.SetFont(ChangelistEntryFont)
			.SetColorAndOpacity(FSlateColor::UseForeground())
		);

		const FLinearColor DocsTint = FLinearColor(0.7f, 0.7f, 0.3f);
		const FLinearColor InfoTint = FLinearColor(0.5f, 0.15f, 0.1f);
		const FLinearColor WebTint = FLinearColor(0.5f, 0.8f, 0.5f);

		// Icons
		Style->Set("Icon.PluginLogo", new IMAGE_BRUSH("../Icon128", Icon128x128));
		Style->Set("Icon.Documentation", new IMAGE_BRUSH("Icons/docs", FVector2D(100, 100), DocsTint));
		Style->Set("Icon.Support", new IMAGE_BRUSH("Icons/support", Icon128x128));
		Style->Set("Icon.Discord", new IMAGE_BRUSH("Icons/discord_32x", Icon32x32)); //FVector2D(245, 240)));
		Style->Set("Icon.Website", new IMAGE_BRUSH("Icons/website", Icon128x128, WebTint));
		Style->Set("Icon.Twitter", new IMAGE_BRUSH("Icons/twitter", FVector2D(183, 183)));
		Style->Set("Icon.Marketplace", new IMAGE_BRUSH("Icons/ue4", Icon128x128));

		Style->Set("ClassThumbnail.KCKantanInfoAsset", new IMAGE_BRUSH("Icons/info", Icon64x64, InfoTint));
		Style->Set("ClassThumbnail.KCKantanDocsAsset", new IMAGE_BRUSH("Icons/docs", Icon64x64, DocsTint));

		return Style;
	}

#undef IMAGE_BRUSH
#undef BOX_BRUSH
#undef BORDER_BRUSH
#undef TTF_FONT
#undef OTF_FONT


	void FKantanInstallationStyleSet::ReloadTextures()
	{
		FSlateApplication::Get().GetRenderer()->ReloadTextureResources();
	}

	const ISlateStyle& FKantanInstallationStyleSet::Get()
	{
		// @TODO: Temp workaround for monolithic build initialization order issue.
		// UObject CDOs are getting constructed before a module that their own module has a 
		// dependency on is initialized... 
		if (!StyleInstance.IsValid())
		{
			//Initialize();
			// Can't do in this case since need plugin name context...
			check(false);
		}
		//

		return *StyleInstance;
	}
}

