// Copyright 2018-2020 Cameron Angus. All rights reserved.

#include "KCKantanInfoAsset.h"


namespace KCKantanInstallation
{
	UClass* FAssetTypeActions_KantanInfo::GetSupportedClass() const
	{
		return UKCKantanInfoAssetBase::StaticClass();
	}

	uint32 FAssetTypeActions_KantanInfo::GetCategories()
	{
		return 0;
	}

	void FAssetTypeActions_KantanInfo::GetActions(const TArray<UObject*>& InObjects, FMenuBuilder& MenuBuilder)
	{

	}

	void FAssetTypeActions_KantanInfo::OpenAssetEditor(const TArray<UObject*>& InObjects, TSharedPtr<IToolkitHost> EditWithinLevelEditor)
	{
		for (auto ObjIt = InObjects.CreateConstIterator(); ObjIt; ++ObjIt)
		{
			if (const auto Asset = Cast< UKCKantanInfoAssetBase >(*ObjIt))
			{
				Asset->PerformAssetAction();
			}
		}
	}
}

