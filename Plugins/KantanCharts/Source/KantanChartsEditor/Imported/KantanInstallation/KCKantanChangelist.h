// Copyright 2018-2020 Cameron Angus. All rights reserved.

#pragma once

#include "CoreMinimal.h"


class FXmlNode;

namespace KCKantanInstallation
{

	struct FKantanChangelistEntry
	{
		int32 Priority;
		FString Title;
		FString Text;

	protected:
		static FKantanChangelistEntry FromXmlNode(const FXmlNode& Node);

		FKantanChangelistEntry(): Priority(0)
		{}

		friend struct FKantanVersionChangelist;
	};

	typedef TSharedPtr< FKantanChangelistEntry > FKantanChangelistEntryItem;

	struct FKantanVersionChangelist
	{
		int32 VersionNumber;
		FString VersionName;
		FDateTime Timestamp;
		TArray< FKantanChangelistEntryItem > Entries;

	protected:
		static FKantanVersionChangelist FromXmlNode(const FXmlNode& Node);

		FKantanVersionChangelist(): VersionNumber(0)
		{}

		friend struct FKantanChangelist;
	};

	typedef TSharedPtr< FKantanVersionChangelist > FKantanVersionChangelistItem;

	struct FKantanChangelist
	{
		TArray< FKantanVersionChangelistItem > Versions;

		static TSharedPtr< FKantanChangelist > ReadFromXml(const FString& XmlFilename);

	protected:
		static FKantanChangelist FromXmlNode(const FXmlNode& Node);
	};

}

