// Copyright 2018-2020 Cameron Angus. All rights reserved.

#include "KCKantanChangelist.h"
#include "XmlFile.h"
#include "XmlNode.h"


#define LOCTEXT_NAMESPACE "KantanChangelist"


namespace KCKantanInstallation
{
	inline TOptional< FString > FindAttribute(const FXmlNode& Node, const FString& Attribute)
	{
		const auto AttribValue = Node.GetAttribute(Attribute);
		return AttribValue.IsEmpty() ? TOptional< FString >() : TOptional< FString >(AttribValue);
	}

	inline bool AssignAttributeAsString(const FXmlNode& Node, const FString& Attribute, FString& Dest)
	{
		if (const auto AttribValue = FindAttribute(Node, Attribute))
		{
			Dest = AttribValue.GetValue();
			return true;
		}
		return false;
	}

	inline bool AssignAttributeAsInt(const FXmlNode& Node, const FString& Attribute, int32& Dest)
	{
		if (const auto AttribValue = FindAttribute(Node, Attribute))
		{
			Dest = FCString::Atoi(*AttribValue.GetValue());
			return true;
		}
		return false;
	}

	inline bool AssignAttributeAsDateTime(const FXmlNode& Node, const FString& Attribute, FDateTime& Dest)
	{
		if (const auto AttribValue = FindAttribute(Node, Attribute))
		{
			FDateTime Result;
			if (FDateTime::ParseIso8601(*AttribValue.GetValue(), Result))
			{
				Dest = Result;
				return true;
			}
		}
		return false;
	}


	namespace Attribs
	{
		namespace Entry
		{
			const FString Priority = TEXT("priority");
			const FString Title = TEXT("title");
		}

		namespace Version
		{
			const FString Number = TEXT("number");
			const FString Name = TEXT("name");
			const FString Timestamp = TEXT("timestamp");
		}
	}

	namespace Tags
	{
		const FString Entry = TEXT("entry");
		const FString Entries = TEXT("entries");
		const FString Version = TEXT("version");
		const FString Versions = TEXT("versions");
		const FString Changelist = TEXT("changelist");
	}


	FKantanChangelistEntry FKantanChangelistEntry::FromXmlNode(const FXmlNode& Node)
	{
		check(Node.GetTag() == Tags::Entry);

		FKantanChangelistEntry Entry;
		AssignAttributeAsInt(Node, Attribs::Entry::Priority, Entry.Priority);
		AssignAttributeAsString(Node, Attribs::Entry::Title, Entry.Title);
		Entry.Text = Node.GetContent();
		return Entry;
	}

	FKantanVersionChangelist FKantanVersionChangelist::FromXmlNode(const FXmlNode& Node)
	{
		check(Node.GetTag() == Tags::Version);

		FKantanVersionChangelist Version;
		AssignAttributeAsInt(Node, Attribs::Version::Number, Version.VersionNumber);
		AssignAttributeAsString(Node, Attribs::Version::Name, Version.VersionName);
		AssignAttributeAsDateTime(Node, Attribs::Version::Timestamp, Version.Timestamp);
		
		if (const auto Entries = Node.FindChildNode(Tags::Entries))
		{
			const auto& Children = Entries->GetChildrenNodes();
			for (const auto Child : Children)
			{
				if (Child && Child->GetTag() == Tags::Entry)
				{
					Version.Entries.Add(MakeShareable(new FKantanChangelistEntry(FKantanChangelistEntry::FromXmlNode(*Child))));
				}
			}

			Algo::Sort(Version.Entries, [](const FKantanChangelistEntryItem& Item1, const FKantanChangelistEntryItem& Item2)
			{
				return Item1->Priority > Item2->Priority;
			});
		}
		return Version;
	}

	FKantanChangelist FKantanChangelist::FromXmlNode(const FXmlNode& Node)
	{
		check(Node.GetTag() == Tags::Changelist);

		FKantanChangelist Changelist;

		if (const auto Versions = Node.FindChildNode(Tags::Versions))
		{
			const auto& Children = Versions->GetChildrenNodes();
			for (const auto Child : Children)
			{
				if (Child && Child->GetTag() == Tags::Version)
				{
					Changelist.Versions.Add(MakeShareable(new FKantanVersionChangelist(FKantanVersionChangelist::FromXmlNode(*Child))));
				}
			}

			Algo::Sort(Changelist.Versions, [](const FKantanVersionChangelistItem& Item1, const FKantanVersionChangelistItem& Item2)
			{
				return Item1->VersionNumber > Item2->VersionNumber;
			});
		}
		return Changelist;
	}

	TSharedPtr< FKantanChangelist > FKantanChangelist::ReadFromXml(const FString& XmlFilename)
	{
		FXmlFile XmlFile(XmlFilename);
		if (XmlFile.IsValid())
		{
			const auto Root = XmlFile.GetRootNode();
			if (Root && Root->GetTag() == Tags::Changelist)
			{
				return MakeShareable(new FKantanChangelist(FKantanChangelist::FromXmlNode(*Root)));
			}
		}

		return nullptr;
	}
}


#undef LOCTEXT_NAMESPACE



