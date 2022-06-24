// Copyright (C) 2015-2018 Cameron Angus. All Rights Reserved.

#pragma once

#include "UObject/ObjectMacros.h"
#include "CoreMinimal.h"
#include "KantanCategoryTypes.generated.h"


UENUM(BlueprintType)
enum class EKantanBarChartOrientation: uint8 {
	Vertical,
	Horizontal,
};

UENUM(BlueprintType)
enum class EKantanBarLabelPosition: uint8 {
	NoLabels,
	Standard,
	Overlaid,
};

UENUM(BlueprintType)
enum class EKantanBarValueExtents: uint8 {
	NoValueLines,
	ZeroLineOnly,
	ZeroAndMaxLines,
};
