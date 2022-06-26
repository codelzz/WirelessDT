#pragma once

#include "RHI.h"
#include "ScenePrivate.h"
#include "ScreenPass.h"
#include "WiTracing/WiTracingSignalCommon.h"

#if RHI_RAYTRACING

#include "GlobalShader.h"
#include "WiTracingTypes.h"

TAutoConsoleVariable<int32> CVarWiTracingSamplesPerPixel(
	TEXT("r.WiTracing.SamplesPerPixel"),
	2,
	TEXT("Sets the maximum number of samples per pixel."),
	ECVF_RenderThreadSafe
);

TAutoConsoleVariable<int32> CVarWiTracingMaxBounces(
	TEXT("r.WiTracing.MaxBounces"),
	5,
	TEXT("Sets the maximum number of wi tracing bounces."),
	ECVF_RenderThreadSafe
);

TAutoConsoleVariable<int32> CVarWiTracingSpectrum(
	TEXT("r.WiTracing.Spectrum"),
	1,
	TEXT("Enables spectrum to visualize radiance (default: 1 (enabled))."),
	ECVF_RenderThreadSafe
);

TAutoConsoleVariable<float> CVarWiTracingWavelength(
	TEXT("r.WiTracing.Wavelength"),
	1,
	TEXT("Sets the wavelength of wi tracing transmitter (default: 1 (meter))"),
	ECVF_RenderThreadSafe
);

TAutoConsoleVariable<float> CVarWiTracingPowerAt1Meter(
	TEXT("r.WiTracing.PowerAt1Meter"),
	0,
	TEXT("Sets the PowerAt1Meter of wi tracing transmitter (default: 1 (watt))"),
	ECVF_RenderThreadSafe
);

TAutoConsoleVariable<float> CVarWiTracingRayGenMode(
	TEXT("r.WiTracing.RayGenMode"),
	0,
	TEXT("Sets the ray generation projection mode (default: 0 (octahedral 360) | 1 (normal))"),
	ECVF_RenderThreadSafe
);

#define MAX_PATH_INTENSITY 1

BEGIN_SHADER_PARAMETER_STRUCT(FWiTracingData, )
SHADER_PARAMETER(uint32, MaxBounces)
SHADER_PARAMETER(uint32, MaxSamples)
SHADER_PARAMETER(float, Wavelength)
SHADER_PARAMETER(float, MaxPathIntensity)		// not sure if necessary
SHADER_PARAMETER(uint32, ApproximateCaustics)
SHADER_PARAMETER(float, BlendFactor)
SHADER_PARAMETER(uint32, TemporalSeed)
SHADER_PARAMETER(uint32, RayGenMode)
SHADER_PARAMETER(FVector3f, RayRotation)
END_SHADER_PARAMETER_STRUCT()

struct FWiTracingConfig
{
	FWiTracingData WiTracingData;
	FIntRect ViewRect;

	bool IsDifferent(const FWiTracingConfig& Other) const
	{
		// In current implementation the WiTracingConfig will be released on each render
		// hence, this function migth never be execured.
		return
			WiTracingData.MaxBounces != Other.WiTracingData.MaxBounces ||
			WiTracingData.MaxSamples != Other.WiTracingData.MaxSamples ||
			WiTracingData.Wavelength != Other.WiTracingData.Wavelength ||
			WiTracingData.ApproximateCaustics != Other.WiTracingData.ApproximateCaustics ||
			WiTracingData.MaxPathIntensity != Other.WiTracingData.MaxPathIntensity ||
			ViewRect != Other.ViewRect ;
	}

	void Init()
	{
		const int32 MaxBounces = CVarWiTracingMaxBounces.GetValueOnRenderThread();
		const int32 MaxSamples = CVarWiTracingSamplesPerPixel.GetValueOnRenderThread();
		const float Wavelength = CVarWiTracingWavelength.GetValueOnRenderThread();
		const int32 RayGenMode = CVarWiTracingRayGenMode.GetValueOnRenderThread();

		WiTracingData.MaxBounces = FMath::Max(MaxBounces, 1);
		WiTracingData.MaxSamples = FMath::Max(MaxSamples, 1);
		WiTracingData.Wavelength = Wavelength > 0 ? Wavelength : 1.0f;
		WiTracingData.RayGenMode = FMath::Clamp(RayGenMode, 0, 1);

		WiTracingData.MaxPathIntensity = MAX_PATH_INTENSITY;
		WiTracingData.ApproximateCaustics = true;
		WiTracingData.BlendFactor = 1.0f / WiTracingData.MaxSamples;

		// [?] Not sure if FDateTime::Now().GetMillisecond() is expensive
		WiTracingData.TemporalSeed = FDateTime::Now().GetMillisecond();

		WiTracingData.RayRotation = FVector3f(1, 0, 0);
	}
};

class FWiTracingRG : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FWiTracingRG)
	SHADER_USE_ROOT_PARAMETER_STRUCT(FWiTracingRG, FGlobalShader)

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return ShouldCompileRayTracingShadersForProject(Parameters.Platform);
	}

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_STRUCT_INCLUDE(FWiTracingData, WiTracingData)
		SHADER_PARAMETER_SRV(RaytracingAccelerationStructure, TLAS)
		SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, ViewUniformBuffer)
		// scene lights
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<FWiTracingTransmitter>, SceneTransmitters)
		SHADER_PARAMETER(uint32, SceneTransmitterCount)
		// output
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float3>, RWRadianceTexture)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float3>, RWPhaseTexture)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<uint32>, RWRadiancePdf)
	END_SHADER_PARAMETER_STRUCT()
};

class FWiTracingParallelAccumulatorCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FWiTracingParallelAccumulatorCS)
	SHADER_USE_PARAMETER_STRUCT(FWiTracingParallelAccumulatorCS, FGlobalShader)

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return ShouldCompileRayTracingShadersForProject(Parameters.Platform);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		//OutEnvironment.CompilerFlags.Add(CFLAG_WarningsAsErrors);
		OutEnvironment.CompilerFlags.Add(CFLAG_AllowTypedUAVLoads);
		OutEnvironment.SetDefine(TEXT("THREADGROUPSIZE_X"), FComputeShaderUtils::kGolden2DGroupSize);
		OutEnvironment.SetDefine(TEXT("THREADGROUPSIZE_Y"), FComputeShaderUtils::kGolden2DGroupSize);
	}

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_RDG_TEXTURE_SRV(Texture2D<float3>, RadianceTexture)
		SHADER_PARAMETER_RDG_TEXTURE_SRV(Texture2D<float3>, PhaseTexture)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<uint32>, RadiancePdf)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float3>, RWParallelAccumulatedRadianceTexture)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float3>, RWParallelAccumulatedPhaseTexture)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<uint32>, RWParallelAccumulatedRadiancePdf)
		END_SHADER_PARAMETER_STRUCT()
};

class FWiTracingAccumulatorCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FWiTracingAccumulatorCS)
	SHADER_USE_PARAMETER_STRUCT(FWiTracingAccumulatorCS, FGlobalShader)

		static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return ShouldCompileRayTracingShadersForProject(Parameters.Platform);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		//OutEnvironment.CompilerFlags.Add(CFLAG_WarningsAsErrors);
		OutEnvironment.CompilerFlags.Add(CFLAG_AllowTypedUAVLoads);
		OutEnvironment.SetDefine(TEXT("THREADGROUPSIZE_X"), FComputeShaderUtils::kGolden2DGroupSize);
		OutEnvironment.SetDefine(TEXT("THREADGROUPSIZE_Y"), FComputeShaderUtils::kGolden2DGroupSize);
	}
	
	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_RDG_TEXTURE_SRV(Texture2D<float3>, RadianceTexture)
		SHADER_PARAMETER_RDG_TEXTURE_SRV(Texture2D<float3>, PhaseTexture)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, RadiancePdf)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<float>, RWRadiance)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<float>, RWPhase)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<uint32>, RWAccumulatedRadiancePdf)
		END_SHADER_PARAMETER_STRUCT()
};


class FWiTracingCompositorPS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FWiTracingCompositorPS)
	SHADER_USE_PARAMETER_STRUCT(FWiTracingCompositorPS, FGlobalShader)

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return ShouldCompileRayTracingShadersForProject(Parameters.Platform);
	}

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_RDG_TEXTURE_SRV(Texture2D<float3>, RadianceTexture)
		SHADER_PARAMETER_RDG_TEXTURE_SRV(Texture2D<float3>, PhaseTexture)
		SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, ViewUniformBuffer)
		SHADER_PARAMETER(int32, SpectrumEnabled)
		RENDER_TARGET_BINDING_SLOTS()
	END_SHADER_PARAMETER_STRUCT()
};

void AddWiTracingPass(
	FRDGBuilder& GraphBuilder,
	const FViewInfo& View,
	FRDGTexture* RadianceTexture,
	FRDGTexture* PhaseTexture,
	FRDGBuffer* RadiancePdfBuffer,
	const FWiTracingConfig& Config,
	FWiTracingScene* WiTracingScene,
	FWiTracingTransmitter* Transmitters,
	uint32 NumTransmitters);

void AddWiTracingParallelAccumulatePass(
	FRDGBuilder& GraphBuilder,
	const FViewInfo& View,
	FRDGTexture* RadianceTexture,
	FRDGTexture* PhaseTexture,
	FRDGBuffer* RadiancePdf,
	FRDGTexture* ParallelAccumulatedRadianceTexture,
	FRDGTexture* ParallelAccumulatedPhaseTexture,
	FRDGBuffer* ParallelAccumulatedRadiancePdf,
	FIntPoint Resolution);

void AddWiTracingAccumulatePass(
	FRDGBuilder& GraphBuilder,
	FRDGTexture* RadianceTexture,
	FRDGTexture* PhaseTexture,
	FRDGBuffer* RadiancePdf,
	FRDGBuffer* Radiance,
	FRDGBuffer* Phase,
	FRDGBuffer* AccumulatedRadiancePdf,
	FWiTracingScene* WiTracingScene);

void AddCompositePass(
	FRDGBuilder& GraphBuilder,
	const FViewInfo& View,
	FRDGTexture* RadianceTexture,
	FRDGTextureRef SceneColorOutputTexture);

#endif // RHI_RAYTRACING