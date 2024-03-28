"""
This script defines the data schema for the Field Image Repository using dataclasses validated using Pydantic.
It includes definitions for storing and manipulating camera and table metadata for the Field image repository.

The schema is composed of two main components:
1. CameraInfo: This dataclass captures select camera exif metadata, such as make, model, exposure settings, and more. This information is crucial for understanding the conditions under which each field image was captured, allowing for better analysis and processing of the image data.
2. FieldMetadata: Represents metadata specific to the field where the image was captured, including details such as plant type, ground cover, and environmental conditions at the time of capture. It also includes a reference to the associated CameraInfo to link each image with its capture conditions.

Together, these dataclasses provide a structured and type-safe way to work with field image metadata in the repository, facilitating easy access, validation, and manipulation of this information.
"""

from pydantic.dataclasses import dataclass


@dataclass
class CameraInfo:
    """A dataclass representing camera information extracted from Field camera exif metadata."""

    Make: str
    Model: str
    Orientation: str
    DateTime: str
    ExifOffset: int
    ExposureTime: str
    FNumber: int | str
    ISOSpeedRatings: int
    ExifVersion: str
    WhiteBalance: str
    ImageWidth: int
    ImageLength: int
    Flash: str
    FocalLength: int | str
    XResolution: int = None
    YResolution: int = None
    ResolutionUnit: str = None
    Software: str = None
    YCbCrPositioning: str = None
    ExposureProgram: str = None
    SensitivityType: str = None
    RecommendedExposureIndex: int = None
    OffsetTime: str = None
    ComponentsConfiguration: str = None
    CompressedBitsPerPixel: int = None
    BrightnessValue: str = None
    ExposureBiasValue: int = None
    MaxApertureValue: str = None
    MeteringMode: str = None
    FlashPixVersion: str = None
    ColorSpace: str = None
    LightSource: str = None
    ExposureMode: str = None
    DigitalZoomRatio: int = None
    FocalLengthIn35mmFilm: int = None
    SceneCaptureType: str = None
    Contrast: str = None
    Saturation: str = None
    Sharpness: str = None
    LensSpecification: str = None
    LensModel: str = None

    @classmethod
    def from_dict(cls, data: dict) -> "CameraInfo":
        # Preprocess and map the dictionary keys to match the dataclass attributes
        mapping = {
            "Image Make": "Make",
            "Image Model": "Model",
            "Image Orientation": "Orientation",
            "Image XResolution": "XResolution",
            "Image YResolution": "YResolution",
            "Image ResolutionUnit": "ResolutionUnit",
            "Image Software": "Software",
            "Image DateTime": "DateTime",
            "Image YCbCrPositioning": "YCbCrPositioning",
            "Image ExifOffset": "ExifOffset",
            "EXIF ExposureTime": "ExposureTime",
            "EXIF FNumber": "FNumber",
            "EXIF ExposureProgram": "ExposureProgram",
            "EXIF ISOSpeedRatings": "ISOSpeedRatings",
            "EXIF ExifVersion": "ExifVersion",
            "EXIF ComponentsConfiguration": "ComponentsConfiguration",
            "EXIF BrightnessValue": "BrightnessValue",
            "EXIF ExposureBiasValue": "ExposureBiasValue",
            "EXIF MaxApertureValue": "MaxApertureValue",
            "EXIF MeteringMode": "MeteringMode",
            "EXIF Flash": "Flash",
            "EXIF FocalLength": "FocalLength",
            "EXIF FlashPixVersion": "FlashPixVersion",
            "EXIF ColorSpace": "ColorSpace",
            "EXIF ExifImageWidth": "ImageWidth",
            "EXIF ExifImageLength": "ImageLength",
            "EXIF ExposureMode": "ExposureMode",
            "EXIF WhiteBalance": "WhiteBalance",
            "EXIF DigitalZoomRatio": "DigitalZoomRatio",
            "EXIF FocalLengthIn35mmFilm": "FocalLengthIn35mmFilm",
            "EXIF SensitivityType": "SensitivityType",
            "EXIF RecommendedExposureIndex": "RecommendedExposureIndex",
            "EXIF OffsetTime": "OffsetTime",
            "EXIF CompressedBitsPerPixel": "CompressedBitsPerPixel",
            "EXIF LightSource": "LightSource",
            "EXIF SceneCaptureType": "SceneCaptureType",
            "EXIF Contrast": "Contrast",
            "EXIF Saturation": "Saturation",
            "EXIF Sharpness": "Sharpness",
            "EXIF LensSpecification": "LensSpecification",
            "EXIF LensModel": "LensModel",
        }

        # Convert string values to the appropriate type as needed
        def convert_value(key, value):
            if key in [
                "XResolution",
                "YResolution",
                "ExifOffset",
                "ISOSpeedRatings",
                "ExposureBiasValue",
                "ExifImageWidth",
                "ExifImageLength",
                "DigitalZoomRatio",
            ]:
                return int(float(value))
            return value

        # Create a new dictionary with the mapped keys and converted values
        kwargs = {
            mapping.get(k, k): convert_value(mapping.get(k, k), v)
            for k, v in data.items()
            if mapping.get(k, k) in cls.__annotations__
        }

        # Create and return a new CameraInfo instance
        return cls(**kwargs)


@dataclass
class FieldMetadata:
    """A dataclass representing metadata for a Field images."""

    Name: str
    SizeMiB: float
    UploadDateTimeUTC: str
    MasterRefID: str
    ImageIndex: int
    UsState: str | None
    PlantType: str | None
    CloudCover: str | None
    GroundResidue: str | None
    GroundCover: str | None
    CoverCropFamily: str | None
    GrowthStage: str | None
    CottonVariety: str | None
    CropOrFallow: str | None
    CropTypeSecondary: str | None
    Species: str | None
    Height: str | None
    SizeClass: str | None
    FlowerFruitOrSeeds: bool | None
    HasMatchingJpgAndRaw: bool | None
    CameraInfo: CameraInfo
