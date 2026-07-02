"""Bedrock deployment target configuration models."""
from __future__ import absolute_import

import hashlib
from enum import Enum
from typing import Optional

from pydantic import BaseModel, field_validator, model_validator


ESCROW_URI_TAG_KEY = "sagemaker.amazonaws.com/forge/escrow-uri"


class DeploymentMode(str, Enum):
    """Deployment behavior when an endpoint with the same name already exists."""

    FAIL_IF_EXISTS = "fail_if_exists"
    UPDATE_IF_EXISTS = "update_if_exists"


class ProvisionedConfig(BaseModel):
    """Configuration for Provisioned Throughput deployments."""

    units: int = 1
    commitment_duration: Optional[str] = None
    deployment_mode: DeploymentMode = DeploymentMode.FAIL_IF_EXISTS
    skip_model_reuse: bool = False

    @field_validator("units")
    @classmethod
    def validate_units(cls, v: int) -> int:
        if v < 1 or v > 99:
            raise ValueError(f"units must be between 1 and 99 (inclusive), got {v}")
        return v

    @field_validator("commitment_duration")
    @classmethod
    def validate_commitment_duration(cls, v: Optional[str]) -> Optional[str]:
        valid_values = {"OneMonth", "SixMonths"}
        if v is not None and v not in valid_values:
            raise ValueError(
                f"commitment_duration must be one of {valid_values} or None, got '{v}'"
            )
        return v


class BedrockTarget(BaseModel):
    """Deployment target configuration for BedrockModelBuilder.deploy().

    Attributes:
        mode: Deployment mode - "on_demand" or "provisioned".
        config: Provisioned-mode settings. Only applicable when mode="provisioned".
        skip_model_reuse: If True, skip escrow tag lookup and always create a new model.
            Applies to both on_demand and provisioned modes. Defaults to False.
    """

    mode: str
    config: Optional[ProvisionedConfig] = None
    skip_model_reuse: bool = False

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        valid_modes = {"on_demand", "provisioned"}
        if v not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got '{v}'")
        return v

    @model_validator(mode="after")
    def validate_config_for_mode(self) -> "BedrockTarget":
        if self.mode == "on_demand" and self.config is not None:
            raise ValueError(
                "config is only applicable to provisioned mode. "
                "Remove config or set mode='provisioned'."
            )
        if self.mode == "provisioned" and self.config is None:
            self.config = ProvisionedConfig()
        return self

    def __init__(self, **data):
        if "config" in data and isinstance(data["config"], dict):
            data["config"] = ProvisionedConfig(**data["config"])
        super().__init__(**data)


def normalize_escrow_tag_value(escrow_uri: str) -> str:
    """Normalize escrow URI for use as a tag value (max 256 chars).

    If the URI is <= 256 chars, returns as-is.
    Otherwise, truncates to 224 chars + "-" + 31 hex chars of SHA-256.
    """
    if len(escrow_uri) <= 256:
        return escrow_uri
    return escrow_uri[:224] + "-" + hashlib.sha256(escrow_uri.encode()).hexdigest()[:31]
