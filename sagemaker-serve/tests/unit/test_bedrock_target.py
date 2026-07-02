"""Property and unit tests for BedrockTarget, ProvisionedConfig, and escrow normalization."""
import hashlib

import pytest
from pydantic import ValidationError

from sagemaker.serve.bedrock_target import (
    BedrockTarget,
    DeploymentMode,
    ProvisionedConfig,
    normalize_escrow_tag_value,
)


# ---------------------------------------------------------------------------
# Property 1: BedrockTarget Mode Validation
# Validates: Requirements 1.1, 1.2
# ---------------------------------------------------------------------------


class TestBedrockTargetModeValidation:
    """Property 1: BedrockTarget Mode Validation.

    For any string, BedrockTarget(mode=value) succeeds iff value in
    {"on_demand", "provisioned"}.

    **Validates: Requirements 1.1, 1.2**
    """

    @pytest.mark.parametrize("mode", ["on_demand", "provisioned"])
    def test_valid_modes_succeed(self, mode):
        target = BedrockTarget(mode=mode)
        assert target.mode == mode

    @pytest.mark.parametrize(
        "mode",
        [
            "invalid",
            "ON_DEMAND",
            "Provisioned",
            "",
            "on-demand",
            "provision",
            "ondemand",
            "provisioned_throughput",
            " on_demand",
            "on_demand ",
        ],
    )
    def test_invalid_modes_raise_value_error(self, mode):
        with pytest.raises(ValidationError):
            BedrockTarget(mode=mode)


# ---------------------------------------------------------------------------
# Property 2: ProvisionedConfig Units Range Validation
# Validates: Requirements 2.1, 2.6
# ---------------------------------------------------------------------------


class TestProvisionedConfigUnitsValidation:
    """Property 2: ProvisionedConfig Units Range Validation.

    For any integer, ProvisionedConfig(units=value) succeeds iff value in [1, 99].

    **Validates: Requirements 2.1, 2.6**
    """

    @pytest.mark.parametrize("units", [1, 2, 50, 98, 99])
    def test_valid_units_succeed(self, units):
        config = ProvisionedConfig(units=units)
        assert config.units == units

    @pytest.mark.parametrize("units", [0, -1, 100, 1000, -100])
    def test_invalid_units_raise_value_error(self, units):
        with pytest.raises(ValidationError):
            ProvisionedConfig(units=units)


# ---------------------------------------------------------------------------
# Property 3: ProvisionedConfig Commitment Duration Validation
# Validates: Requirements 2.2, 2.3
# ---------------------------------------------------------------------------


class TestProvisionedConfigCommitmentDurationValidation:
    """Property 3: ProvisionedConfig Commitment Duration Validation.

    For any non-None string, ProvisionedConfig(commitment_duration=value) succeeds
    iff value in {"OneMonth", "SixMonths"}.

    **Validates: Requirements 2.2, 2.3**
    """

    @pytest.mark.parametrize("duration", ["OneMonth", "SixMonths"])
    def test_valid_commitment_durations_succeed(self, duration):
        config = ProvisionedConfig(commitment_duration=duration)
        assert config.commitment_duration == duration

    def test_none_commitment_duration_succeeds(self):
        config = ProvisionedConfig(commitment_duration=None)
        assert config.commitment_duration is None

    @pytest.mark.parametrize(
        "duration",
        [
            "one_month",
            "six_months",
            "OneYear",
            "",
            "onemonth",
            "ONEMONTH",
            "1month",
            "6months",
            "ThreeMonths",
        ],
    )
    def test_invalid_commitment_durations_raise_value_error(self, duration):
        with pytest.raises(ValidationError):
            ProvisionedConfig(commitment_duration=duration)


# ---------------------------------------------------------------------------
# Property 4: Dict-to-ProvisionedConfig Coercion Round-Trip
# Validates: Requirements 1.5
# ---------------------------------------------------------------------------


class TestDictToProvisionedConfigCoercionRoundTrip:
    """Property 4: Dict-to-ProvisionedConfig Coercion Round-Trip.

    For any valid ProvisionedConfig, model_dump() then
    BedrockTarget(mode="provisioned", config=dict) produces equivalent config.

    **Validates: Requirements 1.5**
    """

    @pytest.mark.parametrize(
        "config_kwargs",
        [
            {"units": 1},
            {"units": 50, "commitment_duration": "OneMonth"},
            {"units": 99, "commitment_duration": "SixMonths"},
            {
                "units": 5,
                "commitment_duration": None,
                "deployment_mode": DeploymentMode.UPDATE_IF_EXISTS,
                "skip_model_reuse": True,
            },
            {
                "units": 1,
                "commitment_duration": None,
                "deployment_mode": DeploymentMode.FAIL_IF_EXISTS,
                "skip_model_reuse": False,
            },
        ],
    )
    def test_round_trip_preserves_config(self, config_kwargs):
        original = ProvisionedConfig(**config_kwargs)
        config_dict = original.model_dump()
        target = BedrockTarget(mode="provisioned", config=config_dict)
        assert target.config.units == original.units
        assert target.config.commitment_duration == original.commitment_duration
        assert target.config.deployment_mode == original.deployment_mode
        assert target.config.skip_model_reuse == original.skip_model_reuse


# ---------------------------------------------------------------------------
# Property 5: Escrow URI Normalization Preserves Lookup Identity
# Validates: Requirements 5.8
# ---------------------------------------------------------------------------


class TestEscrowUriNormalizationProperty:
    """Property 5: Escrow URI Normalization Preserves Lookup Identity.

    For strings >256 chars, result is exactly 256 chars, starts with s[:224],
    has "-" at pos 224, remaining 31 chars are hex prefix of sha256(s).
    For strings <=256, result equals input.

    **Validates: Requirements 5.8**
    """

    @pytest.mark.parametrize(
        "uri",
        [
            "",
            "s3://bucket/key",
            "a" * 100,
            "a" * 255,
            "a" * 256,
        ],
    )
    def test_short_uris_returned_unchanged(self, uri):
        result = normalize_escrow_tag_value(uri)
        assert result == uri

    @pytest.mark.parametrize(
        "uri",
        [
            "x" * 257,
            "s3://my-bucket/very/long/path/" + "z" * 300,
            "a" * 512,
        ],
    )
    def test_long_uris_normalized_to_256_chars(self, uri):
        result = normalize_escrow_tag_value(uri)
        assert len(result) == 256
        assert result[:224] == uri[:224]
        assert result[224] == "-"
        expected_hash = hashlib.sha256(uri.encode()).hexdigest()[:31]
        assert result[225:] == expected_hash


class TestBedrockTargetConfigRejectedOnDemand:
    """Validates: Requirements 1.4"""

    def test_config_rejected_when_mode_on_demand(self):
        with pytest.raises(ValueError, match="config is only applicable to provisioned mode"):
            BedrockTarget(mode="on_demand", config=ProvisionedConfig())


class TestBedrockTargetDefaultProvisionedConfig:
    """Validates: Requirements 1.6"""

    def test_default_provisioned_config_created_when_config_none(self):
        target = BedrockTarget(mode="provisioned")
        assert target.config is not None
        assert target.config.units == 1
        assert target.config.commitment_duration is None
        assert target.config.deployment_mode == DeploymentMode.FAIL_IF_EXISTS
        assert target.config.skip_model_reuse is False


class TestDeploymentModeEnumValues:
    """Validates: Requirements 3.1, 3.2"""

    def test_fail_if_exists_value(self):
        assert DeploymentMode.FAIL_IF_EXISTS == "fail_if_exists"

    def test_update_if_exists_value(self):
        assert DeploymentMode.UPDATE_IF_EXISTS == "update_if_exists"


class TestNormalizeEscrowTagValue:
    """Validates: Requirements 5.8"""

    def test_exactly_256_chars_returns_unchanged(self):
        value = "a" * 256
        result = normalize_escrow_tag_value(value)
        assert result == value
        assert len(result) == 256

    def test_257_chars_returns_256_with_correct_format(self):
        value = "b" * 257
        result = normalize_escrow_tag_value(value)
        assert len(result) == 256
        assert result[:224] == value[:224]
        assert result[224] == "-"
        expected_hash = hashlib.sha256(value.encode()).hexdigest()[:31]
        assert result[225:] == expected_hash

    def test_very_long_string_returns_exactly_256_chars(self):
        value = "c" * 1000
        result = normalize_escrow_tag_value(value)
        assert len(result) == 256
        assert result[:224] == value[:224]
        assert result[224] == "-"
        expected_hash = hashlib.sha256(value.encode()).hexdigest()[:31]
        assert result[225:] == expected_hash
