# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Unit tests for BedrockModelBuilder."""

import json
import pytest
from unittest.mock import Mock, patch
from botocore.exceptions import ClientError

from sagemaker.core.resources import TrainingJob
from sagemaker.core.shapes import ModelArtifacts, OutputDataConfig
from sagemaker.core.utils.utils import Unassigned
from sagemaker.serve.bedrock_model_builder import BedrockModelBuilder, _is_nova_model

MODULE = "sagemaker.serve.bedrock_model_builder"


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_container(recipe_name=None, hub_content_name=None, s3_uri=None):
    """Build a mock container with optional base_model and model_data_source."""
    container = Mock()
    if recipe_name is not None or hub_content_name is not None:
        base_model = Mock()
        base_model.recipe_name = recipe_name
        base_model.hub_content_name = hub_content_name
        container.base_model = base_model
    else:
        container.base_model = None

    if s3_uri:
        s3_data = Mock()
        s3_data.s3_uri = s3_uri
        data_source = Mock()
        data_source.s3_data_source = s3_data
        container.model_data_source = data_source
    else:
        container.model_data_source = None
    return container


def _make_model_package(container):
    """Wrap a container in a mock ModelPackage."""
    pkg = Mock()
    pkg.inference_specification.containers = [container]
    return pkg


def _builder():
    """Return a BedrockModelBuilder(model=None) — no side-effects."""
    return BedrockModelBuilder(model=None)


# ── _is_nova_model ──────────────────────────────────────────────────────────


class TestIsNovaModel:
    def test_nova_via_recipe_name(self):
        assert _is_nova_model(_make_container(recipe_name="amazon-nova-micro-v1")) is True

    def test_nova_via_hub_content_name(self):
        assert _is_nova_model(_make_container(hub_content_name="amazon-nova-lite")) is True

    def test_oss(self):
        assert _is_nova_model(_make_container(recipe_name="llama-3-8b", hub_content_name="llama")) is False

    def test_no_base_model(self):
        assert _is_nova_model(_make_container()) is False

    def test_none_fields(self):
        assert _is_nova_model(_make_container(recipe_name=None, hub_content_name=None)) is False

    def test_case_insensitive(self):
        assert _is_nova_model(_make_container(recipe_name="NOVA-PRO")) is True


# ── __init__ ────────────────────────────────────────────────────────────────


class TestInit:
    def test_none_model(self):
        b = _builder()
        assert b.model is None
        assert b.model_package is None
        assert b.s3_model_artifacts is None

    def test_with_model(self):
        m = Mock()
        with patch.object(BedrockModelBuilder, "_fetch_model_package", return_value=Mock()), \
             patch.object(BedrockModelBuilder, "_get_s3_artifacts", return_value="s3://b/k"):
            b = BedrockModelBuilder(model=m)
        assert b.model is m
        assert b.s3_model_artifacts == "s3://b/k"


# ── Client singletons ──────────────────────────────────────────────────────


class TestClients:
    def test_bedrock_client_cached(self):
        b = _builder()
        b.boto_session = Mock()
        b.boto_session.client.return_value = Mock()
        c1 = b._get_bedrock_client()
        c2 = b._get_bedrock_client()
        assert c1 is c2
        b.boto_session.client.assert_called_once_with("bedrock")

    def test_sagemaker_client_cached(self):
        b = _builder()
        b.boto_session = Mock()
        b.boto_session.client.return_value = Mock()
        c1 = b._get_sagemaker_client()
        c2 = b._get_sagemaker_client()
        assert c1 is c2
        b.boto_session.client.assert_called_once_with("sagemaker")

    def test_injected_bedrock_client(self):
        b = _builder()
        injected = Mock()
        b._bedrock_client = injected
        assert b._get_bedrock_client() is injected


# ── _fetch_model_package ────────────────────────────────────────────────────


# Sentinel classes used to control isinstance checks in _fetch_model_package tests.
class _SentinelA:
    pass


class _SentinelB:
    pass


class _SentinelC:
    pass


class TestFetchModelPackage:
    def test_model_package_returned_directly(self):
        """When model is a ModelPackage, return it as-is."""
        b = _builder()
        b.model = Mock()
        # ModelPackage = type(b.model) so isinstance matches; others are sentinels
        with patch(f"{MODULE}.ModelPackage", type(b.model)), \
             patch(f"{MODULE}.TrainingJob", _SentinelA), \
             patch(f"{MODULE}.ModelTrainer", _SentinelB):
            result = b._fetch_model_package()
        assert result is b.model

    def test_from_training_job(self):
        b = _builder()
        b.model = Mock()
        b.model.output_model_package_arn = "arn:pkg"
        expected = Mock()

        # We need ModelPackage to NOT match but still have a .get() method.
        # Use a class with a get classmethod.
        class _FakeModelPackage:
            @staticmethod
            def get(arn):
                return expected

        with patch(f"{MODULE}.ModelPackage", _FakeModelPackage), \
             patch(f"{MODULE}.TrainingJob", type(b.model)), \
             patch(f"{MODULE}.ModelTrainer", _SentinelA):
            result = b._fetch_model_package()
        assert result is expected

    def test_from_model_trainer(self):
        b = _builder()
        b.model = Mock()
        b.model._latest_training_job.output_model_package_arn = "arn:pkg"
        expected = Mock()

        class _FakeModelPackage:
            @staticmethod
            def get(arn):
                return expected

        with patch(f"{MODULE}.ModelPackage", _FakeModelPackage), \
             patch(f"{MODULE}.TrainingJob", _SentinelA), \
             patch(f"{MODULE}.ModelTrainer", type(b.model)):
            result = b._fetch_model_package()
        assert result is expected

    def test_unknown_type_returns_none(self):
        b = _builder()
        b.model = "unknown"
        assert b._fetch_model_package() is None


# ── _get_s3_artifacts ───────────────────────────────────────────────────────


class TestGetS3Artifacts:
    def test_none_when_no_model_package(self):
        b = _builder()
        b.model_package = None
        assert b._get_s3_artifacts() is None

    def test_oss_returns_s3_uri(self):
        c = _make_container(recipe_name="llama", hub_content_name="llama", s3_uri="s3://b/m.tar.gz")
        b = _builder()
        b.model_package = _make_model_package(c)
        assert b._get_s3_artifacts() == "s3://b/m.tar.gz"

    def test_oss_no_data_source(self):
        c = _make_container(recipe_name="llama", hub_content_name="llama")
        b = _builder()
        b.model_package = _make_model_package(c)
        assert b._get_s3_artifacts() is None

    def test_nova_training_job_delegates_to_manifest(self):
        c = _make_container(recipe_name="nova-micro")
        b = _builder()
        b.model = Mock()
        b.model_package = _make_model_package(c)
        with patch(f"{MODULE}.TrainingJob", type(b.model)), \
             patch.object(BedrockModelBuilder, "_get_checkpoint_uri_from_manifest",
                          return_value="s3://b/ckpt"):
            result = b._get_s3_artifacts()
        assert result == "s3://b/ckpt"

    def test_nova_non_training_job_falls_through(self):
        c = _make_container(recipe_name="nova-micro", s3_uri="s3://b/fallback")
        b = _builder()
        b.model = "not-a-training-job"
        b.model_package = _make_model_package(c)
        assert b._get_s3_artifacts() == "s3://b/fallback"


# ── _get_checkpoint_uri_from_manifest ───────────────────────────────────────


class TestGetCheckpointUri:
    def _make_builder(self, s3_output_path, manifest_body=None, s3_error=None,
                      job_name="myjob"):
        mock_job = Mock()
        mock_job.output_data_config = Mock()
        mock_job.output_data_config.s3_output_path = s3_output_path
        mock_job.training_job_name = job_name

        mock_s3 = Mock()
        # Always set exceptions.NoSuchKey to a real exception class so
        # `except s3_client.exceptions.NoSuchKey` works in the source code.
        mock_s3.exceptions = Mock()
        mock_s3.exceptions.NoSuchKey = ClientError

        if s3_error:
            mock_s3.get_object.side_effect = s3_error
        elif manifest_body is not None:
            body = Mock()
            body.read.return_value = json.dumps(manifest_body).encode()
            mock_s3.get_object.return_value = {"Body": body}

        session = Mock()
        session.client.return_value = mock_s3

        b = _builder()
        b.model = mock_job
        b.boto_session = session
        return b, mock_s3

    def test_success(self):
        b, s3 = self._make_builder(
            "s3://bucket/path/",
            manifest_body={"checkpoint_s3_bucket": "s3://bucket/ckpt/step_4"},
            job_name="myjob",
        )
        with patch(f"{MODULE}.TrainingJob", type(b.model)):
            result = b._get_checkpoint_uri_from_manifest()
        assert result == "s3://bucket/ckpt/step_4"
        s3.get_object.assert_called_once_with(
            Bucket="bucket", Key="path/myjob/output/output/manifest.json"
        )

    def test_missing_checkpoint_key(self):
        b, _ = self._make_builder(
            "s3://bucket/path/",
            manifest_body={"other_key": "value"},
        )
        with patch(f"{MODULE}.TrainingJob", type(b.model)):
            with pytest.raises(ValueError, match="checkpoint_s3_bucket"):
                b._get_checkpoint_uri_from_manifest()

    def test_manifest_not_found(self):
        err = ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        b, _ = self._make_builder("s3://bucket/path/", s3_error=err)
        with patch(f"{MODULE}.TrainingJob", type(b.model)):
            with pytest.raises(ValueError, match="manifest.json not found"):
                b._get_checkpoint_uri_from_manifest()

    def test_not_training_job_raises(self):
        b = _builder()
        b.model = "not-a-training-job"
        with pytest.raises(ValueError, match="TrainingJob"):
            b._get_checkpoint_uri_from_manifest()

    def test_no_s3_output_path_raises(self):
        b, _ = self._make_builder(None)
        with patch(f"{MODULE}.TrainingJob", type(b.model)):
            with pytest.raises(ValueError, match="No S3 output path"):
                b._get_checkpoint_uri_from_manifest()

    def test_invalid_json_raises(self):
        mock_job = Mock()
        mock_job.output_data_config = Mock()
        mock_job.output_data_config.s3_output_path = "s3://bucket/path/"
        mock_job.training_job_name = "myjob"

        body = Mock()
        body.read.return_value = b"not-json"
        mock_s3 = Mock()
        mock_s3.get_object.return_value = {"Body": body}
        mock_s3.download_file.side_effect = Exception("not found")
        mock_s3.exceptions = Mock()
        mock_s3.exceptions.NoSuchKey = ClientError

        session = Mock()
        session.client.return_value = mock_s3

        b = _builder()
        b.model = mock_job
        b.boto_session = session

        with patch(f"{MODULE}.TrainingJob", type(b.model)):
            with pytest.raises(ValueError, match="could not extract from output.tar.gz"):
                b._get_checkpoint_uri_from_manifest()


# ── _wait_for_model_active ──────────────────────────────────────────────────


class TestWaitForModelActive:
    def test_immediate_active(self):
        b = _builder()
        b._bedrock_client = Mock()
        b._bedrock_client.get_custom_model.return_value = {"modelStatus": "Active"}
        b._wait_for_model_active("arn:model")
        b._bedrock_client.get_custom_model.assert_called_once()

    def test_polls_then_active(self):
        b = _builder()
        b._bedrock_client = Mock()
        b._bedrock_client.get_custom_model.side_effect = [
            {"modelStatus": "Creating"},
            {"modelStatus": "Creating"},
            {"modelStatus": "Active"},
        ]
        with patch(f"{MODULE}.time.sleep"):
            b._wait_for_model_active("arn:model", poll_interval=1, max_wait=10)
        assert b._bedrock_client.get_custom_model.call_count == 3

    def test_failed_raises(self):
        b = _builder()
        b._bedrock_client = Mock()
        b._bedrock_client.get_custom_model.return_value = {"modelStatus": "Failed"}
        with pytest.raises(RuntimeError, match="failed"):
            b._wait_for_model_active("arn:model")

    def test_timeout_raises(self):
        b = _builder()
        b._bedrock_client = Mock()
        b._bedrock_client.get_custom_model.return_value = {"modelStatus": "Creating"}
        with patch(f"{MODULE}.time.sleep"):
            with pytest.raises(RuntimeError, match="Timed out"):
                b._wait_for_model_active("arn:model", poll_interval=1, max_wait=2)


# ── create_deployment ───────────────────────────────────────────────────────


class TestCreateDeployment:
    def test_polls_model_then_creates_then_polls_deployment(self):
        b = _builder()
        b._bedrock_client = Mock()
        b._bedrock_client.get_custom_model.return_value = {"modelStatus": "Active"}
        b._bedrock_client.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "arn:dep"
        }
        b._bedrock_client.get_custom_model_deployment.return_value = {"status": "Active"}

        result = b.create_deployment(model_arn="arn:model", deployment_name="dep")

        b._bedrock_client.get_custom_model.assert_called_once()
        b._bedrock_client.create_custom_model_deployment.assert_called_once()
        b._bedrock_client.get_custom_model_deployment.assert_called_once()
        assert result["customModelDeploymentArn"] == "arn:dep"

    def test_passes_extra_kwargs(self):
        b = _builder()
        b._bedrock_client = Mock()
        b._bedrock_client.get_custom_model.return_value = {"modelStatus": "Active"}
        b._bedrock_client.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "arn:dep"
        }
        b._bedrock_client.get_custom_model_deployment.return_value = {"status": "Active"}

        b.create_deployment(model_arn="arn:model", deployment_name="d", commitmentDuration="ONE_MONTH")
        kw = b._bedrock_client.create_custom_model_deployment.call_args[1]
        assert kw["commitmentDuration"] == "ONE_MONTH"

    def test_skips_deployment_polling_when_no_arn_in_response(self):
        b = _builder()
        b._bedrock_client = Mock()
        b._bedrock_client.get_custom_model.return_value = {"modelStatus": "Active"}
        b._bedrock_client.create_custom_model_deployment.return_value = {}

        b.create_deployment(model_arn="arn:model", deployment_name="d")
        b._bedrock_client.get_custom_model_deployment.assert_not_called()

    def test_empty_model_arn_raises(self):
        with pytest.raises(ValueError, match="model_arn is required"):
            _builder().create_deployment(model_arn="", deployment_name="d")

    def test_none_model_arn_raises(self):
        with pytest.raises(ValueError, match="model_arn is required"):
            _builder().create_deployment(model_arn=None, deployment_name="d")


# ── _wait_for_deployment_active ─────────────────────────────────────────────


class TestWaitForDeploymentActive:
    def test_immediate_active(self):
        b = _builder()
        b._bedrock_client = Mock()
        b._bedrock_client.get_custom_model_deployment.return_value = {"status": "Active"}
        b._wait_for_deployment_active("arn:dep")
        b._bedrock_client.get_custom_model_deployment.assert_called_once_with(
            customModelDeploymentIdentifier="arn:dep"
        )

    def test_polls_then_active(self):
        b = _builder()
        b._bedrock_client = Mock()
        b._bedrock_client.get_custom_model_deployment.side_effect = [
            {"status": "Creating"},
            {"status": "Creating"},
            {"status": "Active"},
        ]
        with patch(f"{MODULE}.time.sleep"):
            b._wait_for_deployment_active("arn:dep", poll_interval=1, max_wait=10)
        assert b._bedrock_client.get_custom_model_deployment.call_count == 3

    def test_failed_raises(self):
        b = _builder()
        b._bedrock_client = Mock()
        b._bedrock_client.get_custom_model_deployment.return_value = {"status": "Failed"}
        with pytest.raises(RuntimeError, match="failed"):
            b._wait_for_deployment_active("arn:dep")

    def test_timeout_raises(self):
        b = _builder()
        b._bedrock_client = Mock()
        b._bedrock_client.get_custom_model_deployment.return_value = {"status": "Creating"}
        with patch(f"{MODULE}.time.sleep"):
            with pytest.raises(RuntimeError, match="Timed out"):
                b._wait_for_deployment_active("arn:dep", poll_interval=1, max_wait=2)


# ── deploy ──────────────────────────────────────────────────────────────────


class TestDeploy:
    @pytest.fixture(autouse=True)
    def _stub_role_validation(self):
        """deploy() now validates the provided role via resolve_and_validate_role.

        These tests pass a placeholder role ("r") and exercise the deploy flow, not
        IAM validation, so stub the resolver to echo the provided role back (or fall
        back to an auto-role when none is given). Tests that specifically assert the
        auto-resolve path patch the resolver themselves.
        """
        with patch(
            f"{MODULE}.resolve_and_validate_role",
            side_effect=lambda provided_role, **kwargs: provided_role or "auto-role",
        ):
            yield

    def test_oss_waits_for_import_and_returns_job_details(self):
        """OSS deploy: import job → wait → return job details."""
        c = _make_container(s3_uri="s3://b/m.tar.gz")
        b = _builder()
        b.model_package = _make_model_package(c)
        b.s3_model_artifacts = "s3://b/m.tar.gz"
        b._bedrock_client = Mock()
        b._bedrock_client.create_model_import_job.return_value = {"jobArn": "arn:job"}
        b._bedrock_client.get_model_import_job.return_value = {
            "status": "Completed",
            "importedModelName": "my-imported-model",
            "importedModelArn": "arn:aws:bedrock:us-west-2:123:imported-model/abc",
        }

        with patch(f"{MODULE}.time.sleep"), \
             patch.object(b, "_extract_tar_gz_to_s3", return_value="s3://b/extracted/checkpoints/hf/"):
            result = b.deploy(job_name="j", imported_model_name="m", role_arn="r")

        b._bedrock_client.create_model_import_job.assert_called_once()
        b._bedrock_client.get_model_import_job.assert_called()
        # Should NOT call create_provisioned_model_throughput
        b._bedrock_client.create_provisioned_model_throughput.assert_not_called()
        assert result["status"] == "Completed"
        assert result["importedModelName"] == "my-imported-model"

    def test_oss_does_not_create_provisioned_throughput(self):
        """deploy() for OSS models should never call CreateProvisionedModelThroughput."""
        c = _make_container(s3_uri="s3://b/m.tar.gz")
        b = _builder()
        b.model_package = _make_model_package(c)
        b.s3_model_artifacts = "s3://b/m.tar.gz"
        b._bedrock_client = Mock()
        b._bedrock_client.create_model_import_job.return_value = {"jobArn": "arn:job"}
        b._bedrock_client.get_model_import_job.return_value = {
            "status": "Completed",
            "importedModelName": "m",
        }

        with patch(f"{MODULE}.time.sleep"), \
             patch.object(b, "_extract_tar_gz_to_s3", return_value="s3://b/extracted/checkpoints/hf/"):
            b.deploy(job_name="j", imported_model_name="m", role_arn="r")

        b._bedrock_client.create_provisioned_model_throughput.assert_not_called()
        b._bedrock_client.get_provisioned_model_throughput.assert_not_called()

    def test_nova_full_chain(self):
        c = _make_container(recipe_name="nova-micro", hub_content_name="nova")
        b = _builder()
        b.model_package = _make_model_package(c)
        b.s3_model_artifacts = "s3://b/ckpt"
        b._bedrock_client = Mock()
        b._bedrock_client.create_custom_model.return_value = {"modelArn": "arn:m"}
        b._bedrock_client.get_custom_model.return_value = {"modelStatus": "Active"}
        b._bedrock_client.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "arn:dep"
        }
        b._bedrock_client.get_custom_model_deployment.return_value = {"status": "Active"}

        result = b.deploy(custom_model_name="nova-m", role_arn="r")
        b._bedrock_client.create_custom_model.assert_called_once()
        b._bedrock_client.get_custom_model.assert_called_once()
        b._bedrock_client.create_custom_model_deployment.assert_called_once()
        b._bedrock_client.get_custom_model_deployment.assert_called_once()
        assert result["customModelDeploymentArn"] == "arn:dep"

    def test_nova_via_hub_content_name(self):
        c = _make_container(recipe_name=None, hub_content_name="amazon-nova-lite")
        b = _builder()
        b.model_package = _make_model_package(c)
        b.s3_model_artifacts = "s3://b/ckpt"
        b._bedrock_client = Mock()
        b._bedrock_client.create_custom_model.return_value = {"modelArn": "arn:m"}
        b._bedrock_client.get_custom_model.return_value = {"modelStatus": "Active"}
        b._bedrock_client.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "arn:dep"
        }
        b._bedrock_client.get_custom_model_deployment.return_value = {"status": "Active"}

        result = b.deploy(custom_model_name="n", role_arn="r")
        assert result["customModelDeploymentArn"] == "arn:dep"

    def test_nova_default_deployment_name(self):
        c = _make_container(recipe_name="nova-micro")
        b = _builder()
        b.model_package = _make_model_package(c)
        b.s3_model_artifacts = "s3://b/k"
        b._bedrock_client = Mock()
        b._bedrock_client.create_custom_model.return_value = {"modelArn": "arn"}
        b._bedrock_client.get_custom_model.return_value = {"modelStatus": "Active"}
        b._bedrock_client.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "dep"
        }
        b._bedrock_client.get_custom_model_deployment.return_value = {"status": "Active"}

        b.deploy(custom_model_name="my-model", role_arn="r")
        kw = b._bedrock_client.create_custom_model_deployment.call_args[1]
        assert kw["modelDeploymentName"] == "my-model-deployment"

    def test_nova_with_tags(self):
        c = _make_container(recipe_name="nova-micro")
        b = _builder()
        b.model_package = _make_model_package(c)
        b.s3_model_artifacts = "s3://b/k"
        b._bedrock_client = Mock()
        b._bedrock_client.create_custom_model.return_value = {"modelArn": "arn"}
        b._bedrock_client.get_custom_model.return_value = {"modelStatus": "Active"}
        b._bedrock_client.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "dep"
        }
        b._bedrock_client.get_custom_model_deployment.return_value = {"status": "Active"}

        tags = [{"Key": "env", "Value": "test"}]
        b.deploy(custom_model_name="m", role_arn="r", model_tags=tags)
        kw = b._bedrock_client.create_custom_model.call_args[1]
        assert kw["modelTags"] == tags

    def test_no_model_package_raises(self):
        b = _builder()
        b.model_package = None
        b.s3_model_artifacts = None
        with pytest.raises(ValueError, match="No model source available"):
            b.deploy(job_name="j", role_arn="r")

    def test_nova_missing_custom_model_name_raises(self):
        c = _make_container(recipe_name="nova-micro")
        b = _builder()
        b.model_package = _make_model_package(c)
        b.s3_model_artifacts = "s3://b/k"
        with pytest.raises(ValueError, match="custom_model_name is required"):
            b.deploy(role_arn="r")

    def test_nova_missing_role_arn_auto_resolves(self):
        """When no role_arn is given, a least-privilege Bedrock role is auto-resolved."""
        c = _make_container(recipe_name="nova-micro")
        b = _builder()
        b.model_package = _make_model_package(c)
        b.s3_model_artifacts = "s3://b/k"
        b._bedrock_client = Mock()
        b._bedrock_client.create_custom_model.return_value = {"modelArn": "model-arn"}

        with patch(f"{MODULE}.resolve_and_validate_role", return_value="auto-role") as mock_resolve, \
             patch.object(b, "create_deployment", return_value={"ok": True}) as mock_create_deploy:
            b.deploy(custom_model_name="m")

        mock_resolve.assert_called_once_with(
            provided_role=None,
            role_type="bedrock",
            sagemaker_session=b.sagemaker_session,
        )
        # The auto-resolved role is threaded into the Bedrock create call.
        assert b._bedrock_client.create_custom_model.call_args[1]["roleArn"] == "auto-role"
        mock_create_deploy.assert_called_once()

    def test_oss_missing_role_arn_auto_resolves(self):
        """OSS import path also auto-resolves a Bedrock role when none is provided."""
        c = _make_container()
        b = _builder()
        b.model_package = _make_model_package(c)
        b.s3_model_artifacts = "s3://b/k"
        b._bedrock_client = Mock()
        b._bedrock_client.create_model_import_job.return_value = {"jobArn": "arn"}
        b._bedrock_client.get_model_import_job.return_value = {
            "status": "Completed",
            "importedModelName": "m",
        }

        with patch(f"{MODULE}.resolve_and_validate_role", return_value="auto-role") as mock_resolve, \
             patch(f"{MODULE}.time.sleep"):
            b.deploy(job_name="j", imported_model_name="m")

        mock_resolve.assert_called_once_with(
            provided_role=None,
            role_type="bedrock",
            sagemaker_session=b.sagemaker_session,
        )
        assert b._bedrock_client.create_model_import_job.call_args[1]["roleArn"] == "auto-role"

    def test_oss_strips_none_params(self):
        c = _make_container()
        b = _builder()
        b.model_package = _make_model_package(c)
        b.s3_model_artifacts = "s3://b/k"
        b._bedrock_client = Mock()
        b._bedrock_client.create_model_import_job.return_value = {"jobArn": "arn"}
        b._bedrock_client.get_model_import_job.return_value = {
            "status": "Completed",
            "importedModelName": "m",
        }

        with patch(f"{MODULE}.time.sleep"):
            b.deploy(job_name="j", imported_model_name="m", role_arn="r")

        kw = b._bedrock_client.create_model_import_job.call_args[1]
        assert "importedModelKmsKeyId" not in kw
        assert "clientRequestToken" not in kw

    def test_s3_uri_string_with_custom_model_name_uses_nova_path(self):
        """Direct S3 URI + custom_model_name triggers create_custom_model path."""
        b = BedrockModelBuilder(model="s3://my-bucket/my-checkpoint/")
        b._bedrock_client = Mock()
        b._bedrock_client.create_custom_model.return_value = {"modelArn": "arn:model"}

        with patch.object(b, "create_deployment", return_value={"customModelDeploymentArn": "arn:dep"}) as mock_deploy:
            result = b.deploy(custom_model_name="my-nova-model", role_arn="arn:role")

        b._bedrock_client.create_custom_model.assert_called_once()
        kw = b._bedrock_client.create_custom_model.call_args[1]
        assert kw["modelName"] == "my-nova-model"
        assert kw["modelSourceConfig"] == {"s3DataSource": {"s3Uri": "s3://my-bucket/my-checkpoint/"}}
        assert kw["roleArn"] == "arn:role"
        mock_deploy.assert_called_once_with(model_arn="arn:model", deployment_name="my-nova-model-deployment")

    def test_s3_uri_string_without_custom_model_name_uses_oss_path(self):
        """Direct S3 URI without custom_model_name triggers import job path."""
        b = BedrockModelBuilder(model="s3://my-bucket/my-checkpoint/")
        b._bedrock_client = Mock()
        b._bedrock_client.create_model_import_job.return_value = {"jobArn": "arn:job"}
        b._bedrock_client.get_model_import_job.return_value = {
            "status": "Completed",
            "importedModelName": "my-imported",
        }

        with patch(f"{MODULE}.time.sleep"):
            result = b.deploy(job_name="j", imported_model_name="my-imported", role_arn="arn:role")

        b._bedrock_client.create_model_import_job.assert_called_once()
        kw = b._bedrock_client.create_model_import_job.call_args[1]
        assert kw["modelDataSource"] == {"s3DataSource": {"s3Uri": "s3://my-bucket/my-checkpoint/"}}

    def test_s3_uri_string_invalid_raises(self):
        """Non-S3 string as model raises ValueError."""
        with pytest.raises(ValueError, match="must be an S3 URI"):
            BedrockModelBuilder(model="not-an-s3-uri")

    def test_model_trainer_with_checkpoint_no_model_package_uses_nova_path(self):
        """ModelTrainer with model_artifacts on _latest_training_job but no output_model_package_arn deploys via S3."""
        mock_trainer = Mock()
        mock_training_job = Mock()
        mock_training_job.output_model_package_arn = None
        mock_training_job.model_artifacts = Mock()
        mock_training_job.model_artifacts.s3_model_artifacts = "s3://bucket/hp-job/outputs/checkpoints/step_4/"
        mock_trainer._latest_training_job = mock_training_job

        with patch(f"{MODULE}.ModelPackage", _SentinelA), \
             patch(f"{MODULE}.TrainingJob", _SentinelB), \
             patch(f"{MODULE}.ModelTrainer", type(mock_trainer)), \
             patch(f"{MODULE}.MultiTurnRLTrainer", _SentinelA), \
             patch(f"{MODULE}.AgentRFTJob", _SentinelA), \
             patch(f"{MODULE}.is_restricted_model_package", return_value=False), \
             patch(f"{MODULE}.Session") as mock_session:
            mock_session.return_value.boto_session = Mock()
            b = BedrockModelBuilder(model=mock_trainer)

        assert b.model_package is None
        assert b.s3_model_artifacts == "s3://bucket/hp-job/outputs/checkpoints/step_4/"

    def test_model_trainer_with_checkpoint_deploys_via_create_custom_model(self):
        """ModelTrainer with checkpoint S3 URI can deploy via create_custom_model."""
        b = _builder()
        b.model_package = None
        b.s3_model_artifacts = "s3://bucket/hp-job/outputs/checkpoints/step_4/"
        b._bedrock_client = Mock()
        b._bedrock_client.create_custom_model.return_value = {"modelArn": "arn:model"}

        with patch.object(b, "create_deployment", return_value={"customModelDeploymentArn": "arn:dep"}):
            b.deploy(custom_model_name="my-hp-model", role_arn="arn:role")

        kw = b._bedrock_client.create_custom_model.call_args[1]
        assert kw["modelName"] == "my-hp-model"
        assert kw["modelSourceConfig"] == {
            "s3DataSource": {"s3Uri": "s3://bucket/hp-job/outputs/checkpoints/step_4/"}
        }
        assert kw["roleArn"] == "arn:role"

    def test_model_trainer_no_checkpoint_no_model_package_raises(self):
        """ModelTrainer with neither checkpoint nor model package raises on deploy."""
        b = _builder()
        b.model_package = None
        b.s3_model_artifacts = None
        with pytest.raises(ValueError, match="No model source available"):
            b.deploy(custom_model_name="m", role_arn="r")


# ── _wait_for_import_job_complete ───────────────────────────────────────────


class TestWaitForImportJobComplete:
    def test_immediate_completed(self):
        b = _builder()
        b._bedrock_client = Mock()
        b._bedrock_client.get_model_import_job.return_value = {"status": "Completed"}
        b._wait_for_import_job_complete("arn:job")
        b._bedrock_client.get_model_import_job.assert_called_once_with(
            jobIdentifier="arn:job"
        )

    def test_polls_then_completed(self):
        b = _builder()
        b._bedrock_client = Mock()
        b._bedrock_client.get_model_import_job.side_effect = [
            {"status": "InProgress"},
            {"status": "InProgress"},
            {"status": "Completed"},
        ]
        with patch(f"{MODULE}.time.sleep"):
            b._wait_for_import_job_complete("arn:job", poll_interval=1, max_wait=10)
        assert b._bedrock_client.get_model_import_job.call_count == 3

    def test_failed_raises(self):
        b = _builder()
        b._bedrock_client = Mock()
        b._bedrock_client.get_model_import_job.return_value = {
            "status": "Failed",
            "failureMessage": "Invalid model format",
        }
        with pytest.raises(RuntimeError, match="Invalid model format"):
            b._wait_for_import_job_complete("arn:job")

    def test_failed_unknown_reason(self):
        b = _builder()
        b._bedrock_client = Mock()
        b._bedrock_client.get_model_import_job.return_value = {"status": "Failed"}
        with pytest.raises(RuntimeError, match="Unknown"):
            b._wait_for_import_job_complete("arn:job")

    def test_timeout_raises(self):
        b = _builder()
        b._bedrock_client = Mock()
        b._bedrock_client.get_model_import_job.return_value = {"status": "InProgress"}
        with patch(f"{MODULE}.time.sleep"):
            with pytest.raises(RuntimeError, match="Timed out"):
                b._wait_for_import_job_complete("arn:job", poll_interval=1, max_wait=2)


# ── create_provisioned_throughput ───────────────────────────────────────────


class TestCreateProvisionedThroughput:
    def test_creates_and_polls(self):
        b = _builder()
        b._bedrock_client = Mock()
        b._bedrock_client.create_provisioned_model_throughput.return_value = {
            "provisionedModelArn": "arn:pt"
        }
        b._bedrock_client.get_provisioned_model_throughput.return_value = {
            "status": "InService"
        }

        result = b.create_provisioned_throughput(
            model_id="arn:model", provisioned_model_name="my-pt"
        )

        b._bedrock_client.create_provisioned_model_throughput.assert_called_once_with(
            modelId="arn:model",
            provisionedModelName="my-pt",
            modelUnits=1,
        )
        b._bedrock_client.get_provisioned_model_throughput.assert_called_once()
        assert result["provisionedModelArn"] == "arn:pt"

    def test_passes_commitment_duration(self):
        b = _builder()
        b._bedrock_client = Mock()
        b._bedrock_client.create_provisioned_model_throughput.return_value = {
            "provisionedModelArn": "arn:pt"
        }
        b._bedrock_client.get_provisioned_model_throughput.return_value = {
            "status": "InService"
        }

        b.create_provisioned_throughput(
            model_id="arn:model",
            provisioned_model_name="pt",
            model_units=5,
            commitment_duration="OneMonth",
        )

        kw = b._bedrock_client.create_provisioned_model_throughput.call_args[1]
        assert kw["modelUnits"] == 5
        assert kw["commitmentDuration"] == "OneMonth"

    def test_passes_tags(self):
        b = _builder()
        b._bedrock_client = Mock()
        b._bedrock_client.create_provisioned_model_throughput.return_value = {
            "provisionedModelArn": "arn:pt"
        }
        b._bedrock_client.get_provisioned_model_throughput.return_value = {
            "status": "InService"
        }

        tags = [{"Key": "team", "Value": "ml"}]
        b.create_provisioned_throughput(
            model_id="arn:model", provisioned_model_name="pt", tags=tags
        )

        kw = b._bedrock_client.create_provisioned_model_throughput.call_args[1]
        assert kw["tags"] == tags

    def test_skips_polling_when_no_arn_in_response(self):
        b = _builder()
        b._bedrock_client = Mock()
        b._bedrock_client.create_provisioned_model_throughput.return_value = {}

        b.create_provisioned_throughput(
            model_id="arn:model", provisioned_model_name="pt"
        )
        b._bedrock_client.get_provisioned_model_throughput.assert_not_called()

    def test_empty_model_id_raises(self):
        b = _builder()
        with pytest.raises(ValueError, match="model_id is required"):
            b.create_provisioned_throughput(model_id="", provisioned_model_name="pt")

    def test_none_model_id_raises(self):
        b = _builder()
        with pytest.raises(ValueError, match="model_id is required"):
            b.create_provisioned_throughput(model_id=None, provisioned_model_name="pt")

    def test_empty_provisioned_model_name_raises(self):
        b = _builder()
        with pytest.raises(ValueError, match="provisioned_model_name is required"):
            b.create_provisioned_throughput(
                model_id="arn:model", provisioned_model_name=""
            )

    def test_uses_imported_model_id_from_deploy(self):
        """model_id falls back to _imported_model_id set by deploy()."""
        b = _builder()
        b._imported_model_id = "my-deployed-model"
        b._bedrock_client = Mock()
        b._bedrock_client.create_provisioned_model_throughput.return_value = {
            "provisionedModelArn": "arn:pt"
        }
        b._bedrock_client.get_provisioned_model_throughput.return_value = {
            "status": "InService"
        }

        result = b.create_provisioned_throughput(provisioned_model_name="my-pt")

        kw = b._bedrock_client.create_provisioned_model_throughput.call_args[1]
        assert kw["modelId"] == "my-deployed-model"
        assert result["provisionedModelArn"] == "arn:pt"

    def test_explicit_model_id_overrides_stored(self):
        """Explicit model_id takes precedence over _imported_model_id."""
        b = _builder()
        b._imported_model_id = "stored-model"
        b._bedrock_client = Mock()
        b._bedrock_client.create_provisioned_model_throughput.return_value = {
            "provisionedModelArn": "arn:pt"
        }
        b._bedrock_client.get_provisioned_model_throughput.return_value = {
            "status": "InService"
        }

        b.create_provisioned_throughput(
            model_id="explicit-model", provisioned_model_name="my-pt"
        )

        kw = b._bedrock_client.create_provisioned_model_throughput.call_args[1]
        assert kw["modelId"] == "explicit-model"


# ── _wait_for_provisioned_throughput_in_service ─────────────────────────────


class TestWaitForProvisionedThroughputInService:
    def test_immediate_in_service(self):
        b = _builder()
        b._bedrock_client = Mock()
        b._bedrock_client.get_provisioned_model_throughput.return_value = {
            "status": "InService"
        }
        b._wait_for_provisioned_throughput_in_service("arn:pt")
        b._bedrock_client.get_provisioned_model_throughput.assert_called_once_with(
            provisionedModelId="arn:pt"
        )

    def test_polls_then_in_service(self):
        b = _builder()
        b._bedrock_client = Mock()
        b._bedrock_client.get_provisioned_model_throughput.side_effect = [
            {"status": "Creating"},
            {"status": "Creating"},
            {"status": "InService"},
        ]
        with patch(f"{MODULE}.time.sleep"):
            b._wait_for_provisioned_throughput_in_service(
                "arn:pt", poll_interval=1, max_wait=10
            )
        assert b._bedrock_client.get_provisioned_model_throughput.call_count == 3

    def test_failed_raises(self):
        b = _builder()
        b._bedrock_client = Mock()
        b._bedrock_client.get_provisioned_model_throughput.return_value = {
            "status": "Failed",
            "failureMessage": "Insufficient capacity",
        }
        with pytest.raises(RuntimeError, match="Insufficient capacity"):
            b._wait_for_provisioned_throughput_in_service("arn:pt")

    def test_failed_unknown_reason(self):
        b = _builder()
        b._bedrock_client = Mock()
        b._bedrock_client.get_provisioned_model_throughput.return_value = {
            "status": "Failed"
        }
        with pytest.raises(RuntimeError, match="Unknown"):
            b._wait_for_provisioned_throughput_in_service("arn:pt")

    def test_timeout_raises(self):
        b = _builder()
        b._bedrock_client = Mock()
        b._bedrock_client.get_provisioned_model_throughput.return_value = {
            "status": "Creating"
        }
        with patch(f"{MODULE}.time.sleep"):
            with pytest.raises(RuntimeError, match="Timed out"):
                b._wait_for_provisioned_throughput_in_service(
                    "arn:pt", poll_interval=1, max_wait=2
                )


def _apply_model_artifacts_postprocessing(training_job):
    if (
        training_job.training_job_status == "Completed"
        and isinstance(training_job.model_artifacts, Unassigned)
        and not isinstance(training_job.output_data_config, Unassigned)
        and training_job.output_data_config
    ):
        s3_output_path = training_job.output_data_config.s3_output_path
        if s3_output_path and isinstance(s3_output_path, str):
            synthesized_path = (
                f"{s3_output_path.rstrip('/')}/{training_job.training_job_name}/output/"
            )
            training_job.model_artifacts = ModelArtifacts(
                s3_model_artifacts=synthesized_path
            )
    return training_job


class TestModelArtifactsPostProcessing:
    def test_api_model_artifacts_preserved_no_override(self):
        """Synthesis only activates when model_artifacts is Unassigned."""
        api_s3_path = "s3://real-bucket/real-prefix/model.tar.gz"
        job_name = "completed-job-with-artifacts"

        job = TrainingJob(
            training_job_name=job_name,
            training_job_status="Completed",
            model_artifacts=ModelArtifacts(s3_model_artifacts=api_s3_path),
            output_data_config=OutputDataConfig(s3_output_path="s3://output-bucket/output"),
        )

        _apply_model_artifacts_postprocessing(job)

        assert not isinstance(job.model_artifacts, Unassigned)
        assert job.model_artifacts.s3_model_artifacts == api_s3_path

        synthesized_path = f"s3://output-bucket/output/{job_name}/output/"
        assert job.model_artifacts.s3_model_artifacts != synthesized_path

    def test_api_model_artifacts_preserved_even_with_output_data_config(self):
        api_s3_path = "s3://training-output/job123/output/model.tar.gz"
        job_name = "job-with-both-artifacts-and-output-config"

        job = TrainingJob(
            training_job_name=job_name,
            training_job_status="Completed",
            model_artifacts=ModelArtifacts(s3_model_artifacts=api_s3_path),
            output_data_config=OutputDataConfig(s3_output_path="s3://different-bucket/prefix"),
        )

        _apply_model_artifacts_postprocessing(job)
        assert job.model_artifacts.s3_model_artifacts == api_s3_path

    def test_synthesis_applies_when_model_artifacts_is_unassigned(self):
        job_name = "completed-job-no-artifacts"

        job = TrainingJob(
            training_job_name=job_name,
            training_job_status="Completed",
            output_data_config=OutputDataConfig(s3_output_path="s3://my-bucket/output"),
        )

        assert isinstance(job.model_artifacts, Unassigned)
        _apply_model_artifacts_postprocessing(job)

        assert not isinstance(job.model_artifacts, Unassigned)
        expected_path = f"s3://my-bucket/output/{job_name}/output/"
        assert job.model_artifacts.s3_model_artifacts == expected_path

    @pytest.mark.parametrize("status", ["InProgress", "Failed", "Stopped", "Stopping"])
    def test_non_completed_status_no_synthesis(self, status):
        job = TrainingJob(
            training_job_name=f"job-{status.lower()}",
            training_job_status=status,
            output_data_config=OutputDataConfig(s3_output_path="s3://my-bucket/output"),
        )

        assert isinstance(job.model_artifacts, Unassigned)
        _apply_model_artifacts_postprocessing(job)
        assert isinstance(job.model_artifacts, Unassigned)


class TestGetS3ArtifactsFromTrainingJob:
    def test_training_job_with_valid_model_artifacts(self):
        job = TrainingJob(
            training_job_name="my-job",
            model_artifacts=ModelArtifacts(s3_model_artifacts="s3://bucket/path/output/"),
        )
        b = BedrockModelBuilder(model=job)
        b.model_package = None
        assert b._get_s3_artifacts() == "s3://bucket/path/output/"

    def test_training_job_with_unassigned_model_artifacts(self):
        job = TrainingJob(training_job_name="my-job")
        b = BedrockModelBuilder(model=job)
        b.model_package = None
        assert b._get_s3_artifacts() is None

    def test_model_trainer_with_valid_model_artifacts(self):
        mock_trainer = Mock()
        mock_training_job = Mock()
        mock_training_job.output_model_package_arn = None
        mock_training_job.model_artifacts = Mock()
        mock_training_job.model_artifacts.s3_model_artifacts = "s3://bucket/checkpoint/"
        mock_trainer._latest_training_job = mock_training_job

        with patch(f"{MODULE}.ModelPackage", _SentinelA), \
             patch(f"{MODULE}.TrainingJob", _SentinelB), \
             patch(f"{MODULE}.ModelTrainer", type(mock_trainer)), \
             patch(f"{MODULE}.MultiTurnRLTrainer", _SentinelA), \
             patch(f"{MODULE}.AgentRFTJob", _SentinelA), \
             patch(f"{MODULE}.is_restricted_model_package", return_value=False), \
             patch(f"{MODULE}.Session") as mock_session:
            mock_session.return_value.boto_session = Mock()
            b = BedrockModelBuilder(model=mock_trainer)

        assert b.model_package is None
        assert b.s3_model_artifacts == "s3://bucket/checkpoint/"

    def test_model_trainer_no_latest_training_job(self):
        mock_trainer = Mock()
        mock_trainer._latest_training_job = None

        with patch(f"{MODULE}.ModelPackage", _SentinelA), \
             patch(f"{MODULE}.TrainingJob", _SentinelB), \
             patch(f"{MODULE}.ModelTrainer", type(mock_trainer)), \
             patch(f"{MODULE}.MultiTurnRLTrainer", _SentinelA), \
             patch(f"{MODULE}.AgentRFTJob", _SentinelA), \
             patch(f"{MODULE}.is_restricted_model_package", return_value=False), \
             patch(f"{MODULE}.Session") as mock_session:
            mock_session.return_value.boto_session = Mock()
            b = BedrockModelBuilder(model=mock_trainer)

        assert b.s3_model_artifacts is None


# ── _resolve_escrow_identifier ──────────────────────────────────────────────


class TestResolveEscrowIdentifier:
    def test_returns_model_package_arn_when_available(self):
        b = _builder()
        b.model_package = Mock()
        b.model_package.model_package_arn = "arn:aws:sagemaker:us-west-2:123:model-package/pkg"
        b.s3_model_artifacts = "s3://bucket/path/"
        assert b._resolve_escrow_identifier() == "arn:aws:sagemaker:us-west-2:123:model-package/pkg"

    def test_returns_s3_uri_when_no_model_package(self):
        b = _builder()
        b.model_package = None
        b.s3_model_artifacts = "s3://bucket/checkpoints/step_4/"
        assert b._resolve_escrow_identifier() == "s3://bucket/checkpoints/step_4/"

    def test_returns_s3_uri_when_model_package_has_no_arn(self):
        b = _builder()
        b.model_package = Mock()
        b.model_package.model_package_arn = None
        b.s3_model_artifacts = "s3://bucket/path/"
        assert b._resolve_escrow_identifier() == "s3://bucket/path/"

    def test_returns_none_when_no_source(self):
        b = _builder()
        b.model_package = None
        b.s3_model_artifacts = None
        assert b._resolve_escrow_identifier() is None

    def test_prefers_model_package_arn_over_s3(self):
        b = _builder()
        b.model_package = Mock()
        b.model_package.model_package_arn = "arn:pkg"
        b.s3_model_artifacts = "s3://bucket/path/"
        assert b._resolve_escrow_identifier() == "arn:pkg"


# ── _find_existing_model_by_escrow ──────────────────────────────────────────


class TestFindExistingModelByEscrow:
    def test_returns_arn_when_active_model_found(self):
        b = _builder()
        b.boto_session = Mock()
        tagging_client = Mock()
        b.boto_session.client.return_value = tagging_client
        tagging_client.get_resources.return_value = {
            "ResourceTagMappingList": [
                {"ResourceARN": "arn:aws:bedrock:us-west-2:123:custom-model/my-model"}
            ]
        }
        b._bedrock_client = Mock()
        b._bedrock_client.get_custom_model.return_value = {"modelStatus": "Active"}

        result = b._find_existing_model_by_escrow("s3://bucket/path/")
        assert result == "arn:aws:bedrock:us-west-2:123:custom-model/my-model"

    def test_polls_creating_model_until_active(self):
        b = _builder()
        b.boto_session = Mock()
        tagging_client = Mock()
        b.boto_session.client.return_value = tagging_client
        tagging_client.get_resources.return_value = {
            "ResourceTagMappingList": [
                {"ResourceARN": "arn:aws:bedrock:us-west-2:123:custom-model/creating-model"}
            ]
        }
        b._bedrock_client = Mock()
        b._bedrock_client.get_custom_model.side_effect = [
            {"modelStatus": "Creating"},
            {"modelStatus": "Creating"},
            {"modelStatus": "Active"},
        ]

        with patch(f"{MODULE}.time.sleep"):
            result = b._find_existing_model_by_escrow("s3://bucket/path/")

        assert result == "arn:aws:bedrock:us-west-2:123:custom-model/creating-model"
        assert b._bedrock_client.get_custom_model.call_count == 3

    def test_raises_timeout_on_creating_model(self):
        b = _builder()
        b.boto_session = Mock()
        tagging_client = Mock()
        b.boto_session.client.return_value = tagging_client
        tagging_client.get_resources.return_value = {
            "ResourceTagMappingList": [
                {"ResourceARN": "arn:aws:bedrock:us-west-2:123:custom-model/stuck-model"}
            ]
        }
        b._bedrock_client = Mock()
        b._bedrock_client.get_custom_model.return_value = {"modelStatus": "Creating"}

        with patch(f"{MODULE}.time.sleep"):
            with pytest.raises(TimeoutError, match="did not reach Active"):
                b._find_existing_model_by_escrow("s3://bucket/path/")

    def test_returns_none_on_failed_model(self, caplog):
        b = _builder()
        b.boto_session = Mock()
        tagging_client = Mock()
        b.boto_session.client.return_value = tagging_client
        tagging_client.get_resources.return_value = {
            "ResourceTagMappingList": [
                {"ResourceARN": "arn:aws:bedrock:us-west-2:123:custom-model/failed-model"}
            ]
        }
        b._bedrock_client = Mock()
        b._bedrock_client.get_custom_model.return_value = {"modelStatus": "Failed"}

        import logging
        with caplog.at_level(logging.WARNING):
            result = b._find_existing_model_by_escrow("s3://bucket/path/")

        assert result is None
        assert "Failed" in caplog.text

    def test_returns_none_on_api_failure(self, caplog):
        b = _builder()
        b.boto_session = Mock()
        b.boto_session.client.side_effect = Exception("Network error")

        import logging
        with caplog.at_level(logging.WARNING):
            result = b._find_existing_model_by_escrow("s3://bucket/path/")

        assert result is None
        assert "Network error" in caplog.text

    def test_returns_none_when_no_resources_found(self):
        b = _builder()
        b.boto_session = Mock()
        tagging_client = Mock()
        b.boto_session.client.return_value = tagging_client
        tagging_client.get_resources.return_value = {"ResourceTagMappingList": []}

        result = b._find_existing_model_by_escrow("s3://bucket/path/")
        assert result is None

    def test_returns_none_on_get_custom_model_failure(self, caplog):
        b = _builder()
        b.boto_session = Mock()
        tagging_client = Mock()
        b.boto_session.client.return_value = tagging_client
        tagging_client.get_resources.return_value = {
            "ResourceTagMappingList": [
                {"ResourceARN": "arn:aws:bedrock:us-west-2:123:custom-model/model"}
            ]
        }
        b._bedrock_client = Mock()
        b._bedrock_client.get_custom_model.side_effect = Exception("Access denied")

        import logging
        with caplog.at_level(logging.WARNING):
            result = b._find_existing_model_by_escrow("s3://bucket/path/")

        assert result is None
        assert "Access denied" in caplog.text


# ── _find_or_create_model ───────────────────────────────────────────────────


class TestFindOrCreateModel:
    def test_skips_lookup_when_skip_model_reuse_true(self):
        b = _builder()
        b.model_package = None
        b.s3_model_artifacts = "s3://bucket/path/"
        b._is_rmp = False
        b._bedrock_client = Mock()
        b._bedrock_client.create_custom_model.return_value = {"modelArn": "arn:new-model"}
        b._bedrock_client.get_custom_model.return_value = {"modelStatus": "Active"}

        with patch.object(b, "_find_existing_model_by_escrow") as mock_find:
            result = b._find_or_create_model(
                custom_model_name="my-model",
                role_arn="arn:role",
                model_tags=None,
                skip_model_reuse=True,
            )

        mock_find.assert_not_called()
        assert result == "arn:new-model"

    def test_reuses_existing_model_when_found(self):
        b = _builder()
        b.model_package = None
        b.s3_model_artifacts = "s3://bucket/path/"
        b._is_rmp = False
        b._bedrock_client = Mock()

        with patch.object(
            b, "_find_existing_model_by_escrow", return_value="arn:existing-model"
        ):
            result = b._find_or_create_model(
                custom_model_name="my-model",
                role_arn="arn:role",
                model_tags=None,
                skip_model_reuse=False,
            )

        assert result == "arn:existing-model"
        b._bedrock_client.create_custom_model.assert_not_called()

    def test_always_applies_escrow_tag_on_new_model_creation(self):
        """Property 6: Escrow Tag Always Applied on Model Creation.

        Validates: Requirements 5.6, 8.2
        """
        b = _builder()
        b.model_package = None
        b.s3_model_artifacts = "s3://bucket/path/"
        b._is_rmp = False
        b._bedrock_client = Mock()
        b._bedrock_client.create_custom_model.return_value = {"modelArn": "arn:new-model"}
        b._bedrock_client.get_custom_model.return_value = {"modelStatus": "Active"}

        with patch.object(b, "_find_existing_model_by_escrow", return_value=None):
            b._find_or_create_model(
                custom_model_name="my-model",
                role_arn="arn:role",
                model_tags=None,
                skip_model_reuse=False,
            )

        call_kwargs = b._bedrock_client.create_custom_model.call_args[1]
        tags = call_kwargs["modelTags"]
        escrow_tags = [t for t in tags if t["key"] == "sagemaker.amazonaws.com/forge/escrow-uri"]
        assert len(escrow_tags) == 1
        assert escrow_tags[0]["value"] == "s3://bucket/path/"

    def test_escrow_tag_applied_even_when_skip_model_reuse_true(self):
        """Property 6: Escrow tag applied regardless of skip_model_reuse.

        Validates: Requirements 5.6, 8.2
        """
        b = _builder()
        b.model_package = None
        b.s3_model_artifacts = "s3://bucket/path/"
        b._is_rmp = False
        b._bedrock_client = Mock()
        b._bedrock_client.create_custom_model.return_value = {"modelArn": "arn:new-model"}
        b._bedrock_client.get_custom_model.return_value = {"modelStatus": "Active"}

        b._find_or_create_model(
            custom_model_name="my-model",
            role_arn="arn:role",
            model_tags=None,
            skip_model_reuse=True,
        )

        call_kwargs = b._bedrock_client.create_custom_model.call_args[1]
        tags = call_kwargs["modelTags"]
        escrow_tags = [t for t in tags if t["key"] == "sagemaker.amazonaws.com/forge/escrow-uri"]
        assert len(escrow_tags) == 1
        assert escrow_tags[0]["value"] == "s3://bucket/path/"

    def test_escrow_tag_does_not_duplicate_existing_tags(self):
        b = _builder()
        b.model_package = None
        b.s3_model_artifacts = "s3://bucket/path/"
        b._is_rmp = False
        b._bedrock_client = Mock()
        b._bedrock_client.create_custom_model.return_value = {"modelArn": "arn:new-model"}
        b._bedrock_client.get_custom_model.return_value = {"modelStatus": "Active"}

        existing_tags = [
            {"key": "team", "value": "ml"},
            {"key": "sagemaker.amazonaws.com/forge/escrow-uri", "value": "old-value"},
        ]

        with patch.object(b, "_find_existing_model_by_escrow", return_value=None):
            b._find_or_create_model(
                custom_model_name="my-model",
                role_arn="arn:role",
                model_tags=existing_tags,
                skip_model_reuse=False,
            )

        call_kwargs = b._bedrock_client.create_custom_model.call_args[1]
        tags = call_kwargs["modelTags"]
        escrow_tags = [t for t in tags if t["key"] == "sagemaker.amazonaws.com/forge/escrow-uri"]
        assert len(escrow_tags) == 1
        assert escrow_tags[0]["value"] == "s3://bucket/path/"
        team_tags = [t for t in tags if t["key"] == "team"]
        assert len(team_tags) == 1

    def test_creates_nova_model_with_s3_source(self):
        b = _builder()
        b.model_package = None
        b.s3_model_artifacts = "s3://bucket/ckpt/"
        b._is_rmp = False
        b._bedrock_client = Mock()
        b._bedrock_client.create_custom_model.return_value = {"modelArn": "arn:model"}
        b._bedrock_client.get_custom_model.return_value = {"modelStatus": "Active"}

        with patch.object(b, "_find_existing_model_by_escrow", return_value=None):
            result = b._find_or_create_model(
                custom_model_name="my-nova",
                role_arn="arn:role",
                model_tags=None,
                skip_model_reuse=False,
            )

        assert result == "arn:model"
        call_kwargs = b._bedrock_client.create_custom_model.call_args[1]
        assert call_kwargs["modelSourceConfig"] == {"s3DataSource": {"s3Uri": "s3://bucket/ckpt/"}}
        assert call_kwargs["modelName"] == "my-nova"
        assert call_kwargs["roleArn"] == "arn:role"


# ── _check_existing_deployment ──────────────────────────────────────────────


class TestCheckExistingDeployment:
    def test_returns_arn_on_exact_pt_name_match(self):
        b = _builder()
        b._bedrock_client = Mock()
        b._bedrock_client.list_provisioned_model_throughputs.return_value = {
            "provisionedModelSummaries": [
                {
                    "provisionedModelName": "my-pt-deployment",
                    "provisionedModelArn": "arn:aws:bedrock:us-west-2:123:provisioned-model/my-pt",
                },
            ]
        }

        result = b._check_existing_deployment("my-pt-deployment", is_provisioned=True)
        assert result == "arn:aws:bedrock:us-west-2:123:provisioned-model/my-pt"
        b._bedrock_client.list_provisioned_model_throughputs.assert_called_once_with(
            nameContains="my-pt-deployment"
        )

    def test_returns_none_on_partial_match_only(self):
        b = _builder()
        b._bedrock_client = Mock()
        b._bedrock_client.list_provisioned_model_throughputs.return_value = {
            "provisionedModelSummaries": [
                {
                    "provisionedModelName": "my-pt-deployment-v2",
                    "provisionedModelArn": "arn:aws:bedrock:us-west-2:123:provisioned-model/v2",
                },
                {
                    "provisionedModelName": "my-pt-deployment-old",
                    "provisionedModelArn": "arn:aws:bedrock:us-west-2:123:provisioned-model/old",
                },
            ]
        }

        result = b._check_existing_deployment("my-pt-deployment", is_provisioned=True)
        assert result is None

    def test_returns_none_and_logs_warning_on_api_failure(self, caplog):
        b = _builder()
        b._bedrock_client = Mock()
        b._bedrock_client.list_provisioned_model_throughputs.side_effect = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "Not authorized"}},
            "ListProvisionedModelThroughputs",
        )

        import logging
        with caplog.at_level(logging.WARNING):
            result = b._check_existing_deployment("my-pt", is_provisioned=True)

        assert result is None
        assert "Could not check for existing deployment" in caplog.text

    def test_returns_arn_on_exact_od_name_match(self):
        b = _builder()
        b._bedrock_client = Mock()
        b._bedrock_client.list_custom_model_deployments.return_value = {
            "customModelDeploymentSummaries": [
                {
                    "customModelDeploymentName": "my-od-deploy",
                    "customModelDeploymentArn": "arn:aws:bedrock:us-west-2:123:deployment/od",
                },
            ]
        }

        result = b._check_existing_deployment("my-od-deploy", is_provisioned=False)
        assert result == "arn:aws:bedrock:us-west-2:123:deployment/od"

    def test_returns_none_when_no_summaries(self):
        b = _builder()
        b._bedrock_client = Mock()
        b._bedrock_client.list_provisioned_model_throughputs.return_value = {
            "provisionedModelSummaries": []
        }

        result = b._check_existing_deployment("nonexistent", is_provisioned=True)
        assert result is None


# ── _handle_existing_deployment ─────────────────────────────────────────────


class TestHandleExistingDeployment:
    def test_raises_runtime_error_with_fail_if_exists(self):
        from sagemaker.serve.bedrock_target import DeploymentMode

        b = _builder()
        with pytest.raises(RuntimeError, match="already exists") as exc_info:
            b._handle_existing_deployment(
                existing_arn="arn:aws:bedrock:us-west-2:123:provisioned-model/existing",
                deployment_mode=DeploymentMode.FAIL_IF_EXISTS,
                new_model_arn="arn:aws:bedrock:us-west-2:123:custom-model/new",
                deployment_name="my-deployment",
                is_provisioned=True,
            )
        assert "UPDATE_IF_EXISTS" in str(exc_info.value)
        assert "arn:aws:bedrock:us-west-2:123:provisioned-model/existing" in str(exc_info.value)

    def test_calls_update_provisioned_model_throughput_with_update_if_exists(self):
        from sagemaker.serve.bedrock_target import DeploymentMode

        b = _builder()
        b._bedrock_client = Mock()

        result = b._handle_existing_deployment(
            existing_arn="arn:aws:bedrock:us-west-2:123:provisioned-model/existing",
            deployment_mode=DeploymentMode.UPDATE_IF_EXISTS,
            new_model_arn="arn:aws:bedrock:us-west-2:123:custom-model/new",
            deployment_name="my-deployment",
            is_provisioned=True,
        )

        b._bedrock_client.update_provisioned_model_throughput.assert_called_once_with(
            provisionedModelId="arn:aws:bedrock:us-west-2:123:provisioned-model/existing",
            desiredModelId="arn:aws:bedrock:us-west-2:123:custom-model/new",
        )
        assert result["provisionedModelArn"] == "arn:aws:bedrock:us-west-2:123:provisioned-model/existing"
        assert result["customModelArn"] == "arn:aws:bedrock:us-west-2:123:custom-model/new"
        assert result["status"] == "Updating"

    def test_raises_value_error_with_update_if_exists_on_non_pt(self):
        from sagemaker.serve.bedrock_target import DeploymentMode

        b = _builder()
        with pytest.raises(ValueError, match="only supported for Provisioned Throughput"):
            b._handle_existing_deployment(
                existing_arn="arn:aws:bedrock:us-west-2:123:deployment/od",
                deployment_mode=DeploymentMode.UPDATE_IF_EXISTS,
                new_model_arn="arn:aws:bedrock:us-west-2:123:custom-model/new",
                deployment_name="my-od-deployment",
                is_provisioned=False,
            )

    def test_raises_runtime_error_when_update_call_fails(self):
        from sagemaker.serve.bedrock_target import DeploymentMode

        b = _builder()
        b._bedrock_client = Mock()
        b._bedrock_client.update_provisioned_model_throughput.side_effect = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "Model not compatible"}},
            "UpdateProvisionedModelThroughput",
        )

        with pytest.raises(RuntimeError, match="Failed to update deployment 'my-pt'"):
            b._handle_existing_deployment(
                existing_arn="arn:aws:bedrock:us-west-2:123:provisioned-model/existing",
                deployment_mode=DeploymentMode.UPDATE_IF_EXISTS,
                new_model_arn="arn:aws:bedrock:us-west-2:123:custom-model/new",
                deployment_name="my-pt",
                is_provisioned=True,
            )

# ── _create_and_poll_provisioned_throughput ──────────────────────────────────


class TestCreateAndPollProvisionedThroughput:
    def test_returns_result_on_immediate_in_service(self):
        b = _builder()
        b._bedrock_client = Mock()
        b._bedrock_client.create_provisioned_model_throughput.return_value = {
            "provisionedModelArn": "arn:aws:bedrock:us-west-2:123:provisioned-model/my-pt"
        }
        b._bedrock_client.get_provisioned_model_throughput.return_value = {
            "status": "InService"
        }

        result = b._create_and_poll_provisioned_throughput(
            model_arn="arn:aws:bedrock:us-west-2:123:custom-model/m",
            deployment_name="my-pt",
            units=2,
            commitment_duration="OneMonth",
        )

        b._bedrock_client.create_provisioned_model_throughput.assert_called_once_with(
            modelId="arn:aws:bedrock:us-west-2:123:custom-model/m",
            provisionedModelName="my-pt",
            modelUnits=2,
            commitmentDuration="OneMonth",
        )
        assert result["provisionedModelArn"] == "arn:aws:bedrock:us-west-2:123:provisioned-model/my-pt"
        assert result["customModelArn"] == "arn:aws:bedrock:us-west-2:123:custom-model/m"
        assert result["status"] == "InService"

    def test_raises_runtime_error_on_failed_status(self):
        b = _builder()
        b._bedrock_client = Mock()
        b._bedrock_client.create_provisioned_model_throughput.return_value = {
            "provisionedModelArn": "arn:pt"
        }
        b._bedrock_client.get_provisioned_model_throughput.return_value = {
            "status": "Failed",
            "failureMessage": "Capacity unavailable",
        }

        with pytest.raises(RuntimeError, match="Capacity unavailable"):
            b._create_and_poll_provisioned_throughput(
                model_arn="arn:model",
                deployment_name="my-pt",
                units=1,
                commitment_duration=None,
            )

    def test_raises_runtime_error_on_timeout(self):
        b = _builder()
        b._bedrock_client = Mock()
        b._bedrock_client.create_provisioned_model_throughput.return_value = {
            "provisionedModelArn": "arn:pt"
        }
        b._bedrock_client.get_provisioned_model_throughput.return_value = {
            "status": "Creating"
        }

        with patch(f"{MODULE}.time.sleep"):
            with pytest.raises(RuntimeError, match="Timed out"):
                b._create_and_poll_provisioned_throughput(
                    model_arn="arn:model",
                    deployment_name="my-pt",
                    units=1,
                    commitment_duration=None,
                    poll_interval=1,
                    max_wait=2,
                )

    def test_no_commitment_duration_omitted_from_params(self):
        b = _builder()
        b._bedrock_client = Mock()
        b._bedrock_client.create_provisioned_model_throughput.return_value = {
            "provisionedModelArn": "arn:pt"
        }
        b._bedrock_client.get_provisioned_model_throughput.return_value = {
            "status": "InService"
        }

        b._create_and_poll_provisioned_throughput(
            model_arn="arn:model",
            deployment_name="my-pt",
            units=1,
            commitment_duration=None,
        )

        call_kwargs = b._bedrock_client.create_provisioned_model_throughput.call_args[1]
        assert "commitmentDuration" not in call_kwargs

    def test_polls_creating_then_in_service(self):
        b = _builder()
        b._bedrock_client = Mock()
        b._bedrock_client.create_provisioned_model_throughput.return_value = {
            "provisionedModelArn": "arn:pt"
        }
        b._bedrock_client.get_provisioned_model_throughput.side_effect = [
            {"status": "Creating"},
            {"status": "Creating"},
            {"status": "InService"},
        ]

        with patch(f"{MODULE}.time.sleep"):
            result = b._create_and_poll_provisioned_throughput(
                model_arn="arn:model",
                deployment_name="my-pt",
                units=1,
                commitment_duration=None,
                poll_interval=1,
                max_wait=10,
            )

        assert result["status"] == "InService"
        assert b._bedrock_client.get_provisioned_model_throughput.call_count == 3


# ── deploy() with target parameter (PT and OD routing) ──────────────────────


class TestDeployWithTarget:
    @pytest.fixture(autouse=True)
    def _stub_role_validation(self):
        with patch(
            f"{MODULE}.resolve_and_validate_role",
            side_effect=lambda provided_role, **kwargs: provided_role or "auto-role",
        ):
            yield

    def test_provisioned_mode_creates_pt_deployment(self):
        """deploy(target=BedrockTarget(mode="provisioned")) creates PT deployment.

        Validates: Requirements 4.1
        """
        from sagemaker.serve.bedrock_target import BedrockTarget, ProvisionedConfig

        b = _builder()
        b.model_package = None
        b.s3_model_artifacts = "s3://bucket/ckpt/"

        with patch.object(
            b, "_find_or_create_model", return_value="arn:aws:bedrock:us-west-2:123:custom-model/m"
        ) as mock_find, patch.object(
            b, "_check_existing_deployment", return_value=None
        ) as mock_check, patch.object(
            b,
            "_create_and_poll_provisioned_throughput",
            return_value={
                "provisionedModelArn": "arn:aws:bedrock:us-west-2:123:provisioned-model/pt",
                "modelArn": "arn:aws:bedrock:us-west-2:123:provisioned-model/pt",
                "customModelArn": "arn:aws:bedrock:us-west-2:123:custom-model/m",
                "status": "InService",
            },
        ) as mock_create_pt:
            result = b.deploy(
                custom_model_name="my-model",
                role_arn="arn:role",
                target=BedrockTarget(mode="provisioned", config=ProvisionedConfig(units=2)),
            )

        mock_find.assert_called_once()
        mock_check.assert_called_once_with("my-model-pt", is_provisioned=True)
        mock_create_pt.assert_called_once_with(
            model_arn="arn:aws:bedrock:us-west-2:123:custom-model/m",
            deployment_name="my-model-pt",
            units=2,
            commitment_duration=None,
        )
        assert result["provisionedModelArn"] == "arn:aws:bedrock:us-west-2:123:provisioned-model/pt"
        assert result["status"] == "InService"

    def test_on_demand_mode_uses_od_path(self):
        """deploy(target=BedrockTarget(mode="on_demand")) uses OD path.

        Validates: Requirements 4.2
        """
        from sagemaker.serve.bedrock_target import BedrockTarget

        b = _builder()
        b.model_package = None
        b.s3_model_artifacts = "s3://bucket/ckpt/"

        with patch.object(
            b, "_find_or_create_model", return_value="arn:aws:bedrock:us-west-2:123:custom-model/m"
        ), patch.object(
            b, "_check_existing_deployment", return_value=None
        ), patch.object(
            b,
            "create_deployment",
            return_value={"customModelDeploymentArn": "arn:dep"},
        ) as mock_deploy:
            result = b.deploy(
                custom_model_name="my-model",
                role_arn="arn:role",
                target=BedrockTarget(mode="on_demand"),
            )

        mock_deploy.assert_called_once_with(
            model_arn="arn:aws:bedrock:us-west-2:123:custom-model/m",
            deployment_name="my-model-deployment",
        )
        assert result["customModelDeploymentArn"] == "arn:dep"
        assert result["customModelArn"] == "arn:aws:bedrock:us-west-2:123:custom-model/m"

    def test_deploy_without_target_backward_compat(self):
        """deploy() without target maintains backward compat.

        Validates: Requirements 4.3
        """
        c = _make_container(recipe_name="nova-micro")
        b = _builder()
        b.model_package = _make_model_package(c)
        b.s3_model_artifacts = "s3://b/k"
        b._bedrock_client = Mock()
        b._bedrock_client.create_custom_model.return_value = {"modelArn": "arn:m"}
        b._bedrock_client.get_custom_model.return_value = {"modelStatus": "Active"}
        b._bedrock_client.create_custom_model_deployment.return_value = {
            "customModelDeploymentArn": "arn:dep"
        }
        b._bedrock_client.get_custom_model_deployment.return_value = {"status": "Active"}

        result = b.deploy(custom_model_name="m", role_arn="r")

        b._bedrock_client.create_custom_model.assert_called_once()
        b._bedrock_client.create_custom_model_deployment.assert_called_once()
        assert result["customModelDeploymentArn"] == "arn:dep"

    def test_deploy_return_value_contains_custom_model_arn_provisioned(self):
        """Property 7: Deploy Return Value Contains customModelArn (provisioned).

        Validates: Requirements 7.4
        """
        from sagemaker.serve.bedrock_target import BedrockTarget, ProvisionedConfig

        b = _builder()
        b.model_package = None
        b.s3_model_artifacts = "s3://bucket/ckpt/"

        with patch.object(
            b, "_find_or_create_model", return_value="arn:aws:bedrock:us-west-2:123:custom-model/m"
        ), patch.object(
            b, "_check_existing_deployment", return_value=None
        ), patch.object(
            b,
            "_create_and_poll_provisioned_throughput",
            return_value={
                "provisionedModelArn": "arn:pt",
                "modelArn": "arn:pt",
                "customModelArn": "arn:aws:bedrock:us-west-2:123:custom-model/m",
                "status": "InService",
            },
        ):
            result = b.deploy(
                custom_model_name="my-model",
                role_arn="arn:role",
                target=BedrockTarget(mode="provisioned"),
            )

        assert "customModelArn" in result
        assert result["customModelArn"] == "arn:aws:bedrock:us-west-2:123:custom-model/m"

    def test_deploy_return_value_contains_custom_model_arn_on_demand(self):
        """Property 7: Deploy Return Value Contains customModelArn (on_demand).

        Validates: Requirements 7.4
        """
        from sagemaker.serve.bedrock_target import BedrockTarget

        b = _builder()
        b.model_package = None
        b.s3_model_artifacts = "s3://bucket/ckpt/"

        with patch.object(
            b, "_find_or_create_model", return_value="arn:aws:bedrock:us-west-2:123:custom-model/m"
        ), patch.object(
            b, "_check_existing_deployment", return_value=None
        ), patch.object(
            b,
            "create_deployment",
            return_value={"customModelDeploymentArn": "arn:dep"},
        ):
            result = b.deploy(
                custom_model_name="my-model",
                role_arn="arn:role",
                target=BedrockTarget(mode="on_demand"),
            )

        assert "customModelArn" in result
        assert result["customModelArn"] == "arn:aws:bedrock:us-west-2:123:custom-model/m"

    def test_deploy_with_update_if_exists_routes_to_update(self):
        """deploy() with UPDATE_IF_EXISTS routes to update path.

        Validates: Requirements 4.7, 7.3
        """
        from sagemaker.serve.bedrock_target import (
            BedrockTarget,
            DeploymentMode,
            ProvisionedConfig,
        )

        b = _builder()
        b.model_package = None
        b.s3_model_artifacts = "s3://bucket/ckpt/"

        with patch.object(
            b, "_find_or_create_model", return_value="arn:aws:bedrock:us-west-2:123:custom-model/new"
        ), patch.object(
            b,
            "_check_existing_deployment",
            return_value="arn:aws:bedrock:us-west-2:123:provisioned-model/existing",
        ), patch.object(
            b,
            "_handle_existing_deployment",
            return_value={
                "provisionedModelArn": "arn:aws:bedrock:us-west-2:123:provisioned-model/existing",
                "customModelArn": "arn:aws:bedrock:us-west-2:123:custom-model/new",
                "status": "Updating",
            },
        ) as mock_handle:
            result = b.deploy(
                custom_model_name="my-model",
                role_arn="arn:role",
                target=BedrockTarget(
                    mode="provisioned",
                    config=ProvisionedConfig(deployment_mode=DeploymentMode.UPDATE_IF_EXISTS),
                ),
            )

        mock_handle.assert_called_once_with(
            existing_arn="arn:aws:bedrock:us-west-2:123:provisioned-model/existing",
            deployment_mode=DeploymentMode.UPDATE_IF_EXISTS,
            new_model_arn="arn:aws:bedrock:us-west-2:123:custom-model/new",
            deployment_name="my-model-pt",
            is_provisioned=True,
        )
        assert result["status"] == "Updating"
        assert result["customModelArn"] == "arn:aws:bedrock:us-west-2:123:custom-model/new"

    def test_provisioned_mode_with_custom_deployment_name(self):
        """deploy() with target and deployment_name uses the custom name.

        Validates: Requirements 4.1
        """
        from sagemaker.serve.bedrock_target import BedrockTarget

        b = _builder()
        b.model_package = None
        b.s3_model_artifacts = "s3://bucket/ckpt/"

        with patch.object(
            b, "_find_or_create_model", return_value="arn:model"
        ), patch.object(
            b, "_check_existing_deployment", return_value=None
        ) as mock_check, patch.object(
            b,
            "_create_and_poll_provisioned_throughput",
            return_value={
                "provisionedModelArn": "arn:pt",
                "modelArn": "arn:pt",
                "customModelArn": "arn:model",
                "status": "InService",
            },
        ) as mock_create_pt:
            b.deploy(
                custom_model_name="my-model",
                role_arn="arn:role",
                deployment_name="custom-pt-name",
                target=BedrockTarget(mode="provisioned"),
            )

        mock_check.assert_called_once_with("custom-pt-name", is_provisioned=True)
        mock_create_pt.assert_called_once_with(
            model_arn="arn:model",
            deployment_name="custom-pt-name",
            units=1,
            commitment_duration=None,
        )

    def test_missing_custom_model_name_raises_with_target(self):
        """deploy() with target but no custom_model_name raises ValueError.

        Validates: Requirements 4.1
        """
        from sagemaker.serve.bedrock_target import BedrockTarget

        b = _builder()
        b.model_package = None
        b.s3_model_artifacts = "s3://bucket/ckpt/"

        with pytest.raises(ValueError, match="custom_model_name is required"):
            b.deploy(
                role_arn="arn:role",
                target=BedrockTarget(mode="provisioned"),
            )
