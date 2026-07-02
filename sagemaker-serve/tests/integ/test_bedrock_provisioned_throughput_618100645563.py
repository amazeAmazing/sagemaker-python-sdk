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
"""Integration tests for BedrockModelBuilder deploy() with target — account 618100645563.

Tests both Nova (create_custom_model with modelSourceConfig) and OSS (create_model_import_job)
deployment paths using real S3 checkpoints from SageMaker training jobs.

Nova checkpoint:
    s3://customer-escrow-618100645563-smtj-3ff597fc/
        nilis-micro2-sl-dm-1782838144-4668-20260630164905/step_100
    (Amazon Nova Micro 2 fine-tuned via SageMaker ModelTrainer)

OSS checkpoint:
    s3://nova-forge-testing-zhaoqi-618100645563/
        sft-demo-smtj/output/my-gpt-oss-smtj-20260620104027/output/extracted/
    (Uncompressed HF model with checkpoints/hf_merged/ containing safetensors)

Region: us-east-1
Account: 618100645563
"""
from __future__ import absolute_import

import time
import random
import logging
from datetime import datetime, timezone, timedelta

import boto3
import pytest

from sagemaker.core.helper.session_helper import get_execution_role
from sagemaker.core.resources import TrainingJob
from sagemaker.serve.bedrock_model_builder import BedrockModelBuilder
from sagemaker.serve.bedrock_target import BedrockTarget, DeploymentMode, ProvisionedConfig

logger = logging.getLogger(__name__)

AWS_REGION = "us-east-1"

# Nova checkpoint from a completed SageMaker training job (Nova Lite 2 RLVR)
NOVA_CHECKPOINT_URI = (
    "s3://customer-escrow-618100645563-smtj-3ff597fc/"
    "ealynnh-rlvr-nova-lite-2-20260625151656/step_8"
)

# OSS checkpoint (uncompressed HF format with model files under checkpoints/hf_merged/)
OSS_CHECKPOINT_URI = (
    "s3://nova-forge-testing-zhaoqi-618100645563/"
    "sft-demo-smtj/output/my-gpt-oss-smtj-20260620104027/output/extracted/checkpoints/hf_merged/"
)

# Training job that produced the Nova checkpoint
NOVA_TRAINING_JOB_NAME = "ealynnh-rlvr-nova-lite-2-20260625151656"

# Prefix used for all resources created by this test module.
PT_TEST_PREFIX = "test-pt-integ-618-"
# Resources older than this are considered leaked and reaped on setup.
PT_STALE_AGE = timedelta(hours=2)


@pytest.fixture(scope="module")
def role_arn():
    """IAM role ARN with Bedrock permissions (must trust bedrock.amazonaws.com and have S3 access to escrow buckets)."""
    return "arn:aws:iam::618100645563:role/BedrockDeployModelExecutionRole"


@pytest.fixture(scope="module")
def bedrock_client():
    """Create Bedrock client and eagerly reap leaked test provisioned throughputs."""
    client = boto3.client("bedrock", region_name=AWS_REGION)

    try:
        cutoff = datetime.now(timezone.utc) - PT_STALE_AGE
        paginator_token = None
        while True:
            params = {"maxResults": 100}
            if paginator_token:
                params["nextToken"] = paginator_token
            response = client.list_provisioned_model_throughputs(**params)
            for pt in response.get("provisionedModelSummaries", []):
                name = pt.get("provisionedModelName", "")
                if not name.startswith(PT_TEST_PREFIX):
                    continue
                created = pt.get("creationTime")
                if created and created >= cutoff:
                    continue
                if pt.get("status") not in ("InService", "Failed"):
                    continue
                try:
                    logger.info("Eager cleanup of stale PT: %s", name)
                    client.delete_provisioned_model_throughput(
                        provisionedModelId=pt["provisionedModelArn"]
                    )
                except Exception as e:
                    logger.warning("Eager cleanup failed for %s: %s", name, e)
            paginator_token = response.get("nextToken")
            if not paginator_token:
                break
    except Exception as e:
        logger.warning("Failed to list provisioned throughputs for eager cleanup: %s", e)

    return client


@pytest.fixture(scope="module")
def s3_client():
    """Create S3 client."""
    return boto3.client("s3", region_name=AWS_REGION)


@pytest.fixture(scope="module")
def nova_training_job():
    """Get the Nova training job."""
    return TrainingJob.get(
        training_job_name=NOVA_TRAINING_JOB_NAME, region=AWS_REGION
    )


# ── Nova Tests (create_custom_model with modelSourceConfig) ─────────────────


@pytest.mark.serial
@pytest.mark.slow
class TestNovaProvisionedDeployment:
    """Test deploy() with BedrockTarget for Nova models (provisioned throughput).

    Uses a Nova Micro 2 checkpoint from a completed SageMaker training job.
    The deploy(target=provisioned) path calls:
    1. _find_or_create_model → create_custom_model(modelSourceConfig={s3DataSource:...})
    2. _check_existing_deployment
    3. _create_and_poll_provisioned_throughput
    """

    @pytest.fixture(autouse=True)
    def _setup(self, bedrock_client):
        self._bedrock_client = bedrock_client
        self._provisioned_model_arn = None
        self._custom_model_arn = None
        yield
        self._cleanup()

    def _cleanup(self):
        if self._provisioned_model_arn:
            try:
                logger.info("Deleting PT: %s", self._provisioned_model_arn)
                self._bedrock_client.delete_provisioned_model_throughput(
                    provisionedModelId=self._provisioned_model_arn
                )
            except Exception as e:
                logger.warning("Failed to delete PT %s: %s", self._provisioned_model_arn, e)

        if self._custom_model_arn:
            try:
                logger.info("Deleting custom model: %s", self._custom_model_arn)
                self._bedrock_client.delete_custom_model(
                    modelIdentifier=self._custom_model_arn
                )
            except Exception as e:
                logger.warning("Failed to delete custom model %s: %s", self._custom_model_arn, e)

    def test_deploy_nova_provisioned(self, nova_training_job, role_arn, bedrock_client):
        """Deploy Nova checkpoint to provisioned throughput via target-based flow.

        Verifies:
        1. deploy() creates a custom model from the Nova checkpoint
        2. Creates a provisioned throughput deployment
        3. Returns provisionedModelArn, customModelArn, status=InService
        """
        builder = BedrockModelBuilder(model=nova_training_job)
        assert builder.s3_model_artifacts is not None

        suffix = f"{int(time.time())}-{random.randint(1000, 9999)}"
        custom_model_name = f"{PT_TEST_PREFIX}nova-pt-{suffix}"

        target = BedrockTarget(
            mode="provisioned",
            config=ProvisionedConfig(units=1),
        )

        result = builder.deploy(
            custom_model_name=custom_model_name,
            role_arn=role_arn,
            target=target,
        )

        assert "provisionedModelArn" in result, (
            f"Expected 'provisionedModelArn', got keys: {list(result.keys())}"
        )
        assert result["status"] == "InService", (
            f"Expected InService, got '{result.get('status')}'"
        )
        assert "customModelArn" in result

        self._provisioned_model_arn = result["provisionedModelArn"]
        self._custom_model_arn = result.get("customModelArn")

        pt_response = bedrock_client.get_provisioned_model_throughput(
            provisionedModelId=self._provisioned_model_arn
        )
        assert pt_response["status"] == "InService"


@pytest.mark.serial
@pytest.mark.slow
class TestNovaOnDemandDeployment:
    """Test deploy() with BedrockTarget(mode="on_demand") for Nova models."""

    @pytest.fixture(autouse=True)
    def _setup(self, bedrock_client):
        self._bedrock_client = bedrock_client
        self._custom_model_deployment_arn = None
        self._custom_model_arn = None
        yield
        self._cleanup()

    def _cleanup(self):
        if self._custom_model_deployment_arn:
            try:
                logger.info("Deleting deployment: %s", self._custom_model_deployment_arn)
                self._bedrock_client.delete_custom_model_deployment(
                    customModelDeploymentIdentifier=self._custom_model_deployment_arn
                )
            except Exception as e:
                logger.warning("Failed to delete deployment %s: %s", self._custom_model_deployment_arn, e)

        if self._custom_model_arn:
            try:
                logger.info("Deleting custom model: %s", self._custom_model_arn)
                self._bedrock_client.delete_custom_model(
                    modelIdentifier=self._custom_model_arn
                )
            except Exception as e:
                logger.warning("Failed to delete custom model %s: %s", self._custom_model_arn, e)

    def test_deploy_nova_on_demand(self, nova_training_job, role_arn):
        """Deploy Nova checkpoint to on-demand via target-based flow.

        Verifies:
        1. deploy() creates a custom model from the Nova checkpoint
        2. Creates an on-demand deployment
        3. Returns customModelArn in the response
        """
        builder = BedrockModelBuilder(model=nova_training_job)
        assert builder.s3_model_artifacts is not None

        suffix = f"{int(time.time())}-{random.randint(1000, 9999)}"
        custom_model_name = f"{PT_TEST_PREFIX}nova-od-{suffix}"

        result = builder.deploy(
            custom_model_name=custom_model_name,
            role_arn=role_arn,
            target=BedrockTarget(mode="on_demand"),
        )

        assert "customModelArn" in result, (
            f"Expected 'customModelArn', got keys: {list(result.keys())}"
        )
        assert result["customModelArn"], "customModelArn should be non-empty"

        self._custom_model_arn = result["customModelArn"]
        self._custom_model_deployment_arn = result.get("customModelDeploymentArn")


# ── OSS Tests (create_model_import_job) ─────────────────────────────────────


@pytest.mark.serial
@pytest.mark.slow
class TestOssOnDemandDeployment:
    """Test deploy() for OSS models via the import job path.

    Uses an uncompressed HF model checkpoint (GPT-based OSS model) that was
    fine-tuned via SageMaker and extracted to S3.

    OSS models use create_model_import_job (not create_custom_model), so we
    use the legacy deploy path (without target) which correctly routes through
    the import job flow.

    this tests regression ig
    """

    @pytest.fixture(autouse=True)
    def _setup(self, bedrock_client):
        self._bedrock_client = bedrock_client
        self._imported_model_arn = None
        yield
        self._cleanup()

    def _cleanup(self):
        if self._imported_model_arn:
            try:
                logger.info("Deleting imported model: %s", self._imported_model_arn)
                self._bedrock_client.delete_imported_model(
                    modelIdentifier=self._imported_model_arn
                )
            except Exception as e:
                logger.warning("Failed to delete imported model %s: %s", self._imported_model_arn, e)

    def test_deploy_oss_on_demand(self, role_arn):
        """Deploy OSS model checkpoint via import job path.

        Verifies:
        1. deploy() creates a model import job
        2. Polls until import completes
        3. Returns importedModelArn and status=Completed
        """
        builder = BedrockModelBuilder(model=OSS_CHECKPOINT_URI)

        suffix = f"{int(time.time())}-{random.randint(1000, 9999)}"
        job_name = f"{PT_TEST_PREFIX}oss-import-{suffix}"
        imported_model_name = f"{PT_TEST_PREFIX}oss-model-{suffix}"

        result = builder.deploy(
            job_name=job_name,
            imported_model_name=imported_model_name,
            role_arn=role_arn,
        )

        assert result["status"] == "Completed", (
            f"Expected Completed, got {result.get('status')}"
        )
        assert "importedModelArn" in result or "importedModelName" in result

        self._imported_model_arn = result.get("importedModelArn")


@pytest.mark.serial
@pytest.mark.slow
class TestOssProvisionedDeployment:
    """Test OSS model import followed by provisioned throughput creation.

    Uses an uncompressed HF model checkpoint. The flow:
    1. deploy() via legacy path to import the model
    2. create_provisioned_throughput() on the imported model
    """

    @pytest.fixture(autouse=True)
    def _setup(self, bedrock_client):
        self._bedrock_client = bedrock_client
        self._provisioned_model_arn = None
        self._imported_model_arn = None
        yield
        self._cleanup()

    def _cleanup(self):
        if self._provisioned_model_arn:
            try:
                logger.info("Deleting PT: %s", self._provisioned_model_arn)
                self._bedrock_client.delete_provisioned_model_throughput(
                    provisionedModelId=self._provisioned_model_arn
                )
            except Exception as e:
                logger.warning("Failed to delete PT %s: %s", self._provisioned_model_arn, e)

        if self._imported_model_arn:
            try:
                logger.info("Deleting imported model: %s", self._imported_model_arn)
                self._bedrock_client.delete_imported_model(
                    modelIdentifier=self._imported_model_arn
                )
            except Exception as e:
                logger.warning("Failed to delete imported model %s: %s", self._imported_model_arn, e)

    def test_deploy_oss_provisioned(self, role_arn, bedrock_client):
        """Import OSS model, then create provisioned throughput.

        Verifies:
        1. deploy() imports the OSS model via create_model_import_job
        2. create_provisioned_throughput() on the imported model succeeds
        3. PT reaches InService status
        """
        builder = BedrockModelBuilder(model=OSS_CHECKPOINT_URI)

        suffix = f"{int(time.time())}-{random.randint(1000, 9999)}"
        job_name = f"{PT_TEST_PREFIX}oss-pt-import-{suffix}"
        imported_model_name = f"{PT_TEST_PREFIX}oss-pt-model-{suffix}"

        import_result = builder.deploy(
            job_name=job_name,
            imported_model_name=imported_model_name,
            role_arn=role_arn,
        )

        assert import_result["status"] == "Completed", (
            f"Expected import Completed, got {import_result.get('status')}"
        )
        imported_model_arn = import_result.get("importedModelArn")
        assert imported_model_arn, "Expected importedModelArn in result"
        self._imported_model_arn = imported_model_arn

        pt_name = f"{PT_TEST_PREFIX}oss-pt-{suffix}"
        pt_result = builder.create_provisioned_throughput(
            model_id=imported_model_arn,
            provisioned_model_name=pt_name,
            model_units=1,
        )

        assert "provisionedModelArn" in pt_result, (
            f"Expected 'provisionedModelArn', got keys: {list(pt_result.keys())}"
        )
        self._provisioned_model_arn = pt_result["provisionedModelArn"]

        pt_response = bedrock_client.get_provisioned_model_throughput(
            provisionedModelId=self._provisioned_model_arn
        )
        assert pt_response["status"] == "InService"


# ── Escrow Reuse (Nova) ─────────────────────────────────────────────────────


@pytest.mark.serial
@pytest.mark.slow
class TestNovaEscrowModelReuse:
    """Test escrow-based model reuse with Nova checkpoints.

    Deploys the same Nova training job artifacts twice and verifies the second
    deploy reuses the custom model created by the first (same customModelArn).
    """

    @pytest.fixture(autouse=True)
    def _setup(self, bedrock_client):
        self._bedrock_client = bedrock_client
        self._custom_model_deployment_arns = []
        self._custom_model_arns = []
        yield
        self._cleanup()

    def _cleanup(self):
        for arn in self._custom_model_deployment_arns:
            try:
                logger.info("Deleting deployment: %s", arn)
                self._bedrock_client.delete_custom_model_deployment(
                    customModelDeploymentIdentifier=arn
                )
            except Exception as e:
                logger.warning("Failed to delete deployment %s: %s", arn, e)

        for arn in self._custom_model_arns:
            try:
                logger.info("Deleting custom model: %s", arn)
                self._bedrock_client.delete_custom_model(modelIdentifier=arn)
            except Exception as e:
                logger.warning("Failed to delete custom model %s: %s", arn, e)

    def test_nova_escrow_reuse(self, nova_training_job, role_arn):
        """Deploy same Nova source twice, verify second reuses existing model."""
        builder_1 = BedrockModelBuilder(model=nova_training_job)
        assert builder_1.s3_model_artifacts is not None

        suffix = f"{int(time.time())}-{random.randint(1000, 9999)}"

        result_1 = builder_1.deploy(
            custom_model_name=f"{PT_TEST_PREFIX}nova-reuse-1-{suffix}",
            role_arn=role_arn,
            target=BedrockTarget(mode="on_demand"),
        )

        assert "customModelArn" in result_1
        first_model_arn = result_1["customModelArn"]
        self._custom_model_arns.append(first_model_arn)
        if result_1.get("customModelDeploymentArn"):
            self._custom_model_deployment_arns.append(result_1["customModelDeploymentArn"])

        builder_2 = BedrockModelBuilder(model=nova_training_job)
        result_2 = builder_2.deploy(
            custom_model_name=f"{PT_TEST_PREFIX}nova-reuse-2-{suffix}",
            role_arn=role_arn,
            target=BedrockTarget(mode="on_demand"),
        )

        assert "customModelArn" in result_2
        second_model_arn = result_2["customModelArn"]
        if result_2.get("customModelDeploymentArn"):
            self._custom_model_deployment_arns.append(result_2["customModelDeploymentArn"])
        if second_model_arn not in self._custom_model_arns:
            self._custom_model_arns.append(second_model_arn)

        assert first_model_arn == second_model_arn, (
            f"Expected model reuse. First: {first_model_arn}, Second: {second_model_arn}"
        )


# ── UPDATE_IF_EXISTS (Nova) ─────────────────────────────────────────────────


@pytest.mark.serial
@pytest.mark.slow
class TestNovaUpdateIfExists:
    """Test UPDATE_IF_EXISTS with Nova provisioned throughput deployment."""

    @pytest.fixture(autouse=True)
    def _setup(self, bedrock_client):
        self._bedrock_client = bedrock_client
        self._provisioned_model_arn = None
        self._custom_model_arn = None
        yield
        self._cleanup()

    def _cleanup(self):
        if self._provisioned_model_arn:
            try:
                logger.info("Deleting PT: %s", self._provisioned_model_arn)
                self._bedrock_client.delete_provisioned_model_throughput(
                    provisionedModelId=self._provisioned_model_arn
                )
            except Exception as e:
                logger.warning("Failed to delete PT %s: %s", self._provisioned_model_arn, e)

        if self._custom_model_arn:
            try:
                logger.info("Deleting custom model: %s", self._custom_model_arn)
                self._bedrock_client.delete_custom_model(modelIdentifier=self._custom_model_arn)
            except Exception as e:
                logger.warning("Failed to delete custom model %s: %s", self._custom_model_arn, e)

    def test_nova_update_if_exists(self, nova_training_job, role_arn, bedrock_client):
        """Create Nova PT, then update in-place with UPDATE_IF_EXISTS."""
        builder = BedrockModelBuilder(model=nova_training_job)
        assert builder.s3_model_artifacts is not None

        suffix = f"{int(time.time())}-{random.randint(1000, 9999)}"
        custom_model_name = f"{PT_TEST_PREFIX}nova-update-{suffix}"
        deployment_name = f"{PT_TEST_PREFIX}nova-update-pt-{suffix}"

        first_result = builder.deploy(
            custom_model_name=custom_model_name,
            role_arn=role_arn,
            target=BedrockTarget(mode="provisioned", config=ProvisionedConfig(units=1)),
            deployment_name=deployment_name,
        )

        assert "provisionedModelArn" in first_result
        assert first_result["status"] == "InService"
        first_pt_arn = first_result["provisionedModelArn"]
        self._provisioned_model_arn = first_pt_arn
        self._custom_model_arn = first_result.get("customModelArn")

        second_result = builder.deploy(
            custom_model_name=custom_model_name,
            role_arn=role_arn,
            target=BedrockTarget(
                mode="provisioned",
                config=ProvisionedConfig(
                    units=1,
                    deployment_mode=DeploymentMode.UPDATE_IF_EXISTS,
                ),
            ),
            deployment_name=deployment_name,
        )

        assert "provisionedModelArn" in second_result
        assert second_result["provisionedModelArn"] == first_pt_arn, (
            f"Expected in-place update (same ARN). "
            f"First: {first_pt_arn}, Second: {second_result['provisionedModelArn']}"
        )
        assert second_result.get("status"), "Expected valid status in update result"
