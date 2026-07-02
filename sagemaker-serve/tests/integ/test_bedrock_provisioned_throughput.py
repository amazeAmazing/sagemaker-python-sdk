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
"""Integration tests for BedrockModelBuilder import job polling and provisioned throughput."""
from __future__ import absolute_import

import json
import time
import random
import logging
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse

import boto3
import pytest

from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core.resources import TrainingJob
from sagemaker.serve.bedrock_model_builder import BedrockModelBuilder
from sagemaker.serve.bedrock_target import BedrockTarget, DeploymentMode, ProvisionedConfig

logger = logging.getLogger(__name__)

AWS_REGION = "us-west-2"


@pytest.fixture(scope="module")
def training_job_name():
    """Training job name for testing (OSS model)."""
    return "meta-textgeneration-llama-3-2-1b-instruct-sft-20251201172445"


@pytest.fixture(scope="module")
def role_arn():
    """IAM role ARN with Bedrock permissions."""
    return get_execution_role()


# Prefix used for all provisioned throughputs created by this test module.
PT_TEST_PREFIX = "test-pt-integ-"
# Provisioned throughputs older than this are considered leaked and reaped on setup.
PT_STALE_AGE = timedelta(hours=2)


@pytest.fixture(scope="module")
def bedrock_client():
    """Create Bedrock client and eagerly reap leaked test provisioned throughputs.

    Provisioned throughputs cost money and consume a small, easily-exhausted
    model-unit quota. A test process killed before its teardown runs (CodeBuild
    timeout, worker crash, etc.) leaks its PT, and these accumulate across runs
    until the quota is full and CreateProvisionedModelThroughput starts failing.

    To stay self-healing, on setup we delete any ``test-pt-integ-*`` PT older
    than PT_STALE_AGE. The age guard avoids racing a PT that another concurrent
    run just created.
    """
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
                # Only InService/Failed PTs can be deleted.
                if pt.get("status") not in ("InService", "Failed"):
                    continue
                try:
                    logger.info("Eager cleanup of stale provisioned throughput: %s", name)
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
def training_job(training_job_name):
    """Get the training job."""
    return TrainingJob.get(
        training_job_name=training_job_name, region=AWS_REGION
    )


def _setup_model_files(s3_artifacts_uri, s3_client):
    """Setup required model files for Bedrock deployment.

    Bedrock model import requires HuggingFace-format files (config.json,
    tokenizer.json, etc.) at the root of the S3 model artifacts path.
    Training jobs often store these under checkpoints/hf_merged/, so we
    copy them to the expected location.

    Args:
        s3_artifacts_uri: The S3 URI that BedrockModelBuilder will use for import.
        s3_client: boto3 S3 client.
    """
    parsed = urlparse(s3_artifacts_uri)
    bucket = parsed.netloc
    base_prefix = parsed.path.lstrip("/").rstrip("/")

    hf_merged_prefix = f"{base_prefix}/checkpoints/hf_merged/"
    root_prefix = f"{base_prefix}/"

    files_to_copy = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors",
    ]

    for file in files_to_copy:
        try:
            s3_client.head_object(Bucket=bucket, Key=root_prefix + file)
            logger.info("File already exists: s3://%s/%s%s", bucket, root_prefix, file)
        except Exception:
            try:
                s3_client.copy_object(
                    Bucket=bucket,
                    CopySource={"Bucket": bucket, "Key": hf_merged_prefix + file},
                    Key=root_prefix + file,
                )
                logger.info("Copied %s to root", file)
            except Exception as e:
                logger.warning("Could not copy %s: %s", file, e)

    try:
        s3_client.head_object(Bucket=bucket, Key=root_prefix + "added_tokens.json")
    except Exception:
        try:
            s3_client.put_object(
                Bucket=bucket,
                Key=root_prefix + "added_tokens.json",
                Body=json.dumps({}),
                ContentType="application/json",
            )
            logger.info("Created added_tokens.json")
        except Exception as e:
            logger.warning("Could not create added_tokens.json: %s", e)


@pytest.mark.serial
@pytest.mark.import_model
class TestBedrockImportJobPolling:
    """Test import job polling for OSS models (Option C: deploy only waits for import)."""

    @pytest.fixture(autouse=True)
    def _setup(self, bedrock_client):
        """Store bedrock client and track resources for cleanup."""
        self._bedrock_client = bedrock_client
        self._imported_model_arn = None
        yield
        self._cleanup()

    def _cleanup(self):
        """Clean up Bedrock resources created during the test."""
        if self._imported_model_arn:
            try:
                logger.info("Deleting imported model: %s", self._imported_model_arn)
                self._bedrock_client.delete_imported_model(
                    modelIdentifier=self._imported_model_arn
                )
            except Exception as e:
                logger.warning("Failed to delete imported model: %s", e)

    @pytest.mark.slow
    def test_deploy_oss_model_waits_for_import_completion(
        self, training_job, role_arn, bedrock_client, s3_client
    ):
        """Test that deploy() waits for import job to complete and returns job details.

        This test verifies that BedrockModelBuilder.deploy() for OSS models:
        1. Creates a model import job
        2. Polls until the import job reaches Completed status
        3. Returns the completed job details (model is ready for on-demand invoke)
        4. Does NOT create provisioned throughput
        """
        builder = BedrockModelBuilder(model=training_job)
        assert builder.s3_model_artifacts is not None

        _setup_model_files(builder.s3_model_artifacts, s3_client)

        suffix = f"{int(time.time())}-{random.randint(1000, 9999)}"
        job_name = f"test-import-poll-{suffix}"
        imported_model_name = f"test-import-model-{suffix}"

        result = builder.deploy(
            job_name=job_name,
            imported_model_name=imported_model_name,
            role_arn=role_arn,
        )

        # Verify the result is the completed job details
        assert result["status"] == "Completed", (
            f"Expected Completed, got {result.get('status')}"
        )
        assert "importedModelName" in result
        assert "importedModelArn" in result or "jobArn" in result

        # Track for cleanup
        self._imported_model_arn = result.get("importedModelArn")

        # Verify model can be found (it exists and is ready)
        models = bedrock_client.list_imported_models()
        model_names = [m["modelName"] for m in models.get("modelSummaries", [])]
        assert imported_model_name in model_names


@pytest.mark.serial
@pytest.mark.import_model
class TestBedrockProvisionedThroughput:
    """Test create_provisioned_throughput as a standalone method.

    Uses a pre-existing Bedrock custom model (fine-tuned Llama 3.1 8B) to test
    provisioned throughput creation and polling. The custom model was created via
    Bedrock CreateModelCustomizationJob and persists in the CI account.

    Prerequisites:
        - Account 729646638167, us-west-2
        - PT MU quota for Llama 3.1 8B (requested via Matador/Bedrock team)
        - A pre-existing custom model (see below for how to recreate)

    How to recreate the custom model if it gets deleted:

        1. Ensure training data exists at:
           s3://mc-flows-sdk-testing/pt-test-data/train_llama31.jsonl

           If not, create it (minimal JSONL with prompt/completion pairs):
               echo '{"prompt":"What is ML?","completion":"ML is a subset of AI."}' > /tmp/train.jsonl
               aws s3 cp /tmp/train.jsonl s3://mc-flows-sdk-testing/pt-test-data/train_llama31.jsonl

        2. Create the fine-tuning job:
               aws bedrock create-model-customization-job \\
                   --job-name test-llama31-8b-pt-integ \\
                   --custom-model-name test-llama31-8b-pt-model \\
                   --role-arn arn:aws:iam::729646638167:role/Admin \\
                   --base-model-identifier meta.llama3-1-8b-instruct-v1:0:128k \\
                   --customization-type FINE_TUNING \\
                   --training-data-config '{"s3Uri":"s3://mc-flows-sdk-testing/pt-test-data/train_llama31.jsonl"}' \\
                   --output-data-config '{"s3Uri":"s3://mc-flows-sdk-testing/pt-test-output/"}' \\
                   --hyper-parameters '{"epochCount":"1","batchSize":"1","learningRate":"0.00001"}' \\
                   --region us-west-2

        3. Wait for the job to complete (~2-4 hours for 8B model):
               aws bedrock get-model-customization-job \\
                   --job-identifier <job-arn> --region us-west-2 \\
                   --query "status"

        4. Update CUSTOM_MODEL_ARN below with the outputModelArn from the job.
    """

    # Pre-existing custom model created via Bedrock fine-tuning.
    # Base model: meta.llama3-1-8b-instruct-v1:0:128k
    # This model must exist in account 729646638167, us-west-2.
    CUSTOM_MODEL_ARN = (
        "arn:aws:bedrock:us-west-2:729646638167:custom-model/"
        "meta.llama3-1-8b-instruct-v1:0:128k/k2mjykwgn62p"
    )
    CUSTOM_MODEL_NAME = "test-llama31-8b-pt-model"

    @pytest.fixture(autouse=True)
    def _setup(self, bedrock_client):
        """Store bedrock client and track resources for cleanup."""
        self._bedrock_client = bedrock_client
        self._provisioned_model_arn = None
        yield
        # Always clean up PT, even if test fails
        self._cleanup()

    def _cleanup(self):
        """Clean up provisioned throughput created during the test."""
        if self._provisioned_model_arn:
            try:
                logger.info("Deleting provisioned throughput: %s", self._provisioned_model_arn)
                self._bedrock_client.delete_provisioned_model_throughput(
                    provisionedModelId=self._provisioned_model_arn
                )
                logger.info("Provisioned throughput deleted successfully.")
            except Exception as e:
                logger.warning("Failed to delete provisioned throughput: %s", e)

    @pytest.mark.slow
    def test_create_provisioned_throughput(self, bedrock_client):
        """Test create_provisioned_throughput() with a pre-existing custom model.

        This test verifies:
        1. Calls CreateProvisionedModelThroughput with a custom model ARN
        2. Polls until provisioned throughput reaches InService
        3. Returns the provisioned throughput response
        4. Cleans up the PT after the test
        """
        # Check if the pre-existing custom model exists
        try:
            bedrock_client.get_custom_model(modelIdentifier=self.CUSTOM_MODEL_ARN)
        except Exception:
            pytest.skip(
                f"Pre-existing custom model not found: {self.CUSTOM_MODEL_ARN}. "
                f"Recreate it with: aws bedrock create-model-customization-job "
                f"--job-name test-llama31-8b-pt-integ "
                f"--custom-model-name {self.CUSTOM_MODEL_NAME} "
                f"--role-arn <role> "
                f"--base-model-identifier meta.llama3-1-8b-instruct-v1:0:128k "
                f"--customization-type FINE_TUNING "
                f"--training-data-config '{{\"s3Uri\":\"s3://mc-flows-sdk-testing/pt-test-data/train_llama31.jsonl\"}}' "
                f"--output-data-config '{{\"s3Uri\":\"s3://mc-flows-sdk-testing/pt-test-output/\"}}' "
                f"--hyper-parameters '{{\"epochCount\":\"1\",\"batchSize\":\"1\",\"learningRate\":\"0.00001\"}}' "
                f"--region us-west-2"
            )

        suffix = f"{int(time.time())}-{random.randint(1000, 9999)}"
        provisioned_model_name = f"test-pt-integ-{suffix}"

        builder = BedrockModelBuilder(model=None)

        # Create provisioned throughput
        pt_result = builder.create_provisioned_throughput(
            model_id=self.CUSTOM_MODEL_ARN,
            provisioned_model_name=provisioned_model_name,
            model_units=1,
        )

        # Verify result contains provisioned model ARN
        assert "provisionedModelArn" in pt_result, (
            f"Expected 'provisionedModelArn' in result, got keys: {list(pt_result.keys())}"
        )
        self._provisioned_model_arn = pt_result["provisionedModelArn"]

        # Verify provisioned throughput is InService (create_provisioned_throughput
        # already polls until InService, but double-check)
        pt_response = bedrock_client.get_provisioned_model_throughput(
            provisionedModelId=self._provisioned_model_arn
        )
        assert pt_response["status"] == "InService", (
            f"Expected InService, got {pt_response['status']}"
        )


@pytest.mark.serial
@pytest.mark.slow
class TestBedrockTargetOnDemandDeployment:
    """Test deploy() with BedrockTarget(mode="on_demand").

    Exercises the target-based on-demand deployment flow using a training job
    model source. Verifies that the response includes customModelArn.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, bedrock_client):
        """Track resources for cleanup."""
        self._bedrock_client = bedrock_client
        self._custom_model_deployment_arn = None
        self._custom_model_arn = None
        yield
        self._cleanup()

    def _cleanup(self):
        """Delete deployment first, then custom model."""
        if self._custom_model_deployment_arn:
            try:
                logger.info(
                    "Deleting custom model deployment: %s",
                    self._custom_model_deployment_arn,
                )
                self._bedrock_client.delete_custom_model_deployment(
                    customModelDeploymentIdentifier=self._custom_model_deployment_arn
                )
            except Exception as e:
                logger.warning("Failed to delete custom model deployment: %s", e)

        if self._custom_model_arn:
            try:
                logger.info("Deleting custom model: %s", self._custom_model_arn)
                self._bedrock_client.delete_custom_model(
                    modelIdentifier=self._custom_model_arn
                )
            except Exception as e:
                logger.warning("Failed to delete custom model: %s", e)

    def test_deploy_on_demand_with_bedrock_target(
        self, training_job, role_arn, bedrock_client, s3_client
    ):
        """Test deploy(target=BedrockTarget(mode="on_demand")) creates OD deployment.

        Verifies:
        1. deploy() with on_demand target creates a custom model and deployment
        2. Response contains customModelArn
        3. Resources are cleaned up after test
        """
        builder = BedrockModelBuilder(model=training_job)
        assert builder.s3_model_artifacts is not None

        _setup_model_files(builder.s3_model_artifacts, s3_client)

        suffix = f"{int(time.time())}-{random.randint(1000, 9999)}"
        custom_model_name = f"test-pt-integ-od-{suffix}"

        result = builder.deploy(
            custom_model_name=custom_model_name,
            role_arn=role_arn,
            target=BedrockTarget(mode="on_demand"),
        )

        assert "customModelArn" in result, (
            f"Expected 'customModelArn' in result, got keys: {list(result.keys())}"
        )
        assert result["customModelArn"], "customModelArn should be a non-empty string"

        self._custom_model_arn = result["customModelArn"]
        self._custom_model_deployment_arn = result.get("customModelDeploymentArn")


@pytest.mark.slow
@pytest.mark.serial
class TestBedrockTargetProvisionedDeployment:
    """Test deploy() with BedrockTarget(mode="provisioned") using the target-based flow.

    Uses the same pre-existing Bedrock custom model as TestBedrockProvisionedThroughput.
    This test exercises the integrated deploy(target=...) path rather than the
    standalone create_provisioned_throughput() method.

    The deploy() call with a target triggers:
    1. _find_or_create_model (escrow lookup or model creation)
    2. _check_existing_deployment (PT name collision check)
    3. _create_and_poll_provisioned_throughput (PT creation + polling)
    """

    CUSTOM_MODEL_ARN = (
        "arn:aws:bedrock:us-west-2:729646638167:custom-model/"
        "meta.llama3-1-8b-instruct-v1:0:128k/k2mjykwgn62p"
    )

    @pytest.fixture(autouse=True)
    def _setup(self, bedrock_client):
        """Store bedrock client and track resources for cleanup."""
        self._bedrock_client = bedrock_client
        self._provisioned_model_arn = None
        self._custom_model_arn = None
        yield
        self._cleanup()

    def _cleanup(self):
        """Clean up resources: PT first, then custom models."""
        if self._provisioned_model_arn:
            try:
                logger.info(
                    "Deleting provisioned throughput: %s", self._provisioned_model_arn
                )
                self._bedrock_client.delete_provisioned_model_throughput(
                    provisionedModelId=self._provisioned_model_arn
                )
            except Exception as e:
                logger.warning(
                    "Failed to delete provisioned throughput %s: %s",
                    self._provisioned_model_arn,
                    e,
                )

        if self._custom_model_arn:
            try:
                logger.info("Deleting custom model: %s", self._custom_model_arn)
                self._bedrock_client.delete_custom_model(
                    modelIdentifier=self._custom_model_arn
                )
            except Exception as e:
                logger.warning(
                    "Failed to delete custom model %s: %s",
                    self._custom_model_arn,
                    e,
                )

    def test_deploy_with_provisioned_target(
        self, training_job, role_arn, bedrock_client, s3_client
    ):
        """Test deploy(target=BedrockTarget(mode="provisioned")) end-to-end.

        Verifies:
        1. deploy() with a provisioned target creates a custom model and PT
        2. Returns provisionedModelArn in the result
        3. Returns status="InService" after polling completes
        4. Returns customModelArn referencing the deployed model
        """
        try:
            bedrock_client.get_custom_model(modelIdentifier=self.CUSTOM_MODEL_ARN)
        except Exception:
            pytest.skip(
                f"Pre-existing custom model not found: {self.CUSTOM_MODEL_ARN}. "
                "See TestBedrockProvisionedThroughput docstring for recreation steps."
            )

        builder = BedrockModelBuilder(model=training_job)
        assert builder.s3_model_artifacts is not None

        _setup_model_files(builder.s3_model_artifacts, s3_client)

        suffix = f"{int(time.time())}-{random.randint(1000, 9999)}"
        custom_model_name = f"test-pt-integ-model-{suffix}"

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
            f"Expected 'provisionedModelArn' in result, got keys: {list(result.keys())}"
        )
        assert result["status"] == "InService", (
            f"Expected status 'InService', got '{result.get('status')}'"
        )
        assert "customModelArn" in result, (
            f"Expected 'customModelArn' in result, got keys: {list(result.keys())}"
        )

        self._provisioned_model_arn = result["provisionedModelArn"]
        self._custom_model_arn = result.get("customModelArn")

        pt_response = bedrock_client.get_provisioned_model_throughput(
            provisionedModelId=self._provisioned_model_arn
        )
        assert pt_response["status"] == "InService"


@pytest.mark.slow
@pytest.mark.serial
class TestBedrockTargetEscrowModelReuse:
    """Test that deploying the same S3 artifacts twice reuses the existing custom model.

    Exercises the escrow URI tag-based model reuse logic by deploying the same
    training job artifacts with on-demand target twice. The second deploy should
    find the model created by the first deploy via ResourceGroupsTaggingAPI and
    return the same customModelArn without creating a new model.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, bedrock_client):
        """Track resources for cleanup."""
        self._bedrock_client = bedrock_client
        self._custom_model_deployment_arns = []
        self._custom_model_arns = []
        yield
        self._cleanup()

    def _cleanup(self):
        """Delete deployments first, then custom models."""
        for deployment_arn in self._custom_model_deployment_arns:
            try:
                logger.info(
                    "Deleting custom model deployment: %s", deployment_arn
                )
                self._bedrock_client.delete_custom_model_deployment(
                    customModelDeploymentIdentifier=deployment_arn
                )
            except Exception as e:
                logger.warning(
                    "Failed to delete custom model deployment %s: %s",
                    deployment_arn,
                    e,
                )

        for model_arn in self._custom_model_arns:
            try:
                logger.info("Deleting custom model: %s", model_arn)
                self._bedrock_client.delete_custom_model(
                    modelIdentifier=model_arn
                )
            except Exception as e:
                logger.warning(
                    "Failed to delete custom model %s: %s", model_arn, e
                )

    def test_escrow_model_reuse_same_artifacts(
        self, training_job, role_arn, bedrock_client, s3_client
    ):
        """Test that deploying the same source twice reuses the custom model.

        Verifies:
        1. First deploy creates a custom model with an escrow tag
        2. Second deploy finds the existing model via escrow lookup
        3. Both deploys return the same customModelArn
        """
        builder_1 = BedrockModelBuilder(model=training_job)
        assert builder_1.s3_model_artifacts is not None

        _setup_model_files(builder_1.s3_model_artifacts, s3_client)

        suffix = f"{int(time.time())}-{random.randint(1000, 9999)}"
        custom_model_name_1 = f"test-pt-integ-escrow-1-{suffix}"

        result_1 = builder_1.deploy(
            custom_model_name=custom_model_name_1,
            role_arn=role_arn,
            target=BedrockTarget(mode="on_demand"),
        )

        assert "customModelArn" in result_1, (
            f"Expected 'customModelArn' in first result, got keys: "
            f"{list(result_1.keys())}"
        )
        first_model_arn = result_1["customModelArn"]
        assert first_model_arn, "First customModelArn should be non-empty"

        self._custom_model_arns.append(first_model_arn)
        if result_1.get("customModelDeploymentArn"):
            self._custom_model_deployment_arns.append(
                result_1["customModelDeploymentArn"]
            )

        # Second deploy with a new builder pointing to same source
        builder_2 = BedrockModelBuilder(model=training_job)
        custom_model_name_2 = f"test-pt-integ-escrow-2-{suffix}"

        result_2 = builder_2.deploy(
            custom_model_name=custom_model_name_2,
            role_arn=role_arn,
            target=BedrockTarget(mode="on_demand"),
        )

        assert "customModelArn" in result_2, (
            f"Expected 'customModelArn' in second result, got keys: "
            f"{list(result_2.keys())}"
        )
        second_model_arn = result_2["customModelArn"]
        assert second_model_arn, "Second customModelArn should be non-empty"

        if result_2.get("customModelDeploymentArn"):
            self._custom_model_deployment_arns.append(
                result_2["customModelDeploymentArn"]
            )
        # Track second model ARN only if different (for cleanup safety)
        if second_model_arn not in self._custom_model_arns:
            self._custom_model_arns.append(second_model_arn)

        assert first_model_arn == second_model_arn, (
            f"Expected second deploy to reuse model from first deploy. "
            f"First ARN: {first_model_arn}, Second ARN: {second_model_arn}"
        )


@pytest.mark.slow
@pytest.mark.serial
class TestBedrockTargetUpdateIfExists:
    """Test deploy() with UPDATE_IF_EXISTS deployment mode.

    Creates a PT deployment, then deploys again to the same deployment name with
    UPDATE_IF_EXISTS mode, verifying the existing PT is updated in-place rather
    than recreated.

    Uses the same pre-existing Bedrock custom model as TestBedrockProvisionedThroughput.
    """

    CUSTOM_MODEL_ARN = (
        "arn:aws:bedrock:us-west-2:729646638167:custom-model/"
        "meta.llama3-1-8b-instruct-v1:0:128k/k2mjykwgn62p"
    )

    @pytest.fixture(autouse=True)
    def _setup(self, bedrock_client):
        """Store bedrock client and track resources for cleanup."""
        self._bedrock_client = bedrock_client
        self._provisioned_model_arn = None
        self._custom_model_arn = None
        yield
        self._cleanup()

    def _cleanup(self):
        """Clean up resources: PT first, then custom models."""
        if self._provisioned_model_arn:
            try:
                logger.info(
                    "Deleting provisioned throughput: %s", self._provisioned_model_arn
                )
                self._bedrock_client.delete_provisioned_model_throughput(
                    provisionedModelId=self._provisioned_model_arn
                )
            except Exception as e:
                logger.warning(
                    "Failed to delete provisioned throughput %s: %s",
                    self._provisioned_model_arn,
                    e,
                )

        if self._custom_model_arn:
            try:
                logger.info("Deleting custom model: %s", self._custom_model_arn)
                self._bedrock_client.delete_custom_model(
                    modelIdentifier=self._custom_model_arn
                )
            except Exception as e:
                logger.warning(
                    "Failed to delete custom model %s: %s",
                    self._custom_model_arn,
                    e,
                )

    def test_deploy_update_if_exists(
        self, training_job, role_arn, bedrock_client, s3_client
    ):
        """Test UPDATE_IF_EXISTS updates an existing PT deployment in-place.

        Verifies:
        1. First deploy creates a PT and returns provisionedModelArn
        2. Second deploy to the same deployment name with UPDATE_IF_EXISTS
           returns the same provisionedModelArn (in-place update, not recreated)
        3. Second response has a valid status
        """
        try:
            bedrock_client.get_custom_model(modelIdentifier=self.CUSTOM_MODEL_ARN)
        except Exception:
            pytest.skip(
                f"Pre-existing custom model not found: {self.CUSTOM_MODEL_ARN}. "
                "See TestBedrockProvisionedThroughput docstring for recreation steps."
            )

        builder = BedrockModelBuilder(model=training_job)
        assert builder.s3_model_artifacts is not None

        _setup_model_files(builder.s3_model_artifacts, s3_client)

        suffix = f"{int(time.time())}-{random.randint(1000, 9999)}"
        custom_model_name = f"test-pt-integ-update-{suffix}"
        deployment_name = f"{PT_TEST_PREFIX}update-{suffix}"

        # First deploy: create a new PT
        first_target = BedrockTarget(
            mode="provisioned",
            config=ProvisionedConfig(units=1),
        )
        first_result = builder.deploy(
            custom_model_name=custom_model_name,
            role_arn=role_arn,
            target=first_target,
            deployment_name=deployment_name,
        )

        assert "provisionedModelArn" in first_result, (
            f"Expected 'provisionedModelArn' in first result, got keys: "
            f"{list(first_result.keys())}"
        )
        assert first_result["status"] == "InService", (
            f"Expected first deploy status 'InService', got '{first_result.get('status')}'"
        )

        first_pt_arn = first_result["provisionedModelArn"]
        self._provisioned_model_arn = first_pt_arn
        self._custom_model_arn = first_result.get("customModelArn")

        # Second deploy: update the same PT in-place with UPDATE_IF_EXISTS
        second_target = BedrockTarget(
            mode="provisioned",
            config=ProvisionedConfig(
                units=1,
                deployment_mode=DeploymentMode.UPDATE_IF_EXISTS,
            ),
        )
        second_result = builder.deploy(
            custom_model_name=custom_model_name,
            role_arn=role_arn,
            target=second_target,
            deployment_name=deployment_name,
        )

        assert "provisionedModelArn" in second_result, (
            f"Expected 'provisionedModelArn' in second result, got keys: "
            f"{list(second_result.keys())}"
        )
        assert second_result["provisionedModelArn"] == first_pt_arn, (
            f"Expected same PT ARN (in-place update), got "
            f"first={first_pt_arn}, second={second_result['provisionedModelArn']}"
        )
        assert second_result.get("status"), (
            "Expected a valid status in second deploy result"
        )
