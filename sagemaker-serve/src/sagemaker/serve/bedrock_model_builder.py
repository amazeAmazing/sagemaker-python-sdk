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
"""Holds the BedrockModelBuilder class."""
from __future__ import absolute_import

import json
import os
import time
import logging
from datetime import datetime, timezone

from sagemaker.serve.utils.model_package_utils import is_restricted_model_package
from sagemaker.serve.bedrock_target import (
    ESCROW_URI_TAG_KEY,
    BedrockTarget,
    DeploymentMode,
    normalize_escrow_tag_value,
)

from typing import Optional, Dict, Any, Union
from urllib.parse import urlparse

from sagemaker.core.helper.session_helper import Session
from sagemaker.core.helper.iam_role_resolver import resolve_and_validate_role
from sagemaker.core.resources import TrainingJob, ModelPackage
from sagemaker.core.utils.utils import Unassigned

from sagemaker.train.model_trainer import ModelTrainer
from sagemaker.train.base_trainer import BaseTrainer
from sagemaker.train.multi_turn_rl_trainer import MultiTurnRLTrainer
from sagemaker.train.agent_rft_job import AgentRFTJob
from sagemaker.core.telemetry.telemetry_logging import _telemetry_emitter
from sagemaker.core.telemetry.constants import Feature

logger = logging.getLogger(__name__)

def _is_nova_model(container) -> bool:
    """Determine whether a model package container represents a Nova model.

    Checks both recipe_name and hub_content_name for the "nova" substring.

    Args:
        container: A container from ModelPackage.inference_specification.containers.

    Returns:
        True if the container represents a Nova model, False otherwise.
    """
    base_model = getattr(container, "base_model", None)
    if not base_model:
        return False

    recipe_name = getattr(base_model, "recipe_name", None) or ""
    hub_content_name = getattr(base_model, "hub_content_name", None) or ""

    return "nova" in recipe_name.lower() or "nova" in hub_content_name.lower()


_BEDROCK_API_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "bedrock_api_logs")


def _log_bedrock_api_call(api_name: str, params: Dict[str, Any], response: Dict[str, Any]):
    """Log a Bedrock API call to a JSON file in bedrock_api_logs/."""
    log_dir = os.path.normpath(_BEDROCK_API_LOG_DIR)
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{api_name}_{timestamp}.json"
    filepath = os.path.join(log_dir, filename)
    log_entry = {
        "timestamp": timestamp,
        "api": api_name,
        "request": params,
        "response": {k: v for k, v in response.items() if k != "ResponseMetadata"},
    }
    with open(filepath, "w") as f:
        json.dump(log_entry, f, indent=2, default=str)
    logger.info("Bedrock API call logged to %s", filepath)
    print(f"[BedrockModelBuilder] API call logged to: {filepath}")


class BedrockModelBuilder:
    """Builder class for deploying models to Amazon Bedrock.

    This class provides functionality to deploy SageMaker models to Bedrock
    using either model import jobs or custom model creation, depending on
    the model type (Nova models vs. other models).

    Args:
        model: The model to deploy. Can be a ModelTrainer, MultiTurnRLTrainer,
            TrainingJob, ModelPackage instance, or an S3 URI string pointing
            to model artifacts (e.g., ``"s3://bucket/checkpoint/step_4/"``).
    """

    def __init__(
        self,
        model: Optional[Union[str, ModelTrainer, MultiTurnRLTrainer, AgentRFTJob, TrainingJob, ModelPackage]] = None,
    ):
        """Initialize BedrockModelBuilder.

        Args:
            model: Model to deploy. Accepts a ModelTrainer, MultiTurnRLTrainer,
                AgentRFTJob, TrainingJob, ModelPackage, or S3 URI string.
        """
        self._bedrock_client = None
        self._sagemaker_client = None
        self._imported_model_id = None
        self.sagemaker_session = Session()
        self.boto_session = self.sagemaker_session.boto_session

        if isinstance(model, str):
            if not model.startswith("s3://"):
                raise ValueError(
                    f"When 'model' is a string, it must be an S3 URI starting with 's3://'. "
                    f"Got: '{model}'"
                )
            self.model = None
            self.model_package = None
            self._is_rmp = False
            self.s3_model_artifacts = model.rstrip("/") + "/"
        else:
            self.model = model
            self.model_package = self._fetch_model_package() if model else None
            self._is_rmp = is_restricted_model_package(self.model_package)
            self.s3_model_artifacts = self._get_s3_artifacts() if model else None

    def _get_bedrock_client(self):
        """Get or create Bedrock client singleton.

        Returns:
            boto3.client: Bedrock client instance.
        """
        if self._bedrock_client is None:
            self._bedrock_client = self.boto_session.client("bedrock")
        return self._bedrock_client

    def _get_sagemaker_client(self):
        """Get or create SageMaker client singleton.

        Returns:
            boto3.client: SageMaker client instance.
        """
        if self._sagemaker_client is None:
            self._sagemaker_client = self.boto_session.client("sagemaker")
        return self._sagemaker_client

    def _is_nova_model_for_telemetry(self) -> bool:
        """Check if the model is a Nova model for telemetry tracking."""
        try:
            if not self.model_package:
                return False
            container = self.model_package.inference_specification.containers[0]
            return _is_nova_model(container)
        except Exception:
            return False

    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="BedrockModelBuilder.deploy")
    def deploy(
        self,
        job_name: Optional[str] = None,
        imported_model_name: Optional[str] = None,
        custom_model_name: Optional[str] = None,
        role_arn: Optional[str] = None,
        job_tags: Optional[list] = None,
        imported_model_tags: Optional[list] = None,
        model_tags: Optional[list] = None,
        client_request_token: Optional[str] = None,
        imported_model_kms_key_id: Optional[str] = None,
        deployment_name: Optional[str] = None,
        target: Optional["BedrockTarget"] = None,
        skip_model_reuse: bool = False,
    ) -> Dict[str, Any]:
        """Deploy the model to Bedrock.

        When ``target`` is provided, uses the integrated target-based deployment flow:
        escrow-based model reuse, deployment existence checks, and routing to either
        provisioned throughput or on-demand based on target.mode.

        When ``target`` is None (default), uses the legacy flow which automatically
        detects Nova vs OSS models and deploys to on-demand.

        Args:
            job_name: Name for the model import job (OSS models, legacy flow only).
            imported_model_name: Name for the imported model (OSS models, legacy flow only).
            custom_model_name: Name for the custom model. Required when using target.
            role_arn: IAM role ARN with permissions for Bedrock operations.
                If not provided, auto-resolves a least-privilege Bedrock role.
            job_tags: Tags for the import job (OSS models, legacy flow only).
            imported_model_tags: Tags for the imported model (OSS models, legacy flow only).
            model_tags: Tags for the custom model.
            client_request_token: Unique token for idempotency (OSS models, legacy flow only).
            imported_model_kms_key_id: KMS key ID for encryption (OSS models, legacy flow only).
            deployment_name: Name for the deployment. Auto-generated if not provided.
            target: Deployment target configuration. When provided, enables escrow-based
                model reuse and routes to provisioned throughput or on-demand based on
                target.mode. See BedrockTarget for details.
            skip_model_reuse: If True, skip escrow tag-based model lookup and always
                create a new custom model. The escrow tag is still applied to the new
                model for future reuse. Defaults to False.

        Returns:
            When target is provided with mode="provisioned":
                Dict with provisionedModelArn, modelArn, customModelArn, status.
            When target is provided with mode="on_demand":
                Dict with customModelDeploymentArn, customModelArn, and deployment details.
            When target is None (legacy Nova flow):
                The create_custom_model_deployment response.
            When target is None (legacy OSS flow):
                The completed get_model_import_job response.

        Raises:
            ValueError: If no model source is available or required parameters are missing.
            RuntimeError: If deployment fails, times out, or conflicts with FAIL_IF_EXISTS.
        """
        if not self.model_package and not self.s3_model_artifacts:
            raise ValueError(
                "No model source available. Provide a valid model object, an S3 URI string, "
                "or set 's3_model_artifacts' during initialization."
            )

        spec = getattr(self.model_package, "inference_specification", None) if self.model_package else None
        containers = getattr(spec, "containers", None) if spec else None
        container = containers[0] if containers else None
        is_nova = _is_nova_model(container) if container else False

        # Direct S3 URI without model package: use Nova path if custom_model_name
        # is provided, otherwise fall through to OSS import path.
        if not self.model_package and self.s3_model_artifacts:
            if custom_model_name or target is not None:
                is_nova = True

        if self._is_rmp or is_nova:
            if not custom_model_name:
                raise ValueError("custom_model_name is required for Nova model deployment.")

            role_arn = resolve_and_validate_role(
                provided_role=role_arn,
                role_type="bedrock",
                sagemaker_session=self.sagemaker_session,
            )

            is_provisioned = target.mode == "provisioned" if target else False
            config = target.config if target and is_provisioned else None
            effective_skip = skip_model_reuse or (target.skip_model_reuse if target else False)
            deployment_mode = config.deployment_mode if config else DeploymentMode.FAIL_IF_EXISTS

            model_arn = self._find_or_create_model(
                custom_model_name=custom_model_name,
                role_arn=role_arn,
                model_tags=model_tags,
                skip_model_reuse=effective_skip,
            )

            if deployment_name:
                deploy_name = deployment_name
            elif is_provisioned:
                deploy_name = f"{custom_model_name}-pt"
            else:
                deploy_name = f"{custom_model_name}-deployment"

            if target is not None:
                existing_arn = self._check_existing_deployment(
                    deploy_name, is_provisioned=is_provisioned
                )
                if existing_arn:
                    return self._handle_existing_deployment(
                        existing_arn=existing_arn,
                        deployment_mode=deployment_mode,
                        new_model_arn=model_arn,
                        deployment_name=deploy_name,
                        is_provisioned=is_provisioned,
                    )

            if is_provisioned:
                return self._create_and_poll_provisioned_throughput(
                    model_arn=model_arn,
                    deployment_name=deploy_name,
                    units=config.units,
                    commitment_duration=config.commitment_duration,
                )

            response = self.create_deployment(model_arn=model_arn, deployment_name=deploy_name)
            if target is not None and "customModelArn" not in response:
                response["customModelArn"] = model_arn
            return response
        else:
            # OSS import path
            if target is not None and target.mode == "provisioned":
                raise ValueError(
                    "Provisioned Throughput is not supported for OSS models imported via "
                    "create_model_import_job. Only models created via CreateCustomModel "
                    "(Nova or RMP-based) support Provisioned Throughput. "
                    "For OSS models, use mode='on_demand' or deploy without a target."
                )
            if target is not None:
                raise ValueError(
                    "BedrockTarget with mode='on_demand' is not supported for OSS model import. "
                    "OSS models are deployed on-demand by default. "
                    "Use deploy() without target, or provide custom_model_name "
                    "to use the Nova/RMP CreateCustomModel path."
                )

            role_arn = resolve_and_validate_role(
                provided_role=role_arn,
                role_type="bedrock",
                sagemaker_session=self.sagemaker_session,
            )
            model_data_source = {"s3DataSource": {"s3Uri": self.s3_model_artifacts}}
            if self.s3_model_artifacts.endswith(".tar.gz") or self.s3_model_artifacts.endswith(".tar.gz/"):
                extracted_uri = self._extract_tar_gz_to_s3(self.s3_model_artifacts.rstrip("/"))
                resolved_uri = self._resolve_hf_model_path(extracted_uri)
                model_data_source = {"s3DataSource": {"s3Uri": resolved_uri}}
            if not job_name:
                job_name = f"{imported_model_name or 'import'}-{int(time.time())}"
            params = {
                "jobName": job_name,
                "importedModelName": imported_model_name,
                "roleArn": role_arn,
                "modelDataSource": model_data_source,
                "jobTags": job_tags,
                "importedModelTags": imported_model_tags,
                "clientRequestToken": client_request_token,
                "importedModelKmsKeyId": imported_model_kms_key_id,
            }
            params = {k: v for k, v in params.items() if v is not None}

            logger.info("Creating model import job for OSS model deployment")
            import_response = self._get_bedrock_client().create_model_import_job(**params)
            _log_bedrock_api_call("create_model_import_job", params, import_response)

            job_arn = import_response.get("jobArn")
            self._wait_for_import_job_complete(job_arn)

            job_details = self._get_bedrock_client().get_model_import_job(
                jobIdentifier=job_arn
            )
            self._imported_model_id = job_details.get("importedModelName")
            return job_details

    def create_deployment(
        self,
        model_arn: str,
        deployment_name: Optional[str] = None,
        poll_interval: int = 60,
        max_wait: int = 3600,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create a deployment for a Nova custom model.

        Polls the model status until it becomes Active before creating the deployment,
        then polls the deployment status until it becomes Active.

        Args:
            model_arn: ARN of the custom model to deploy.
            deployment_name: Name for the deployment.
            poll_interval: Seconds between status checks. Defaults to 60 for model,
                30 for deployment.
            max_wait: Maximum seconds to wait per polling phase. Defaults to 3600.
            **kwargs: Additional parameters for create_custom_model_deployment.

        Returns:
            Response from Bedrock create_custom_model_deployment API.

        Raises:
            RuntimeError: If the model or deployment fails or times out.
            ValueError: If model_arn is not provided.
        """
        if not model_arn:
            raise ValueError("model_arn is required for create_deployment.")

        self._wait_for_model_active(model_arn, poll_interval=poll_interval, max_wait=max_wait)

        params = {
            "modelDeploymentName": deployment_name,
            "modelArn": model_arn,
            **{k: v for k, v in kwargs.items() if v is not None},
        }
        params = {k: v for k, v in params.items() if v is not None}

        logger.info("Creating deployment %s for model %s", deployment_name, model_arn)
        response = self._get_bedrock_client().create_custom_model_deployment(**params)
        logger.warning(
            "Bedrock create_custom_model_deployment request: %s, response: %s", params, response
        )
        _log_bedrock_api_call("create_custom_model_deployment", params, response)

        deployment_arn = response.get("customModelDeploymentArn")
        if deployment_arn:
            self._wait_for_deployment_active(
                deployment_arn, poll_interval=poll_interval, max_wait=max_wait
            )

        return response

    def create_provisioned_throughput(
        self,
        model_id: Optional[str] = None,
        provisioned_model_name: str = None,
        model_units: int = 1,
        commitment_duration: Optional[str] = None,
        tags: Optional[list] = None,
        poll_interval: int = 60,
        max_wait: int = 3600,
    ) -> Dict[str, Any]:
        """Create provisioned throughput for an imported model on Bedrock.

        Calls CreateProvisionedModelThroughput and polls until the provisioned
        throughput reaches InService status.

        Args:
            model_id: ARN or name of the model. If not provided, uses the model
                ID from the most recent deploy() call.
            provisioned_model_name: Name for the provisioned throughput resource.
            model_units: Number of model units to provision. Defaults to 1.
            commitment_duration: Commitment duration. Valid values: 'OneMonth',
                'SixMonths'. If not provided, no commitment is set (on-demand).
            tags: Tags for the provisioned throughput resource.
            poll_interval: Seconds between status checks. Defaults to 60.
            max_wait: Maximum seconds to wait. Defaults to 3600.

        Returns:
            Response from Bedrock create_provisioned_model_throughput API.

        Raises:
            RuntimeError: If the provisioned throughput fails or times out.
            ValueError: If model_id cannot be determined or provisioned_model_name
                is not provided.
        """
        resolved_model_id = model_id or self._imported_model_id
        if not resolved_model_id:
            raise ValueError(
                "model_id is required for create_provisioned_throughput. "
                "Either pass it explicitly or call deploy() first."
            )
        if not provisioned_model_name:
            raise ValueError(
                "provisioned_model_name is required for create_provisioned_throughput."
            )

        params = {
            "modelId": resolved_model_id,
            "provisionedModelName": provisioned_model_name,
            "modelUnits": model_units,
        }
        if commitment_duration:
            params["commitmentDuration"] = commitment_duration
        if tags:
            params["tags"] = tags

        logger.info(
            "Creating provisioned throughput '%s' for model %s with %d model units",
            provisioned_model_name,
            resolved_model_id,
            model_units,
        )
        response = self._get_bedrock_client().create_provisioned_model_throughput(**params)

        provisioned_model_arn = response.get("provisionedModelArn")
        if provisioned_model_arn:
            self._wait_for_provisioned_throughput_in_service(
                provisioned_model_arn, poll_interval=poll_interval, max_wait=max_wait
            )

        return response

    def _wait_for_import_job_complete(
        self, job_arn: str, poll_interval: int = 60, max_wait: int = 3600
    ):
        """Poll Bedrock until the model import job reaches Completed status.

        Args:
            job_arn: ARN of the model import job.
            poll_interval: Seconds between status checks. Defaults to 60.
            max_wait: Maximum seconds to wait. Defaults to 3600.

        Raises:
            RuntimeError: If the import job fails or times out.
        """
        elapsed = 0
        status = None
        while elapsed < max_wait:
            resp = self._get_bedrock_client().get_model_import_job(jobIdentifier=job_arn)
            status = resp.get("status")
            logger.info("Import job status: %s (elapsed %ds)", status, elapsed)
            if status == "Completed":
                return
            if status == "Failed":
                failure_reason = resp.get("failureMessage", "Unknown")
                raise RuntimeError(
                    f"Model import job {job_arn} failed. Reason: {failure_reason}"
                )
            time.sleep(poll_interval)
            elapsed += poll_interval
        raise RuntimeError(
            f"Timed out after {max_wait}s waiting for import job {job_arn} to complete. "
            f"Last status: {status}"
        )

    def _wait_for_provisioned_throughput_in_service(
        self, provisioned_model_arn: str, poll_interval: int = 60, max_wait: int = 3600
    ):
        """Poll Bedrock until provisioned throughput reaches InService status.

        Args:
            provisioned_model_arn: ARN of the provisioned model throughput.
            poll_interval: Seconds between status checks. Defaults to 60.
            max_wait: Maximum seconds to wait. Defaults to 3600.

        Raises:
            RuntimeError: If the provisioned throughput fails or times out.
        """
        elapsed = 0
        status = None
        while elapsed < max_wait:
            resp = self._get_bedrock_client().get_provisioned_model_throughput(
                provisionedModelId=provisioned_model_arn
            )
            status = resp.get("status")
            logger.info("Provisioned throughput status: %s (elapsed %ds)", status, elapsed)
            if status == "InService":
                return
            if status == "Failed":
                failure_reason = resp.get("failureMessage", "Unknown")
                raise RuntimeError(
                    f"Provisioned throughput {provisioned_model_arn} failed. "
                    f"Reason: {failure_reason}"
                )
            time.sleep(poll_interval)
            elapsed += poll_interval
        raise RuntimeError(
            f"Timed out after {max_wait}s waiting for provisioned throughput "
            f"{provisioned_model_arn} to become InService. Last status: {status}"
        )

    def _wait_for_model_active(
        self, model_arn: str, poll_interval: int = 60, max_wait: int = 3600
    ):
        """Poll Bedrock until the custom model reaches Active status.

        Args:
            model_arn: ARN of the custom model.
            poll_interval: Seconds between status checks.
            max_wait: Maximum seconds to wait.

        Raises:
            RuntimeError: If the model status is Failed or the wait times out.
        """
        elapsed = 0
        status = None
        while elapsed < max_wait:
            resp = self._get_bedrock_client().get_custom_model(modelIdentifier=model_arn)
            status = resp.get("modelStatus")
            logger.info("Custom model status: %s (elapsed %ds)", status, elapsed)
            if status == "Active":
                return
            if status == "Failed":
                raise RuntimeError(
                    f"Custom model {model_arn} failed. Cannot proceed with deployment."
                )
            time.sleep(poll_interval)
            elapsed += poll_interval
        raise RuntimeError(
            f"Timed out after {max_wait}s waiting for custom model {model_arn} to become Active. "
            f"Last status: {status}"
        )

    def _wait_for_deployment_active(
        self, deployment_arn: str, poll_interval: int = 30, max_wait: int = 3600
    ):
        """Poll Bedrock until the custom model deployment reaches Active status.

        Args:
            deployment_arn: ARN of the custom model deployment.
            poll_interval: Seconds between status checks. Defaults to 30.
            max_wait: Maximum seconds to wait. Defaults to 3600.

        Raises:
            RuntimeError: If the deployment status is Failed or the wait times out.
        """
        elapsed = 0
        status = None
        while elapsed < max_wait:
            resp = self._get_bedrock_client().get_custom_model_deployment(
                customModelDeploymentIdentifier=deployment_arn
            )
            status = resp.get("status")
            logger.info("Deployment status: %s (elapsed %ds)", status, elapsed)
            if status == "Active":
                return
            if status == "Failed":
                raise RuntimeError(
                    f"Deployment {deployment_arn} failed."
                )
            time.sleep(poll_interval)
            elapsed += poll_interval
        raise RuntimeError(
            f"Timed out after {max_wait}s waiting for deployment {deployment_arn} to become Active. "
            f"Last status: {status}"
        )

    def _resolve_escrow_identifier(self) -> Optional[str]:
        """Determine the escrow identifier from the model source.

        Resolution order:
        1. Model package ARN (for RMP or model_package-based models)
        2. Checkpoint URI from manifest (for Nova TrainingJob models)
        3. S3 artifact path (for direct S3 URI or training job output)

        Returns:
            Escrow identifier string, or None if cannot be determined.
        """
        if self.model_package and hasattr(self.model_package, "model_package_arn"):
            arn = self.model_package.model_package_arn
            if arn:
                return arn
        if self.s3_model_artifacts:
            if isinstance(self.model, TrainingJob):
                try:
                    checkpoint_uri = self._get_checkpoint_uri_from_manifest()
                    if checkpoint_uri:
                        return checkpoint_uri
                except Exception:
                    pass
            return self.s3_model_artifacts
        return None

    def _find_existing_model_by_escrow(self, escrow_identifier: str) -> Optional[str]:
        """Query ResourceGroupsTaggingAPI for an existing model with matching escrow tag.

        Args:
            escrow_identifier: Raw escrow URI (will be normalized internally).

        Returns:
            Model ARN if an Active model is found, None otherwise.

        Raises:
            TimeoutError: If a Creating model doesn't reach Active within 900s.
        """
        try:
            tagging_client = self.boto_session.client("resourcegroupstaggingapi")
            normalized_value = normalize_escrow_tag_value(escrow_identifier)
            response = tagging_client.get_resources(
                TagFilters=[{"Key": ESCROW_URI_TAG_KEY, "Values": [normalized_value]}],
                ResourceTypeFilters=["bedrock:custom-model"],
            )
        except Exception as e:
            logger.warning("Could not query ResourceGroupsTaggingAPI: %s. Proceeding without.", e)
            return None

        resources = response.get("ResourceTagMappingList", [])
        if not resources:
            return None

        model_arn = resources[0].get("ResourceARN")
        if not model_arn:
            return None

        try:
            bedrock_client = self._get_bedrock_client()
            resp = bedrock_client.get_custom_model(modelIdentifier=model_arn)
            status = resp.get("modelStatus")
        except Exception as e:
            logger.warning(
                "Could not get status for model %s: %s. Proceeding without.", model_arn, e
            )
            return None

        if status == "Active":
            return model_arn

        if status == "Failed":
            logger.warning("Existing model %s is in Failed status. Will create a new model.", model_arn)
            return None

        if status == "Creating":
            elapsed = 0
            poll_interval = 30
            max_wait = 900
            while elapsed < max_wait:
                time.sleep(poll_interval)
                elapsed += poll_interval
                try:
                    resp = bedrock_client.get_custom_model(modelIdentifier=model_arn)
                    status = resp.get("modelStatus")
                except Exception as e:
                    logger.warning(
                        "Error polling model %s status: %s. Proceeding without.", model_arn, e
                    )
                    return None
                if status == "Active":
                    return model_arn
                if status == "Failed":
                    logger.warning(
                        "Model %s transitioned to Failed while waiting. Will create a new model.",
                        model_arn,
                    )
                    return None
            raise TimeoutError(
                f"Model {model_arn} did not reach Active status within {max_wait}s."
            )

        logger.warning("Model %s has unexpected status '%s'. Proceeding without.", model_arn, status)
        return None

    def _find_or_create_model(
        self,
        custom_model_name: str,
        role_arn: str,
        model_tags: Optional[list],
        skip_model_reuse: bool,
    ) -> str:
        """Find existing model via escrow tag or create a new one.

        Args:
            custom_model_name: Name for the custom model (if creating new).
            role_arn: IAM role ARN for model creation.
            model_tags: Additional tags for the model.
            skip_model_reuse: If True, skip escrow lookup.

        Returns:
            Model ARN (either existing or newly created).
        """
        escrow_id = self._resolve_escrow_identifier()

        if not skip_model_reuse and escrow_id:
            existing_arn = self._find_existing_model_by_escrow(escrow_id)
            if existing_arn:
                logger.info("Reusing existing model %s via escrow tag.", existing_arn)
                return existing_arn

        tags = list(model_tags) if model_tags else []
        if escrow_id:
            normalized_value = normalize_escrow_tag_value(escrow_id)
            tags = [t for t in tags if t.get("key") != ESCROW_URI_TAG_KEY]
            tags.append({"key": ESCROW_URI_TAG_KEY, "value": normalized_value})

        spec = (
            getattr(self.model_package, "inference_specification", None)
            if self.model_package
            else None
        )
        containers = getattr(spec, "containers", None) if spec else None
        container = containers[0] if containers else None
        is_nova = _is_nova_model(container) if container else False

        if not self.model_package and self.s3_model_artifacts:
            is_nova = True

        if self._is_rmp or is_nova:
            if self._is_rmp:
                params = {
                    "modelName": custom_model_name,
                    "customModelDataSource": {
                        "modelPackageArnDataSource": {
                            "modelPackageArn": self.model_package.model_package_arn
                        }
                    },
                    "roleArn": role_arn,
                }
            else:
                s3_uri = self.s3_model_artifacts
                if isinstance(self.model, TrainingJob):
                    try:
                        checkpoint_uri = self._get_checkpoint_uri_from_manifest()
                        if checkpoint_uri:
                            s3_uri = checkpoint_uri
                    except Exception as e:
                        logger.warning(
                            "Could not resolve checkpoint from manifest: %s. "
                            "Using s3_model_artifacts path.", e
                        )
                params = {
                    "modelName": custom_model_name,
                    "modelSourceConfig": {
                        "s3DataSource": {"s3Uri": s3_uri}
                    },
                    "roleArn": role_arn,
                }
            if tags:
                params["modelTags"] = tags
            params = {k: v for k, v in params.items() if v is not None}

            logger.info("Creating custom model %s", custom_model_name)
            response = self._get_bedrock_client().create_custom_model(**params)
            _log_bedrock_api_call("create_custom_model", params, response)
            model_arn = response.get("modelArn")
            self._wait_for_model_active(model_arn)
            return model_arn
        else:
            model_data_source = {"s3DataSource": {"s3Uri": self.s3_model_artifacts}}
            if self.s3_model_artifacts.endswith(
                ".tar.gz"
            ) or self.s3_model_artifacts.endswith(".tar.gz/"):
                extracted_uri = self._extract_tar_gz_to_s3(
                    self.s3_model_artifacts.rstrip("/")
                )
                resolved_uri = self._resolve_hf_model_path(extracted_uri)
                model_data_source = {"s3DataSource": {"s3Uri": resolved_uri}}

            job_name = f"{custom_model_name}-import-{int(time.time())}"
            params = {
                "jobName": job_name,
                "importedModelName": custom_model_name,
                "roleArn": role_arn,
                "modelDataSource": model_data_source,
            }
            if tags:
                params["importedModelTags"] = tags
            params = {k: v for k, v in params.items() if v is not None}

            logger.info("Creating model import job for %s", custom_model_name)
            response = self._get_bedrock_client().create_model_import_job(**params)
            _log_bedrock_api_call("create_model_import_job", params, response)

            job_arn = response.get("jobArn")
            self._wait_for_import_job_complete(job_arn)

            job_details = self._get_bedrock_client().get_model_import_job(
                jobIdentifier=job_arn
            )
            return job_details.get("importedModelArn")

    def _check_existing_deployment(
        self,
        deployment_name: str,
        is_provisioned: bool,
    ) -> Optional[str]:
        """Check if a deployment with the given name already exists.

        Args:
            deployment_name: Name to search for.
            is_provisioned: True for PT, False for OD.

        Returns:
            Existing deployment ARN if found with exact name match, None otherwise.
        """
        try:
            bedrock_client = self._get_bedrock_client()
            if is_provisioned:
                response = bedrock_client.list_provisioned_model_throughputs(
                    nameContains=deployment_name
                )
                for summary in response.get("provisionedModelSummaries", []):
                    if summary.get("provisionedModelName") == deployment_name:
                        return summary.get("provisionedModelArn")
            else:
                response = bedrock_client.list_custom_model_deployments(
                    nameContains=deployment_name
                )
                for summary in response.get("customModelDeploymentSummaries", []):
                    if summary.get("customModelDeploymentName") == deployment_name:
                        return summary.get("customModelDeploymentArn")
        except Exception as e:
            logger.warning(
                "Could not check for existing deployment '%s': %s. Proceeding without.",
                deployment_name,
                e,
            )
            return None
        return None

    def _handle_existing_deployment(
        self,
        existing_arn: str,
        deployment_mode: DeploymentMode,
        new_model_arn: str,
        deployment_name: str,
        is_provisioned: bool,
    ) -> Optional[Dict[str, Any]]:
        """Handle conflict with an existing deployment.

        Args:
            existing_arn: ARN of the existing deployment.
            deployment_mode: How to handle the conflict.
            new_model_arn: ARN of the new model to deploy.
            deployment_name: Name of the deployment.
            is_provisioned: True for PT, False for OD.

        Returns:
            Result dict if deployment was updated in-place.

        Raises:
            RuntimeError: If FAIL_IF_EXISTS and deployment exists, or update fails.
            ValueError: If UPDATE_IF_EXISTS on non-PT deployment.
        """
        if deployment_mode == DeploymentMode.FAIL_IF_EXISTS:
            raise RuntimeError(
                f"Deployment '{deployment_name}' already exists (ARN: {existing_arn}). "
                f"Use deployment_mode=UPDATE_IF_EXISTS to update in-place."
            )

        if not is_provisioned:
            raise ValueError(
                "In-place update is only supported for Provisioned Throughput deployments."
            )

        try:
            bedrock_client = self._get_bedrock_client()
            bedrock_client.update_provisioned_model_throughput(
                provisionedModelId=existing_arn,
                desiredModelId=new_model_arn,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to update deployment '{deployment_name}': {e}"
            ) from e

        return {
            "provisionedModelArn": existing_arn,
            "modelArn": existing_arn,
            "customModelArn": new_model_arn,
            "status": "Updating",
        }

    def _create_and_poll_provisioned_throughput(
        self,
        model_arn: str,
        deployment_name: str,
        units: int,
        commitment_duration: Optional[str],
        poll_interval: int = 60,
        max_wait: int = 3600,
    ) -> Dict[str, Any]:
        """Create a provisioned throughput deployment and poll until InService.

        Args:
            model_arn: Custom model ARN to deploy.
            deployment_name: Name for the provisioned throughput resource.
            units: Number of model units.
            commitment_duration: Optional commitment ("OneMonth" or "SixMonths").
            poll_interval: Seconds between polls. Capped at 60.
            max_wait: Maximum wait time in seconds.

        Returns:
            Dict with provisionedModelArn, modelArn, customModelArn, status.

        Raises:
            RuntimeError: If deployment fails or times out.
        """
        params = {
            "modelId": model_arn,
            "provisionedModelName": deployment_name,
            "modelUnits": units,
        }
        if commitment_duration is not None:
            params["commitmentDuration"] = commitment_duration

        logger.info(
            "Creating provisioned throughput '%s' for model %s with %d units",
            deployment_name,
            model_arn,
            units,
        )
        response = self._get_bedrock_client().create_provisioned_model_throughput(**params)
        _log_bedrock_api_call("create_provisioned_model_throughput", params, response)

        provisioned_model_arn = response.get("provisionedModelArn")

        elapsed = 0
        status = None
        while elapsed < max_wait:
            resp = self._get_bedrock_client().get_provisioned_model_throughput(
                provisionedModelId=provisioned_model_arn
            )
            status = resp.get("status")
            logger.info(
                "Provisioned throughput '%s' status: %s (elapsed %ds)",
                deployment_name,
                status,
                elapsed,
            )
            if status == "InService":
                return {
                    "provisionedModelArn": provisioned_model_arn,
                    "modelArn": provisioned_model_arn,
                    "customModelArn": model_arn,
                    "status": status,
                }
            if status == "Failed":
                failure_reason = resp.get("failureMessage", "Unknown")
                raise RuntimeError(
                    f"Provisioned throughput '{deployment_name}' failed. "
                    f"Reason: {failure_reason}"
                )
            time.sleep(min(poll_interval, 60))
            elapsed += min(poll_interval, 60)

        raise RuntimeError(
            f"Timed out after {max_wait}s waiting for provisioned throughput "
            f"'{deployment_name}' to become InService. Last status: {status}"
        )

    def _fetch_model_package(self) -> Optional[ModelPackage]:
        """Fetch the ModelPackage from the provided model.

        Extracts ModelPackage from ModelTrainer, MultiTurnRLTrainer, TrainingJob,
        or returns the ModelPackage directly if that's what was provided.

        Returns:
            ModelPackage instance or None if no model was provided.
        """
        if isinstance(self.model, ModelPackage):
            return self.model
        if isinstance(self.model, TrainingJob):
            arn = getattr(self.model, 'output_model_package_arn', None)
            if arn and isinstance(arn, str):
                try:
                    return ModelPackage.get(arn)
                except Exception:
                    pass
            # No valid model package ARN — _get_s3_artifacts will resolve.
            return None
        if isinstance(self.model, (MultiTurnRLTrainer, AgentRFTJob)):
            arn = self.model.output_model_package_arn
            if not arn:
                job_name = None
                if isinstance(self.model, AgentRFTJob):
                    job_name = self.model.job_name
                elif hasattr(self.model, "_latest_job") and self.model._latest_job:
                    job_name = self.model._latest_job.job_name
                if job_name:
                    from sagemaker.core.resources import Job
                    job = Job.get(
                        job_name=job_name, job_category="AgentRFT"
                    )
                    config = json.loads(job.job_config_document) if job.job_config_document else {}
                    arn = config.get("ServiceOutput", {}).get("OutputModelPackageArn")
            if not arn:
                raise ValueError(
                    "Model has no output_model_package_arn. "
                    "Ensure training has completed successfully."
                )
            return ModelPackage.get(arn)
        if isinstance(self.model, ModelTrainer):
            mp_arn = getattr(self.model, '_latest_training_job', None)
            if mp_arn:
                mp_arn = getattr(mp_arn, 'output_model_package_arn', None)
            if mp_arn:
                return ModelPackage.get(mp_arn)
            # No model package (e.g., HyperPod) — _get_s3_artifacts will resolve.
            return None
        return None

    def _get_s3_artifacts(self) -> Optional[str]:
        """Extract S3 URI of model artifacts from the model package or training job.

        Resolution priority:
        1. If model_package exists and is a Nova model from a TrainingJob, fetches
           checkpoint URI from manifest.json in training job output.
        2. If model_package exists, returns the model data source S3 URI, resolving
           to the hf_merged checkpoint directory if it exists (required for Bedrock import).
        3. If no model_package and model is a TrainingJob, reads model_artifacts.s3_model_artifacts.
        4. If no model_package and model is a ModelTrainer/BaseTrainer, reads
           _latest_training_job.model_artifacts.s3_model_artifacts.

        Returns:
            S3 URI string of the model artifacts, or None if not available.
        """
        if self.model_package:
            if self._is_rmp:
                return None

            container = self.model_package.inference_specification.containers[0]
            is_nova = _is_nova_model(container)

            if is_nova and isinstance(self.model, TrainingJob):
                return self._get_checkpoint_uri_from_manifest()

            if hasattr(container, "model_data_source") and container.model_data_source:
                data_source = container.model_data_source
                if hasattr(data_source, "s3_data_source") and data_source.s3_data_source:
                    s3_uri = data_source.s3_data_source.s3_uri
                    if s3_uri:
                        return self._resolve_hf_model_path(s3_uri)
            return None

        # No model_package — resolve from model_artifacts directly.
        if isinstance(self.model, TrainingJob):
            artifacts = getattr(self.model, 'model_artifacts', None)
            if artifacts and not isinstance(artifacts, Unassigned):
                s3_path = getattr(artifacts, 's3_model_artifacts', None)
                if s3_path and isinstance(s3_path, str):
                    logger.info(
                        "Resolved S3 artifacts from TrainingJob model_artifacts: %s", s3_path
                    )
                    return s3_path
            return None

        # ModelTrainer or BaseTrainer — resolve from _latest_training_job.model_artifacts.
        if isinstance(self.model, (ModelTrainer, BaseTrainer)):
            training_job = getattr(self.model, '_latest_training_job', None)
            if not training_job:
                return None
            artifacts = getattr(training_job, 'model_artifacts', None)
            if artifacts and not isinstance(artifacts, Unassigned):
                s3_path = getattr(artifacts, 's3_model_artifacts', None)
                if s3_path and isinstance(s3_path, str):
                    logger.info(
                        "Resolved S3 artifacts from trainer's training job: %s", s3_path
                    )
                    return s3_path
            return None

        return None

    def _resolve_hf_model_path(self, s3_uri: str) -> str:
        """Resolve the HuggingFace model directory within model artifacts.

        MTRL training jobs produce checkpoints under checkpoints/:
        - hf_merged/ contains full merged weights (config.json + model shards)
        - hf/ contains LoRA adapter only (adapter_config.json + adapter_model.safetensors)

        The s3_uri from the model package already includes the trailing model/ prefix,
        so this method appends checkpoints/hf_merged/ or checkpoints/hf/ directly.

        This method checks for hf_merged first (preferred for Bedrock import),
        then falls back to hf (LoRA adapter), then the original URI.

        Args:
            s3_uri: Base S3 URI from the model package container (typically ends with model/).

        Returns:
            S3 URI pointing to the resolved model directory.
        """
        s3_uri = s3_uri.rstrip("/") + "/"
        parsed_base = urlparse(s3_uri)
        bucket = parsed_base.netloc
        s3_client = self.boto_session.client("s3")

        print(f"[BedrockModelBuilder] Base s3_uri from model package: {s3_uri}")

        hf_merged_uri = s3_uri + "checkpoints/hf_merged/"
        merged_config_key = urlparse(hf_merged_uri).path.lstrip("/") + "config.json"
        print(f"[BedrockModelBuilder] Probing for hf_merged: s3://{bucket}/{merged_config_key}")
        try:
            s3_client.head_object(Bucket=bucket, Key=merged_config_key)
            logger.info("Found merged HF model at %s", hf_merged_uri)
            print(f"[BedrockModelBuilder] Found hf_merged checkpoint, using: {hf_merged_uri}")
            return hf_merged_uri
        except Exception as e:
            print(f"[BedrockModelBuilder] hf_merged not found: {e}")

        hf_lora_uri = s3_uri + "checkpoints/hf/"
        lora_config_key = urlparse(hf_lora_uri).path.lstrip("/") + "adapter_config.json"
        try:
            s3_client.head_object(Bucket=bucket, Key=lora_config_key)
            logger.info("Found LoRA adapter at %s", hf_lora_uri)
            return hf_lora_uri
        except Exception:
            pass

        logger.info("No hf_merged or hf checkpoint found, using base path: %s", s3_uri)
        return s3_uri.rstrip("/")

    def _extract_tar_gz_to_s3(self, tar_gz_uri: str) -> str:
        """Extract a model.tar.gz from S3 and upload contents to a sibling S3 prefix.

        Streams the tar.gz, extracts all files, and uploads them to an
        ``extracted/`` directory alongside the original tar.gz. Skips
        extraction if the output directory already has content (idempotent).

        Args:
            tar_gz_uri: S3 URI to a .tar.gz file.

        Returns:
            S3 URI prefix where files were extracted.
        """
        import tarfile

        parsed = urlparse(tar_gz_uri)
        bucket = parsed.netloc
        tar_key = parsed.path.lstrip("/")

        parent_prefix = tar_key.rsplit("/", 1)[0] + "/"
        extract_prefix = parent_prefix + "extracted/"
        extract_uri = f"s3://{bucket}/{extract_prefix}"

        s3_client = self.boto_session.client("s3")

        # Idempotent — skip if already extracted
        try:
            resp = s3_client.list_objects_v2(Bucket=bucket, Prefix=extract_prefix, MaxKeys=1)
            if resp.get("KeyCount", 0) > 0:
                logger.info("Extracted directory already exists at %s", extract_uri)
                return extract_uri
        except Exception:
            pass

        logger.warning(
            "Model artifacts are in tar.gz format. Extracting to %s. "
            "This may take several minutes for large archives.",
            extract_uri,
        )

        response = s3_client.get_object(Bucket=bucket, Key=tar_key)
        stream = response["Body"]
        stream.seekable = lambda: False

        extracted_count = 0
        with tarfile.open(fileobj=stream, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                dest_key = extract_prefix + member.name
                size_mb = member.size / (1024 * 1024)
                extracted_count += 1
                logger.info(
                    "Extracting [%d]: %s (%.1f MB)", extracted_count, member.name, size_mb
                )
                s3_client.put_object(Bucket=bucket, Key=dest_key, Body=f.read())

        if extracted_count == 0:
            raise RuntimeError(f"No files found in {tar_gz_uri}.")

        logger.info("Extracted %d files to %s", extracted_count, extract_uri)
        return extract_uri

    def _get_checkpoint_uri_from_manifest(self) -> Optional[str]:
        """Get checkpoint URI from manifest.json for Nova models.

        Resolution order:
        1. Try reading manifest.json directly at output/output/manifest.json (serverless)
        2. If not found, download output/output.tar.gz and extract manifest.json (SMTJ)

        Returns:
            Checkpoint URI from manifest.json.

        Raises:
            ValueError: If manifest.json cannot be found or parsed, or if the
                model is not a TrainingJob instance.
        """
        if not isinstance(self.model, TrainingJob):
            raise ValueError("Model must be a TrainingJob instance for Nova models")

        output_data_config = getattr(self.model, "output_data_config", None)
        s3_output_path = getattr(output_data_config, "s3_output_path", None)
        if not s3_output_path:
            raise ValueError("No S3 output path found in training job output_data_config")

        output_path = s3_output_path.rstrip("/")
        job_name = self.model.training_job_name

        s3_client = self.boto_session.client("s3")

        # Try serverless format: output/output/manifest.json as raw file
        manifest_path = f"{output_path}/{job_name}/output/output/manifest.json"
        parsed = urlparse(manifest_path)
        bucket = parsed.netloc
        manifest_key = parsed.path.lstrip("/")

        logger.info("Looking for manifest at s3://%s/%s", bucket, manifest_key)

        manifest = None
        try:
            response = s3_client.get_object(Bucket=bucket, Key=manifest_key)
            manifest = json.loads(response["Body"].read().decode("utf-8"))
        except Exception:
            logger.info("Raw manifest.json not found, trying output.tar.gz")

        # Try SMTJ format: manifest.json inside output.tar.gz
        if manifest is None:
            import tarfile
            import tempfile
            tar_key = f"{output_path}/{job_name}/output/output.tar.gz".replace(
                f"s3://{bucket}/", ""
            )
            tar_key = parsed.path.lstrip("/").rsplit("/manifest.json", 1)[0].rsplit("/output", 1)[0]
            tar_key = f"{tar_key}/output.tar.gz"

            logger.info("Looking for output.tar.gz at s3://%s/%s", bucket, tar_key)
            try:
                with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp_file:
                    s3_client.download_file(bucket, tar_key, tmp_file.name)
                    with tarfile.open(tmp_file.name, "r:gz") as tar:
                        manifest_file = tar.extractfile("manifest.json")
                        if manifest_file is None:
                            raise ValueError("manifest.json not found inside output.tar.gz")
                        manifest = json.loads(manifest_file.read().decode("utf-8"))
            except Exception as e:
                raise ValueError(
                    f"manifest.json not found at s3://{bucket}/{manifest_key} "
                    f"and could not extract from output.tar.gz: {e}"
                )

        logger.info("Manifest content: %s", manifest)
        checkpoint_uri = manifest.get("checkpoint_s3_bucket")
        if not checkpoint_uri:
            raise ValueError(
                "'checkpoint_s3_bucket' not found in manifest. "
                "Available keys: %s" % list(manifest.keys())
            )

        logger.info("Checkpoint URI: %s", checkpoint_uri)
        return checkpoint_uri
