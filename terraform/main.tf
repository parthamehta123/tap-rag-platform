terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.60"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "environment" {
  type    = string
  default = "dev"
}

variable "project" {
  type    = string
  default = "tap-rag-platform"
}

locals {
  name = "${var.project}-${var.environment}"
  tags = {
    Project     = var.project
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# ── S3: Chroma snapshots, feedback, eval reports, LoRA adapters ──
resource "aws_s3_bucket" "artifacts" {
  bucket = "${local.name}-artifacts"
  tags   = local.tags
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ── ECR ──
resource "aws_ecr_repository" "api" {
  name                 = "${local.name}/api"
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration { scan_on_push = true }
  tags = local.tags
}

resource "aws_ecr_repository" "ui" {
  name                 = "${local.name}/ui"
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration { scan_on_push = true }
  tags = local.tags
}

# ── IAM for ECS task (Bedrock + S3) ──
data "aws_iam_policy_document" "ecs_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ecs_task" {
  name               = "${local.name}-ecs-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
  tags               = local.tags
}

data "aws_iam_policy_document" "ecs_task" {
  statement {
    sid = "BedrockInvoke"
    actions = [
      "bedrock:InvokeModel",
      "bedrock:InvokeModelWithResponseStream",
      "bedrock:ApplyGuardrail",
    ]
    resources = ["*"]
  }
  statement {
    sid = "S3Artifacts"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:ListBucket",
    ]
    resources = [
      aws_s3_bucket.artifacts.arn,
      "${aws_s3_bucket.artifacts.arn}/*",
    ]
  }
  statement {
    sid       = "ECRPull"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "ecs_task" {
  name   = "${local.name}-ecs-task-policy"
  role   = aws_iam_role.ecs_task.id
  policy = data.aws_iam_policy_document.ecs_task.json
}

resource "aws_iam_role" "ecs_execution" {
  name               = "${local.name}-ecs-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
  tags               = local.tags
}

resource "aws_iam_role_policy_attachment" "ecs_execution" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# ── Bedrock Guardrail ──
resource "aws_bedrock_guardrail" "tap" {
  name                      = "${local.name}-guardrail"
  blocked_input_messaging   = "Request blocked by TAP guardrail."
  blocked_outputs_messaging = "Response blocked by TAP guardrail."
  description               = "PII + off-topic filters for TAP RAG"

  content_policy_config {
    filters_config {
      type            = "HATE"
      input_strength  = "HIGH"
      output_strength = "HIGH"
    }
    filters_config {
      type            = "VIOLENCE"
      input_strength  = "HIGH"
      output_strength = "HIGH"
    }
  }

  topic_policy_config {
    topics_config {
      name       = "off-topic"
      definition = "Requests unrelated to TAP CloudOps, threat intel, or infrastructure"
      examples   = ["Write me a poem", "Help me with my taxes"]
      type       = "DENY"
    }
  }

  tags = local.tags
}

# ── CloudWatch log groups ──
resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${local.name}/api"
  retention_in_days = 30
  tags              = local.tags
}

resource "aws_cloudwatch_log_group" "ui" {
  name              = "/ecs/${local.name}/ui"
  retention_in_days = 30
  tags              = local.tags
}

# ── ECS Cluster ──
resource "aws_ecs_cluster" "main" {
  name = local.name
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
  tags = local.tags
}

# Outputs for CI/CD
output "s3_bucket" {
  value = aws_s3_bucket.artifacts.bucket
}

output "ecr_api_url" {
  value = aws_ecr_repository.api.repository_url
}

output "ecr_ui_url" {
  value = aws_ecr_repository.ui.repository_url
}

output "ecs_cluster_name" {
  value = aws_ecs_cluster.main.name
}

output "bedrock_guardrail_id" {
  value = aws_bedrock_guardrail.tap.guardrail_id
}

output "ecs_task_role_arn" {
  value = aws_iam_role.ecs_task.arn
}
