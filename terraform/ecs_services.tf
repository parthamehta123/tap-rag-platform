# Optional ECS Fargate service definitions (wire VPC/subnets via tfvars)

variable "vpc_id" {
  type    = string
  default = ""
}

variable "private_subnet_ids" {
  type    = list(string)
  default = []
}

variable "public_subnet_ids" {
  type    = list(string)
  default = []
}

variable "api_image" {
  type    = string
  default = ""
}

variable "ui_image" {
  type    = string
  default = ""
}

# Task definitions are applied when images + networking are provided.
# See terraform/environments/dev/terraform.tfvars.example
