# GeoIP Terraform Module

## Location

Module path: `modules/geoip`

## What It Provisions

- S3 bucket for MaxMind MMDB sync
- `aws_route53` records for GeoIP service discovery
- IAM role for ECS task download of MMDB files
- CloudWatch alarms on sync lag > 24h

## Usage

```hcl
module "geoip" {
  source      = "./modules/geoip"
  environment = "dev"
  vpc_id      = var.vpc_id
}
```

## State

Terraform state is stored in S3 backend `s3://tap-tf-state/geoip/terraform.tfstate`.
Do **not** run `terraform state rm` on active resources without elevated approval.
