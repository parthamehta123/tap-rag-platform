# TAP CloudOps Overview

Talos Analytics Platform (TAP) processes billions of telemetry events daily —
file hashes with prevalence counts, authentication logs, and DNS queries.

Engineers use the RAG chatbot for natural-language access to CloudOps knowledge:
Terraform configs, Spark jobs, ClickHouse schemas, and operational runbooks.

## Key Principles

1. Ground answers in retrieved docs (no hallucination)
2. Cite sources for every operational claim
3. Route destructive agent actions through sandbox + PR approval
4. Keep threat intel in-VPC via Bedrock VPC endpoints
