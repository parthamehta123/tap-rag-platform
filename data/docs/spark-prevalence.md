# TAP Spark Prevalence Job

## Overview

The weekly prevalence aggregation job runs on Databricks (PySpark) and writes
parquet partitions to S3:

```
s3://tap-datalake/prevalence/dt=YYYY-MM-DD/
```

## Schedule

- Cadence: weekly (Sunday 02:00 UTC)
- Cluster: Databricks job cluster `tap-prevalence-weekly`
- Driver: `i3.xlarge`, Workers: 4x `i3.xlarge`

## Output Schema

| Column | Type | Description |
|--------|------|-------------|
| hash | string | SHA256 |
| prev_week_count | long | Prior week prevalence |
| curr_week_count | long | Current week prevalence |
| delta_pct | double | Percent change |
| vertical | string | Industry vertical |

## Downstream

Anomaly classification (LoRA) consumes these partitions and emits
`SUSPICIOUS` / `BENIGN` / `INVESTIGATE` labels for analyst triage.
