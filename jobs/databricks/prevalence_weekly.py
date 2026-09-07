"""
Databricks / PySpark job: weekly TAP prevalence aggregation → S3 parquet.

Deploy as a Databricks Workflow notebook or `spark-submit` job.
Reads raw telemetry, aggregates prevalence deltas, writes partitioned parquet
consumed by the LoRA anomaly classifier and RAG feedback loops.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def build_spark(app_name: str = "tap-prevalence-weekly") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .getOrCreate()
    )


def aggregate_prevalence(spark: SparkSession, input_path: str, output_path: str, run_date: str) -> None:
    raw = spark.read.parquet(input_path)
    # Expect columns: hash, event_ts, vertical, count
    weekly = (
        raw.withColumn("week", F.date_trunc("week", F.col("event_ts")))
        .groupBy("hash", "vertical", "week")
        .agg(F.sum("count").alias("week_count"))
    )

    curr = weekly.filter(F.col("week") == F.lit(run_date))
    prev = weekly.filter(F.col("week") == F.date_sub(F.lit(run_date), 7)).withColumnRenamed(
        "week_count", "prev_week_count"
    )

    joined = (
        curr.alias("c")
        .join(prev.alias("p"), on=["hash", "vertical"], how="left")
        .select(
            F.col("c.hash").alias("hash"),
            F.col("c.vertical").alias("vertical"),
            F.coalesce(F.col("p.prev_week_count"), F.lit(0)).alias("prev_week_count"),
            F.col("c.week_count").alias("curr_week_count"),
        )
        .withColumn(
            "delta_pct",
            F.when(F.col("prev_week_count") == 0, F.lit(None)).otherwise(
                (F.col("curr_week_count") - F.col("prev_week_count"))
                / F.col("prev_week_count")
                * 100.0
            ),
        )
        .withColumn("dt", F.lit(run_date))
    )

    (
        joined.write.mode("overwrite")
        .partitionBy("dt")
        .parquet(output_path)
    )
    print(f"Wrote prevalence aggregates → {output_path}/dt={run_date}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="s3://…/raw/hash_events/")
    parser.add_argument("--output", required=True, help="s3://…/prevalence/")
    parser.add_argument(
        "--run-date",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="Week start date YYYY-MM-DD",
    )
    args = parser.parse_args()
    spark = build_spark()
    aggregate_prevalence(spark, args.input, args.output, args.run_date)
    spark.stop()


if __name__ == "__main__":
    main()
