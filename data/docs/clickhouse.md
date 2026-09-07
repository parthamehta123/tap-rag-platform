# TAP ClickHouse Access

## Dev Cluster

The TapCloudOps ClickHouse **dev** cluster is reached via:

```bash
clickhouse-client --host ch-dev.tap.internal --port 9000
```

Credentials live in Vault at path `secret/tap/clickhouse/dev`.

## Prod Cluster

Production endpoints:

- Host: `ch-prod.tap.internal`
- Port: `9000` (native), `8123` (HTTP)
- Vault: `secret/tap/clickhouse/prod`

## Common Tables

| Table | Purpose |
|-------|---------|
| `tap.hash_prevalence` | Weekly hash prevalence aggregates |
| `tap.authlog_events` | Authentication telemetry |
| `tap.dns_queries` | DNS resolution events |

## Troubleshooting

If connections time out, confirm you are on the Talos VPN and that security group
`sg-tap-clickhouse-dev` allows your workstation CIDR.
