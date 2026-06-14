# XGB autonomous agent — infrastructure (OpenTofu)

Least-privilege AWS access for the autonomous Claude Code agent that searches
for the best XGB model overnight (on a Mac mini) and deploys a **$50-capped**
live experiment to a dedicated EC2 box. Modeled on `crypto_portfolio/infra/terraform`.

Full design + the off-AWS steps (HL wallet, funding, fee measurement) live in
[`../../autonomous_agent/SETUP.md`](../../autonomous_agent/SETUP.md). This module
covers the AWS resources only.

## What it creates

| Resource | Purpose |
|---|---|
| `xgb-hl-dev-experiment-instance` role + instance profile | EC2 role: reads ONLY `/agent/hl/*` ($50 key), denies `/bot/*`, S3 pull/results |
| `xgb-hl-dev-agent-research` role | What the Mac mini assumes: S3 data read, own prefixes write, SSM **only** to the tagged experiment box; denies secrets/live-box/compute |
| `xgb-hl-dev-agent-bootstrap` user | Minimal user that can only `sts:AssumeRole` the agent role (its key lives on the Mac mini) |
| `/agent/hl/private_key`, `/agent/hl/wallet_address` | SSM params (key value set out-of-band; TF never stores it) |
| `aws_instance` `xgb-hl-dev-experiment` | The capped experiment box, tagged `agent-access=xgb-experiment`, SSM-only |
| S3 `.keep` markers | `research/`, `deploy/experiment/`, `xgb_bot_experiment/` |

The live bot host (`i-04e6b054a8d920a83`) and the main wallet key (`/bot/*`) are
**explicitly denied** to the agent.

## State backend

Reuses the existing backend bucket `catorce-crypto-platform-tfstate` (DynamoDB lock
`crypto-platform-dev-tfstate-lock`) with an isolated key `xgb-hyperliquid/dev/terraform.tfstate`.
No new bootstrap needed.

## Apply

```bash
cd hft/xgb_hyperliquid/infra/terraform
cp terraform.tfvars.example terraform.tfvars   # fill in owner + experiment_wallet_address
export AWS_PROFILE=admin

tofu init
tofu plan      # REVIEW before apply
tofu apply
```

## After apply (manual — by design)

These stay out of Terraform on purpose (secrets must not enter state):

```bash
# 1. Real experiment key (TF created only a placeholder)
aws ssm put-parameter --name /agent/hl/private_key --type SecureString \
  --value '<EXPERIMENT_API_PRIVATE_KEY>' --overwrite --region us-east-1

# 2. Bootstrap access key (kept out of state) -> goes to the Mac mini
aws iam create-access-key --user-name xgb-hl-dev-agent-bootstrap

# 3. Verify the boundaries (SETUP.md Part 8): the 4 denied actions must fail.
```

`tofu output mac_mini_aws_config` prints the `~/.aws/config` profile block for the Mac mini.

## Notes

- Resource-creation of the bootstrap **access key** is intentionally NOT in TF (would
  put the secret in state). Same for the private key **value**.
- The experiment instance is tagged `agent-access=xgb-experiment`; the agent's SSM
  permission keys on that tag, so retagging an instance is how you (de)authorize it.
- `tofu destroy` removes the SSM params too — re-put the key after any recreate.
