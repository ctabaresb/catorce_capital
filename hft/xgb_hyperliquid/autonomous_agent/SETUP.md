# Autonomous XGB Agent — Setup runbook (MacPro / AWS)

Setup que se ejecuta **en la MacPro** (máquina de admin + dev). La **Mac mini** (donde correrá
el agente de noche) solo recibe las llaves del agente al final (Parte 10).

Modelo de seguridad (recordatorio):
- La única cosa que puede mover dinero es una **private key de HL**. El agente **nunca** la tiene.
- El tope duro es el **capital fondeado ($50)** en una **wallet separada** — impuesto por el exchange.
- El agente jamás toca la instancia/wallet live (`i-04e6b054a8d920a83`, wallet `0x1265c59…`).
- Las fees se manejan con **doble costo** (`COST_STAKED` para decisiones/selección, `COST_EXPERIMENT`
  para el tracking real de los $50). Las fees son deterministas → el contrafactual "¿rentable en mi
  cuenta con HYPE?" se computa exacto.

> **Terraform es la vía canónica para las Partes 3-7** (SSM params, S3, IAM, instancia,
> guardrails). Corre el módulo en `../infra/terraform/` (`tofu plan` → `tofu apply`) en vez
> de los comandos CLI de esas partes. Lo que **sí** sigue siendo manual: Parte 0 (tooling),
> Parte 1-2 (wallet HL + fees), poner la private key real con `aws ssm put-parameter`,
> `aws iam create-access-key` del bootstrap user, Parte 8 (verificar) y Parte 9 (perfil).
> Los comandos CLI de abajo quedan como referencia / alternativa sin Terraform.

Variables base (córrelas primero en cada terminal):
```bash
export AWS_PROFILE=admin            # tu perfil de admin de AWS
export REGION=us-east-1
export ACCT=454851577001
export LIVE_INSTANCE=i-04e6b054a8d920a83   # INTOCABLE para el agente
```

---

## Parte 0 — Tooling local en la MacPro

```bash
# 0.1 AWS CLI v2 (debe ser v2)
aws --version                       # si no es 2.x → brew install awscli

# 0.2 Session Manager plugin (necesario para `aws ssm start-session`)
brew install --cask session-manager-plugin
session-manager-plugin               # debe imprimir "The Session Manager plugin was installed successfully"

# 0.3 jq (usado en la verificación)
which jq || brew install jq

# 0.4 Python 3.12 + deps del pipeline (ya los usas; solo verifica)
python3.12 --version
python3.12 -c "import xgboost, pandas, numpy, boto3, hyperliquid; print('deps OK')"
```

---

## Parte 1 — Wallet de Hyperliquid (fuera de AWS)

1. Crea un **wallet nuevo** (semilla nueva, EOA distinto a `0x1265c59…`).
2. Deposita **exactamente $50 USDC** en ese wallet en HL. ← tope duro de pérdida.
3. Genera una **API wallet** (agent wallet) para esa cuenta. Anota:
   - private key de la API wallet → `<EXPERIMENT_API_PRIVATE_KEY>`
   - dirección de la cuenta principal del experimento → `<EXPERIMENT_WALLET_ADDR>`

> NO la linkees a tu staking (sería de suma cero con tu wallet principal). Queda sin descuento de
> staking a propósito; eso lo compensamos con el doble costo en la evaluación.

---

## Parte 2 — Medir fees reales (define los dos CostModel)

```bash
cd ~/Documents/GitHub/catorce_capital/hft/xgb_hyperliquid

# Tu cuenta REAL con HYPE staking → COST_STAKED
python3 hl_fee_check.py --wallet 0x1265c59536ee727eDB942EBF30fA1878BB659847

# La wallet del experimento (sin staking, tier 0) → COST_EXPERIMENT
python3 hl_fee_check.py --wallet <EXPERIMENT_WALLET_ADDR>
```
Anota los bps taker/maker por lado de cada una. Esperado aprox (confírmalo con la salida real):
- `COST_STAKED   ≈ CostModel(4.05, 4.05, 0.0) = 8.10 bps RT`  (Bronze 10% staking; = tu `COST_OBSERVED`)
- `COST_EXPERIMENT ≈ CostModel(4.5, 4.5, 0.0) = 9.00 bps RT`  (sin staking, tier 0 de volumen)

Si tienes más de 100 HYPE stakeados (Silver/Gold/…), tu descuento real es mayor y la brecha
crece — por eso se mide, no se asume.

---

## Parte 3 — SSM Parameter Store (la llave de $50)

```bash
aws ssm put-parameter --region $REGION --name /agent/hl/private_key \
  --type SecureString --value '<EXPERIMENT_API_PRIVATE_KEY>'

aws ssm put-parameter --region $REGION --name /agent/hl/wallet_address \
  --type String --value '<EXPERIMENT_WALLET_ADDR>'

aws ssm get-parameters-by-path --region $REGION --path /agent/hl --query 'Parameters[].Name'
```

---

## Parte 4 — Prefijos S3 (marcadores)

```bash
printf '' | aws s3 cp - s3://hyperliquid-orderbook/research/.keep
printf '' | aws s3 cp - s3://hyperliquid-orderbook/deploy/experiment/.keep
printf '' | aws s3 cp - s3://hyperliquid-orderbook/xgb_bot_experiment/.keep
```

---

## Parte 5 — IAM

### 5a. Rol de la instancia de experimento
```bash
cat > /tmp/trust-ec2.json <<'JSON'
{ "Version":"2012-10-17","Statement":[
  {"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"} ] }
JSON

cat > /tmp/policy-instance.json <<JSON
{ "Version":"2012-10-17","Statement":[
  {"Sid":"ReadExperimentKeyOnly","Effect":"Allow",
   "Action":["ssm:GetParameter","ssm:GetParameters"],
   "Resource":"arn:aws:ssm:$REGION:$ACCT:parameter/agent/hl/*"},
  {"Sid":"HardDenyMainBotKey","Effect":"Deny",
   "Action":["ssm:GetParameter","ssm:GetParameters","ssm:GetParametersByPath"],
   "Resource":"arn:aws:ssm:$REGION:$ACCT:parameter/bot/*"},
  {"Sid":"DecryptViaSSM","Effect":"Allow","Action":["kms:Decrypt"],
   "Resource":"*","Condition":{"StringEquals":{"kms:ViaService":"ssm.$REGION.amazonaws.com"}}},
  {"Sid":"PullModelsAndCode","Effect":"Allow",
   "Action":["s3:GetObject","s3:ListBucket"],
   "Resource":[
     "arn:aws:s3:::hyperliquid-orderbook",
     "arn:aws:s3:::hyperliquid-orderbook/deploy/experiment/*",
     "arn:aws:s3:::hyperliquid-orderbook/research/*"]},
  {"Sid":"WriteLiveResults","Effect":"Allow","Action":["s3:PutObject"],
   "Resource":"arn:aws:s3:::hyperliquid-orderbook/xgb_bot_experiment/*"} ] }
JSON

aws iam create-role --role-name xgb-experiment-instance-role \
  --assume-role-policy-document file:///tmp/trust-ec2.json
aws iam put-role-policy --role-name xgb-experiment-instance-role \
  --policy-name xgb-experiment-instance --policy-document file:///tmp/policy-instance.json
aws iam attach-role-policy --role-name xgb-experiment-instance-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore

aws iam create-instance-profile --instance-profile-name xgb-experiment-instance-profile
aws iam add-role-to-instance-profile \
  --instance-profile-name xgb-experiment-instance-profile \
  --role-name xgb-experiment-instance-role
```

### 5b. Usuario bootstrap (sus llaves van a la Mac mini)
```bash
aws iam create-user --user-name xgb-agent-bootstrap
```

### 5c. Rol del agente (lo que asume la Mac mini)
```bash
cat > /tmp/trust-agent.json <<JSON
{ "Version":"2012-10-17","Statement":[
  {"Effect":"Allow",
   "Principal":{"AWS":"arn:aws:iam::$ACCT:user/xgb-agent-bootstrap"},
   "Action":"sts:AssumeRole"} ] }
JSON

cat > /tmp/policy-agent.json <<JSON
{ "Version":"2012-10-17","Statement":[
  {"Sid":"ListDataPrefixes","Effect":"Allow","Action":["s3:ListBucket"],
   "Resource":["arn:aws:s3:::hyperliquid-orderbook","arn:aws:s3:::bitso-orderbook"],
   "Condition":{"StringLike":{"s3:prefix":[
     "hyperliquid_dom_parquet/*","hyperliquid_metrics_parquet/*",
     "hyperliquid_dom_snapshots/*","hyperliquid_metrics_snapshots/*",
     "data/lead_lag/*","research/*","deploy/experiment/*","xgb_bot_experiment/*"]}}},
  {"Sid":"ReadMarketData","Effect":"Allow","Action":["s3:GetObject"],
   "Resource":[
     "arn:aws:s3:::hyperliquid-orderbook/hyperliquid_dom_parquet/*",
     "arn:aws:s3:::hyperliquid-orderbook/hyperliquid_metrics_parquet/*",
     "arn:aws:s3:::hyperliquid-orderbook/hyperliquid_dom_snapshots/*",
     "arn:aws:s3:::hyperliquid-orderbook/hyperliquid_metrics_snapshots/*",
     "arn:aws:s3:::bitso-orderbook/data/lead_lag/*"]},
  {"Sid":"ReadResearchAndResults","Effect":"Allow","Action":["s3:GetObject"],
   "Resource":[
     "arn:aws:s3:::hyperliquid-orderbook/research/*",
     "arn:aws:s3:::hyperliquid-orderbook/deploy/experiment/*",
     "arn:aws:s3:::hyperliquid-orderbook/xgb_bot_experiment/*"]},
  {"Sid":"WriteCandidatesAndResearch","Effect":"Allow","Action":["s3:PutObject"],
   "Resource":[
     "arn:aws:s3:::hyperliquid-orderbook/research/*",
     "arn:aws:s3:::hyperliquid-orderbook/deploy/experiment/*"]},
  {"Sid":"SSMonlyExperimentTaggedInstances","Effect":"Allow",
   "Action":["ssm:StartSession"],
   "Resource":"arn:aws:ec2:$REGION:$ACCT:instance/*",
   "Condition":{"StringEquals":{"ssm:resourceTag/Project":"xgb-experiment"}}},
  {"Sid":"SSMSessionDoc","Effect":"Allow","Action":["ssm:StartSession"],
   "Resource":"arn:aws:ssm:$REGION:$ACCT:document/SSM-SessionManagerRunShell"},
  {"Sid":"ManageOwnSessions","Effect":"Allow",
   "Action":["ssm:TerminateSession","ssm:ResumeSession"],
   "Resource":"arn:aws:ssm:*:$ACCT:session/\${aws:userid}-*"},
  {"Sid":"DescribeForConnect","Effect":"Allow",
   "Action":["ssm:DescribeInstanceInformation","ssm:DescribeSessions","ec2:DescribeInstances"],
   "Resource":"*"},
  {"Sid":"NeverShellTheLiveBox","Effect":"Deny",
   "Action":["ssm:StartSession","ssm:SendCommand"],
   "Resource":"arn:aws:ec2:$REGION:$ACCT:instance/$LIVE_INSTANCE"},
  {"Sid":"HardDenyOwnSecretReads","Effect":"Deny",
   "Action":["ssm:GetParameter","ssm:GetParameters","ssm:GetParametersByPath","secretsmanager:GetSecretValue"],
   "Resource":"*"},
  {"Sid":"HardDenyComputeAndIAM","Effect":"Deny",
   "Action":["ec2:RunInstances","ec2:TerminateInstances","ec2:StartInstances","ec2:StopInstances",
             "iam:*","s3:DeleteObject","s3:DeleteBucket","s3:PutBucketPolicy"],
   "Resource":"*"} ] }
JSON

aws iam create-role --role-name xgb-agent-research-role \
  --assume-role-policy-document file:///tmp/trust-agent.json --max-session-duration 43200
aws iam put-role-policy --role-name xgb-agent-research-role \
  --policy-name xgb-agent-research --policy-document file:///tmp/policy-agent.json
```

### 5d. Política del usuario bootstrap + llaves
```bash
cat > /tmp/policy-bootstrap.json <<JSON
{ "Version":"2012-10-17","Statement":[
  {"Effect":"Allow","Action":"sts:AssumeRole",
   "Resource":"arn:aws:iam::$ACCT:role/xgb-agent-research-role"} ] }
JSON

aws iam put-user-policy --user-name xgb-agent-bootstrap \
  --policy-name assume-research-role --policy-document file:///tmp/policy-bootstrap.json

# Genera las llaves — GUÁRDALAS (van a la Mac mini, NO a git)
aws iam create-access-key --user-name xgb-agent-bootstrap
```

---

## Parte 6 — Instancia de experimento

```bash
AMI=$(aws ssm get-parameter --region $REGION \
  --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
  --query Parameter.Value --output text)

SG=$(aws ec2 create-security-group --region $REGION \
  --group-name xgb-experiment-sg --description "xgb experiment egress only" \
  --query GroupId --output text)

aws ec2 run-instances --region $REGION \
  --image-id $AMI --instance-type t3.small \
  --iam-instance-profile Name=xgb-experiment-instance-profile \
  --security-group-ids $SG \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Project,Value=xgb-experiment},{Key=Name,Value=xgb-experiment}]' \
  --query 'Instances[0].InstanceId' --output text
```
Requiere salida a internet (subnet pública con IGW, NAT, o VPC endpoints de ssm/ssmmessages/ec2messages).
Verifica que aparezca en SSM (1-2 min):
```bash
aws ssm describe-instance-information --region $REGION \
  --query "InstanceInformationList[].[InstanceId,PingStatus]" --output table
```

---

## Parte 7 — Guardrails

```bash
aws cloudtrail describe-trails --region $REGION --query 'trailList[].Name'
```
- Si no hay trail → créalo (CloudTrail → Create trail, management events).
- Crea un AWS Budget de alerta (~$20/mes).

---

## Parte 8 — VERIFICAR las fronteras (no te lo saltes)

```bash
CREDS=$(aws sts assume-role \
  --role-arn arn:aws:iam::$ACCT:role/xgb-agent-research-role \
  --role-session-name verify --query Credentials --output json)
export AWS_ACCESS_KEY_ID=$(echo $CREDS | jq -r .AccessKeyId)
export AWS_SECRET_ACCESS_KEY=$(echo $CREDS | jq -r .SecretAccessKey)
export AWS_SESSION_TOKEN=$(echo $CREDS | jq -r .SessionToken)

# DEBE FUNCIONAR:
aws s3 ls s3://hyperliquid-orderbook/hyperliquid_metrics_parquet/ | head

# DEBEN FALLAR con AccessDenied:
aws ssm get-parameter --name /agent/hl/private_key --with-decryption --region $REGION
aws ssm get-parameter --name /bot/hl/private_key   --with-decryption --region $REGION
aws ssm start-session --target $LIVE_INSTANCE --region $REGION
aws s3 cp /etc/hostname s3://hyperliquid-orderbook/xgb_models/live_v8/hack.txt

unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
```
4 deniegos + el `s3 ls` OK = barreras bien puestas.

---

## Parte 9 — Preparar el perfil del agente (para la Mac mini)

Las access keys de la Parte 5d van a la **Mac mini** en `~/.aws/credentials` + `~/.aws/config`:
```ini
# ~/.aws/credentials  (Mac mini)
[xgb-agent-bootstrap]
aws_access_key_id = <ACCESS_KEY_ID>
aws_secret_access_key = <SECRET>

# ~/.aws/config  (Mac mini)
[profile xgb-agent]
role_arn = arn:aws:iam::454851577001:role/xgb-agent-research-role
source_profile = xgb-agent-bootstrap
region = us-east-1
```
El agente usa `AWS_PROFILE=xgb-agent` → el SDK asume el rol y refresca creds cortas solo.
(Si la Mac mini y la MacPro fueran la misma máquina, mismo perfil, sin transferir nada.)

---

## Lo que construyo yo después (NO es tu setup manual)

1. **Cost presets** en `data/targets.py`: `COST_STAKED` y `COST_EXPERIMENT` (de la Parte 2), y
   wiring de `COST_STAKED` en sweep/holdout/retrain + en el decision-cost del bot.
   - ⚠️ Tienes WIP sin commitear en `retrain_no_bnvol.py` y `xgb_bot.py` (migración v8 a 8.10).
     Lo reconcilio con cuidado para no pisarlo.
2. **Eval de doble costo**: por cada fill, `net_staked` (go/no-go) y `net_experiment` ($50 real).
3. **Harness de instrumentación shadow**: diff features live vs training (cero riesgo, alta señal).
4. **Launcher del EC2**: systemd que clampa `--size/--max_loss` y corre live capado.
5. **Harness autónomo de Claude Code** para la Mac mini (loop nocturno + allowlist de bash).
