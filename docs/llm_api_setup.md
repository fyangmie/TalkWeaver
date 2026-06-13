# Optional LLM API Setup

## Why It Is Optional

TalkWeaver can run its complete correction audit with deterministic glossary
rules. An external LLM is optional and is reserved for controlled correction
experiments where provider, model, prompt, cost, and privacy conditions are
recorded.

Tests never require network access or an API key. A failed real API call must
not be reported as successful LLM correction.

## Configuration

Create a local configuration file:

```bash
cp .env.example .env
```

Edit `.env`:

```env
LLM_PROVIDER=deepseek
LLM_API_KEY=replace_with_your_real_key
LLM_MODEL=deepseek-chat
LLM_BASE_URL=https://api.deepseek.com
LLM_TEMPERATURE=0
LLM_TIMEOUT_SECONDS=30
```

`qwen` and `openai` are also supported through OpenAI-compatible
chat-completion endpoints. Corresponding examples are commented in
`.env.example`.

The real `.env` file is ignored by Git. Never paste credentials into source
code, command history, screenshots, result CSVs, issue comments, or commits.
Runtime metadata stores only a masked key indicator.

## Correction Modes

### `rule_fallback`

No API call is attempted. Corrections are limited to retrieved glossary
substitutions:

```bash
python scripts/run_llm_correction_smoke.py --mode rule_fallback
```

Expected metadata:

```text
correction_mode=rule_fallback
api_used=False
fallback_used=False
```

### `llm`

A valid API configuration is required. Missing configuration, network
failure, invalid JSON, or an unsupported rewrite causes a nonzero exit. It
does not silently fall back:

```bash
python scripts/run_llm_correction_smoke.py --mode llm
```

### `llm_with_rule_fallback`

TalkWeaver attempts the configured API and uses deterministic rules only when
the API is unavailable or its response fails grounding validation:

```bash
python scripts/run_llm_correction_smoke.py \
  --mode llm_with_rule_fallback
```

Interpretation:

- `api_used=true, fallback_used=false`: a validated API response was used;
- `api_used=true, fallback_used=true`: an API call was attempted but rejected
  or failed, then rules were used;
- `api_used=false, fallback_used=true`: API configuration was unavailable,
  so rules were used;
- `api_used=false, fallback_used=false`: intentional `rule_fallback`.

## Recorded Metadata

Every `CorrectionAudit` records:

- correction mode;
- provider and model;
- prompt version;
- temperature;
- whether an API call was attempted;
- whether fallback was used;
- unsupported changes;
- review state and hallucination risk.

Current prompt version:

```text
talkweaver.correction.v1
```

The API key is never included in the audit.

## Cost And Privacy

API calls may incur provider charges. Keep smoke tests short, use temperature
zero, and review provider pricing before batch experiments.

Do not send private, restricted, confidential, or non-consented meeting
transcripts to an external provider. Public dataset licenses may also limit
third-party processing. Use deterministic fallback unless the audio and
transcript are permitted for the selected API.

## Troubleshooting

Missing configuration in strict mode produces:

```text
Real LLM correction requested but LLM API configuration is incomplete.
```

Confirm `.env` exists locally and that the selected provider, model, base URL,
and API key are valid. Do not weaken strict `llm` mode to hide failures. Use
`llm_with_rule_fallback` only when an explicitly labeled fallback is
acceptable for the experiment.
