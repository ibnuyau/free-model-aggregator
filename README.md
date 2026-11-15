# Free Model Aggregator

One server that serves the OpenAI Chat Completions API and the Anthropic Messages API
over the ten free providers of a trial, in JSON and in server-sent events.

## How it works

- Reads the brief at `$MAKEFUN_PROMPT_FILE` (provider directory, quality table, workload).
- Probes every provider's catalog route in its own wire and builds one catalog, which
  `node main.ts --list` prints and the register body carries, byte for byte.
- Registers its port and catalog with the env, then serves `GET /v1/models`
  (OpenAI shape for a Bearer token, Anthropic shape for an `x-api-key`),
  `POST /v1/chat/completions` and `POST /v1/messages`.
- For each request it picks the best free model that meets the request's requirements,
  ranked by the quality table's `rating_milli`; a JSON-mode OpenAI request only goes to
  a provider whose wire echoes JSON mode (openai or gemini). When a provider refuses
  (402, 429 or 5xx) it retries the same provider once — a refusal may be one scheduled
  failure — then moves to the next-best candidate; a provider that refuses twice running
  is left alone for the rest of the trial. Calls to one provider run one at a time.
- Request and response translation covers all three upstream wires (openai, anthropic,
  gemini) in both directions, JSON and SSE: the system prompt, user text, tool names,
  `max_tokens`, stop sequences, image bytes and JSON mode all round-trip, and the
  provider's answer text is passed through verbatim. Usage and stop reasons are
  translated into the client's wire.
- Polls `GET /work` and exits 0 when the env says the workload is done. No wall clock
  in any answer and no unseeded randomness, so a replay reproduces the run.

## Files

- `main.ts` — the server: brief, catalog, routing, register and poll.
- `lib/brief.ts` — the brief's three documents, ratings, requirements.
- `lib/catalog.ts` — provider catalog probes and the catalog document.
- `lib/translate.ts` — the client wires and the upstream wires, JSON and SSE.

## Grade locally

```
node <kit>/cli/makefun.mjs grade-local --kit <kit> --bot . --driver process --instances 3
```

Practice trials 0–2 score 99 (quality 99, served 100, wire 100, catalog 100,
failover 100), with the replay matching byte for byte.

## Run it

```
node main.ts --list   # print the catalog and exit
node main.ts          # serve a trial (needs MAKEFUN_PROMPT_FILE and MAKEFUN_ENV_URL)
```
