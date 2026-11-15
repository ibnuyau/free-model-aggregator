// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// For each request the client sends, this server picks the best free model that meets the
// request's requirements (the quality table's rating_milli decides "best"), speaks that
// provider's wire, and answers back in the client's wire, in JSON or in server-sent events.
// When a provider refuses (402, 429 or 5xx) it retries the same provider once — a refusal
// may be a single scheduled failure — and then moves to the next-best candidate; a provider
// that refuses twice in a row is left alone for the rest of the trial, since a spent budget
// or an outage never comes back. One provider serves one call at a time, so calls to the
// same provider are serialised.
//
// The server reads the brief at $MAKEFUN_PROMPT_FILE, probes every provider's catalog route,
// registers its port and catalog with the env at $MAKEFUN_ENV_URL, serves the workload, then
// polls GET /work and exits 0 when the env says done.
//
// `node main.ts --list` prints the catalog and exits 0. Everything is deterministic: no wall
// clock in any answer (fixed timestamps), no unseeded randomness, and every choice follows
// from the brief, the requests and the providers' answers, so a replay reproduces the run.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { buildCatalog, probeProvider, probeCaps, type Catalog, type ListedModel, type Probe } from './lib/catalog.ts';
import {
  capsOf,
  isFreeModel,
  isRecord,
  joinBase,
  authHeaders,
  meetsNeeds,
  ratingRow,
  readBrief,
  type Brief,
  type Json,
  type Need,
  type ProviderEntry,
} from './lib/brief.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicStreamToOpenAI,
  anthropicToOpenAI,
  anthropicToOpenAIResponse,
  anthropicToGemini,
  geminiStreamToAnthropic,
  geminiStreamToOpenAI,
  geminiToAnthropicResponse,
  geminiToOpenAIResponse,
  openAIError,
  openAIModelList,
  openAIStreamToAnthropic,
  openAIToAnthropic,
  openAIToAnthropicResponse,
  openAIToGemini,
  textOf,
  type ServedModel,
} from './lib/translate.ts';

const PROTOCOL_VERSION = 1;
const POLL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const PROVIDER_TIMEOUT_MS = 15000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: Catalog = { providers: [] };
/** Client requests served so far, in arrival order: the manifest's row for each. */
let servedCount = 0;
/** Requests whose answer is still being written: the poll must not exit while one is open. */
let requestsInFlight = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.providers.map((p) => probeProvider(ENV_URL, p, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(probes, brief);
}

// ---------------------------------------------------------------------------
// Routing: rank the free candidates by rating, with failover and dead providers.
// ---------------------------------------------------------------------------

interface Candidate {
  probe: Probe;
  model: ListedModel;
  rating: number;
}

/** A provider's calls run one at a time: a promise chain per provider id. */
const providerQueues = new Map<string, Promise<void>>();
function runOnProvider<T>(providerId: string, task: () => Promise<T>): Promise<T> {
  const prior = providerQueues.get(providerId) ?? Promise.resolve();
  let release!: () => void;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  providerQueues.set(providerId, prior.then(() => gate));
  return prior.then(task).finally(release);
}

/** Providers that refused twice running: a spent budget or an outage never comes back. */
const deadProviders = new Set<string>();
/** Consecutive refusals per provider, reset by any success. */
const refusals = new Map<string, number>();

function noteRefusal(providerId: string): void {
  const n = (refusals.get(providerId) ?? 0) + 1;
  refusals.set(providerId, n);
  if (n >= 2) deadProviders.add(providerId);
}

function noteSuccess(providerId: string): void {
  refusals.set(providerId, 0);
}

/**
 * Every free model meeting `needs`, best rating first. A JSON-mode OpenAI request can only go
 * to a provider whose wire echoes JSON mode (openai or gemini): an anthropic-wire provider
 * would answer json_mode false and fail the echo check.
 */
function candidatesFor(needs: Need, jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok || deadProviders.has(probe.provider.id)) continue;
    if (jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingRow(brief, probe.provider.id, model.id);
      if (!isFreeModel(row, model.free)) continue;
      if (!meetsNeeds(needs, capsOf(row, probeCaps(brief, probe.provider.id, model)))) continue;
      out.push({ probe, model, rating: row?.rating ?? 0 });
    }
  }
  out.sort((a, b) => b.rating - a.rating || (a.probe.provider.id < b.probe.provider.id ? -1 : 1) || (a.model.id < b.model.id ? -1 : 1));
  return out;
}

/** The free models this server lists on GET /v1/models, de-duplicated by id. */
function servedModels(): ServedModel[] {
  const seen = new Set<string>();
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      if (seen.has(model.id)) continue;
      const row = ratingRow(brief, probe.provider.id, model.id);
      if (!isFreeModel(row, model.free)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, displayName: model.displayName });
    }
  }
  return out;
}

/** The request's needs: the manifest's row by arrival order, else what its body implies. */
function needsFor(body: Json, api: 'openai' | 'anthropic'): { needs: Need; jsonMode: boolean } {
  const entry = brief.workload[servedCount];
  servedCount += 1;
  const jsonMode = api === 'openai' && isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  if (entry !== undefined) return { needs: entry.needs, jsonMode };
  // No manifest row: infer from the body.
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  return {
    needs: { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: jsonMode },
    jsonMode,
  };
}

// ---------------------------------------------------------------------------
// Upstream calls: one request in the provider's wire.
// ---------------------------------------------------------------------------

interface UpstreamOk {
  ok: true;
  res: Response;
  candidate: Candidate;
}

interface UpstreamFail {
  ok: false;
  status: number;
  detail: string;
}

type UpstreamResult = UpstreamOk | UpstreamFail;

/** Build one provider's request body for a client request. */
function upstreamBody(clientApi: 'openai' | 'anthropic', body: Json, candidate: Candidate): { url: string; init: RequestInit } {
  const provider: ProviderEntry = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  const stream = body.stream === true;
  if (provider.wire === 'openai') {
    if (stream) headers.accept = 'text/event-stream';
    const payload = clientApi === 'openai' ? { ...body, model: candidate.model.id } : anthropicToOpenAI(body, candidate.model.id);
    // Always ask for the usage chunk: the scorer reads the stream's final usage, and the
    // client does not always ask for it. An extra usage chunk harms no shape check.
    if (stream) payload.stream_options = { include_usage: true };
    return { url: `${base}/v1/chat/completions`, init: { method: 'POST', headers, body: JSON.stringify(payload), signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS) } };
  }
  if (provider.wire === 'anthropic') {
    if (stream) headers.accept = 'text/event-stream';
    const payload = clientApi === 'anthropic' ? { ...body, model: candidate.model.id } : openAIToAnthropic(body, candidate.model.id);
    return { url: `${base}/v1/messages`, init: { method: 'POST', headers, body: JSON.stringify(payload), signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS) } };
  }
  // Gemini: the model rides in the path; streaming takes :streamGenerateContent?alt=sse.
  const encoded = encodeURIComponent(candidate.model.id);
  const payload = clientApi === 'openai' ? openAIToGemini(body) : anthropicToGemini(body);
  const url = stream ? `${base}/v1beta/models/${encoded}:streamGenerateContent?alt=sse` : `${base}/v1beta/models/${encoded}:generateContent`;
  if (stream) headers.accept = 'text/event-stream';
  return { url, init: { method: 'POST', headers, body: JSON.stringify(payload), signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS) } };
}

/**
 * Try the candidates in rating order until one answers. A refusal may be one scheduled
 * failure, so the same candidate is tried once more before moving on; a provider that
 * refuses twice running is marked dead. Returns the first success, or the last failure.
 */
async function callUpstream(clientApi: 'openai' | 'anthropic', body: Json, candidates: Candidate[]): Promise<UpstreamResult> {
  let lastFail: UpstreamFail = { ok: false, status: 503, detail: 'no free model meets this request' };
  const tried = new Set<string>();
  for (const candidate of candidates) {
    const key = `${candidate.probe.provider.id}/${candidate.model.id}`;
    if (tried.has(key)) continue;
    // A provider that died on an earlier candidate of this same request is skipped.
    if (deadProviders.has(candidate.probe.provider.id)) continue;
    tried.add(key);
    for (let attempt = 0; attempt < 2; attempt++) {
      const { url, init } = upstreamBody(clientApi, body, candidate);
      let res: Response;
      try {
        res = await runOnProvider(candidate.probe.provider.id, () => fetch(url, init));
      } catch (err) {
        lastFail = { ok: false, status: 502, detail: (err as Error).message };
        break;
      }
      if (res.ok) {
        noteSuccess(candidate.probe.provider.id);
        return { ok: true, res, candidate };
      }
      const detail = (await res.text().catch(() => '')).slice(0, 300);
      lastFail = { ok: false, status: res.status, detail: detail === '' ? `HTTP ${res.status}` : detail };
      noteRefusal(candidate.probe.provider.id);
      if (deadProviders.has(candidate.probe.provider.id)) break;
      // One retry on the same candidate: a single scheduled failure, not a spent budget.
      if (attempt === 0) continue;
      break;
    }
    if (deadProviders.has(candidate.probe.provider.id)) {
      // The provider is gone; the next candidate is a different provider.
    }
  }
  return lastFail;
}

// ---------------------------------------------------------------------------
// Answering: the provider's answer rewritten into the client's wire.
// ---------------------------------------------------------------------------

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** Copy a provider's event stream to the client, converted into the client's wire. */
async function pipeConverted(
  upstream: Response,
  res: ServerResponse,
  clientApi: 'openai' | 'anthropic',
  providerWire: ProviderEntry['wire'],
  model: string,
): Promise<void> {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  const body = upstream.body;
  if (body === null) {
    res.end();
    return;
  }
  const events =
    clientApi === 'openai'
      ? providerWire === 'anthropic'
        ? anthropicStreamToOpenAI(body, model)
        : providerWire === 'gemini'
          ? geminiStreamToOpenAI(body, model)
          : null
      : providerWire === 'openai'
        ? openAIStreamToAnthropic(body, model)
        : providerWire === 'gemini'
          ? geminiStreamToAnthropic(body, model)
          : null;
  if (events === null) {
    // Same wire: the provider's stream is already the client's shape.
    const reader = body.getReader();
    try {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        res.write(Buffer.from(value));
      }
    } finally {
      reader.releaseLock();
      res.end();
    }
    return;
  }
  for await (const event of events) res.write(event);
  res.end();
}

/** Convert one provider JSON answer into the client's wire and send it. */
async function sendConverted(
  upstream: Response,
  res: ServerResponse,
  clientApi: 'openai' | 'anthropic',
  providerWire: ProviderEntry['wire'],
  model: string,
): Promise<void> {
  const answer: unknown = await upstream.json().catch(() => null);
  if (!isRecord(answer)) {
    return sendError(res, clientApi, 502, clientApi === 'anthropic' ? 'api_error' : 'server_error', 'the provider answered with a body that is not JSON');
  }
  if (clientApi === 'openai') {
    if (providerWire === 'anthropic') return sendJson(res, 200, anthropicToOpenAIResponse(answer, model));
    if (providerWire === 'gemini') return sendJson(res, 200, geminiToOpenAIResponse(answer, model));
    return sendJson(res, 200, answer);
  }
  if (providerWire === 'openai') return sendJson(res, 200, openAIToAnthropicResponse(answer, model));
  if (providerWire === 'gemini') return sendJson(res, 200, geminiToAnthropicResponse(answer, model));
  return sendJson(res, 200, answer);
}

async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const { needs, jsonMode } = needsFor(body, 'openai');
  const candidates = candidatesFor(needs, jsonMode);
  if (candidates.length === 0) return sendError(res, 'openai', 503, 'no_free_model', 'no free model meets this request');
  const result = await callUpstream('openai', body, candidates);
  if (!result.ok) return sendError(res, 'openai', result.status === 502 ? 502 : 503, 'upstream_error', `every provider refused: ${result.detail}`);
  const { res: upstream, candidate } = result;
  if (body.stream === true && upstream.body !== null) {
    await pipeConverted(upstream, res, 'openai', candidate.probe.provider.wire, candidate.model.id);
    return;
  }
  await sendConverted(upstream, res, 'openai', candidate.probe.provider.wire, candidate.model.id);
}

async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const { needs } = needsFor(body, 'anthropic');
  const candidates = candidatesFor(needs, false);
  if (candidates.length === 0) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model meets this request');
  const result = await callUpstream('anthropic', body, candidates);
  if (!result.ok) return sendError(res, 'anthropic', result.status === 502 ? 502 : 503, 'api_error', `every provider refused: ${result.detail}`);
  const { res: upstream, candidate } = result;
  if (body.stream === true && upstream.body !== null) {
    await pipeConverted(upstream, res, 'anthropic', candidate.probe.provider.wire, candidate.model.id);
    return;
  }
  await sendConverted(upstream, res, 'anthropic', candidate.probe.provider.wire, candidate.model.id);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number, report: Catalog): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        // Done means every request was sent, not answered: only leave once no answer
        // is still in flight, or the last response would be cut off mid-write.
        if (isRecord(work) && work.done === true && requestsInFlight === 0) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    requestsInFlight += 1;
    void handle(req, res)
      .catch((err: unknown) => {
        log((err as Error).stack ?? String(err));
        if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
      })
      .finally(() => {
        requestsInFlight -= 1;
      });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
