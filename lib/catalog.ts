// Probing each provider's catalog route and building the one catalog document.
//
// The catalog is what `--list` prints and what the register body carries, byte for byte:
// every provider that answered, every model it listed, with the quality table's free flag
// and context length where it has a row. Its base_url is the directory's own path (/p/<id>),
// never the joined URL: the env's origin differs between a live pass and its replay, and the
// catalog is part of the state a replay must reproduce byte for byte.

import { authHeaders, isFreeModel, joinBase, ratingRow, type Brief, type Caps, type ProviderEntry, type RatingRow } from './brief.ts';

export interface ListedModel {
  id: string;
  displayName: string;
  free: boolean | null;
  context: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

export interface Probe {
  provider: ProviderEntry;
  ok: boolean;
  status: number;
  models: ListedModel[];
  error: string | null;
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: 'openai' | 'anthropic' | 'gemini';
  models: CatalogModel[];
}

export interface Catalog {
  providers: CatalogProvider[];
}

const asString = (v: unknown): string => (typeof v === 'string' ? v : '');
const asNumberOrNull = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const asBoolOrNull = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null);

/** A listing's price block says a model is free when every price it names is zero. */
function zeroPriced(v: Record<string, unknown>): boolean | null {
  const pricing = v.pricing;
  if (pricing === null || typeof pricing !== 'object' || Array.isArray(pricing)) return null;
  const values = Object.values(pricing).flatMap((x) => {
    if (typeof x === 'number') return [x];
    if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
    return [];
  });
  if (values.length === 0) return null;
  return values.every((n) => Number.isFinite(n) && n === 0);
}

/** One model as a listing wrote it, in whichever wire's spelling. */
function listedModel(wire: ProviderEntry['wire'], v: unknown): ListedModel | null {
  if (v === null || typeof v !== 'object' || Array.isArray(v)) return null;
  const rec = v as Record<string, unknown>;
  const id = wire === 'gemini' ? asString(rec.name).replace(/^models\//, '') : asString(rec.id);
  if (id === '') return null;
  return {
    id,
    displayName: asString(rec.display_name) || asString(rec.displayName) || id,
    free: asBoolOrNull(rec.free) ?? zeroPriced(rec),
    context: asNumberOrNull(rec.context_length) ?? asNumberOrNull(rec.context_window) ?? asNumberOrNull(rec.inputTokenLimit),
    tools: asBoolOrNull(rec.tools),
    vision: asBoolOrNull(rec.vision),
    json: asBoolOrNull(rec.json),
  };
}

/** The models one catalog route's answer listed, in its wire's shape. */
function listedModels(wire: ProviderEntry['wire'], body: unknown): ListedModel[] {
  if (body === null || typeof body !== 'object' || Array.isArray(body)) return [];
  const rec = body as Record<string, unknown>;
  const list = wire === 'gemini' ? rec.models : rec.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = listedModel(wire, v);
    return m === null ? [] : [m];
  });
}

/** Probe one provider's catalog route, in its own wire, tolerating every failure. */
export async function probeProvider(envUrl: string, provider: ProviderEntry, timeoutMs = 5000): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) return { provider, ok: false, status: res.status, models: [], error: `GET ${url} answered HTTP ${res.status}` };
    const body: unknown = await res.json();
    return { provider, ok: true, status: res.status, models: listedModels(provider.wire, body), error: null };
  } catch (err) {
    return { provider, ok: false, status: 0, models: [], error: `GET ${url} failed: ${(err as Error).message}` };
  }
}

/** The catalog the probes and the quality table make. */
export function buildCatalog(probes: readonly Probe[], brief: Brief): Catalog {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    const sorted = [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    for (const model of sorted) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row: RatingRow | null = ratingRow(brief, probe.provider.id, model.id);
      models.push({
        id: model.id,
        free: isFreeModel(row, model.free),
        context_length: row?.context ?? model.context ?? null,
      });
    }
    providers.push({
      id: probe.provider.id,
      base_url: probe.provider.base,
      key_env: probe.provider.keyEnv,
      wire: probe.provider.wire,
      models,
    });
  }
  return { providers };
}

/** One probed model's capabilities, from the quality table first and the listing second. */
export function probeCaps(brief: Brief, providerId: string, model: ListedModel): Caps {
  const row = ratingRow(brief, providerId, model.id);
  return {
    context: row?.context ?? model.context ?? null,
    tools: row?.tools ?? model.tools ?? null,
    vision: row?.vision ?? model.vision ?? null,
    json: row?.json ?? model.json ?? null,
  };
}
