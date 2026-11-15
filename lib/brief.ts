// The trial brief: three JSON documents embedded in one text file, found by schema.
//
//   free-model-aggregator-01/providers/v1  the ten providers (id, wire, auth, key, base path)
//   free-model-aggregator-01/ratings/v1    every model's rating and capabilities, and whether free
//   free-model-aggregator-01/workload/v1   the twenty requests, in order, with their requirements
//
// Everything here is pure and deterministic: no clock, no randomness.

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

/** Any JSON object. */
export type Json = Record<string, unknown>;
export const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

export interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  /** Base path (e.g. /p/groq) or an absolute URL. */
  base: string;
  keyEnv: string;
  key: string;
}

export interface RatingRow {
  model: string;
  provider: string;
  rating: number;
  context: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface Need {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  minContext?: number;
}

export interface WorkItem {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  needs: Need;
}

export interface Brief {
  providers: ProviderEntry[];
  ratings: RatingRow[];
  /** Best rating per request id, from the table's `best` list. */
  best: Map<string, number>;
  workload: WorkItem[];
  error: string | null;
}

const asString = (v: unknown): string => (typeof v === 'string' ? v : '');
const asNumberOrNull = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const asBoolOrNull = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null);

/** Scan `text` for top-level JSON objects and return the ones carrying a string `schema`. */
function schemaDocs(text: string): Json[] {
  const docs: Json[] = [];
  for (let i = 0; i < text.length; i++) {
    if (text[i] !== '{') continue;
    // Walk a balanced object, skipping string contents.
    let depth = 0;
    let inStr = false;
    let end = -1;
    for (let j = i; j < text.length; j++) {
      const ch = text[j];
      if (inStr) {
        if (ch === '\\') j += 1;
        else if (ch === '"') inStr = false;
      } else if (ch === '"') inStr = true;
      else if (ch === '{') depth += 1;
      else if (ch === '}') {
        depth -= 1;
        if (depth === 0) {
          end = j;
          break;
        }
      }
    }
    if (end === -1) break;
    try {
      const parsed: unknown = JSON.parse(text.slice(i, end + 1));
      if (isRecord(parsed) && typeof parsed.schema === 'string') docs.push(parsed);
    } catch {
      // Not a document on its own; keep scanning.
    }
    i = end;
  }
  return docs;
}

function parseProvider(v: unknown): ProviderEntry | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth: AuthScheme =
    v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
  const base = asString(v.path) || asString(v.base_path) || asString(v.base_url) || asString(v.base);
  return { id, name: asString(v.name) || id, wire, auth, base, keyEnv: asString(v.key_env), key: asString(v.key) };
}

function parseRating(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = asString(v.model);
  const provider = asString(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating: typeof v.rating_milli === 'number' && Number.isFinite(v.rating_milli) ? v.rating_milli : 0,
    context: asNumberOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function parseWork(v: unknown): WorkItem | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  if (id === '') return null;
  const needs = isRecord(v.requires) ? v.requires : {};
  const out: Need = {
    tools: needs.tools === true,
    vision: needs.vision === true,
    json: needs.json === true,
  };
  if (typeof needs.min_context === 'number' && Number.isFinite(needs.min_context)) out.minContext = needs.min_context;
  return { id, api: v.api === 'anthropic' ? 'anthropic' : 'openai', stream: v.stream === true, weight: v.weight === 2 ? 2 : 1, needs: out };
}

/** Read the brief at `path`. A missing file yields an empty brief with `error` set. */
export function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], best: new Map(), workload: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = schemaDocs(text);
  const dirDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/providers/v1' && Array.isArray(d.providers));
  if (dirDoc === undefined || !Array.isArray(dirDoc.providers)) {
    return { ...empty, error: 'the brief holds no free-model-aggregator-01/providers/v1 document' };
  }
  const providers = dirDoc.providers.flatMap((v) => {
    const p = parseProvider(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/ratings/v1' && Array.isArray(d.ratings));
  const ratings: RatingRow[] =
    ratingsDoc === undefined || !Array.isArray(ratingsDoc.ratings)
      ? []
      : ratingsDoc.ratings.flatMap((v) => {
          const r = parseRating(v);
          return r === null ? [] : [r];
        });
  const best = new Map<string, number>();
  if (ratingsDoc !== undefined && Array.isArray(ratingsDoc.best)) {
    for (const b of ratingsDoc.best) {
      if (isRecord(b) && asString(b.id) !== '') best.set(b.id, asNumberOrNull(b.rating_milli) ?? 0);
    }
  }
  const workloadDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/workload/v1' && Array.isArray(d.manifest));
  const workload: WorkItem[] =
    workloadDoc === undefined || !Array.isArray(workloadDoc.manifest)
      ? []
      : workloadDoc.manifest.flatMap((v) => {
          const w = parseWork(v);
          return w === null ? [] : [w];
        });
  return { providers, ratings, best, workload, error: null };
}

/** The quality table's row for one provider's model, if it has one. */
export function ratingRow(brief: Brief, providerId: string, modelId: string): RatingRow | null {
  return brief.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/** What a model can do, from the quality table first and a listing's fields second. */
export interface Caps {
  context: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

export function capsOf(row: RatingRow | null, listed: { context: number | null; tools: boolean | null; vision: boolean | null; json: boolean | null }): Caps {
  return {
    context: row?.context ?? listed.context ?? null,
    tools: row?.tools ?? listed.tools ?? null,
    vision: row?.vision ?? listed.vision ?? null,
    json: row?.json ?? listed.json ?? null,
  };
}

/** Whether a model is free: the table's row where it has one, else the listing's own word. */
export function isFreeModel(row: RatingRow | null, listedFree: boolean | null): boolean {
  if (row !== null) return row.free;
  return listedFree ?? false;
}

/** Whether a model's capabilities meet one request's needs. An unknown capability never meets one. */
export function meetsNeeds(needs: Need, caps: Caps): boolean {
  if (needs.tools === true && caps.tools !== true) return false;
  if (needs.vision === true && caps.vision !== true) return false;
  if (needs.json === true && caps.json !== true) return false;
  if (needs.minContext !== undefined && (caps.context === null || caps.context < needs.minContext)) return false;
  return true;
}

/** Join a provider base path to the env origin; an absolute base is used as it stands. */
export function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers a provider wants its key in. */
export function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

export { asBoolOrNull, asNumberOrNull, asString };
