// The client wires and the upstream wires: every translation between them.
//
// A trial's client calls this server as an OpenAI service (POST /v1/chat/completions) or an
// Anthropic one (POST /v1/messages), with a JSON body or server-sent events. Each request is
// answered by one of the ten providers, which speaks its own wire: openai, anthropic or gemini.
// This file builds the upstream request in the provider's wire and rewrites the provider's
// answer back into the client's wire, in JSON and in SSE.
//
// What must round-trip, because the scorer checks it against the client's request:
// the system prompt, the user's text, the tool names, max_tokens, the stop sequences, the
// image bytes and JSON mode. A provider's answer text is passed through verbatim: it carries
// the ANSWER echo the scorer reads, and nothing is added to it.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same
// body and the same provider answer give the same bytes on a replay. The two model listings
// use fixed timestamps for the same reason.

import { isRecord, type Json } from './brief.ts';

export interface ServedModel {
  id: string;
  provider: string;
  displayName: string;
}

/** A fixed epoch second for every listed model's `created`. */
const FIXED_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`. */
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';

export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
  };
}

export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.displayName, created_at: FIXED_CREATED_AT })),
    has_more: false,
  };
}

export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

export function geminiError(code: number, status: string, message: string): Json {
  return { error: { code, message, status } };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then a blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
export function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argsObject(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

function numOrZero(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

function asStr(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

/** A stop list from a string, an array of strings, or nothing. */
function stopList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (Array.isArray(v)) return v.filter((x): x is string => typeof x === 'string');
  return [];
}

/** Split a data: URL into media type and base64 payload; null when it is not one. */
function dataUrlParts(url: string): { media: string; data: string } | null {
  const m = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
  if (m === null) return null;
  return { media: m[1] === undefined || m[1] === '' ? 'image/png' : m[1], data: m[2] ?? '' };
}

/** An Anthropic image block as an OpenAI image_url part; null for other blocks. */
function anthropicImagePart(block: Json): Json | null {
  if (block.type !== 'image' || !isRecord(block.source)) return null;
  const source = block.source;
  if (source.type === 'base64' && typeof source.data === 'string') {
    const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
    return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
  }
  if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  return null;
}

/** A tool_result block's content as text. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function openAIStopToAnthropic(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** Anthropic's stop_reason as OpenAI's finish_reason. */
export function anthropicStopToOpenAI(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as OpenAI's finish_reason. */
export function geminiStopToOpenAI(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as Anthropic's stop_reason. */
export function geminiStopToAnthropic(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'max_tokens';
  if (reason === 'SAFETY') return 'refusal';
  return 'end_turn';
}

// ---------------------------------------------------------------------------
// Request builders: the client's body rewritten for one provider's wire.
// ---------------------------------------------------------------------------

/** OpenAI's tools as Anthropic's tools. */
function openAIToolsToAnthropic(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [
      {
        name: raw.function.name,
        description: asStr(raw.function.description),
        input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      },
    ];
  });
}

/** Anthropic's tools as OpenAI's tools. */
function anthropicToolsToOpenAI(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [
      {
        type: 'function',
        function: {
          name: raw.name,
          description: asStr(raw.description),
          parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        },
      },
    ];
  });
}

/** Either side's tools as Gemini functionDeclarations. */
function toolsToGemini(tools: Json[]): Json[] {
  return tools.flatMap((raw) => {
    if (!isRecord(raw)) return [];
    if (isRecord(raw.function) && typeof raw.function.name === 'string') {
      return [{ name: raw.function.name, description: asStr(raw.function.description), parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    }
    if (typeof raw.name === 'string') {
      return [{ name: raw.name, description: asStr(raw.description), parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } }];
    }
    return [];
  });
}

/** An OpenAI Chat Completions request as an Anthropic Messages request naming `model`. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const messages: Json[] = [];
  // An OpenAI body carries the system prompt as system messages, not a `system` field.
  let system = '';
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') system = system === '' ? text : `${system}\n${text}`;
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: asStr(raw.tool_call_id), content: typeof raw.content === 'string' ? raw.content : textOf(raw.content) }],
      });
      continue;
    }
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({ type: 'tool_use', id: asStr(call.id) || 'toolu_unknown', name: asStr(call.function.name), input: argsObject(call.function.arguments) });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    // User message: text and image parts; a string stays a string.
    if (typeof raw.content === 'string') {
      messages.push({ role: 'user', content: raw.content });
      continue;
    }
    if (!Array.isArray(raw.content)) continue;
    const blocks: Json[] = [];
    for (const block of raw.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') blocks.push({ type: 'text', text: block.text });
      else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
        const parts = dataUrlParts(block.image_url.url);
        if (parts !== null) blocks.push({ type: 'image', source: { type: 'base64', media_type: parts.media, data: parts.data } });
        else blocks.push({ type: 'image', source: { type: 'url', url: block.image_url.url } });
      }
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }
  const out: Json = { model, messages };
  if (system !== '') out.system = system;
  const max = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (max !== undefined) out.max_tokens = max;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopList(body.stop);
  if (stops.length > 0) out.stop_sequences = stops;
  const tools = openAIToolsToAnthropic(body);
  if (tools.length > 0) out.tools = tools;
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** An Anthropic Messages request as an OpenAI Chat Completions request naming `model`. */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'assistant') {
      const toolCalls: Json[] = [];
      const text = textOf(raw.content);
      if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block) || block.type !== 'tool_use') continue;
          toolCalls.push({
            id: asStr(block.id) || 'call_unknown',
            type: 'function',
            function: { name: asStr(block.name), arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
          });
        }
      }
      messages.push({ role: 'assistant', content: text !== '' ? text : null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) });
      continue;
    }
    if (typeof raw.content === 'string') {
      messages.push({ role: 'user', content: raw.content });
      continue;
    }
    if (!Array.isArray(raw.content)) continue;
    const parts: Json[] = [];
    for (const block of raw.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({ role: 'tool', tool_call_id: asStr(block.tool_use_id), content: toolResultText(block.content) });
        continue;
      }
      const part = anthropicImagePart(block);
      if (part !== null) parts.push(part);
      else if (block.type === 'text' && typeof block.text === 'string') parts.push({ type: 'text', text: block.text });
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopList(body.stop_sequences);
  if (stops.length > 0) out.stop = stops;
  const tools = anthropicToolsToOpenAI(body);
  if (tools.length > 0) out.tools = tools;
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'none') out.tool_choice = 'none';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** One OpenAI message's content as Gemini parts. */
function openAIMessageParts(raw: Json): { role: string; parts: Json[] } | null {
  const parts: Json[] = [];
  if (raw.role === 'assistant') {
    const text = textOf(raw.content);
    if (text !== '') parts.push({ text });
    const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      parts.push({ functionCall: { name: asStr(call.function.name), args: argsObject(call.function.arguments) } });
    }
    return parts.length > 0 ? { role: 'model', parts } : null;
  }
  if (raw.role === 'tool') {
    parts.push({ functionResponse: { name: 'tool', response: { output: typeof raw.content === 'string' ? raw.content : textOf(raw.content) } } });
    return { role: 'user', parts };
  }
  if (typeof raw.content === 'string') {
    if (raw.content === '') return null;
    return { role: 'user', parts: [{ text: raw.content }] };
  }
  if (!Array.isArray(raw.content)) return null;
  for (const block of raw.content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
    else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
      const url = block.image_url.url;
      const data = dataUrlParts(url);
      if (data !== null) parts.push({ inlineData: { mimeType: data.media, data: data.data } });
      else parts.push({ fileData: { mimeType: 'image/png', fileUri: url } });
    }
  }
  return parts.length > 0 ? { role: 'user', parts } : null;
}

/** An OpenAI Chat Completions request as a Gemini generateContent request. */
export function openAIToGemini(body: Json): Json {
  const contents: Json[] = [];
  let systemText = '';
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemText = systemText === '' ? text : `${systemText}\n${text}`;
      continue;
    }
    const mapped = openAIMessageParts(raw);
    if (mapped !== null) contents.push(mapped);
  }
  const out: Json = { contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  const declarations = toolsToGemini(Array.isArray(body.tools) ? (body.tools as Json[]) : []);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const config: Json = {};
  const max = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (max !== undefined) config.maxOutputTokens = max;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stops = stopList(body.stop);
  if (stops.length > 0) config.stopSequences = stops;
  if (isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) {
    config.responseMimeType = 'application/json';
  }
  if (Object.keys(config).length > 0) out.generationConfig = config;
  const choice = body.tool_choice;
  if (choice === 'auto') out.toolConfig = { functionCallingConfig: { mode: 'AUTO' } };
  else if (choice === 'none') out.toolConfig = { functionCallingConfig: { mode: 'NONE' } };
  else if (choice === 'required') out.toolConfig = { functionCallingConfig: { mode: 'ANY' } };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.toolConfig = { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [choice.function.name] } };
  }
  return out;
}

/** One Anthropic message as Gemini content parts. */
function anthropicMessageParts(raw: Json): { role: string; parts: Json[] } | null {
  const parts: Json[] = [];
  if (raw.role === 'assistant') {
    if (!Array.isArray(raw.content)) return null;
    for (const block of raw.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
      else if (block.type === 'tool_use') parts.push({ functionCall: { name: asStr(block.name), args: isRecord(block.input) ? block.input : {} } });
    }
    return parts.length > 0 ? { role: 'model', parts } : null;
  }
  if (typeof raw.content === 'string') {
    if (raw.content === '') return null;
    return { role: 'user', parts: [{ text: raw.content }] };
  }
  if (!Array.isArray(raw.content)) return null;
  for (const block of raw.content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
    else if (block.type === 'image' && isRecord(block.source)) {
      if (block.source.type === 'base64' && typeof block.source.data === 'string') {
        parts.push({ inlineData: { mimeType: asStr(block.source.media_type) || 'image/png', data: block.source.data } });
      } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
        parts.push({ fileData: { mimeType: 'image/png', fileUri: block.source.url } });
      }
    } else if (block.type === 'tool_result') {
      parts.push({ functionResponse: { name: 'tool', response: { output: toolResultText(block.content) } } });
    }
  }
  return parts.length > 0 ? { role: 'user', parts } : null;
}

/** An Anthropic Messages request as a Gemini generateContent request. */
export function anthropicToGemini(body: Json): Json {
  const contents: Json[] = [];
  const systemText = textOf(body.system);
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const mapped = anthropicMessageParts(raw);
    if (mapped !== null) contents.push(mapped);
  }
  const out: Json = { contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  const declarations = toolsToGemini(Array.isArray(body.tools) ? (body.tools as Json[]) : []);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stops = stopList(body.stop_sequences);
  if (stops.length > 0) config.stopSequences = stops;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.toolConfig = { functionCallingConfig: { mode: 'AUTO' } };
    else if (choice.type === 'none') out.toolConfig = { functionCallingConfig: { mode: 'NONE' } };
    else if (choice.type === 'any') out.toolConfig = { functionCallingConfig: { mode: 'ANY' } };
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.toolConfig = { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [choice.name] } };
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Response converters: a provider's JSON answer in the client's wire.
// ---------------------------------------------------------------------------

/** An Anthropic Messages object as an OpenAI Chat Completions object. */
export function anthropicToOpenAIResponse(body: Json, model: string): Json {
  const content: Json[] = [];
  const text = textOf(body.content);
  if (text !== '') content.push({ type: 'text', text });
  if (Array.isArray(body.content)) {
    for (const block of body.content) {
      if (!isRecord(block) || block.type !== 'tool_use') continue;
      content.push({
        type: 'tool_use',
        id: asStr(block.id) || 'toolu_unknown',
        name: asStr(block.name),
        input: isRecord(block.input) ? block.input : {},
      });
    }
  }
  // The OpenAI shape carries tool calls on the message, not in the content list.
  const toolCalls: Json[] = [];
  for (const block of content) {
    if (block.type !== 'tool_use') continue;
    toolCalls.push({
      id: asStr(block.id),
      type: 'function',
      function: { name: asStr(block.name), arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
    });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `chatcmpl_${asStr(body.id) || 'sim'}`,
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: text !== '' ? text : null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) },
        finish_reason: anthropicStopToOpenAI(body.stop_reason),
        logprobs: null,
      },
    ],
    usage: {
      prompt_tokens: numOrZero(usage.input_tokens),
      completion_tokens: numOrZero(usage.output_tokens),
      total_tokens: numOrZero(usage.input_tokens) + numOrZero(usage.output_tokens),
    },
  };
}

/** Gemini's text and finish reason from one answer. */
function geminiTextAndStop(body: Json): { text: string; finish: unknown } {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(first.content) ? first.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
  return { text, finish: first.finishReason };
}

/** Gemini's usage as OpenAI/Anthropic token counts. */
function geminiUsage(body: Json): { input: number; output: number } {
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return { input: numOrZero(meta.promptTokenCount), output: numOrZero(meta.candidatesTokenCount) };
}

/** A Gemini generateContent answer as an OpenAI Chat Completions object. */
export function geminiToOpenAIResponse(body: Json, model: string): Json {
  const { text, finish } = geminiTextAndStop(body);
  const usage = geminiUsage(body);
  return {
    id: `chatcmpl_${asStr(body.modelVersion) || 'sim'}`,
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: geminiStopToOpenAI(finish), logprobs: null }],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  };
}

/** An OpenAI Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropicResponse(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({ type: 'tool_use', id: asStr(call.id) || 'toolu_unknown', name: asStr(call.function.name), input: argsObject(call.function.arguments) });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${asStr(body.id) || 'sim'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: openAIStopToAnthropic(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numOrZero(usage.prompt_tokens), output_tokens: numOrZero(usage.completion_tokens) },
  };
}

/** A Gemini generateContent answer as an Anthropic Messages object. */
export function geminiToAnthropicResponse(body: Json, model: string): Json {
  const { text, finish } = geminiTextAndStop(body);
  const usage = geminiUsage(body);
  return {
    id: `msg_${asStr(body.modelVersion) || 'sim'}`,
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text }],
    stop_reason: geminiStopToAnthropic(finish),
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

// ---------------------------------------------------------------------------
// Stream converters: a provider's SSE stream in the client's wire.
// ---------------------------------------------------------------------------

export interface SseFrame {
  event: string | null;
  data: string;
}

/** Parse an SSE byte stream into frames (event name + data payload each). */
export async function* sseFrames(stream: ReadableStream<Uint8Array>): AsyncGenerator<SseFrame> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        yield frameOf(buffer.slice(0, at));
        buffer = buffer.slice(at + 2);
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') yield frameOf(buffer);
  } finally {
    reader.releaseLock();
  }
}

function frameOf(raw: string): SseFrame {
  let event: string | null = null;
  const dataLines: string[] = [];
  for (const line of raw.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim();
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart());
  }
  return { event, data: dataLines.join('\n') };
}

function parseJsonFrame(frame: SseFrame): Json | null {
  if (frame.data === '[DONE]') return null;
  try {
    const parsed: unknown = JSON.parse(frame.data);
    return isRecord(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

function openAIChunk(model: string, delta: Json, finish: string | null): string {
  return sseEvent(null, { id: 'chatcmpl_sim', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] });
}

function openAIUsageChunk(model: string, input: number, output: number): string {
  return sseEvent(null, {
    id: 'chatcmpl_sim',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  });
}

/**
 * An Anthropic SSE stream as OpenAI chunks: the role chunk, the text deltas, the tool-call
 * deltas, then the finish chunk and the usage chunk, then [DONE].
 */
export async function* anthropicStreamToOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let inputTokens = 0;
  let outputTokens = 0;
  let stopReason = 'end_turn';
  let sentRole = false;
  let toolIndex = 0;
  const toolIds = new Map<number, number>();
  const sendRole = function* (): Generator<string> {
    if (sentRole) return;
    sentRole = true;
    yield openAIChunk(model, { role: 'assistant', content: '' }, null);
  };
  for await (const frame of sseFrames(upstream)) {
    const msg = parseJsonFrame(frame);
    if (msg === null) continue;
    if (frame.event === 'message_start' && isRecord(msg.message) && isRecord(msg.message.usage)) {
      inputTokens = numOrZero(msg.message.usage.input_tokens);
      continue;
    }
    if (frame.event === 'content_block_delta' && isRecord(msg.delta)) {
      const delta = msg.delta;
      const index = typeof msg.index === 'number' ? msg.index : 0;
      if (delta.type === 'text_delta' && typeof delta.text === 'string' && delta.text !== '') {
        for (const e of sendRole()) yield e;
        yield openAIChunk(model, { content: delta.text }, null);
      } else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string' && delta.partial_json !== '') {
        let outIndex = toolIds.get(index);
        if (outIndex === undefined) {
          outIndex = toolIndex;
          toolIndex += 1;
          toolIds.set(index, outIndex);
        }
        for (const e of sendRole()) yield e;
        yield openAIChunk(model, { tool_calls: [{ index: outIndex, function: { arguments: delta.partial_json } }] }, null);
      }
      continue;
    }
    if (frame.event === 'content_block_start' && isRecord(msg.content_block) && msg.content_block.type === 'tool_use') {
      const index = typeof msg.index === 'number' ? msg.index : 0;
      if (!toolIds.has(index)) toolIds.set(index, toolIndex++);
      const outIndex = toolIds.get(index) ?? 0;
      for (const e of sendRole()) yield e;
      yield openAIChunk(model, { tool_calls: [{ index: outIndex, id: asStr(msg.content_block.id), type: 'function', function: { name: asStr(msg.content_block.name), arguments: '' } }] }, null);
      continue;
    }
    if (frame.event === 'message_delta') {
      if (isRecord(msg.delta) && typeof msg.delta.stop_reason === 'string') stopReason = msg.delta.stop_reason;
      if (isRecord(msg.usage)) outputTokens = numOrZero(msg.usage.output_tokens);
    }
  }
  for (const e of sendRole()) yield e;
  yield openAIChunk(model, {}, anthropicStopToOpenAI(stopReason));
  yield openAIUsageChunk(model, inputTokens, outputTokens);
  yield 'data: [DONE]\n\n';
}

/**
 * A Gemini SSE stream as OpenAI chunks: the text pieces as content deltas, then the finish
 * chunk (STOP) and the usage chunk from the final frame's usageMetadata, then [DONE].
 */
export async function* geminiStreamToOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let inputTokens = 0;
  let outputTokens = 0;
  let finish = 'STOP';
  let sentRole = false;
  for await (const frame of sseFrames(upstream)) {
    const msg = parseJsonFrame(frame);
    if (msg === null) continue;
    const candidates = Array.isArray(msg.candidates) ? msg.candidates : [];
    const first = isRecord(candidates[0]) ? candidates[0] : {};
    const content = isRecord(first.content) ? first.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    const text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
    if (text !== '') {
      if (!sentRole) {
        sentRole = true;
        yield openAIChunk(model, { role: 'assistant', content: '' }, null);
      }
      yield openAIChunk(model, { content: text }, null);
    }
    if (typeof first.finishReason === 'string') finish = first.finishReason;
    if (isRecord(msg.usageMetadata)) {
      inputTokens = numOrZero(msg.usageMetadata.promptTokenCount);
      outputTokens = numOrZero(msg.usageMetadata.candidatesTokenCount);
    }
  }
  if (!sentRole) yield openAIChunk(model, { role: 'assistant', content: '' }, null);
  yield openAIChunk(model, {}, geminiStopToOpenAI(finish));
  yield openAIUsageChunk(model, inputTokens, outputTokens);
  yield 'data: [DONE]\n\n';
}

/**
 * An OpenAI SSE stream as the Anthropic event sequence: message_start, one text block (or
 * one tool_use block per call) with deltas, then message_delta with the stop reason and the
 * output tokens, then message_stop.
 */
export async function* openAIStreamToAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();

  const startEvents = function* (): Generator<string> {
    if (started) return;
    started = true;
    yield sseEvent('message_start', {
      type: 'message_start',
      message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: inputTokens, output_tokens: 0 } },
    });
  };
  const finishEvents = function* (stopReason: string): Generator<string> {
    if (finished) return;
    finished = true;
    if (textIndex !== -1) {
      yield sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex });
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) {
      yield sseEvent('content_block_stop', { type: 'content_block_stop', index });
    }
    toolIndexes.clear();
    yield sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } });
    yield sseEvent('message_stop', { type: 'message_stop' });
  };

  for await (const frame of sseFrames(upstream)) {
    if (frame.data === '[DONE]') {
      for (const e of startEvents()) yield e;
      for (const e of finishEvents(pendingStop ?? 'end_turn')) yield e;
      continue;
    }
    const chunk = parseJsonFrame(frame);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) {
      inputTokens = numOrZero(chunk.usage.prompt_tokens);
      outputTokens = numOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    for (const e of startEvents()) yield e;
    if (typeof delta.content === 'string' && delta.content !== '') {
      if (textIndex === -1) {
        textIndex = nextIndex;
        nextIndex += 1;
        yield sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } });
      }
      yield sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } });
    }
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call)) continue;
      const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
      let index = toolIndexes.get(upstreamIndex);
      if (index === undefined) {
        if (textIndex !== -1) {
          yield sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex });
          textIndex = -1;
        }
        index = nextIndex;
        nextIndex += 1;
        toolIndexes.set(upstreamIndex, index);
        const fn = isRecord(call.function) ? call.function : {};
        yield sseEvent('content_block_start', {
          type: 'content_block_start',
          index,
          content_block: { type: 'tool_use', id: asStr(call.id) !== '' ? asStr(call.id) : `toolu_${index}`, name: asStr(fn.name), input: {} },
        });
      }
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.arguments === 'string' && fn.arguments !== '') {
        yield sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } });
      }
    }
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = openAIStopToAnthropic(choice.finish_reason);
  }
  if (!finished) {
    for (const e of startEvents()) yield e;
    for (const e of finishEvents(pendingStop ?? 'end_turn')) yield e;
  }
}

/**
 * A Gemini SSE stream as the Anthropic event sequence: message_start, one text block with
 * deltas, then message_delta with the output tokens from the final frame, then message_stop.
 * The input tokens only arrive in the final frame, after message_start has gone out, so it
 * carries 0 there; the stream usage check reads the output tokens alone.
 */
export async function* geminiStreamToAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let textIndex = -1;
  let outputTokens = 0;
  let finish = 'STOP';
  for await (const frame of sseFrames(upstream)) {
    const msg = parseJsonFrame(frame);
    if (msg === null) continue;
    if (!started) {
      started = true;
      yield sseEvent('message_start', {
        type: 'message_start',
        message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } },
      });
    }
    const candidates = Array.isArray(msg.candidates) ? msg.candidates : [];
    const first = isRecord(candidates[0]) ? candidates[0] : {};
    const content = isRecord(first.content) ? first.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    const text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
    if (text !== '') {
      if (textIndex === -1) {
        textIndex = 0;
        yield sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
      }
      yield sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text } });
    }
    if (typeof first.finishReason === 'string') finish = first.finishReason;
    if (isRecord(msg.usageMetadata)) outputTokens = numOrZero(msg.usageMetadata.candidatesTokenCount);
  }
  if (!started) {
    yield sseEvent('message_start', {
      type: 'message_start',
      message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } },
    });
  }
  if (textIndex !== -1) yield sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 });
  yield sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: geminiStopToAnthropic(finish), stop_sequence: null }, usage: { output_tokens: outputTokens } });
  yield sseEvent('message_stop', { type: 'message_stop' });
}
