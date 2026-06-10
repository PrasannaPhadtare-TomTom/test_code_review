/**
 * AI Update Set Reviewer
 *
 * Flow:
 *  1. Fetch Update Set metadata + all code changes from ServiceNow
 *  2. Identify any referenced Script Includes for deeper context
 *  3. Fetch linked Jira ticket for requirements + acceptance criteria
 *  4. AI (GitHub Models / gpt-4.1) reviews all code against best practices
 *  5. Generates a decisional score (0-100) + PUSH / HOLD / DO NOT PUSH
 *  6. Creates a record in u_ai_code_review table with the full HTML review
 *     and also posts a brief work note on the Update Set linking to it
 */

import OpenAI from 'openai';

// ── Environment ───────────────────────────────────────────────────────────────
const {
  GITHUB_TOKEN,
  AI_API_TOKEN,
  REVIEW_MODEL,
  REVIEW_API_ENDPOINT,
  FORCE_CHUNKED_REVIEW,
  REVIEW_CHUNK_SIZE,
  REQUESTED_BY,
  AI_TOKEN_SOURCE,
  UPDATE_SET_SYS_ID,
  UPDATE_SET_NAME,
  JIRA_TICKET,
  SERVICENOW_INSTANCE,
  SERVICENOW_USERNAME,
  SERVICENOW_PASSWORD,
  JIRA_URL,
  JIRA_EMAIL,
  JIRA_TOKEN,
} = process.env;

const FORCE_CHUNKED = ['1', 'true', 'yes', 'on'].includes(String(FORCE_CHUNKED_REVIEW || '').trim().toLowerCase());
const CHUNK_SIZE = Math.max(2, Math.min(25, parseInt(String(REVIEW_CHUNK_SIZE || '8'), 10) || 8));

// Validate sys_id format before doing anything
if (UPDATE_SET_SYS_ID && !/^[a-f0-9]{32}$/.test(UPDATE_SET_SYS_ID)) {
  console.error(`Invalid UPDATE_SET_SYS_ID format: "${UPDATE_SET_SYS_ID}". Expected a 32-char hex string.`);
  process.exit(1);
}

if (!GITHUB_TOKEN || !UPDATE_SET_SYS_ID || !SERVICENOW_INSTANCE || !SERVICENOW_USERNAME || !SERVICENOW_PASSWORD) {
  console.error('Missing required env vars: GITHUB_TOKEN, UPDATE_SET_SYS_ID, SERVICENOW_INSTANCE, SERVICENOW_USERNAME, SERVICENOW_PASSWORD');
  process.exit(1);
}

// ── AI client (provider-aware) ────────────────────────────────────────────────
// The API key: prefer AI_API_TOKEN (set when using non-GitHub-Models providers),
// fall back to GITHUB_TOKEN (legacy / GitHub Models default).
const API_KEY      = AI_API_TOKEN || GITHUB_TOKEN;
const MODEL_ID     = REVIEW_MODEL || 'gpt-4.1';
const API_ENDPOINT = REVIEW_API_ENDPOINT || 'https://models.inference.ai.azure.com';

// Derive a human-readable provider label for the audit trail.
function deriveProviderLabel(endpoint, model) {
  const ep = endpoint.toLowerCase();
  if (ep.includes('anthropic.com') || model.startsWith('claude'))   return `Anthropic / ${model}`;
  if (ep.includes('openai.com') && !ep.includes('azure'))           return `OpenAI / ${model}`;
  if (ep.includes('inference.ai') || ep.includes('azure.com'))      return `GitHub Models / ${model}`;
  if (ep.includes('googleapis.com') || model.startsWith('gemini'))  return `Google AI / ${model}`;
  try { return `${new URL(endpoint).hostname} / ${model}`; } catch { return `AI / ${model}`; }
}
const REVIEW_ENGINE = deriveProviderLabel(API_ENDPOINT, MODEL_ID);

// Build the OpenAI-compatible client.
// Anthropic's v1 API uses x-api-key + anthropic-version instead of Bearer auth.
const clientOptions = { baseURL: API_ENDPOINT, apiKey: API_KEY };
if (API_ENDPOINT.includes('anthropic.com')) {
  clientOptions.defaultHeaders = {
    'x-api-key':          API_KEY,
    'anthropic-version':  '2023-06-01',
  };
}
const openai = new OpenAI(clientOptions);

// ── ServiceNow REST helpers ───────────────────────────────────────────────────
const snowBase = `https://${SERVICENOW_INSTANCE}.service-now.com/api/now`;
// NOTE: snowAuth is intentionally never logged or included in error messages
const snowAuth = 'Basic ' + Buffer.from(`${SERVICENOW_USERNAME}:${SERVICENOW_PASSWORD}`).toString('base64');

const FETCH_TIMEOUT_MS = 30_000; // 30 s — prevents hung requests stalling the agent

async function snowGet(path, params = {}) {
  const url = new URL(`${snowBase}${path}`);
  for (const [k, v] of Object.entries(params)) url.searchParams.set(k, String(v));
  const res = await fetch(url.toString(), {
    headers: { Authorization: snowAuth, Accept: 'application/json' },
    signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
  });
  if (!res.ok) {
    // Deliberately omit auth header / credential values from the error message
    throw new Error(`SNOW GET ${path} -> HTTP ${res.status}`);
  }
  return (await res.json()).result;
}

async function snowPost(path, body) {
  const res = await fetch(`${snowBase}${path}`, {
    method: 'POST',
    headers: { Authorization: snowAuth, 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
  });
  if (!res.ok) {
    throw new Error(`SNOW POST ${path} -> HTTP ${res.status}`);
  }
  return (await res.json()).result;
}

async function snowPatch(path, body) {
  const res = await fetch(`${snowBase}${path}`, {
    method: 'PATCH',
    headers: { Authorization: snowAuth, 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
  });
  if (!res.ok) {
    throw new Error(`SNOW PATCH ${path} -> HTTP ${res.status}`);
  }
  return (await res.json()).result;
}

// ── XML parsing helpers ───────────────────────────────────────────────────────
// Uses a simple but safe approach: extract one tag at a time with a bounded match.
// For very large payloads a proper XML parser would be more robust, but this is
// sufficient for ServiceNow update XML which has predictable shallow structure.
function extractXmlTag(xml, tag) {
  if (!xml) return null;
  // Bound the content match to 200 KB to avoid catastrophic backtracking on huge payloads
  const re = new RegExp(`<${tag}(?:\\s[^>]*)?>([\\s\\S]{0,204800}?)<\\/${tag}>`, 'i');
  const m  = xml.match(re);
  if (!m) return null;
  return m[1]
    .replace(/<!\[CDATA\[|\]\]>/g, '')
    .replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&').replace(/&quot;/g, '"')
    .trim();
}

const CODE_FIELDS    = ['script', 'condition', 'advanced_condition', 'html', 'css', 'body', 'template', 'message'];
const CHANGES_LIMIT  = 150; // max records fetched from sys_update_xml in one call
const CHANGE_PAGE_DEFAULT = 25;
const CHANGE_PAGE_MAX = 50;
const CODE_FIELD_MAX_CHARS = 4000;
const TOOL_RESULT_MAX_CHARS = 18000;

// In-memory cache: record_name -> full parsed artifact
// Assumption: single-run lifetime only — do not reuse this module across multiple Update Sets
const changeCodeCache = new Map();

function parsePayload(payload, type, targetName) {
  const extracted = {};
  for (const field of CODE_FIELDS) {
    const val = extractXmlTag(payload, field);
    if (val && val.length > 5) extracted[field] = val;
  }
  const name   = extractXmlTag(payload, 'name') || targetName;
  const when   = extractXmlTag(payload, 'when');
  const active = extractXmlTag(payload, 'active');
  const table  = extractXmlTag(payload, 'collection');
  return {
    type, name,
    ...(when   ? { when }   : {}),
    ...(active ? { active } : {}),
    ...(table  ? { table }  : {}),
    ...extracted,
  };
}

// ── Jira helper ───────────────────────────────────────────────────────────────
function extractADFText(node) {
  if (!node) return null;
  if (typeof node === 'string') return node;
  if (node.type === 'text') return node.text ?? '';
  if (Array.isArray(node.content)) return node.content.map(extractADFText).filter(Boolean).join(' ');
  return null;
}


async function fetchJiraTicket(ticketId) {
  if (!JIRA_URL || !JIRA_EMAIL || !JIRA_TOKEN) return 'Jira credentials not configured.';
  // NOTE: jiraAuth is intentionally never logged
  const jiraAuth = Buffer.from(`${JIRA_EMAIL}:${JIRA_TOKEN}`).toString('base64');
  const res = await fetch(
    `${JIRA_URL.replace(/\/$/, '')}/rest/api/3/issue/${encodeURIComponent(ticketId)}`,
    {
      headers: { Authorization: `Basic ${jiraAuth}`, Accept: 'application/json' },
      signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
    }
  );
  if (!res.ok) return `Jira ticket ${ticketId} not found (HTTP ${res.status}).`;
  const d = await res.json();

  // IMPORTANT: customfield_10016 is Story Points in many Jira instances.
  // Verify the correct field ID for Acceptance Criteria in your Jira project
  // by inspecting /rest/api/3/issue/{key}?expand=names and checking field names.
  // Common alternatives: customfield_10034, customfield_10053, customfield_10054
  const AC_FIELD = 'customfield_10034'; // <-- adjust for your Jira instance

  return JSON.stringify({
    id:                 d.key,
    summary:            d.fields?.summary,
    description:        extractADFText(d.fields?.description),
    acceptanceCriteria: extractADFText(d.fields?.[AC_FIELD]),
    status:             d.fields?.status?.name,
    priority:           d.fields?.priority?.name,
    type:               d.fields?.issuetype?.name,
  });
}

// ── HTML review formatter ─────────────────────────────────────────────────────
// Converts the structured review args into clean HTML for the u_description field
function buildReviewHtml({ score, recommendation, good_points = [], issues = [], jira_alignment, summary }) {
  const badgeColour = {
    'PUSH':                 '#007a33',
    'PUSH WITH MINOR FIXES':'#5a7a00',
    'HOLD FOR REVIEW':      '#b87000',
    'DO NOT PUSH':          '#b80000',
  }[recommendation] ?? '#555';

  const severityColour = { CRITICAL: '#b80000', HIGH: '#c94a00', MEDIUM: '#b87000', LOW: '#005b99', INFO: '#555' };

  const escHtml = s => String(s ?? '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

  const SNIPPET_MAX = 350;
  const clip = (s) => s && s.length > SNIPPET_MAX ? s.slice(0, SNIPPET_MAX) + '\n\u2026' : s;
  const codeBlock = (bg, border, label, code) =>
    `<div style="background:${bg};border-left:3px solid ${border};padding:8px 10px;` +
    `margin:6px 0 0;font-family:monospace;font-size:0.82em;white-space:pre-wrap;overflow-wrap:anywhere">` +
    `<span style="font-family:sans-serif;font-size:0.9em;font-weight:bold">${label}</span>\n${escHtml(code)}</div>`;

  const issueCards = issues.map(i => {
    const sc      = severityColour[i.severity] ?? '#555';
    const snippet = clip(i.code_snippet);
    const fixed   = clip(i.fixed_snippet);
    return `
    <div style="border:1px solid #e0e0e0;border-radius:4px;margin-bottom:10px;overflow:hidden">
      <div style="background:${sc};color:#fff;padding:7px 12px;font-size:0.9em">
        <strong>${escHtml(i.severity)}</strong> &mdash; ${escHtml(i.location)}
      </div>
      <div style="padding:10px 12px">
        <p style="margin:0 0 6px">${escHtml(i.description)}</p>
        ${snippet ? codeBlock('#fff5f5', '#e88', '\u274c Problematic:', snippet) : ''}
        ${fixed   ? codeBlock('#f5fff8', '#6c6', '\u2705 Fixed:',       fixed)   : ''}
        ${i.suggestion && !fixed ? `<p style="margin:6px 0 0;font-size:0.88em;color:#555"><em>${escHtml(i.suggestion)}</em></p>` : ''}
      </div>
    </div>`;
  }).join('');

  const goodItems = good_points.map(g => `<li>${escHtml(g)}</li>`).join('');

  return `
<div style="font-family:Arial,sans-serif;font-size:14px;line-height:1.5;color:#333;max-width:900px">

  <div style="background:${badgeColour};color:#fff;padding:12px 18px;border-radius:4px;margin-bottom:16px">
    <span style="font-size:1.4em;font-weight:bold">${escHtml(recommendation)}</span>
    &nbsp;&nbsp;
    <span style="font-size:1.1em">Score: ${escHtml(String(score))} / 100</span>
  </div>

  <h3 style="margin-top:0">Executive Summary</h3>
  <p style="white-space:pre-wrap">${escHtml(summary)}</p>

  <h3>What Is Good</h3>
  <ul>${goodItems || '<li><em>None noted</em></li>'}</ul>

  <h3>Issues Found (${issues.length})</h3>
  ${issues.length ? issueCards : '<p><em>No issues found.</em></p>'}

  ${jira_alignment ? `
  <h3>Jira Ticket Alignment</h3>
  <p>${escHtml(jira_alignment)}</p>` : ''}

  <hr style="margin-top:24px;border:none;border-top:1px solid #ddd"/>
  <p style="color:#888;font-size:0.85em">
    Reviewed by AI &mdash; ${escHtml(REVIEW_ENGINE)} &mdash; ${new Date().toUTCString()}
  </p>

</div>`;
}

// ── Tool definitions ──────────────────────────────────────────────────────────
const tools = [
  {
    type: 'function',
    function: {
      name: 'get_update_set_details',
      description: 'Fetch Update Set metadata: name, description, scope, state, release date, owner. Also inspect the description for Jira ticket IDs (pattern [A-Z]+-[0-9]+).',
      parameters: { type: 'object', properties: {} },
    },
  },
  {
    type: 'function',
    function: {
      name: 'get_update_set_changes',
      description: `Fetch changes in the Update Set that contain code. Returns a PAGE of metadata + a short preview. Use offset/limit to iterate pages and avoid model token overflow. Call get_change_code(record_name) for each record to get full code.`,
      parameters: {
        type: 'object',
        properties: {
          offset: { type: 'number', description: '0-based page start index. Default 0.' },
          limit: { type: 'number', description: `Page size. Default ${CHANGE_PAGE_DEFAULT}, max ${CHANGE_PAGE_MAX}.` },
        },
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'get_change_code',
      description: 'Fetch code for a single Update Set change by its record_name. Returns truncated fields by default to reduce token usage.',
      parameters: {
        type: 'object',
        properties: {
          record_name: { type: 'string', description: 'The record_name value from get_update_set_changes' },
          max_chars_per_field: { type: 'number', description: `Optional per-field cap. Default ${CODE_FIELD_MAX_CHARS}.` },
        },
        required: ['record_name'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'get_existing_script_include',
      description: 'Fetch an existing Script Include by name from the ServiceNow instance. Use this to get the full code of a utility/helper referenced by code in the Update Set.',
      parameters: {
        type: 'object',
        properties: { name: { type: 'string', description: 'Script Include name' } },
        required: ['name'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'get_jira_ticket',
      description: 'Fetch a Jira ticket by ID (e.g. TT-1234). Returns summary, description, acceptance criteria. Use to validate the Update Set meets requirements.',
      parameters: {
        type: 'object',
        properties: { ticket_id: { type: 'string', description: 'Jira ticket ID, e.g. TT-1234' } },
        required: ['ticket_id'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'post_review_to_update_set',
      description: [
        'Save the complete AI review.',
        'Creates a record in the u_ai_code_review table with full HTML details,',
        'then posts a brief work note on the Update Set linking to it.',
        'Call this ONCE at the very end with ALL findings.',
      ].join(' '),
      parameters: {
        type: 'object',
        properties: {
          score:          { type: 'number', description: '0-100 decisional score. 90-100=PUSH, 70-89=PUSH WITH MINOR FIXES, 50-69=HOLD FOR REVIEW, 0-49=DO NOT PUSH' },
          recommendation: { type: 'string', enum: ['PUSH', 'PUSH WITH MINOR FIXES', 'HOLD FOR REVIEW', 'DO NOT PUSH'] },
          good_points:    { type: 'array', items: { type: 'string' }, description: 'What is done well' },
          issues: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                severity:      { type: 'string', enum: ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO'] },
                location:      { type: 'string', description: 'Exact artifact type + name, e.g. "Business Rule: ValidateCase" or "Script Include: CaseUtils — processRequest()"' },
                description:   { type: 'string', description: 'Specific problem naming the exact variable/function/line involved — never generic' },
                code_snippet:  { type: 'string', description: 'The exact problematic lines copied verbatim from get_change_code output. Required for CRITICAL/HIGH/MEDIUM.' },
                fixed_snippet: { type: 'string', description: 'The corrected code showing exactly what to write instead. Write actual code, not advice.' },
                suggestion:    { type: 'string', description: 'Additional context or explanation if needed beyond fixed_snippet' },
              },
            },
          },
          jira_alignment: { type: 'string', description: 'How well the Update Set meets the Jira ticket requirements' },
          summary:        { type: 'string', description: 'Executive summary (2-3 paragraphs): (1) what each changed artifact does, named explicitly; (2) overall quality with specific evidence from the code — quote artifact names and observations; (3) top findings. Never use generic phrases like "overall good quality" without supporting specifics.' },
        },
        required: ['score', 'recommendation', 'good_points', 'issues', 'summary'],
      },
    },
  },
];

// ── Tool execution ────────────────────────────────────────────────────────────
// Tracks whether post_review_to_update_set was successfully called
let reviewPosted = false;
let tokenLimitEvents = 0;

function clampNumber(value, fallback, min, max) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.min(max, Math.floor(n)));
}

function clipText(value, maxChars) {
  if (typeof value !== 'string') return value;
  if (value.length <= maxChars) return value;
  return `${value.slice(0, maxChars)}\n... [TRUNCATED ${value.length - maxChars} chars to control token usage]`;
}

function normalizeToolResult(toolName, result) {
  let content = typeof result === 'string' ? result : JSON.stringify(result);
  const maxChars = toolName === 'get_change_code' ? TOOL_RESULT_MAX_CHARS : Math.floor(TOOL_RESULT_MAX_CHARS * 0.7);
  if (content.length > maxChars) {
    content = `${content.slice(0, maxChars)}\n... [TRUNCATED ${content.length - maxChars} chars from tool output to control model context size]`;
  }
  return content;
}

function chunkArray(items, size) {
  const out = [];
  for (let i = 0; i < items.length; i += size) out.push(items.slice(i, i + size));
  return out;
}

function artifactCategoryFromLocation(location) {
  const text = String(location || '').trim();
  if (!text) return 'Uncategorized';
  const idx = text.indexOf(':');
  if (idx <= 0) return 'Uncategorized';
  return text.slice(0, idx).trim() || 'Uncategorized';
}

function buildChunkPlan(issues) {
  const byCategory = new Map();
  for (const issue of issues) {
    const key = artifactCategoryFromLocation(issue?.location);
    if (!byCategory.has(key)) byCategory.set(key, []);
    byCategory.get(key).push(issue);
  }

  const groups = [];
  for (const [category, categoryIssues] of byCategory.entries()) {
    const parts = chunkArray(categoryIssues, CHUNK_SIZE);
    parts.forEach((partIssues, idx) => {
      groups.push({
        category,
        part: idx + 1,
        totalParts: parts.length,
        issues: partIssues,
      });
    });
  }
  return groups;
}

function collectDeclaredSymbolsFromChanges() {
  const declared = new Set();

  for (const artifact of changeCodeCache.values()) {
    for (const field of CODE_FIELDS) {
      const src = artifact?.[field];
      if (typeof src !== 'string' || !src) continue;

      let m;
      const fnDecl = /function\s+([A-Za-z_$][\w$]*)\s*\(/g;
      while ((m = fnDecl.exec(src)) !== null) declared.add(m[1]);

      const classCreateDecl = /var\s+([A-Za-z_$][\w$]*)\s*=\s*Class\.create\s*\(/g;
      while ((m = classCreateDecl.exec(src)) !== null) declared.add(m[1]);

      const methodDecl = /([A-Za-z_$][\w$]*)\s*:\s*function\s*\(/g;
      while ((m = methodDecl.exec(src)) !== null) declared.add(m[1]);
    }
  }

  return declared;
}

function extractPotentialMissingSymbol(issue) {
  const text = `${String(issue?.location || '')} ${String(issue?.description || '')} ${String(issue?.suggestion || '')}`;

  let m = text.match(/script include[^A-Za-z0-9_$]*['\"]?([A-Za-z_$][\w$]*)['\"]?/i);
  if (m && m[1]) return m[1];

  m = text.match(/['\"]([A-Za-z_$][\w$]*)['\"]\s+is\s+referenced\s+but\s+not\s+present/i);
  if (m && m[1]) return m[1];

  return null;
}

function hasUnresolvedDependencyIssue(issues, declaredSymbols) {
  return (issues || []).some((i) => {
    const location = String(i?.location || '').toLowerCase();
    const description = String(i?.description || '').toLowerCase();
    const suggestion = String(i?.suggestion || '').toLowerCase();
    const text = `${location} ${description} ${suggestion}`;
    const candidate = extractPotentialMissingSymbol(i);
    if (candidate && declaredSymbols && declaredSymbols.has(candidate)) return false;
    return (
      text.includes('not found in the instance') ||
      text.includes('not found in instance') ||
      text.includes('referenced but not present') ||
      text.includes('script include') && text.includes('not found')
    );
  });
}

function applyRecommendationGuards({ score, recommendation, issues, summary }) {
  let guardedScore = Number(score);
  let guardedRecommendation = recommendation;
  const guardNotes = [];
  const declaredSymbols = collectDeclaredSymbolsFromChanges();

  // Hard guard: unresolved dependencies should never be approved as PUSH.
  if (hasUnresolvedDependencyIssue(issues, declaredSymbols) && guardedRecommendation === 'PUSH') {
    guardedRecommendation = 'HOLD FOR REVIEW';
    guardedScore = Math.min(Number.isFinite(guardedScore) ? guardedScore : 69, 69);
    guardNotes.push('Guard applied: unresolved dependency detected; recommendation downgraded from PUSH to HOLD FOR REVIEW.');
  }

  const guardedSummary = guardNotes.length
    ? `${guardNotes.join(' ')}\n\n${summary || ''}`.trim()
    : summary;

  return {
    score: Number.isFinite(guardedScore) ? guardedScore : score,
    recommendation: guardedRecommendation,
    summary: guardedSummary,
  };
}

async function executeTool(name, args) {
  switch (name) {

    case 'get_update_set_details': {
      const record = await snowGet(`/table/sys_update_set/${UPDATE_SET_SYS_ID}`, {
        sysparm_fields: 'name,description,state,application,sys_created_by,sys_created_on,release_date,sys_scope',
      });
      return JSON.stringify(record);
    }

    case 'get_update_set_changes': {
      const offset = clampNumber(args.offset, 0, 0, CHANGES_LIMIT);
      const limit = clampNumber(args.limit, CHANGE_PAGE_DEFAULT, 1, CHANGE_PAGE_MAX);
      const changes = await snowGet('/table/sys_update_xml', {
        sysparm_query:  `update_set=${UPDATE_SET_SYS_ID}`,
        sysparm_fields: 'name,type,payload,action,target_name,category,sys_created_by,sys_updated_on',
        sysparm_limit:  CHANGES_LIMIT,
      });

      if (changes.length === CHANGES_LIMIT) {
        console.warn(`[WARN] Fetched exactly ${CHANGES_LIMIT} changes — the Update Set may have more records that were NOT reviewed.`);
      }

      const summary = changes.map((c) => {
        const artifact = parsePayload(c.payload, c.type, c.target_name || c.name);
        const meta = {
          record_name:   c.name,
          type:          c.type,
          category:      c.category,
          action:        c.action,
          created_by:    typeof c.sys_created_by === 'object' ? c.sys_created_by?.display_value : c.sys_created_by,
          artifact_name: artifact.name,
          ...(artifact.when   ? { when: artifact.when }     : {}),
          ...(artifact.active ? { active: artifact.active } : {}),
          ...(artifact.table  ? { table: artifact.table }   : {}),
        };
        const codeFields = CODE_FIELDS.filter(f => artifact[f]);
        if (codeFields.length) {
          changeCodeCache.set(c.name, artifact);
          meta.has_code     = true;
          meta.code_fields  = codeFields;
          meta.code_preview = artifact[codeFields[0]].slice(0, 100);
        }
        return meta;
      });

      const withCode = summary.filter(p => p.has_code);
      console.log(`     Found ${changes.length} changes, ${withCode.length} contain code`);
      // Return ONLY code-containing records and page the result to avoid token overflows.
      const paged = withCode.slice(offset, offset + limit);
      const nextOffset = offset + paged.length;
      const hasMore = nextOffset < withCode.length;
      return JSON.stringify({
        total_records: withCode.length,
        offset,
        limit,
        returned: paged.length,
        has_more: hasMore,
        next_offset: hasMore ? nextOffset : null,
        records: paged,
      });
    }

    case 'get_change_code': {
      const { record_name } = args;
      const maxCharsPerField = clampNumber(args.max_chars_per_field, CODE_FIELD_MAX_CHARS, 500, 12000);
      if (!record_name) return 'Error: record_name is required.';
      const artifact = changeCodeCache.get(record_name);
      if (!artifact) return `No code found for record '${record_name}'. Make sure you called get_update_set_changes first.`;
      console.log(`     Returning full code for: ${record_name}`);
      const trimmed = { ...artifact };
      const truncatedFields = [];
      for (const field of CODE_FIELDS) {
        if (typeof trimmed[field] === 'string' && trimmed[field].length > maxCharsPerField) {
          trimmed[field] = clipText(trimmed[field], maxCharsPerField);
          truncatedFields.push(field);
        }
      }
      return JSON.stringify({
        ...trimmed,
        _meta: {
          max_chars_per_field: maxCharsPerField,
          truncated_fields: truncatedFields,
        },
      });
    }

    case 'get_existing_script_include': {
      const results = await snowGet('/table/sys_script_include', {
        sysparm_query:  `name=${args.name}`,
        sysparm_fields: 'name,script,description,active,access',
        sysparm_limit:  '1',
      });
      if (!results?.length) return `Script Include '${args.name}' not found.`;
      const r = results[0];
      return JSON.stringify({ name: r.name, description: r.description, active: r.active, script: r.script });
    }

    case 'get_jira_ticket': {
      return await fetchJiraTicket(args.ticket_id);
    }

    // ── Main output tool — writes to u_ai_code_review + brief work note ────────
    case 'post_review_to_update_set': {
      const { score, recommendation, good_points = [], issues = [], jira_alignment, summary } = args;
      const guarded = applyRecommendationGuards({ score, recommendation, issues, summary });
      const finalScore = guarded.score;
      const finalRecommendation = guarded.recommendation;
      const finalSummary = guarded.summary;

      // 1. Build HTML review content
      const htmlContent = buildReviewHtml({
        score: finalScore,
        recommendation: finalRecommendation,
        good_points,
        issues,
        jira_alignment,
        summary: finalSummary,
      });

      // 2. Persist review in one record or in category-based chunks for large payloads.
      const needsMultiPart = FORCE_CHUNKED || htmlContent.length > 7800 || issues.length > 20;
      let reviewSysId = 'unknown';
      const childReviewIds = [];

      if (!needsMultiPart) {
        const reviewRecord = await snowPost('/table/u_ai_code_review', {
          u_update_set: UPDATE_SET_SYS_ID,
          u_review_engine: REVIEW_ENGINE,
          u_description: htmlContent.slice(0, 8000),
          u_emergency_override: false,
        });
        reviewSysId = reviewRecord?.sys_id ?? 'unknown';
        console.log(`     u_ai_code_review record created: ${reviewSysId}`);
      } else {
        const groups = buildChunkPlan(issues);
        console.log(`     Multi-part mode active. Creating ${groups.length} child review record(s).`);

        for (const g of groups) {
          const childSummary = [
            `Chunked review for ${g.category} findings.`,
            `Part ${g.part} of ${g.totalParts}.`,
            `Contains ${g.issues.length} issue(s).`,
          ].join(' ');

          const childHtml = buildReviewHtml({
            score: finalScore,
            recommendation: finalRecommendation,
            good_points: [],
            issues: g.issues,
            jira_alignment,
            summary: childSummary,
          });

          const childRecord = await snowPost('/table/u_ai_code_review', {
            u_update_set: UPDATE_SET_SYS_ID,
            u_review_engine: REVIEW_ENGINE,
            u_description: childHtml.slice(0, 8000),
            u_emergency_override: false,
          });
          const childId = childRecord?.sys_id || 'unknown';
          childReviewIds.push(childId);
          console.log(`     child review created: ${childId} (${g.category} ${g.part}/${g.totalParts})`);
        }

        const issueCountByCategory = new Map();
        for (const issue of issues) {
          const category = artifactCategoryFromLocation(issue?.location);
          issueCountByCategory.set(category, (issueCountByCategory.get(category) || 0) + 1);
        }
        const categoryLines = Array.from(issueCountByCategory.entries())
          .map(([cat, count]) => `- ${cat}: ${count} issue(s)`)
          .join('\n');
        const childLines = childReviewIds.map((id, i) => `${i + 1}. ${id}`).join('\n');

        const summaryHtml = buildReviewHtml({
          score: finalScore,
          recommendation: finalRecommendation,
          good_points,
          issues: [],
          jira_alignment,
          summary: [
            finalSummary,
            '',
            `This review was split into ${childReviewIds.length} child record(s) due to payload size constraints.`,
            'Issue distribution by artifact category:',
            categoryLines || '- None',
            '',
            'Child review sys_id list:',
            childLines || 'None',
          ].join('\n'),
        });

        const summaryRecord = await snowPost('/table/u_ai_code_review', {
          u_update_set: UPDATE_SET_SYS_ID,
          u_review_engine: REVIEW_ENGINE,
          u_description: summaryHtml.slice(0, 8000),
          u_emergency_override: false,
        });
        reviewSysId = summaryRecord?.sys_id ?? 'unknown';
        console.log(`     summary review created: ${reviewSysId}`);
      }

      // 3. Post a brief work note on the Update Set itself so it shows in the activity stream
      const badge = { PUSH: '[PASS]', 'PUSH WITH MINOR FIXES': '[WARN]', 'HOLD FOR REVIEW': '[HOLD]', 'DO NOT PUSH': '[FAIL]' }[finalRecommendation] ?? '[?]';
      const briefNote = [
        `${badge} AI Code Review complete — Score: ${finalScore}/100 — ${finalRecommendation}`,
        `Full review details: navigate to Code Reviews and filter by this Update Set (record sys_id: ${reviewSysId}).`,
        ...(childReviewIds.length ? [`Additional child reviews (${childReviewIds.length}): ${childReviewIds.join(', ')}`] : []),
        `Reviewed by: ${REVIEW_ENGINE} on ${new Date().toUTCString()}`,
      ].join('\n');

      await snowPatch(`/table/sys_update_set/${UPDATE_SET_SYS_ID}`, { work_notes: briefNote });

      console.log(`\n  Review posted. Score: ${finalScore}/100 | ${finalRecommendation} | u_ai_code_review: ${reviewSysId}`);
      reviewPosted = true;
      return `Review saved to u_ai_code_review (sys_id: ${reviewSysId}). Score: ${finalScore}/100, Recommendation: ${finalRecommendation}.`;
    }

    default:
      throw new Error(`Unknown tool: ${name}`);
  }
}

// ── System Prompt ─────────────────────────────────────────────────────────────
const SYSTEM_PROMPT = `You are a senior ServiceNow architect and code reviewer.
Review a ServiceNow Update Set and produce a decisional score on whether it is safe to push to production.

=== REQUIRED WORKFLOW ===
1. Call get_update_set_details
   -> Read name, description, scope
   -> Extract any Jira ticket ID from the description (pattern [A-Z]+-[0-9]+, e.g. TT-1234)
2. Call get_update_set_changes
  -> Returns metadata + short preview PAGE for code-containing changes
  -> If has_more=true, call again with next_offset until done
   -> Identify all records where has_code=true
3. For EACH record where has_code=true, call get_change_code(record_name) one at a time
  -> Review code thoroughly against the checklist below
  -> If _meta.truncated_fields has entries, focus on critical logic and avoid requesting huge payloads repeatedly
4. For Script Includes referenced in code but NOT in the Update Set, call get_existing_script_include
5. If a Jira ticket ID was found, call get_jira_ticket
6. Analyse everything thoroughly
7. Call post_review_to_update_set ONCE with complete findings — this is mandatory, do not stop without calling it

=== REVIEW CHECKLIST ===

ServiceNow Best Practices:
- GlideRecord queries must have setLimit(); use encoded queries (never string concat)
- Client Scripts: no synchronous GlideRecord; must use GlideAjax
- Script Includes: class-based pattern (Class.create()), correct scope
- Business Rules: correct trigger (before/after/async); no heavy logic in sync rules
- No hardcoded sys_ids, URLs, or instance-specific values
- Proper null checks before accessing object properties
- No deprecated APIs

Security (OWASP + ServiceNow):
- No SQL injection via GlideRecord string concatenation
- No XSS in UI Pages / Jelly / HTML — escape user input
- ACLs not bypassed without justification
- No sensitive data in gs.log(), gs.print()
- No hardcoded credentials, tokens, or API keys

Code Quality:
- DRY — no duplicated logic that belongs in a utility Script Include
- Functions are reasonably sized and single-purpose
- Error handling: try/catch around external calls
- Consistent naming conventions
- No dead code or commented-out blocks

Jira Alignment (if ticket available):
- Does the code implement what the ticket describes?
- Are all acceptance criteria addressed?
- Any over-engineering or missing requirements?

=== SCORING ===
90-100 -> PUSH
70-89  -> PUSH WITH MINOR FIXES
50-69  -> HOLD FOR REVIEW
0-49   -> DO NOT PUSH

=== EVIDENCE RULES (mandatory — vague findings will waste the reviewer's time) ===
Every issue you report MUST follow ALL of these rules:

1. code_snippet: Paste the EXACT lines verbatim from what get_change_code returned.
   Never paraphrase. Never invent. Copy-paste only.

2. fixed_snippet: Write the corrected code showing exactly what to type instead.
   WRONG: "Add a null check before accessing the property"
   RIGHT: paste the corrected 3-5 lines with the null check actually written in

3. location: Include artifact type + exact name + function if applicable.
   WRONG: "Business Rule"
   RIGHT: "Business Rule: ValidateCase — before insert on incident"

4. description: Name the exact variable/function/API call involved.
   WRONG: "Missing error handling"
   RIGHT: "rm.execute() on line 12 has no try/catch — any HTTP timeout will crash the Business Rule and silently fail the transaction"

5. Do NOT invent issues. If the code is clean, say so clearly and score it high.
   INFO severity = observation only, no action required. Use sparingly (0-2 max).

6. summary must name each artifact reviewed and cite specific observations.
   WRONG: "The Update Set follows good practices overall."
   RIGHT: "The Business Rule 'ValidateCase' correctly uses encoded queries but calls gr.getValue() without checking isValidRecord(). The Script Include 'CaseUtils' is well-structured with proper Class.create() pattern, though processRequest() lacks try/catch around the REST call on line 34."

Update Set sys_id: ${UPDATE_SET_SYS_ID}
${UPDATE_SET_NAME ? `Update Set name: ${UPDATE_SET_NAME}` : ''}
${JIRA_TICKET ? `Provided Jira ticket: ${JIRA_TICKET}` : '(extract Jira ticket from description if present)'}`;

// ── Agent Loop ────────────────────────────────────────────────────────────────
const messages = [
  { role: 'system', content: SYSTEM_PROMPT },
  {
    role: 'user',
    content: `Review Update Set${UPDATE_SET_NAME ? ` "${UPDATE_SET_NAME}"` : ''} (sys_id: ${UPDATE_SET_SYS_ID}). Follow the required workflow and post results back to ServiceNow. You MUST call post_review_to_update_set at the end.`,
  },
];

console.log(`\n  AI Update Set Reviewer`);
console.log(`  Update Set : ${UPDATE_SET_NAME || UPDATE_SET_SYS_ID}`);
console.log(`  Instance   : ${SERVICENOW_INSTANCE}`);
console.log(`  Engine     : ${REVIEW_ENGINE}`);
console.log(`  Chunk Mode : ${FORCE_CHUNKED ? 'FORCED' : 'AUTO'}`);
console.log(`  Chunk Size : ${CHUNK_SIZE}`);
if (REQUESTED_BY) console.log(`  Requested By: ${REQUESTED_BY}`);
if (AI_TOKEN_SOURCE) console.log(`  Token Src  : ${AI_TOKEN_SOURCE}`);
if (JIRA_TICKET) console.log(`  Jira Ticket: ${JIRA_TICKET}`);
console.log();

// ── Context compression ───────────────────────────────────────────────────────
// Once the model produces a new assistant message it has already consumed the
// tool results from the previous round. Compressing those results keeps the
// running context well below the model's token ceiling for large Update Sets.
// Each tool result is tagged with the iteration it was produced in so we only
// compress results from completed (fully consumed) rounds.
function compressConsumedToolResults(messages, currentIteration) {
  for (const m of messages) {
    if (
      m.role === 'tool' &&
      m._iteration !== undefined &&
      m._iteration < currentIteration - 1 && // leave the most-recent round intact
      typeof m.content === 'string' &&
      m.content.length > 100
    ) {
      m.content = '[processed by model — content removed to save context]';
    }
  }
}

const MAX_ITERATIONS = 20;
let iteration = 0;

while (iteration < MAX_ITERATIONS) {
  iteration++;
  console.log(`[${iteration}/${MAX_ITERATIONS}] Calling model...`);

  let response;
  try {
    response = await openai.chat.completions.create({
      model:       MODEL_ID,
      messages,
      tools,
      tool_choice: 'auto',
      max_tokens:  4096,
      // Low temperature for deterministic, consistent code review output
      temperature: 0.1,
    });
  } catch (apiErr) {
    if (apiErr?.status === 413 || apiErr?.error?.code === 'tokens_limit_reached') {
      tokenLimitEvents++;
      console.warn('  [WARN] Token limit reached — truncating older tool results and retrying...');
      for (const m of messages) {
        if (m.role === 'tool' && typeof m.content === 'string' && m.content.length > 800) {
          m.content = m.content.slice(0, 800) + '\n[... TRUNCATED due to token limit ...]';
        }
      }
      response = await openai.chat.completions.create({
        model: MODEL_ID, messages, tools, tool_choice: 'auto', max_tokens: 4096, temperature: 0.1,
      });
    } else {
      throw apiErr;
    }
  }

  const choice  = response.choices[0];
  const message = choice.message;
  messages.push(message);
  compressConsumedToolResults(messages, iteration);

  if (choice.finish_reason === 'stop' || !message.tool_calls?.length) {
    console.log('\n  Agent finished.');
    if (message.content) console.log('  Note:', message.content);
    break;
  }

  for (const toolCall of message.tool_calls) {
    let args = {};
    let parseError = null;
    try {
      args = JSON.parse(toolCall.function.arguments);
    } catch (e) {
      parseError = e.message;
    }

    if (parseError) {
      // Surface the parse failure back to the model so it can retry with valid args
      console.error(`  [ERROR] Failed to parse args for ${toolCall.function.name}: ${parseError}`);
      messages.push({
        role: 'tool',
        tool_call_id: toolCall.id,
        content: `Error: could not parse tool arguments — ${parseError}. Please retry with valid JSON.`,
        _iteration: iteration,
      });
      continue;
    }

    console.log(`  -> ${toolCall.function.name}(${Object.keys(args).length ? JSON.stringify(args) : ''})`);

    let result;
    try {
      result = await executeTool(toolCall.function.name, args);
    } catch (err) {
      result = `Error: ${err.message}`;
      console.error(`     ${result}`);
    }

    messages.push({ role: 'tool', tool_call_id: toolCall.id, content: normalizeToolResult(toolCall.function.name, result), _iteration: iteration });
  }
}

if (iteration >= MAX_ITERATIONS) {
  console.warn('\n  [WARN] Max iterations reached.');
}

// ── Safety net — always leave a trace if the review was never posted ──────────
if (!reviewPosted) {
  console.error('  [ERROR] post_review_to_update_set was never called. Posting fallback error note to Update Set.');
  if (tokenLimitEvents > 0) {
    try {
      const partialSummary = [
        'Partial AI review: the agent hit model token/context limits while processing a large Update Set.',
        'This is not a full-quality gate result.',
        'Recommendation: split the Update Set or re-run review with smaller batches.',
      ].join(' ');

      const partialHtml = buildReviewHtml({
        score: 55,
        recommendation: 'HOLD FOR REVIEW',
        good_points: ['Automated review started successfully.', 'Code changes were fetched and processing began.'],
        issues: [{
          severity: 'HIGH',
          location: 'AI Review Pipeline',
          description: 'Model context limit reached before final consolidated review could be posted.',
          suggestion: 'Split the Update Set into smaller logical chunks and re-run the review.',
        }],
        summary: partialSummary,
      });

      const fallbackReview = await snowPost('/table/u_ai_code_review', {
        u_update_set: UPDATE_SET_SYS_ID,
        u_review_engine: REVIEW_ENGINE,
        u_description: partialHtml.slice(0, 8000),
        u_emergency_override: false,
      });

      await snowPatch(`/table/sys_update_set/${UPDATE_SET_SYS_ID}`, {
        work_notes: [
          '[AI Review] Partial review posted due to model token/context limits.',
          `Recommendation: HOLD FOR REVIEW. Score: 55/100.`,
          `u_ai_code_review sys_id: ${fallbackReview?.sys_id || 'unknown'}`,
          'Action: split the Update Set into smaller batches and re-run the review.',
          `Engine: ${REVIEW_ENGINE} | Time: ${new Date().toUTCString()}`,
        ].join('\n'),
      });

      console.warn('  [WARN] Partial review posted due to token limits. Exiting without fatal failure.');
      process.exit(0);
    } catch (partialErr) {
      console.error('  [ERROR] Failed to post partial review fallback:', partialErr.message);
    }
  }

  try {
    await snowPatch(`/table/sys_update_set/${UPDATE_SET_SYS_ID}`, {
      work_notes: [
        '[AI Review] ERROR: The AI review agent did not complete successfully.',
        `Reason: ${iteration >= MAX_ITERATIONS ? 'Max iterations (' + MAX_ITERATIONS + ') reached' : 'Agent stopped before posting review'}.`,
        'Please re-trigger the review or contact your platform team.',
        `Engine: ${REVIEW_ENGINE} | Time: ${new Date().toUTCString()}`,
      ].join('\n'),
    });
  } catch (fallbackErr) {
    console.error('  [ERROR] Even the fallback work note failed:', fallbackErr.message);
  }
  process.exit(1);
}

console.log('\nDone.');
