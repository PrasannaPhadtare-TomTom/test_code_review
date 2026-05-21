/**
 * AI Update Set Reviewer
 *
 * Flow:
 *  1. Fetch Update Set metadata + all code changes from ServiceNow
 *  2. Identify any referenced Script Includes for deeper context
 *  3. Fetch linked Jira ticket for requirements + acceptance criteria
 *  4. AI (GitHub Models / gpt-4o) reviews all code against best practices
 *  5. Generates a decisional score (0-100) + PUSH / HOLD / DO NOT PUSH
 *  6. Posts the full review as work notes back to the Update Set in ServiceNow
 */

import OpenAI from 'openai';

// ── Environment ───────────────────────────────────────────────────────────────
const {
  GITHUB_TOKEN,
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

if (!GITHUB_TOKEN || !UPDATE_SET_SYS_ID || !SERVICENOW_INSTANCE || !SERVICENOW_USERNAME || !SERVICENOW_PASSWORD) {
  console.error('Missing required env vars: GITHUB_TOKEN, UPDATE_SET_SYS_ID, SERVICENOW_INSTANCE, SERVICENOW_USERNAME, SERVICENOW_PASSWORD');
  process.exit(1);
}

// ── GitHub Models (Copilot) client ────────────────────────────────────────────
const openai = new OpenAI({
  baseURL: 'https://models.inference.ai.azure.com',
  apiKey: GITHUB_TOKEN,
});

// ── ServiceNow REST helpers ───────────────────────────────────────────────────
const snowBase = `https://${SERVICENOW_INSTANCE}.service-now.com/api/now`;
const snowAuth = 'Basic ' + Buffer.from(`${SERVICENOW_USERNAME}:${SERVICENOW_PASSWORD}`).toString('base64');

async function snowGet(path, params = {}) {
  const url = new URL(`${snowBase}${path}`);
  for (const [k, v] of Object.entries(params)) url.searchParams.set(k, v);
  const res = await fetch(url.toString(), {
    headers: { Authorization: snowAuth, Accept: 'application/json' },
  });
  if (!res.ok) throw new Error(`SNOW GET ${path} -> HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`);
  return (await res.json()).result;
}

async function snowPatch(path, body) {
  const res = await fetch(`${snowBase}${path}`, {
    method: 'PATCH',
    headers: { Authorization: snowAuth, 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`SNOW PATCH ${path} -> HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`);
  return (await res.json()).result;
}

// ── XML parsing helpers ───────────────────────────────────────────────────────
function extractXmlTag(xml, tag) {
  if (!xml) return null;
  const re = new RegExp(`<${tag}(?:\\s[^>]*)?>([\\s\\S]*?)<\/${tag}>`, 'i');
  const m = xml.match(re);
  if (!m) return null;
  return m[1]
    .replace(/<!\[CDATA\[|\]\]>/g, '')
    .replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&').replace(/&quot;/g, '"')
    .trim();
}

const CODE_FIELDS = ['script', 'condition', 'advanced_condition', 'html', 'css', 'body', 'template', 'message'];

// In-memory cache: record_name -> full parsed artifact (populated by get_update_set_changes)
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
  return { type, name,
    ...(when   ? { when }   : {}),
    ...(active ? { active } : {}),
    ...(table  ? { table }  : {}),
    ...extracted };
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
  const auth = Buffer.from(`${JIRA_EMAIL}:${JIRA_TOKEN}`).toString('base64');
  const res = await fetch(`${JIRA_URL.replace(/\/$/, '')}/rest/api/3/issue/${encodeURIComponent(ticketId)}`, {
    headers: { Authorization: `Basic ${auth}`, Accept: 'application/json' },
  });
  if (!res.ok) return `Jira ticket ${ticketId} not found (HTTP ${res.status}).`;
  const d = await res.json();
  return JSON.stringify({
    id: d.key,
    summary: d.fields?.summary,
    description: extractADFText(d.fields?.description),
    acceptanceCriteria: extractADFText(d.fields?.customfield_10016),
    status: d.fields?.status?.name,
    priority: d.fields?.priority?.name,
    type: d.fields?.issuetype?.name,
  });
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
      description: 'Fetch ALL changes in the Update Set. Returns metadata and a short code preview for each change. For records with has_code=true, call get_change_code(record_name) to get the full code.',
      parameters: { type: 'object', properties: {} },
    },
  },
  {
    type: 'function',
    function: {
      name: 'get_change_code',
      description: 'Fetch the full code for a single Update Set change by its record_name. Call this once per code-containing record after get_update_set_changes.',
      parameters: {
        type: 'object',
        properties: { record_name: { type: 'string', description: 'The record_name value from get_update_set_changes' } },
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
      description: 'Post the complete AI review as work notes on the ServiceNow Update Set. Call this ONCE at the very end with all findings.',
      parameters: {
        type: 'object',
        properties: {
          score: { type: 'number', description: '0-100 decisional score. 90-100=PUSH, 70-89=PUSH WITH MINOR FIXES, 50-69=HOLD FOR REVIEW, 0-49=DO NOT PUSH' },
          recommendation: { type: 'string', enum: ['PUSH', 'PUSH WITH MINOR FIXES', 'HOLD FOR REVIEW', 'DO NOT PUSH'] },
          good_points: { type: 'array', items: { type: 'string' }, description: 'What is done well' },
          issues: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                severity:    { type: 'string', enum: ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO'] },
                location:    { type: 'string', description: 'e.g. "Business Rule: MyRule"' },
                description: { type: 'string' },
                suggestion:  { type: 'string', description: 'How to fix, with code snippet if possible' },
              },
            },
          },
          jira_alignment: { type: 'string', description: 'How well the Update Set meets the Jira ticket requirements' },
          summary:        { type: 'string', description: 'Executive summary: what the US does, overall quality, key findings (2-3 paragraphs)' },
        },
        required: ['score', 'recommendation', 'good_points', 'issues', 'summary'],
      },
    },
  },
];

// ── Tool execution ────────────────────────────────────────────────────────────
async function executeTool(name, args) {
  switch (name) {

    case 'get_update_set_details': {
      const record = await snowGet(`/table/sys_update_set/${UPDATE_SET_SYS_ID}`, {
        sysparm_fields: 'name,description,state,application,sys_created_by,sys_created_on,release_date,sys_scope',
      });
      return JSON.stringify(record);
    }

    case 'get_update_set_changes': {
      const changes = await snowGet('/table/sys_update_xml', {
        sysparm_query:  `update_set=${UPDATE_SET_SYS_ID}`,
        sysparm_fields: 'name,type,payload,action,target_name,category,sys_created_by,sys_updated_on',
        sysparm_limit:  '150',
      });
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
        // Store full code in cache; expose only a 200-char preview here
        const codeFields = CODE_FIELDS.filter(f => artifact[f]);
        if (codeFields.length) {
          changeCodeCache.set(c.name, artifact);
          meta.has_code     = true;
          meta.code_fields  = codeFields;
          meta.code_preview = artifact[codeFields[0]].slice(0, 200);
        }
        return meta;
      });
      const withCode = summary.filter(p => p.has_code);
      console.log(`     Found ${changes.length} changes, ${withCode.length} contain code`);
      return JSON.stringify(summary);
    }

    case 'get_change_code': {
      const { record_name } = args;
      const artifact = changeCodeCache.get(record_name);
      if (!artifact) return `No code found for record '${record_name}'. Make sure you called get_update_set_changes first.`;
      console.log(`     Returning full code for: ${record_name}`);
      return JSON.stringify(artifact);
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

    case 'post_review_to_update_set': {
      const { score, recommendation, good_points = [], issues = [], jira_alignment, summary } = args;
      const badge = { PUSH: '[PASS]', 'PUSH WITH MINOR FIXES': '[WARN]', 'HOLD FOR REVIEW': '[HOLD]', 'DO NOT PUSH': '[FAIL]' }[recommendation] ?? '[?]';
      const sep = '-'.repeat(60);
      const goodSection = good_points.length ? good_points.map(g => `  + ${g}`).join('\n') : '  (none noted)';
      const issueSection = issues.length
        ? issues.map(i =>
            `  [${i.severity}] ${i.location}\n  Problem: ${i.description}${i.suggestion ? '\n  Fix:     ' + i.suggestion : ''}`
          ).join('\n\n')
        : '  No issues found.';

      const workNote = [
        `${badge} AI UPDATE SET REVIEW --- Score: ${score}/100 --- ${recommendation}`,
        sep, '',
        summary, '',
        sep, 'WHAT IS GOOD:',
        goodSection, '',
        sep, `ISSUES FOUND (${issues.length}):`,
        issueSection,
        ...(jira_alignment ? ['', sep, 'JIRA TICKET ALIGNMENT:', `  ${jira_alignment}`] : []),
        '', sep,
        `Reviewed by AI (GitHub Copilot / gpt-4o) on ${new Date().toUTCString()}`,
      ].join('\n');

      await snowPatch(`/table/sys_update_set/${UPDATE_SET_SYS_ID}`, { work_notes: workNote });
      console.log(`\n  Review posted. Score: ${score}/100 | ${recommendation}`);
      return `Review posted. Score: ${score}/100, Recommendation: ${recommendation}.`;
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
   -> Returns metadata + short preview for every change
   -> Identify all records where has_code=true
3. For EACH record where has_code=true, call get_change_code(record_name) one at a time
   -> Review the full code thoroughly against the checklist below
4. For Script Includes referenced in code but NOT in the Update Set, call get_existing_script_include
5. If a Jira ticket ID was found, call get_jira_ticket
6. Analyse everything thoroughly
7. Call post_review_to_update_set ONCE with complete findings

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

Update Set sys_id: ${UPDATE_SET_SYS_ID}
${UPDATE_SET_NAME ? `Update Set name: ${UPDATE_SET_NAME}` : ''}
${JIRA_TICKET ? `Provided Jira ticket: ${JIRA_TICKET}` : '(extract Jira ticket from description if present)'}`;

// ── Agent Loop ────────────────────────────────────────────────────────────────
const messages = [
  { role: 'system', content: SYSTEM_PROMPT },
  {
    role: 'user',
    content: `Review Update Set${UPDATE_SET_NAME ? ` "${UPDATE_SET_NAME}"` : ''} (sys_id: ${UPDATE_SET_SYS_ID}). Follow the required workflow and post results back to ServiceNow.`,
  },
];

console.log(`\n  AI Update Set Reviewer`);
console.log(`  Update Set : ${UPDATE_SET_NAME || UPDATE_SET_SYS_ID}`);
console.log(`  Instance   : ${SERVICENOW_INSTANCE}`);
if (JIRA_TICKET) console.log(`  Jira Ticket: ${JIRA_TICKET}`);
console.log();

const MAX_ITERATIONS = 20;
let iteration = 0;

while (iteration < MAX_ITERATIONS) {
  iteration++;
  console.log(`[${iteration}/${MAX_ITERATIONS}] Calling model...`);

  let response;
  try {
    response = await openai.chat.completions.create({
      model: 'gpt-4.1',
      messages,
      tools,
      tool_choice: 'auto',
      max_tokens: 4096,
      temperature: 0.1,
    });
  } catch (apiErr) {
    if (apiErr?.status === 413 || apiErr?.error?.code === 'tokens_limit_reached') {
      console.warn('  [WARN] Token limit reached — truncating oldest tool results and retrying...');
      // Trim the content of all tool messages in-place to at most 3000 chars
      for (const m of messages) {
        if (m.role === 'tool' && typeof m.content === 'string' && m.content.length > 3000) {
          m.content = m.content.slice(0, 3000) + '\n[... TRUNCATED due to token limit ...]';
        }
      }
      response = await openai.chat.completions.create({
        model: 'gpt-4.1',
        messages,
        tools,
        tool_choice: 'auto',
        max_tokens: 4096,
        temperature: 0.1,
      });
    } else {
      throw apiErr;
    }
  }

  const choice  = response.choices[0];
  const message = choice.message;
  messages.push(message);

  if (choice.finish_reason === 'stop' || !message.tool_calls?.length) {
    console.log('\n  Agent finished.');
    if (message.content) console.log('  Note:', message.content);
    break;
  }

  for (const toolCall of message.tool_calls) {
    let args = {};
    try { args = JSON.parse(toolCall.function.arguments); } catch { /* ignore */ }

    console.log(`  -> ${toolCall.function.name}(${Object.keys(args).length ? JSON.stringify(args) : ''})`);

    let result;
    try {
      result = await executeTool(toolCall.function.name, args);
    } catch (err) {
      result = `Error: ${err.message}`;
      console.error(`     ${result}`);
    }

    messages.push({ role: 'tool', tool_call_id: toolCall.id, content: result });
  }
}

if (iteration >= MAX_ITERATIONS) console.warn('\n  Max iterations reached.');
console.log('\nDone.');
