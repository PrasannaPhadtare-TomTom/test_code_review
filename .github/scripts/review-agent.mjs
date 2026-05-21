/**
 * AI Code Review Agent
 * Uses GitHub Models API (Copilot) + MCP servers (GitHub, Filesystem)
 * + Jira & ServiceNow REST APIs for full contextual code review.
 */

import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import OpenAI from 'openai';

// ── Environment ──────────────────────────────────────────────────────────────
const {
  GITHUB_TOKEN,
  PR_NUMBER,
  REPO_OWNER,
  REPO_NAME,
  WORKSPACE_PATH,
  JIRA_URL,
  JIRA_EMAIL,
  JIRA_TOKEN,
  SERVICENOW_INSTANCE,
  SERVICENOW_USERNAME,
  SERVICENOW_PASSWORD,
} = process.env;

if (!GITHUB_TOKEN || !PR_NUMBER || !REPO_OWNER || !REPO_NAME) {
  console.error('❌ Missing required environment variables.');
  process.exit(1);
}

const prNumber = parseInt(PR_NUMBER, 10);

// ── GitHub Models (Copilot) client ───────────────────────────────────────────
// GitHub Models is OpenAI-compatible and uses GITHUB_TOKEN — no separate key needed.
const openai = new OpenAI({
  baseURL: 'https://models.inference.ai.azure.com',
  apiKey: GITHUB_TOKEN,
});

// ── MCP Clients ──────────────────────────────────────────────────────────────
async function createMCPClient(name, command, args, extraEnv = {}) {
  const transport = new StdioClientTransport({
    command,
    args,
    env: { ...process.env, ...extraEnv },
  });
  const client = new Client({ name, version: '1.0.0' }, { capabilities: {} });
  await client.connect(transport);
  return client;
}

console.log('🔌 Connecting to MCP servers...');

const githubMCP = await createMCPClient(
  'github-mcp',
  'npx',
  ['-y', '@modelcontextprotocol/server-github'],
  { GITHUB_PERSONAL_ACCESS_TOKEN: GITHUB_TOKEN }
);

const fsMCP = await createMCPClient(
  'fs-mcp',
  'npx',
  ['-y', '@modelcontextprotocol/server-filesystem', WORKSPACE_PATH || process.cwd()]
);

console.log('✅ MCP servers connected.');

// ── Tool registration ─────────────────────────────────────────────────────────
const { tools: rawGithubTools } = await githubMCP.listTools();
const { tools: rawFsTools } = await fsMCP.listTools();

/**
 * Convert an MCP tool definition to OpenAI function-calling format.
 * @param {object} tool   - MCP tool descriptor
 * @param {string} prefix - namespace prefix ('github' | 'fs')
 */
function toOpenAITool(tool, prefix) {
  return {
    type: 'function',
    function: {
      name: `${prefix}__${tool.name}`,
      description: tool.description ?? '',
      parameters: tool.inputSchema ?? { type: 'object', properties: {} },
    },
  };
}

// Custom tools for Jira and ServiceNow (handled inline — no extra MCP server needed)
const customTools = [
  {
    type: 'function',
    function: {
      name: 'jira__get_ticket',
      description:
        'Fetch a Jira ticket by ID (e.g. PROJ-123). Returns summary, description, acceptance criteria, status, and priority. Use this when the PR title or body references a Jira ticket.',
      parameters: {
        type: 'object',
        properties: {
          ticket_id: {
            type: 'string',
            description: 'Jira ticket ID, e.g. TT-1234',
          },
        },
        required: ['ticket_id'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'servicenow__get_incident',
      description:
        'Fetch a ServiceNow incident by number (e.g. INC0001234). Returns short_description, state, and priority. Use this when the PR references a SNOW incident.',
      parameters: {
        type: 'object',
        properties: {
          incident_number: {
            type: 'string',
            description: 'ServiceNow incident number, e.g. INC0001234',
          },
        },
        required: ['incident_number'],
      },
    },
  },
];

const allTools = [
  ...rawGithubTools.map((t) => toOpenAITool(t, 'github')),
  ...rawFsTools.map((t) => toOpenAITool(t, 'fs')),
  ...customTools,
];

// ── Tool execution ────────────────────────────────────────────────────────────
async function callMCPTool(toolName, args) {
  if (toolName.startsWith('github__')) {
    const name = toolName.slice('github__'.length);
    const result = await githubMCP.callTool({ name, arguments: args });
    return result.content?.[0]?.text ?? JSON.stringify(result.content);
  }

  if (toolName.startsWith('fs__')) {
    const name = toolName.slice('fs__'.length);
    const result = await fsMCP.callTool({ name, arguments: args });
    return result.content?.[0]?.text ?? JSON.stringify(result.content);
  }

  if (toolName === 'jira__get_ticket') {
    return await fetchJiraTicket(args.ticket_id);
  }

  if (toolName === 'servicenow__get_incident') {
    return await fetchServiceNowIncident(args.incident_number);
  }

  throw new Error(`Unknown tool: ${toolName}`);
}

// ── Jira REST API ─────────────────────────────────────────────────────────────
async function fetchJiraTicket(ticketId) {
  if (!JIRA_URL || !JIRA_EMAIL || !JIRA_TOKEN) {
    return `Jira credentials not configured. Cannot fetch ${ticketId}.`;
  }
  const auth = Buffer.from(`${JIRA_EMAIL}:${JIRA_TOKEN}`).toString('base64');
  const url = `${JIRA_URL.replace(/\/$/, '')}/rest/api/3/issue/${encodeURIComponent(ticketId)}`;
  const res = await fetch(url, {
    headers: { Authorization: `Basic ${auth}`, Accept: 'application/json' },
  });
  if (!res.ok) {
    return `Could not fetch Jira ticket ${ticketId}: HTTP ${res.status}`;
  }
  const data = await res.json();
  return JSON.stringify({
    id: data.key,
    summary: data.fields?.summary,
    description: extractADFText(data.fields?.description),
    acceptanceCriteria: extractADFText(data.fields?.customfield_10016),
    status: data.fields?.status?.name,
    priority: data.fields?.priority?.name,
    type: data.fields?.issuetype?.name,
  });
}

/** Recursively extract plain text from Atlassian Document Format (ADF). */
function extractADFText(node) {
  if (!node) return null;
  if (typeof node === 'string') return node;
  if (node.type === 'text') return node.text ?? '';
  if (Array.isArray(node.content)) {
    return node.content.map(extractADFText).filter(Boolean).join(' ');
  }
  return null;
}

// ── ServiceNow REST API ───────────────────────────────────────────────────────
async function fetchServiceNowIncident(incidentNumber) {
  if (!SERVICENOW_INSTANCE || !SERVICENOW_USERNAME || !SERVICENOW_PASSWORD) {
    return `ServiceNow credentials not configured. Cannot fetch ${incidentNumber}.`;
  }
  const auth = Buffer.from(`${SERVICENOW_USERNAME}:${SERVICENOW_PASSWORD}`).toString('base64');
  const url = `https://${SERVICENOW_INSTANCE}.service-now.com/api/now/table/incident?sysparm_query=number=${encodeURIComponent(incidentNumber)}&sysparm_fields=number,short_description,description,state,priority&sysparm_limit=1`;
  const res = await fetch(url, {
    headers: { Authorization: `Basic ${auth}`, Accept: 'application/json' },
  });
  if (!res.ok) {
    return `Could not fetch SNOW incident ${incidentNumber}: HTTP ${res.status}`;
  }
  const data = await res.json();
  const record = data.result?.[0];
  if (!record) return `ServiceNow incident ${incidentNumber} not found.`;
  return JSON.stringify(record);
}

// ── System prompt ─────────────────────────────────────────────────────────────
const SYSTEM_PROMPT = `You are a senior code reviewer specialising in TypeScript and JavaScript.
Your task is to perform a thorough, actionable review of GitHub PR #${prNumber} in ${REPO_OWNER}/${REPO_NAME}.

## Review areas (cover ALL of them)
1. **Bugs & Logic Errors** — off-by-one, null/undefined, async/await misuse, unhandled promises, race conditions
2. **Security (OWASP Top 10)** — injection, broken auth, sensitive data exposure, XSS, insecure dependencies, misconfiguration
3. **Code Quality** — DRY, naming, cyclomatic complexity, TypeScript strict typing, dead code
4. **Test Coverage** — missing unit/integration tests, untested edge cases, test quality
5. **Ticket Validation**
   - Extract Jira IDs (pattern [A-Z]+-[0-9]+) from the PR title and body → call jira__get_ticket
   - Extract SNOW numbers (pattern INC[0-9]+) from the PR title and body → call servicenow__get_incident
   - Verify the code change actually addresses the linked ticket's requirements

## Workflow
1. Call github__get_pull_request to read PR metadata and description
2. Call github__list_pull_request_files to list changed files
3. For key files, call fs__read_file to get full context beyond the diff
4. Fetch any linked Jira tickets or ServiceNow incidents
5. Post the complete review with github__create_pull_request_review
   - Use event "REQUEST_CHANGES" if critical/high issues exist, "COMMENT" otherwise
   - Include inline comments (with path + position) for specific line issues
   - Include a summary body covering all 5 review areas

## Comment format
Use severity labels: [CRITICAL] [HIGH] [MEDIUM] [LOW] [INFO]
Always explain WHY it is an issue and HOW to fix it with a code snippet where possible.`;

// ── Agent loop ────────────────────────────────────────────────────────────────
const messages = [
  { role: 'system', content: SYSTEM_PROMPT },
  {
    role: 'user',
    content: `Please review PR #${prNumber} in ${REPO_OWNER}/${REPO_NAME}. Cover all 5 review areas and post the full review as a GitHub PR review.`,
  },
];

const MAX_ITERATIONS = 30;
let iteration = 0;

console.log(`\n🤖 Starting AI Code Review — PR #${prNumber} in ${REPO_OWNER}/${REPO_NAME}\n`);

while (iteration < MAX_ITERATIONS) {
  iteration++;
  console.log(`[${iteration}/${MAX_ITERATIONS}] Calling model...`);

  const response = await openai.chat.completions.create({
    model: 'gpt-4o',
    messages,
    tools: allTools,
    tool_choice: 'auto',
    max_tokens: 4096,
    temperature: 0.2, // Lower temp for more deterministic reviews
  });

  const choice = response.choices[0];
  const message = choice.message;
  messages.push(message);

  // Model finished — no more tool calls
  if (choice.finish_reason === 'stop' || !message.tool_calls?.length) {
    console.log('\n✅ Review complete.');
    if (message.content) {
      console.log('\nFinal message from model:\n', message.content);
    }
    break;
  }

  // Execute each tool call in the response
  for (const toolCall of message.tool_calls) {
    const toolName = toolCall.function.name;
    let toolArgs;
    try {
      toolArgs = JSON.parse(toolCall.function.arguments);
    } catch {
      toolArgs = {};
    }

    console.log(`  → ${toolName}(${JSON.stringify(toolArgs)})`);

    let toolResult;
    try {
      toolResult = await callMCPTool(toolName, toolArgs);
    } catch (err) {
      toolResult = `Error: ${err.message}`;
      console.error(`  ✗ ${toolResult}`);
    }

    messages.push({
      role: 'tool',
      tool_call_id: toolCall.id,
      content: typeof toolResult === 'string' ? toolResult : JSON.stringify(toolResult),
    });
  }
}

if (iteration >= MAX_ITERATIONS) {
  console.warn('⚠️  Max iterations reached. Review may be incomplete.');
}

// ── Cleanup ───────────────────────────────────────────────────────────────────
await githubMCP.close();
await fsMCP.close();
console.log('\n🔌 MCP connections closed.');
