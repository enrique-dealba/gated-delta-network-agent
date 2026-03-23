/**
 * Experiment Enforcer — prevents the agent from stopping mid-experiment-loop.
 *
 * Parses the target iteration count from conversation history, tracks progress
 * via counting unique results.tsv rows (including confirmation runs) to match
 * actual compute budget consumed. Forces continuation on agent_end if iterations
 * remain. Compaction-safe via monotonic high water mark.
 */

import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";

const ENFORCER_STATUS_HEX = "#FFDE21";

function hexToAnsi(hex: string): string {
  const h = hex.replace("#", "");
  return `\x1b[38;2;${parseInt(h.slice(0, 2), 16)};${parseInt(h.slice(2, 4), 16)};${parseInt(h.slice(4, 6), 16)}m`;
}

function colorizeEnforcerStatus(text: string): string {
  return `${hexToAnsi(ENFORCER_STATUS_HEX)}${text}\x1b[0m`;
}

/**
 * Scan the full branch and count unique experiment rows seen across all messages.
 *
 * Matches TSV-style rows where a 3-digit experiment number is the first
 * non-whitespace on a line, followed by whitespace and a decimal val_bpb.
 * Whitespace is normalized before dedup so the same row displayed via
 * `cat results.tsv` (tabs) and tool output (spaces) counts only once.
 *
 * Does NOT match printf commands — those use different formatting and would
 * double-count rows that also appear when results.tsv is read back.
 */
export function scanBranchRowCount(ctx: any): number {
  const globalSeen = new Set<string>();

  for (const entry of ctx.sessionManager.getBranch()) {
    if (entry.type !== "message") continue;
    const msg = entry.message;

    const parts: string[] = [];

    if (msg.role === "toolResult") {
      for (const c of msg.content || []) {
        if (c.type === "text") parts.push(c.text);
      }
    } else if (msg.role === "user" || msg.role === "assistant") {
      if (typeof msg.content === "string") {
        parts.push(msg.content);
      } else if (Array.isArray(msg.content)) {
        for (const c of msg.content) {
          if (c.type === "text") parts.push(c.text);
        }
      }
    }

    // Capture bash command inputs from tool calls
    if (msg.role === "assistant" && Array.isArray(msg.content)) {
      for (const block of msg.content) {
        if (block.type === "tool_use" && block.input?.command) {
          parts.push(block.input.command);
        }
      }
    }

    const content = parts.join("\n");

    // Match TSV rows: 3-digit experiment number at line start, whitespace, decimal val_bpb.
    // Normalize whitespace so tabs vs spaces dedup correctly across different reads.
    for (const m of content.matchAll(/^\s*(\d{3}[\s\t]+\d+\.\d+.*\S)[\s]*$/gm)) {
      globalSeen.add(m[1].replace(/\s+/g, "\t"));
    }
  }

  return globalSeen.size;
}

export default function (pi: ExtensionAPI) {
  let targetRowCount: number | null = null;
  let startRowCount = 0;
  let requestedIters = 0;
  let lastNudgedAtRow: number | null = null;
  // High water mark — monotonically increasing, survives compaction
  let highWaterMark = 0;

  const parseTargetIters = (text: string): number | null => {
    // Match "-iters <N>", "--iters <N>", "-iters=<N>", or "--iters=<N>"
    const match = text.match(/--?iters(?:\s+|=)(\d+)/);
    return match ? parseInt(match[1], 10) : null;
  };

  const getCurrentRowCount = (ctx: any): number => {
    // Scan branch but never let the count decrease (compaction-safe)
    const scanned = scanBranchRowCount(ctx);
    if (scanned > highWaterMark) {
      highWaterMark = scanned;
    }
    return highWaterMark;
  };

  pi.on("session_start", async (_event, ctx) => {
    targetRowCount = null;
    startRowCount = 0;
    requestedIters = 0;
    lastNudgedAtRow = null;
    // On session start, seed high water mark from whatever is in the branch
    highWaterMark = scanBranchRowCount(ctx);
    ctx.ui.setStatus("experiment-enforcer", undefined);
  });

  // Parse target iterations from user input
  pi.on("input", async (event, ctx) => {
    lastNudgedAtRow = null;
    if (event.text) {
      const parsed = parseTargetIters(event.text);
      if (parsed !== null) {
        startRowCount = getCurrentRowCount(ctx);
        requestedIters = parsed;
        targetRowCount = startRowCount + parsed;
        ctx.ui.setStatus(
          "experiment-enforcer",
          colorizeEnforcerStatus(` Experiment Enforcer: 0/${requestedIters} iterations queued`)
        );
      }
    }
    return { action: "continue" as const };
  });

  // On agent_end, check if we're done
  pi.on("agent_end", async (_event, ctx) => {
    if (targetRowCount === null) return;

    const currentRows = getCurrentRowCount(ctx);
    const completed = currentRows - startRowCount;

    if (currentRows < targetRowCount) {
      // Only nudge if the agent made progress since the last nudge.
      // If no progress, the agent is stuck — don't infinite-loop.
      if (lastNudgedAtRow !== null && currentRows <= lastNudgedAtRow) {
        return;
      }

      lastNudgedAtRow = currentRows;
      const remaining = targetRowCount - currentRows;

      ctx.ui.setStatus(
        "experiment-enforcer",
        colorizeEnforcerStatus(` Experiment Enforcer: ${completed}/${requestedIters} — ${remaining} remaining`)
      );

      pi.sendMessage(
        {
          customType: "experiment-enforcer-nudge",
          content:
            `⚠️ You have completed ${completed}/${requestedIters} experiment iterations. ` +
            `${remaining} iterations remain. You were instructed to run ${requestedIters} iterations NON-STOP. ` +
            `Do NOT stop. Do NOT ask the user. Continue immediately with the next experiment.`,
          display: true,
        },
        { triggerTurn: true }
      );
    } else {
      ctx.ui.setStatus(
        "experiment-enforcer",
        colorizeEnforcerStatus(` Experiment Enforcer: ${completed}/${requestedIters} complete`)
      );
    }
  });
}
