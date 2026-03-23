/**
 * Session Footer — Status footer with model, thinking level, context, tokens, cost
 */

import type { AssistantMessage } from "@mariozechner/pi-ai";
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { truncateToWidth, visibleWidth } from "@mariozechner/pi-tui";

function sanitizeStatusText(text: string): string {
  return text.replace(/[\r\n\t]/g, " ").replace(/ +/g, " ").trim();
}

function fmtTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return `${n}`;
}

const RAINBOW = ["#b281d6","#d787af","#febc38","#e4c00f","#89d281","#00afaf","#178fb9","#b281d6"];

function hexToAnsi(hex: string): string {
  const h = hex.replace("#", "");
  return `\x1b[38;2;${parseInt(h.slice(0,2),16)};${parseInt(h.slice(2,4),16)};${parseInt(h.slice(4,6),16)}m`;
}

function rainbow(text: string): string {
  let result = "";
  let i = 0;
  for (const ch of text) {
    if (ch === " " || ch === ":") result += ch;
    else { result += hexToAnsi(RAINBOW[i % RAINBOW.length]) + ch; i++; }
  }
  return result + "\x1b[0m";
}

const THINKING_LABELS: Record<string, string> = {
  off: "off", minimal: "min", low: "low", medium: "med", high: "high", xhigh: "xhigh",
};

export default function (pi: ExtensionAPI) {
  pi.on("session_start", async (_event, ctx) => {
    // Force thinking level for this session (clamped to model capabilities)
    pi.setThinkingLevel("medium"); // or "high", "medium", "low", "xhigh", "off"
    ctx.ui.setFooter((tui, theme, footerData) => {
      const unsub = footerData.onBranchChange(() => tui.requestRender());

      return {
        dispose: unsub,
        invalidate() {},
        render(width: number): string[] {
          let tokIn = 0, tokOut = 0, cost = 0;
          let thinkingLevel = "off";

          for (const entry of ctx.sessionManager.getBranch()) {
            if (entry.type === "thinking_level_change" && (entry as any).thinkingLevel) {
              thinkingLevel = (entry as any).thinkingLevel;
            }
            if (entry.type === "message" && entry.message.role === "assistant") {
              const m = entry.message as AssistantMessage;
              if (m.stopReason === "error" || m.stopReason === "aborted") continue;
              tokIn += m.usage.input;
              tokOut += m.usage.output;
              cost += m.usage.cost.total;
            }
          }

          // Model name (strip "Claude " prefix for brevity)
          let model = ctx.model?.name || ctx.model?.id || "no-model";
          if (model.startsWith("Claude ")) model = model.slice(7);

          // Thinking level
          const label = THINKING_LABELS[thinkingLevel] || thinkingLevel;
          const thinkStr = `think:${label}`;
          const thinkColored = (thinkingLevel === "high" || thinkingLevel === "xhigh" || thinkingLevel == "medium")
            ? rainbow(thinkStr)
            : theme.fg("dim", thinkStr);

          // Context bar
          const usage = ctx.getContextUsage();
          const pct = usage?.percent ?? 0;
          const filled = Math.max(0, Math.min(10, Math.round(pct / 10)));
          const bar = "#".repeat(filled) + "-".repeat(10 - filled);

          const l1Left =
            theme.fg("accent", ` ${model}`) +
            theme.fg("dim", " · ") +
            thinkColored +
            theme.fg("dim", " · ") +
            theme.fg("warning", "[") +
            theme.fg("success", bar.slice(0, filled)) +
            theme.fg("dim", bar.slice(filled)) +
            theme.fg("warning", "]") +
            theme.fg("dim", " ") +
            theme.fg("accent", `${Math.round(pct)}%`);

          const l1Right =
            theme.fg("success", `↑${fmtTokens(tokIn)}`) +
            theme.fg("dim", " ") +
            theme.fg("accent", `↓${fmtTokens(tokOut)}`) +
            theme.fg("dim", " ") +
            (cost < 0.01
              ? theme.fg("warning", `$${cost.toFixed(4)}`)
              : theme.fg("warning", `$${cost.toFixed(2)}`)) +
            theme.fg("dim", " ");

          const pad = " ".repeat(Math.max(1, width - visibleWidth(l1Left) - visibleWidth(l1Right)));
          const line1 = truncateToWidth(l1Left + pad + l1Right, width);

          const extensionStatuses = footerData.getExtensionStatuses();
          if (extensionStatuses.size === 0) return [line1];

          const statusLine = Array.from(extensionStatuses.entries())
            .sort(([a], [b]) => a.localeCompare(b))
            .map(([, text]) => sanitizeStatusText(text))
            .filter((text) => text.length > 0)
            .join(" ");

          if (!statusLine) return [line1];
          return [line1, truncateToWidth(statusLine, width, theme.fg("dim", "..."))];
        },
      };
    });
  });
}
