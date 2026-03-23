/// <reference types="bun-types" />
import { describe, expect, test } from "bun:test";
import { scanBranchRowCount } from "./experiment-enforcer";

const SAMPLE_TSV = `experiment	val_bpb	memory_gb	status	category	description	run_of	num_steps
001	1.770823	18.5	keep	training	baseline run 1	-	5240
002	1.649805	18.5	keep	training	baseline run 2	-	4752
003	1.746396	18.5	keep	training	baseline run 3	-	5283
004	1.667600	18.5	confirm	chunk	chunk_size 64 + neumann iters sync (run1)	004	3695
004	1.667463	18.5	confirm	chunk	chunk_size 64 + neumann iters sync (run2)	004	3665
004	1.711581	18.5	confirm	chunk	chunk_size 64 + neumann iters sync (run3)	004	3591
004	1.682215	18.5	keep	chunk	marginal keep by 3-run mean	-	3591
005	1.725082	18.5	discard	gate	activation swap to sigmoid gates	-	3667
006	1.721901	18.5	discard	numeric	RMS-normalize K_c before overlap and solves	-	3543
007	1.657283	18.5	confirm	solve	single-iter Neumann solve (run1)	007	4459
007	1.652688	18.5	confirm	solve	single-iter Neumann solve (run2)	007	4395
007	1.651412	18.5	confirm	solve	single-iter Neumann solve (run3)	007	4413
007	1.653794	18.5	keep	solve	marginal keep by 3-run mean	-	4413
008	1.648815	18.5	confirm	arch	per-head output gate after composition (run1)	008	4443
008	1.658205	18.5	confirm	arch	per-head output gate after composition (run2)	008	4446
008	1.658399	18.5	confirm	arch	per-head output gate after composition (run3)	008	4408
008	1.655140	18.5	discard	arch	discard after 3-run mean check	-	4408
009	1.648660	18.5	confirm	state	chunk-boundary smoothing in state update (run1)	009	4587
009	1.651529	18.5	confirm	state	chunk-boundary smoothing in state update (run2)	009	4372
009	1.666451	18.5	confirm	state	chunk-boundary smoothing in state update (run3)	009	4456
009	1.655547	18.5	discard	state	discard after 3-run mean check	-	4456
010	1.747894	18.5	discard	chunk	chunk_size 512 coarse chunks	-	5538`;

function mkCtx(messages: any[]) {
  return {
    sessionManager: {
      getBranch() {
        return messages.map((message) => ({ type: "message", message }));
      },
    },
  };
}

describe("scanBranchRowCount", () => {
  test("counts experiment rows in TSV output", () => {
    const ctx = mkCtx([
      {
        role: "toolResult",
        content: [{ type: "text", text: SAMPLE_TSV }],
      },
    ]);

    expect(scanBranchRowCount(ctx)).toBe(22);
  });

  test("dedups tab vs space rows and ignores printf command text", () => {
    const rowWithTabs = "001\t1.770823\t18.5\tkeep\ttraining\tbaseline run 1\t-\t5240";
    const rowWithSpaces = "001 1.770823 18.5 keep training baseline run 1 - 5240";

    const ctx = mkCtx([
      {
        role: "toolResult",
        content: [{ type: "text", text: rowWithTabs }],
      },
      {
        role: "assistant",
        content: [
          { type: "text", text: rowWithSpaces },
          { type: "tool_use", input: { command: `printf "${rowWithTabs}\\n"` } },
        ],
      },
    ]);

    expect(scanBranchRowCount(ctx)).toBe(1);
  });
});

test("ignores training step logs", () => {
  const ctx = mkCtx([
    {
      role: "toolResult",
      content: [{ type: "text", text: [
        "step 02341 (42.8%) | loss: 5.282514 | lrm: 1.00 | dt: 53ms | tok/sec: 153,379",
        "step 00007 (0.1%) | loss: 8.123456 | dt: 51ms",
        "001\t1.770823\t18.5\tkeep\ttraining\tbaseline run 1\t-\t5240",
      ].join("\n") }],
    },
  ]);
  expect(scanBranchRowCount(ctx)).toBe(1);
});
