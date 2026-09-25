// Run with `node --test tests/js/`. Exercises the pure parser of src/jev_bench/web/static/js/markdown.js.
import assert from "node:assert/strict";
import test from "node:test";

const { parseInline, parseMarkdown } = await import("../../src/jev_bench/web/static/js/markdown.js");

const text = (value) => ({ type: "text", text: value });

const INLINE_CASES = [
  ["plain text", "just words", [text("just words")]],
  ["code is literal", "run `**x**` now", [text("run "), { type: "code", text: "**x**" }, text(" now")]],
  ["bold", "a **b** c", [text("a "), { type: "strong", children: [text("b")] }, text(" c")]],
  ["underscore bold", "__b__", [{ type: "strong", children: [text("b")] }]],
  ["italic", "*i* and _j_", [{ type: "em", children: [text("i")] }, text(" and "), { type: "em", children: [text("j")] }]],
  ["snake_case is not italic", "use needs_reply_now here", [text("use needs_reply_now here")]],
  ["bold wins over italic at the same place", "**x** *y*", [{ type: "strong", children: [text("x")] }, text(" "), { type: "em", children: [text("y")] }]],
  ["nested emphasis", "**a *b* c**", [{ type: "strong", children: [text("a "), { type: "em", children: [text("b")] }, text(" c")] }]],
  ["link", "see [docs](https://x.test/a)", [text("see "), { type: "link", href: "https://x.test/a", children: [text("docs")] }]],
  ["email refs", "e017 and e1001, not e01 or the001", [{ type: "ref", ref: "e017" }, text(" and "), { type: "ref", ref: "e1001" }, text(", not e01 or the001")]],
  ["html stays text", "<b>x</b> & <script>", [text("<b>x</b> & <script>")]],
  ["unclosed markers stay text", "a ** b * c ` d", [text("a ** b * c ` d")]],
  ["empty", "", []],
];

for (const [name, input, expected] of INLINE_CASES) {
  test(`parseInline: ${name}`, () => assert.deepEqual(parseInline(input), expected));
}

const BLOCK_CASES = [
  ["headings", "# One\n### Three ###", [{ type: "heading", level: 1, children: [text("One")] }, { type: "heading", level: 3, children: [text("Three")] }]],
  ["paragraph lines join", "a\nb\n\nc", [{ type: "paragraph", children: [text("a b")] }, { type: "paragraph", children: [text("c")] }]],
  ["rule", "a\n\n---\n\nb", [{ type: "paragraph", children: [text("a")] }, { type: "rule" }, { type: "paragraph", children: [text("b")] }]],
  ["fenced code keeps text", "```json\n{\"a\": 1}\n  **x**\n```", [{ type: "code", lang: "json", text: "{\"a\": 1}\n  **x**" }]],
  ["unclosed fence runs to the end", "```\ncode", [{ type: "code", lang: "", text: "code" }]],
  ["quote", "> **Note**\n> more", [{ type: "quote", children: [{ type: "paragraph", children: [{ type: "strong", children: [text("Note")] }, text(" more")] }] }]],
  [
    "bullet list with nesting and continuation",
    "- one\n  still one\n  - inner\n- two",
    [
      {
        type: "list",
        ordered: false,
        start: null,
        items: [
          { children: [text("one still one")], sublists: [{ type: "list", ordered: false, start: null, items: [{ children: [text("inner")], sublists: [] }] }] },
          { children: [text("two")], sublists: [] },
        ],
      },
    ],
  ],
  [
    "loose numbered list keeps its start",
    "3. a\n\n4. b\n\nafter",
    [
      { type: "list", ordered: true, start: 3, items: [{ children: [text("a")], sublists: [] }, { children: [text("b")], sublists: [] }] },
      { type: "paragraph", children: [text("after")] },
    ],
  ],
  [
    "table with alignment and ragged rows",
    "| Run | κ |\n|:---|---:|\n| R1 | 0.8 | extra |\n| R2 |",
    [
      {
        type: "table",
        align: ["left", "right"],
        header: [[text("Run")], [text("κ")]],
        rows: [
          [[text("R1")], [text("0.8")]],
          [[text("R2")], []],
        ],
      },
    ],
  ],
  ["a pipe without a separator row is a paragraph", "a | b", [{ type: "paragraph", children: [text("a | b")] }]],
  ["windows newlines", "# T\r\n\r\nx", [{ type: "heading", level: 1, children: [text("T")] }, { type: "paragraph", children: [text("x")] }]],
  ["empty and blank", "\n  \n", []],
];

for (const [name, input, expected] of BLOCK_CASES) {
  test(`parseMarkdown: ${name}`, () => assert.deepEqual(parseMarkdown(input), expected));
}

test("parseMarkdown: a paragraph stops at the next block", () => {
  const types = parseMarkdown("text\n# H\ntext\n- item\ntext\n```\nc\n```").map((block) => block.type);
  assert.deepEqual(types, ["paragraph", "heading", "paragraph", "list", "paragraph", "code"]);
});

test("parseMarkdown: tolerates null and non-strings", () => {
  assert.deepEqual(parseMarkdown(null), []);
  assert.deepEqual(parseMarkdown(42), [{ type: "paragraph", children: [text("42")] }]);
});
