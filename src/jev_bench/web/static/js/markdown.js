/**
 * Safe Markdown parser for analysis reports: the subset LLMs write (ATX headings, paragraphs, nested
 * bullet and numbered lists, pipe tables, fenced code, block quotes, rules; inline code, bold, italic,
 * links and e001… email refs) into a plain tree that markdown-render.js turns into DOM nodes. It never
 * produces HTML: raw HTML in the text stays literal text.
 * Exports: parseInline, parseMarkdown.
 */

const INLINE = [
  ["code", /`([^`]+)`/],
  ["strong", /\*\*(?=\S)([\s\S]*?\S)\*\*|__(?=\S)([\s\S]*?\S)__/],
  ["em", /\*(?=[^\s*])([^*]*?[^\s*])\*|(?<!\w)_(?=\S)([^_]*?\S)_(?!\w)/],
  ["link", /\[([^\]]+)\]\(([^)\s]+)\)/],
  ["ref", /\be\d{3,}\b/],
];
const HEADING = /^(#{1,6})\s+(.*?)\s*#*\s*$/;
const FENCE = /^\s*(`{3,}|~{3,})\s*([\w+#.-]*)\s*$/;
const RULE = /^\s*([-*_])(\s*\1){2,}\s*$/;
const LIST_ITEM = /^(\s*)([-*+]|\d{1,9}[.)])\s+(.*)$/;
const QUOTE = /^\s*>\s?/;
const SEPARATOR = /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$/;

export function parseInline(text) {
  const nodes = [];
  let rest = String(text ?? "");
  while (rest) {
    const found = earliest(rest);
    if (!found) break;
    pushText(nodes, rest.slice(0, found.match.index));
    nodes.push(inlineNode(found));
    rest = rest.slice(found.match.index + found.match[0].length);
  }
  pushText(nodes, rest);
  return nodes;
}

function earliest(text) {
  let best = null;
  for (const [type, pattern] of INLINE) {
    const match = pattern.exec(text);
    if (match && (best === null || match.index < best.match.index)) best = { type, match };
  }
  return best;
}

function inlineNode({ type, match }) {
  if (type === "code") return { type, text: match[1] };
  if (type === "ref") return { type, ref: match[0] };
  if (type === "link") return { type, href: match[2], children: parseInline(match[1]) };
  return { type, children: parseInline(match[1] ?? match[2]) };
}

function pushText(nodes, text) {
  if (text) nodes.push({ type: "text", text });
}

export function parseMarkdown(text) {
  const lines = text === null || text === undefined ? [] : String(text).replace(/\r\n?/g, "\n").split("\n");
  const blocks = [];
  let index = 0;
  while (index < lines.length) {
    const [block, next] = readBlock(lines, index);
    if (block) blocks.push(block);
    index = next;
  }
  return blocks;
}

const READERS = [readBlank, readFence, readHeading, readRule, readTable, readQuote, readList, readParagraph];

function readBlock(lines, index) {
  for (const reader of READERS) {
    const result = reader(lines, index);
    if (result) return result;
  }
  return [null, index + 1];
}

function readBlank(lines, index) {
  return lines[index].trim() ? null : [null, index + 1];
}

function readFence(lines, index) {
  const open = FENCE.exec(lines[index]);
  if (!open) return null;
  const close = new RegExp(`^\\s*${open[1][0]}{${open[1].length},}\\s*$`);
  let end = index + 1;
  while (end < lines.length && !close.test(lines[end])) end += 1;
  return [{ type: "code", lang: open[2], text: lines.slice(index + 1, end).join("\n") }, end + 1];
}

function readHeading(lines, index) {
  const match = HEADING.exec(lines[index]);
  return match ? [{ type: "heading", level: match[1].length, children: parseInline(match[2]) }, index + 1] : null;
}

function readRule(lines, index) {
  return RULE.test(lines[index]) ? [{ type: "rule" }, index + 1] : null;
}

const splitRow = (line) => line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((cell) => cell.trim());
const isTableStart = (lines, index) => lines[index].includes("|") && index + 1 < lines.length && lines[index + 1].includes("|") && SEPARATOR.test(lines[index + 1]);

function alignment(cell) {
  const left = cell.startsWith(":");
  const right = cell.endsWith(":");
  return left && right ? "center" : right ? "right" : left ? "left" : null;
}

function readTable(lines, index) {
  if (!isTableStart(lines, index)) return null;
  const header = splitRow(lines[index]);
  const width = header.length;
  const align = splitRow(lines[index + 1]).slice(0, width).map(alignment);
  let end = index + 2;
  const rows = [];
  while (end < lines.length && lines[end].trim() && lines[end].includes("|")) {
    const cells = splitRow(lines[end]);
    rows.push(Array.from({ length: width }, (_, column) => parseInline(cells[column] ?? "")));
    end += 1;
  }
  return [{ type: "table", align, header: header.map(parseInline), rows }, end];
}

function readQuote(lines, index) {
  if (!QUOTE.test(lines[index])) return null;
  let end = index;
  while (end < lines.length && QUOTE.test(lines[end])) end += 1;
  const inner = lines.slice(index, end).map((line) => line.replace(QUOTE, ""));
  return [{ type: "quote", children: parseMarkdown(inner.join("\n")) }, end];
}

function readList(lines, index) {
  const first = LIST_ITEM.exec(lines[index]);
  return first ? parseList(lines, index, first[1].length) : null;
}

function nextFilled(lines, index) {
  let next = index;
  while (next < lines.length && !lines[next].trim()) next += 1;
  return next;
}

function parseList(lines, index, indent) {
  const marker = LIST_ITEM.exec(lines[index])[2];
  const ordered = /\d/.test(marker);
  const items = [];
  let at = index;
  while (at < lines.length) {
    const step = listStep(lines, at, indent, items);
    if (step === null) break;
    at = step;
  }
  const finished = items.map((item) => ({ children: parseInline(item.raw), sublists: item.sublists }));
  return [{ type: "list", ordered, start: ordered ? Number.parseInt(marker, 10) : null, items: finished }, at];
}

function listStep(lines, at, indent, items) {
  const line = lines[at];
  const match = LIST_ITEM.exec(line);
  if (match && match[1].length === indent) {
    items.push({ raw: match[3], sublists: [] });
    return at + 1;
  }
  if (!items.length || (match && match[1].length < indent)) return null;
  if (match) {
    const [sublist, next] = parseList(lines, at, match[1].length);
    items.at(-1).sublists.push(sublist);
    return next;
  }
  if (!line.trim()) return continuesAfterBlank(lines, at, indent);
  if (!/^\s/.test(line)) return null;
  items.at(-1).raw += ` ${line.trim()}`;
  return at + 1;
}

function continuesAfterBlank(lines, at, indent) {
  const next = nextFilled(lines, at);
  const match = next < lines.length ? LIST_ITEM.exec(lines[next]) : null;
  return match && match[1].length >= indent ? next : null;
}

const startsBlock = (lines, index) =>
  !lines[index].trim() || FENCE.test(lines[index]) || HEADING.test(lines[index]) || RULE.test(lines[index]) || LIST_ITEM.test(lines[index]) || QUOTE.test(lines[index]) || isTableStart(lines, index);

function readParagraph(lines, index) {
  let end = index + 1;
  while (end < lines.length && !startsBlock(lines, end)) end += 1;
  const text = lines.slice(index, end).map((line) => line.trim()).join(" ");
  return [{ type: "paragraph", children: parseInline(text) }, end];
}
