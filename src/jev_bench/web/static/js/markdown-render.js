/**
 * Renders parsed Markdown (markdown.js) into h() nodes: headings two levels below the page's own, lists,
 * bordered tables with column alignment, code blocks, quotes and rules. Link URLs go through h()'s scheme
 * check, and an e001… ref becomes a link when `refHref(ref)` returns one.
 * Exports: renderMarkdown.
 */
import { h } from "./dom.js";
import { parseMarkdown } from "./markdown.js";

export function renderMarkdown(text, { refHref = () => null } = {}) {
  const context = { refHref };
  context.inline = (nodes) => nodes.map((node) => INLINE_RENDERERS[node.type](node, context));
  context.block = (block) => BLOCK_RENDERERS[block.type](block, context);
  return parseMarkdown(text).map(context.block);
}

const ALIGN_CLASS = { left: "text-start", center: "text-center", right: "text-end" };
const listStart = (block) => (block.start !== null && block.start !== 1 ? block.start : null);

const BLOCK_RENDERERS = {
  heading: (block, { inline }) => h(`h${Math.min(6, block.level + 2)}`, { class: "md-heading" }, inline(block.children)),
  paragraph: (block, { inline }) => h("p", {}, inline(block.children)),
  list: (block, context) => h(block.ordered ? "ol" : "ul", { start: listStart(block) }, block.items.map((item) => h("li", {}, context.inline(item.children), item.sublists.map(context.block)))),
  table: renderTable,
  code: (block) => h("pre", { class: "md-code" }, h("code", {}, block.text)),
  quote: (block, { block: render }) => h("blockquote", { class: "md-quote" }, block.children.map(render)),
  rule: () => h("hr"),
};

function renderTable(block, { inline }) {
  const cell = (tag, nodes, column) => h(tag, { class: ALIGN_CLASS[block.align[column]] ?? null }, inline(nodes));
  const head = h("thead", {}, h("tr", {}, block.header.map((nodes, column) => cell("th", nodes, column))));
  const body = h("tbody", {}, block.rows.map((row) => h("tr", {}, row.map((nodes, column) => cell("td", nodes, column)))));
  return h("div", { class: "table-responsive" }, h("table", { class: "table table-sm table-bordered md-table" }, head, body));
}

const INLINE_RENDERERS = {
  text: (node) => node.text,
  code: (node) => h("code", {}, node.text),
  strong: (node, { inline }) => h("strong", {}, inline(node.children)),
  em: (node, { inline }) => h("em", {}, inline(node.children)),
  link: (node, { inline }) => h("a", { href: node.href, target: "_blank", rel: "noopener noreferrer" }, inline(node.children)),
  ref: (node, { refHref }) => refLink(node.ref, refHref(node.ref)),
};

function refLink(ref, href) {
  return href ? h("a", { href, class: "md-ref", title: "Open this email in the Explorer" }, ref) : h("span", { class: "md-ref" }, ref);
}
