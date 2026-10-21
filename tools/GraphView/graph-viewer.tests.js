"use strict";

// A small XML/element adapter keeps model tests offline without a browser or npm.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const html = fs.readFileSync(path.join(__dirname, "graph-viewer.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

class Element {
  constructor(name, attributes = {}) {
    this.localName = name;
    this.values = new Map(Object.entries(attributes));
    this.children = [];
    this.style = {};
    this.dataset = {};
    this.listeners = new Map();
    this.classes = new Set();
    this.classList = {
      add: name => this.classes.add(name), remove: name => this.classes.delete(name),
      contains: name => this.classes.has(name),
      toggle: (name, value) => value ? this.classes.add(name) : this.classes.delete(name),
    };
    this.disabled = false;
    this.hidden = false;
    this.checked = false;
    this.value = "";
    this._text = "";
  }
  get attributes() { return [...this.values].map(([name, value]) => ({ name, value })); }
  get options() { return this.children; }
  get selectedOptions() { return [this.children.find(item => item.value === this.value) || this.children[0]]; }
  get firstElementChild() { return this.children[0]; }
  get textContent() { return this._text + this.children.map(item => item.textContent).join(""); }
  set textContent(value) { this._text = String(value); this.children = []; }
  getAttribute(name) { return this.values.get(name) ?? null; }
  hasAttribute(name) { return this.values.has(name); }
  setAttribute(name, value) { this.values.set(name, String(value)); }
  removeAttribute(name) { this.values.delete(name); }
  append(...items) { for (const item of items) { item.parentElement = this; this.children.push(item); } }
  prepend(...items) { this.children.unshift(...items); }
  add(item) { this.append(item); }
  replaceChildren(...items) { this.children = []; this._text = ""; this.append(...items); }
  replaceChild(next, previous) { this.children[this.children.indexOf(previous)] = next; }
  getElementsByTagName(tag) {
    return this.children.flatMap(item => [...(item.localName === tag ? [item] : []), ...item.getElementsByTagName(tag)]);
  }
  querySelector(tag) { return this.getElementsByTagName(tag)[0] || null; }
  querySelectorAll(tag) { return this.getElementsByTagName(tag); }
  addEventListener(name, listener) { this.listeners.set(name, listener); }
  getBoundingClientRect() { return { width: 1200, height: 800, left: 0, top: 0 }; }
}

const decode = text => text.replace(/&quot;/g, '"').replace(/&lt;/g, "<")
  .replace(/&gt;/g, ">").replace(/&amp;/g, "&");
class XmlDocument {
  constructor(root) { this.documentElement = new Element(root); }
  createElement(name) { return new Element(name); }
  importNode(element) {
    const copy = new Element(element.localName, Object.fromEntries(element.values));
    copy._text = element._text;
    copy.append(...element.children.map(item => this.importNode(item)));
    return copy;
  }
  querySelector(name) {
    return this.documentElement.localName === name ? this.documentElement : this.documentElement.querySelector(name);
  }
}
class DOMParser {
  parseFromString(xml) {
    const stack = [], roots = [];
    for (const token of xml.match(/<[^>]*>|[^<]+/g) || []) {
      if (token.startsWith("<?") || token.startsWith("<!--")) continue;
      if (token.startsWith("</")) { stack.pop(); continue; }
      if (token.startsWith("<")) {
        const name = token.match(/^<([\w.-]+)/)[1], attributes = {};
        for (const match of token.matchAll(/([\w.-]+)="([^"]*)"/g)) attributes[match[1]] = decode(match[2]);
        const element = new Element(name, attributes);
        if (stack.length) stack.at(-1).append(element); else roots.push(element);
        if (!token.endsWith("/>")) stack.push(element);
      } else if (stack.length) stack.at(-1)._text += decode(token);
    }
    const document = new XmlDocument("unused");
    document.documentElement = roots[0];
    return document;
  }
}
class XMLSerializer {
  serializeToString(element) {
    return `<${element.localName}${element.attributes.map(a => ` ${a.name}="${a.value.replace(/"/g, "&quot;")}"`).join("")}>` +
      element._text + element.children.map(item => this.serializeToString(item)).join("") + `</${element.localName}>`;
  }
}

function viewer() {
  const controls = new Map();
  const $ = id => {
    if (!controls.has(id)) controls.set(id, new Element(id));
    return controls.get(id);
  };
  $("layout-priority").value = "execution";
  $("empty").append(new Element("p"));
  const document = {
    getElementById: $, createElement: name => new Element(name),
    createElementNS: (_, name) => new Element(name),
    implementation: { createDocument: (_, name) => new XmlDocument(name) },
    querySelectorAll: () => [],
  };
  const context = vm.createContext({ document, DOMParser, XMLSerializer, Element,
    Option: function (label, value) { const option = new Element("option"); option.textContent = label; option.value = value; return option; },
    setTimeout, clearTimeout, performance, DOMException, console });
  // Expose real parser/model functions, without attaching page input handlers.
  const prefix = script.slice(0, script.indexOf('  $("file").addEventListener'));
  vm.runInContext(prefix + `
    globalThis.api = { state, parseXml, registerRegionViews, buildGraph, graphType, tensor,
      overview, select, setLanguage, openFile, renderProgressively, setRenderBusy,
      renderSteps, captureView,
      stubRender: value => { render = value; },
      stubSteps: value => { renderSteps = value; } };
  })();`, context);
  return { ...context.api, $, context };
}

const tensor = '<Tensor elementType="Float32" layout="dense"><Dimension value="N"/><Dimension value="64"/></Tensor>';
const resource = (id, attributes = "") => `<Resource id="${id}" category="custom.resource" ${attributes}>${tensor}</Resource>`;
const metadata = (path = "model/left") => `<Metadata>
  <Entry key="architecture.path" value="${path}"/><Entry key="architecture.name" value="Module"/>
  <Entry key="architecture.description" value="Description"/><Entry key="architecture.formula" value="y = x"/>
  <Entry key="architecture.repeatGroup" value="true"/></Metadata>`;
const binding = (property, targetProperty, access = "Read", extra = "") =>
  `<Binding property="${property}" targetKind="Graph" targetProperty="${targetProperty}" access="${access}" ${extra}/>`;
const operation = (id, bindings = "", extra = "") => `<Operation id="${id}" category="custom.operation" operation="core.copy" version="1">
  <Attributes/>${extra}${bindings}</Operation>`;
const graph = (id, actions = "", resources = "", flows = "", extra = "") => `<Graph id="${id}" ${extra}>
  <Resources>${resources}</Resources><Actions>${actions}</Actions><GraphFlow>${flows}</GraphFlow></Graph>`;
const packageXml = (stage = "business", options = {}) => `<GraphPackage version="${options.version || 1}" stage="${stage}" ${options.noMain ? "" : `main="${options.main || "main"}"`}>
  ${stage === "execution" ? '<Target backend="D3D12"/>' : ""}
  <Metadata><Entry key="note" value="package metadata"/></Metadata><Graphs>${options.graphs || graph("main",
    operation("a", binding("input", "input") + binding("output", "x", "Write"), metadata()) +
    operation("b", binding("input", "x") + binding("output", "output", "Write"), metadata("model/right")),
    resource("input", 'graphIO="In"') + resource("x") + resource("output", 'graphIO="Out"'),
    '<Then fromGraph="main" from="a" toGraph="main" to="b"/>')}</Graphs></GraphPackage>`;

async function load(xml, checkpoint = async () => {}) {
  const v = viewer(), parsed = await v.parseXml(xml, checkpoint);
  Object.assign(v.state, parsed, { rootResources: parsed.resources });
  v.$("scope").add({ value: "root", textContent: "Main" });
  v.$("scope").value = "root";
  v.registerRegionViews();
  return v;
}

for (const stage of ["business", "logical", "execution"]) {
  test(`${stage}: package stage, Tensor, metadata, custom categories and real bindings`, async () => {
    const v = await load(packageXml(stage));
    assert.match(v.graphType(), stage === "business" ? /业务/ : stage === "logical" ? /逻辑/ : /执行/);
    const graph = v.buildGraph();
    assert.equal(graph.nodes.filter(item => !item.external).length, 2);
    assert.ok(graph.edges.some(edge => edge.bindings.some(item => item.access === "Order")));
    const a = graph.nodes.find(item => item.id === "a");
    assert.equal(a.element.getAttribute("category"), "custom.operation");
    assert.equal(a.metadata.querySelector("Formula").textContent, "y = x");
    assert.equal(v.state.package.originals.get(a.element).getAttribute("version"), "1");
    assert.match(v.tensor(v.state.resources.get("input")), /N.*64/);
    v.state.contextId = "root";
    v.overview();
    assert.match(v.$("details").textContent, /package metadata/);
    if (stage === "execution") assert.match(v.$("details").textContent, /D3D12/);
  });
}

test("package version/stage and obsolete graph roots are rejected", async () => {
  for (const xml of [packageXml("business", { version: "2" }), packageXml("other"), '<Graph/>'])
    await assert.rejects(viewer().parseXml(xml, async () => {}));
});

test("missing main is diagnosed without inferring a unique root; all graphs remain selectable", async () => {
  const v = await load(packageXml("business", { noMain: true }));
  assert.equal(v.state.contexts.get("root").nodes.length, 0);
  assert.ok(v.state.contexts.has("@graph:main"));
  assert.match(v.state.package.issues.join(), /main is missing/);
  v.$("scope").value = "@graph:main";
  assert.equal(v.buildGraph().nodes.filter(item => !item.external).length, 2);
});

test("unknown main, unresolved calls and bindings retain actions and diagnostic placeholders", async () => {
  const xml = packageXml("business", { main: "absent", graphs: graph("editable",
    '<Call id="missing-call" target="missing"/>' + operation("pending", binding("state", "missing", "ReadWrite"))) });
  const v = await load(xml);
  v.$("scope").value = "@graph:editable";
  const displayed = v.buildGraph();
  assert.equal(displayed.nodes.filter(item => !item.external).length, 2);
  assert.ok(displayed.nodes.some(item => item.kind === "Unresolved"));
  assert.equal(v.state.contexts.get("@graph:editable").nodes.find(item => item.id === "pending").bindings.length, 1);
  assert.equal(v.state.package.issues.length, 3);
});

test("GraphIO, scope, state, permission, optional usage, bindingKey and Storage are independent", async () => {
  const xml = packageXml("execution", { graphs: graph("main", operation("a", binding("state", "s", "ReadWrite")),
    resource("s", 'graphIO="InOut" scope="Global" isState="true" permission="Read" bindingKey="state-key" usage="Weight"')
      .replace("</Resource>", '<Storage kind="LocalBuffer" slot="shared"/></Resource>') + resource("unused")) });
  const v = await load(xml), s = v.state.resources.get("s");
  for (const [name, value] of Object.entries({ graphIO: "InOut", scope: "Global", isState: "true", permission: "Read", usage: "Weight", bindingKey: "state-key" }))
    assert.equal(s.getAttribute(name), value);
  assert.equal(s.querySelector("Storage").getAttribute("slot"), "shared");
  const hidden = v.buildGraph();
  assert.ok(!hidden.nodes.some(item => item.external && item.id === "s"));
  assert.ok(hidden.nodes.some(item => item.id === "unused"));
  const a = hidden.nodes.find(item => item.id === "a");
  assert.equal(a.bindings.length, 1);
  assert.equal(a.ports.filter(item => item.resource === "s").length, 2);
  assert.equal(a.bindings[0].access, "ReadWrite");
  v.select(a);
  assert.equal((v.$("details").textContent.match(/<Binding /g) || []).length, 1);
  assert.match(v.$("details").textContent, /graphIOInOutscopeGlobalisStatetruepermissionReadusageWeight/);
  v.$("weights").checked = true;
  const shown = v.buildGraph();
  assert.ok(shown.nodes.some(item => item.external && item.id === "s"));
  assert.equal(v.state.resources.get("unused").getAttribute("scope"), "Call");
  assert.equal(v.state.resources.get("unused").getAttribute("isState"), "false");
  assert.equal(v.state.resources.get("unused").getAttribute("usage"), null);
});

test("flat auxiliary graphs, repeated calls, Call and explicit Graph bindings resolve independently", async () => {
  const xml = packageXml("logical", { graphs: graph("main",
    '<Call id="one" target="helper"/><Call id="two" target="helper"/>' +
    operation("read", '<Binding property="value" targetKind="Call" target="two" targetProperty="out" access="Read"/>' +
      binding("other", "out", "Read", 'target="helper"')), resource("local")) +
    graph("helper", operation("body", binding("out", "out", "Write")), resource("out", 'graphIO="Out"'), "", 'compileBoundary="Atomic"') +
    graph("unused-helper", operation("independent")) });
  const v = await load(xml), displayed = v.buildGraph();
  assert.ok(v.state.contexts.has("@graph:unused-helper"));
  for (const id of ["one", "two"]) assert.equal(displayed.nodes.find(item => item.id === id).drillTarget, "@graph:helper");
  assert.ok(v.state.resources.has("Call:two/out"));
  assert.ok(v.state.resources.has("Graph:helper/out"));
  v.$("scope").value = "@graph:helper";
  assert.ok(v.buildGraph().nodes.some(item => item.id === "body"));
});

test("Then is a partial order, not XML sequence, and logic priority hides only order edges", async () => {
  const xml = packageXml("business", { graphs: graph("main",
    operation("b", binding("input", "x")) + operation("a", binding("output", "x", "Write")) + operation("independent"),
    resource("x"), '<Then fromGraph="main" from="a" toGraph="main" to="b"/>') });
  const v = await load(xml), execution = v.buildGraph();
  assert.deepEqual(Array.from(v.state.contexts.get("root").nodes, item => item.id), ["a", "independent", "b"]);
  assert.ok(execution.edges.some(edge => edge.from.id === "a" && edge.to.id === "b" && edge.bindings[0].access === "Read"));
  const data = execution.edges.filter(edge => edge.bindings[0].access !== "Order").length;
  v.$("layout-priority").value = "logical";
  const logical = v.buildGraph();
  assert.equal(logical.edges.length, data);
  assert.ok(logical.nodes.find(item => item.id === "independent").rank === 0);
});

test("cross-graph Then retains qualified endpoints and missing references are diagnosed", async () => {
  const v = await load(packageXml("business", { graphs: graph("main", operation("a"), "",
    '<Then fromGraph="main" from="a" toGraph="helper" to="b"/><Then fromGraph="main" from="a" toGraph="missing" to="x"/>') +
    graph("helper", operation("b")) }));
  const displayed = v.buildGraph();
  assert.ok(displayed.edges.some(edge => edge.to.id === "helper/b"));
  assert.match(v.state.package.issues.join(), /missing reference/);
});

test("cyclic Then source is diagnosed but actions are not discarded", async () => {
  const v = await load(packageXml("business", { graphs: graph("main", operation("a") + operation("b"), "",
    '<Then fromGraph="main" from="a" toGraph="main" to="b"/><Then fromGraph="main" from="b" toGraph="main" to="a"/>') }));
  assert.match(v.state.package.issues.join(), /cyclic Then/);
  assert.equal(v.buildGraph().nodes.length, 2);
});

test("execution Sources, Precision and Implementation are retained in original details", async () => {
  const v = await load(packageXml("execution", { graphs: graph("main", operation("a", "", `
    <Precision arithmetic="Float32" accumulator="Float32"/>
    <Implementation collection="123" instruction="copy" target="GPU" arithmetic="Float32" accumulator="Float32">
      <Dispatch threadsX="64" threadsY="1" threadsZ="1" groupsX="2" groupsY="1" groupsZ="1"/>
    </Implementation><Sources><Source document="logical.xml" graph="model" node="original"/></Sources>`)) }));
  const a = v.buildGraph().nodes[0], original = v.state.package.originals.get(a.element);
  assert.equal(original.querySelector("Source").getAttribute("node"), "original");
  assert.equal(a.element.querySelector("Dispatch").getAttribute("threadsX"), "64");
  assert.equal(a.element.querySelector("Precision").getAttribute("arithmetic"), "Float32");
  v.select(a);
  assert.match(v.$("details").textContent, /Sourceslogical.xmlmodel\/original/);
});

test("architecture grouping preserves interleaved bindings, formula, repeat and raw action IDs", async () => {
  const v = await load(packageXml("logical"));
  v.$("collapse").checked = true;
  v.$("scope").value = "@architecture";
  const architecture = v.buildGraph();
  assert.equal(architecture.nodes.filter(item => item.region).length, 2);
  assert.ok(architecture.hasGroups);
  v.$("scope").value = "root";
  assert.ok(v.buildGraph().nodes.some(item => item.id === "a"));
  assert.ok(v.state.views.size > 2);
});

test("invalid architecture paths report a concrete error", async () => {
  await assert.rejects(load(packageXml().replace("model/left", "model//left")), /architecture.path/);
});

test("production IL embeds GraphPackage and retains package stage and backend", async () => {
  const v = await load(`<ExecutionPlan schema="sharp-inference" version="1" kind="il">${packageXml("execution")}</ExecutionPlan>`);
  assert.match(v.graphType(), /IL/);
  assert.equal(v.state.package.root.getAttribute("stage"), "execution");
});

test("production CPU/GPU plans remain independent consumers", async () => {
  for (const kind of ["cpu", "gpu"]) {
    const xml = `<ExecutionPlan schema="sharp-inference" version="1" kind="${kind}">
      <Slots/><Definitions><Definition id="body" kind="Function"><Parameters/><Body/></Definition></Definitions>
      <EntryPoints><Entry name="entry" definition="body"><Arguments/></Entry></EntryPoints></ExecutionPlan>`;
    const v = await load(xml);
    assert.equal(v.state.plan.kind, kind);
  }
});

test("experimental plans remain independent consumers", async () => {
  for (const kind of ["il", "cpu", "gpu"]) {
    const xml = `<ExecutionPlan schema="rwkv7-experiment" version="1" kind="${kind}">
      <Defaults order="document-sequential"/><Identity/><Model/><Resources/><Inputs/><Outputs/><GraphState/>
      <Definitions/><EntryPoints><Entry name="entry"><Region id="entry"/></Entry></EntryPoints></ExecutionPlan>`;
    const v = await load(xml);
    assert.equal(v.state.plan.kind, kind);
  }
});

test("large conversion yields every 64 actions with progress and permits cancellation", async () => {
  const actions = Array.from({ length: 640 }, (_, i) => operation(`bulk.${i}`)).join("");
  const xml = packageXml("execution", { graphs: graph("main", actions) });
  const phases = [], v = await load(xml, async (phase, done, total) => { phases.push([phase, done, total]); });
  assert.equal(v.state.contexts.get("root").nodes.length, 640);
  assert.ok(phases.some(([, done, total]) => done === 64 && total === 640));
  let checkpoints = 0;
  await assert.rejects(viewer().parseXml(xml, async () => {
    if (++checkpoints === 3) throw new DOMException("Superseded", "AbortError");
  }), { name: "AbortError" });
});

test("new file supersedes pending conversion without replacing the newer state", async () => {
  const v = viewer();
  v.stubRender(() => { v.state.viewScope = "root"; return true; });
  const pending = v.openFile({ name: "superseded.xml", text: async () => packageXml("execution", {
    graphs: graph("main", Array.from({ length: 640 }, (_, i) => operation(`bulk.${i}`)).join("")),
  }) });
  const latest = v.openFile({ name: "latest.xml", text: async () => packageXml("business") });
  await Promise.all([pending, latest]);
  assert.equal(v.state.filename, "latest.xml");
  assert.equal(v.state.package.root.getAttribute("stage"), "business");
  assert.ok(v.$("load-progress").hidden);
});

test("batched render restores busy controls and checks generation cancellation", async () => {
  const v = viewer();
  const phases = [];
  v.stubSteps(function* () {
    for (let i = 0; i < 4; i++) { phases.push(i); yield { phase: "createNodes", completed: i, total: 4 }; }
    return true;
  });
  v.state.renderGeneration = 1;
  const result = await v.renderProgressively(1);
  assert.equal(result, true);
  assert.equal(phases.length, 4);
  assert.equal(v.state.renderBusy, null);
  v.state.renderGeneration = 2;
  await v.renderProgressively(1);
  assert.ok(phases.length <= 5);
});

test("viewer script compiles and retains bilingual/offline/navigation/drag controls", () => {
  new vm.Script(script);
  for (const token of ["setLanguage", "captureView", "travel", "startNodeDrag", "routeEdges", "renderProgressively"])
    assert.ok(script.includes(token));
  assert.ok(!/<script[^>]+src=/.test(html));
});
