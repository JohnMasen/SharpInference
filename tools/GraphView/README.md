# GraphView

An offline, dependency-free SharpInference XML graph viewer. Open
[graph-viewer.html](graph-viewer.html) directly in a browser; no server, build,
package installation or network connection is required. Files stay in the browser
and are never uploaded.

Use **中文** / **English** to switch the interface language. Switching preserves
the current view, history, display options, zoom, selection and canvas position.
XML-authored names, descriptions and formulas stay in their original language.
Each page load starts in Chinese.

## Opening a graph package

Choose **Open XML file** or drag an XML file onto the canvas.
The source contract is `src/SharpInference.Graphs/GraphPackageXml.cs`:

```xml
<GraphPackage version="1" stage="business" main="main">
  <Metadata><Entry key="description" value="Editable example" /></Metadata>
  <Graphs>
    <Graph id="main" name="Example">
      <Resources>
        <Resource id="input" category="resource" graphIO="In">
          <Tensor elementType="Float32" layout="dense"><Dimension value="N" /></Tensor>
        </Resource>
        <Resource id="output" category="resource" graphIO="Out">
          <Tensor elementType="Float32" layout="dense"><Dimension value="N" /></Tensor>
        </Resource>
      </Resources>
      <Actions>
        <Operation id="copy" category="operation" operation="core.copy" version="1">
          <Attributes />
          <Metadata>
            <Entry key="architecture.path" value="model/copy" />
            <Entry key="architecture.formula" value="y = x" />
          </Metadata>
          <Binding property="input" targetKind="Graph" targetProperty="input" access="Read" />
          <Binding property="output" targetKind="Graph" targetProperty="output" access="Write" />
        </Operation>
      </Actions>
      <GraphFlow />
    </Graph>
  </Graphs>
</GraphPackage>
```

- The heading explicitly identifies **Business**, **Logical**, or **Execution**
  from `stage="business|logical|execution"`. Execution packages can declare
  `<Target backend="..." />`; this is an execution extension, not a stage.
- `Graphs/Graph` is a flat collection. `main` is explicit: the viewer never infers
  a main graph, even with exactly one graph. Missing main/graph/resource/call/Then
  references are reported as editable-source issues. Auxiliary graphs remain
  selectable; unresolved bindings retain diagnostic resources, and actions are
  not silently removed. Cyclic Then constraints are diagnosed rather than
  deleting their actions.
- `<Call id="..." target="helper">` refers to another flat graph. Multiple call
  sites remain separate nodes and navigate to the same reusable definition.
  Uncalled auxiliary graphs are also available in the scope selector.
- Resource `graphIO` (`Internal`, `In`, `Out`, `InOut`), `scope`
  (`Call`, `Session`, `Global`), optional `usage` (`Temporary`, `Weight`,
  `Constant`), independent `isState`, `permission`, `bindingKey`, and `category`
  are retained separately. Defaults are Internal, Call, unspecified usage,
  false state and ReadWrite permission. A state is not inferred from Session
  scope, nor a weight from Global scope. Weight visibility follows **usage**
  only. Unbound resources are still visible. Execution `Storage` and Tensor
  `Dimension value` declarations are retained.
- A Binding belongs directly to its action. `property` is the action property;
  `targetKind="Graph"` with omitted `target` refers to the current graph,
  while an explicit target names a flat graph. `targetKind="Call"` names a
  direct call instance. `targetProperty` names the target resource.
  `ReadWrite` remains **one Binding** in details and projects to input and
  output ports; it is not split into fabricated source bindings or edge IDs.
  Instance resources are qualified by target kind and call ID in the display.
- `GraphFlow/Then fromGraph/from/toGraph/to` expresses a **partial order**,
  not a sequential schedule. Local Then constraints are topologically ordered
  for the data-access display; unrelated actions remain independent. Qualified
  cross-graph Then endpoints are displayed as navigable boundary nodes.
  Logic-priority layout hides pure Then edges, without changing source bindings.
  Resource edges are a viewing projection of accesses, not a compiler or a
  proof of execution order for an under-constrained editable source.
- Operations retain `operation` + `version`, Attributes, optional Precision and
  execution Implementation/Dispatch. Execution action `Sources/Source`
  (`document`, `graph`, `node`) is retained in the original XML details.
  Package, graph and action `Metadata/Entry key/value` is preserved.

Superseded graph formats are not supported. Re-export using GraphPackage;
there is no fallback parser for older Graph documents or standalone architecture
trees.

## Architecture metadata

`Metadata/Entry` values `architecture.path`, `architecture.name`,
`architecture.role`, `architecture.description`, and `architecture.formula`
create viewer-only display containers and documentation. Each slash-separated
path prefix becomes a container. A common leading segment represents the graph
root; unannotated actions remain at the root. Empty path segments are rejected.
Generated container IDs avoid collisions with action IDs.

`step`, `defaultView`, `defaultCollapsed`, and `repeatGroup` (also accepted with
an `architecture.` prefix) retain their display behavior. For example,
`architecture.repeatGroup=true` marks a toggleable repeated group and
`architecture.defaultCollapsed=false` transparently expands a container.
When the computation context has `step=token`, optional purple dashed state
edges connect the last state write to the next token's first read; `isState`
controls this independently of scope/usage. These edges do not rank this step.

Interleaved path groups retain access order; aggregate display cycles share a
layout row without dropping edges. Logical graph `compileBoundary`
(`Auto`, `Atomic`, `Transparent`) is preserved as compilation intent, not a
display hierarchy or an implied synchronization edge.

## Standalone execution plans

Production `ExecutionPlan schema="sharp-inference" version="1" kind="il|cpu|gpu"`
remains a separate consumer. An IL plan embeds a **GraphPackage**.
CPU/GPU target plans retain EntryPoints, orchestration Calls, Functions, Kernels,
Dispatches and Barriers. Double-click a call to enter its definition.
Details retain original plan XML, thread/grid configuration, formal/actual
bindings and offsets. ResourcePlan summaries show local slots, mappings, static
views and payload capacity: these are declarations, not measurements or allocated
memory. Resource kinds propagate through actual call bindings, preserving weight
visibility for normalized formal slot names.

Earlier experimental `ExecutionPlan schema="rwkv7-experiment" version="1"`
IL/CPU/GPU plans also remain supported. They open at the entry's Region hierarchy;
enter a layer definition, then a Function/Kernel to inspect internal operations.
GPU Kernels are purple; barriers are amber. Dashed invocation-order edges are
separate from resource edges; Unit declarations appear only at their Invoke sites.
Multiple entry points remain in the scope selector. Details retain fingerprints,
contracts, budgets, resource mappings and partition examples.

Compact target tensors support comma-separated `shape="64,64"` and earlier
space-separated shapes. Static `viewOf` resources with `byteOffset="0"`,
`access="Read"` and `binding="frame-static"` follow base producers/weight
visibility and are excluded from slot budgets. Unsupported plan schemas and
missing/duplicate unit references are rejected. This tool does not compile,
allocate, optimize, measure performance or execute VM code.

## Controls and large graphs

- **Execution-order first** (default) ranks data and ordering edges top-to-bottom.
  **Logic first** hides pure order edges and uses neighbor-center sweeps to reduce
  crossings. Fork/join branches can share rows. Switching retains zoom/selection
  position where possible, with separate dragged positions for each mode.
- Click a node for details; double-click a module/call to enter it. **View all
  operations** bypasses intermediate architecture levels. Click an edge to
  highlight its line, arrow and both endpoints.
- Drag empty canvas to pan, wheel to zoom, drag nodes to reposition. Ports are
  labeled blocks on the top/bottom edges. Routes attach to ports and update after
  dragging, with outside lanes for long connections. Automatic layout leaves
  96 pixels between nodes in a row and 140 pixels between node bodies vertically.
- **Fit to canvas** shows the entire graph; **Readable view (F)** restores 100%
  zoom centered on the selection. F is ignored in editable fields and does not
  intercept Ctrl+F. Search + **Next** locates nodes.
- **Alt+Left / Alt+Right** navigates history, restoring canvas, options and
  selection. Opening a file resets history. **Reset layout** affects only the
  current mode/options. Positions are session-local and never modify XML.
- Weights/constants, repeated groups and cross-token state have separate toggles.
  Option changes keep the selected node (or nearest surviving node) in place.

Loading reports phase-local progress for parsing, conversion, view registration
and rendering. Conversion checkpoints every 64 actions yield to input and allow
another file to supersede loading. Resource/access indexes are built per graph.
XML parsing, layout and final browser paint remain synchronous; initial file
display renders synchronously. This is not viewport-only rendering.

Switching into a view with more than 500 operations/calls reports layout,
edge/node creation and routing progress. It checks an approximately 12 ms work
budget every 64 objects/edges and yields on budget exhaustion or phase changes.
SVG objects are constructed/routed off-DOM, then attached together. Navigation
and display controls are disabled during switching; a new file cancels it.
Small views remain synchronous. This bounds batches, not one complex path search.

## Offline regression tests

With Node.js installed, run from the repository root:

```powershell
node --test .\tools\GraphView\graph-viewer.tests.js
```

No npm packages or browser are required. Tests execute the viewer's actual parser,
registration, display-model and cancellation functions through a small XML/DOM
adapter. Coverage includes the three stages, explicit/missing main, unresolved
editable references, flat/repeated-call navigation, Graph/Call binding resolution,
ReadWrite ports, independent resource semantics/weight filtering, Then partial
orders, metadata/Source/Implementation retention, architecture grouping,
production/experimental plans, 64-action progress, superseded loads and render
generation cancellation. SVG appearance, actual pointer interactions and browser
paint timing still require manual browser checks.
