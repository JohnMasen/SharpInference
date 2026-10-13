# GraphView

An offline, dependency-free SharpInference XML graph viewer. Open
[graph-viewer.html](graph-viewer.html) directly in a browser; no server, build,
package installation or network connection is required.

Use the **中文** / **English** buttons at the top right to switch the interface
language. Switching preserves the current view, navigation history, display
options, zoom, selection and canvas position. XML-authored names, descriptions
and formulas stay in their original language. Each page load starts in Chinese.

## Opening a graph

Choose **Open XML file** or drag an XML file onto the canvas. The file stays in
the browser and is never uploaded.

File loading shows phase-local progress for XML parsing, graph conversion,
display-model construction, navigation registration, and rendering.
Logical/execution-graph conversion yields to the browser every 64 nodes so
progress and input can be serviced;
opening another file supersedes the previous load. Resource and access/dependency
indexes are built once per definition instead of rescanning the full graph for
every operation. Progress counts represent work items in the current phase,
not a time estimate. XML parsing, graph layout, and final browser layout/paint
remain synchronous; initial file display also renders synchronously.
This is not viewport-only rendering. Switching into a view containing more than
500 operations/calls now also reports layout, edge/node creation, and routing
progress, checking its roughly 12 ms work budget every 64 created objects or
routed edges and yielding when the budget is reached or a new phase starts.
SVG objects are constructed and routed off-DOM, then attached together to avoid
repeated browser layout/paint during each batch. Navigation/display
controls are temporarily disabled during that switch; opening another file
supersedes the in-progress render. Small views remain synchronous. This bounds
batch sizes, not the runtime of one unusually complex path search.

- [Current RWKV6 execution graph](../../artifacts/rwkv6-d3d12-baseline.xml)
- [Current RWKV7 execution graph](../../artifacts/rwkv7-d3d12-baseline.xml)

These local artifacts are optional, git-ignored examples, not tool dependencies.
Use version 2 `LogicalGraphDefinition` / `ExecutionGraphProgram` exports for the
current architecture. The execution-graph examples above include
`architecture.path` annotations; the viewer reconstructs their module hierarchy
without requiring a separate architecture XML. Old version 1 graph and
execution-plan examples have been removed from the local artifacts directory.
Flat `Nodes` graphs and structured `Region` / `LayerDefinition` / `Call` graphs
remain supported for compatibility. The former standalone `Architecture` tree
is not supported.
The heading and status identify the loaded file as a **Logical graph**, an
**Execution graph**, or an **Execution graph · IL / CPU / GPU**. Files without a
declared graph kind are labeled as unspecified, not guessed to be logical graphs.

## Legacy execution plans (compatibility)

The viewer accepts production `ExecutionPlan` XML
(`schema="sharp-inference"`, `version="1"`, `kind="il|cpu|gpu"`).
These legacy formats are not the current version 2 graph architecture.

Production target views display explicit EntryPoints, orchestration Calls,
Functions, Kernels, Dispatches and Barriers. Double-click a call to enter its
definition. Details retain the original execution-plan XML, thread/grid
configuration, formal/actual bindings and offsets. ResourcePlan summaries show
concrete local slots, resource mappings, static views and payload capacity;
these are declarations, not measurements or allocated memory.
Resource kinds propagate through actual call bindings, so weight visibility
continues to work when reusable definitions use normalized formal slot names.

The viewer also accepts the earlier experimental `ExecutionPlan` XML format
(`schema="rwkv7-experiment"`, `version="1"`) for IL, CPU and GPU plans.

Experimental plans open at the entry's Region hierarchy. Double-click a layer call to inspect
its definition, then a Function or Kernel to inspect the exact internal operations.
GPU Kernels are purple and synchronization barriers are amber. Invocation order
is shown as dashed control edges, separately from data-resource connections.
Unit declarations are not executed twice: only their `Invoke` sites appear in
the containing graph. Multiple entry points are available in the scope selector.

The details panel shows plan stage, target, fingerprints, default contracts,
frame-local slot counts/payload budgets, resource-to-slot mappings, and the
cross-host partition interface example. Kernel details include group/thread
configuration and the original Unit XML; operations retain their original XML.
Compact tensor declarations (`elementType` / `shape`) are supported, with
comma-separated dimensions such as `shape="64,64"`. Earlier space-separated
experimental shapes can still be viewed.
Target resources with `viewOf`, `byteOffset="0"`, `access="Read"`, and
`binding="frame-static"` are static views, not allocation or execution nodes.
The details panel lists these views; data edges track their base producers and
weight visibility follows the base resource. Views are excluded from slot budgets.

This is visualization, not compilation, resource allocation, performance
measurement, or VM execution. Unsupported plan schemas and missing/duplicate
unit references are rejected. The viewer builds an internal display model without
modifying the file or re-optimizing the plan. Existing Graph XML behavior is
unchanged.

## Controls

- **Layout priority** switches between **Execution-order first** (the default)
  and **Logic first**. Execution-order first keeps data and execution-order edges
  in the top-to-bottom ranking. Logic first hides pure execution-order edges,
  ranks nodes by their real input/output relationships, and uses repeated
  neighbor-center sweeps to reduce crossings and align connected rows. Independent
  branches can share a row rather than being serialized by scheduling dependencies.
  Resource producers are still resolved in the original execution order: neither
  mode changes resource bindings, XML, compilation, or execution semantics.
  Cross-token state edges remain optional and do not determine either layout.
  Both modes work in architecture and operator/call views. Switching preserves
  zoom and the selected node's screen position where possible; each mode keeps
  separate dragged positions, and navigation history and language changes retain
  the selected mode. **Reset layout** resets only the current mode/display options.
- Click a node for details; double-click a Region or layer call to explore it.
- Click an edge to highlight its line, arrow and both endpoints.
- **View all operations** bypasses a Region's intermediate semantic levels.
- Drag to pan; scroll to zoom; search and use **Next** to locate matches.
- Drag a node to reposition it; drag empty canvas space to pan. Connected edges
  update while dragging. Nodes keep a small clearance to leave room for routing.
  Automatic layout leaves 96 pixels between nodes in the same row and 140 pixels
  between node bodies in adjacent rows, providing space for ports and routing lanes.
  Positions are retained per view/display-option combination during the current
  file session, including navigation and language changes; they do not modify
  the XML. **Reset layout** restores the current view's automatic positions.
- Long connections use separate outside lanes rather than passing behind
  intermediate nodes. Arrowheads terminate at the actual target, and multiple
  connections use separate attachment points. Routing also updates after a drag.
- Input ports are labeled blocks on the top edge of each node; output ports
  sit on the bottom edge. Connections attach to the corresponding blocks.
  Nodes widen to accommodate their ports. Long names are shortened on the
  blocks; hover to see the full name and resource ID. ReadWrite bindings have
  both input and output blocks. Selecting an edge highlights its two ports.
  Region boundary names use Architecture port metadata when available.
- When weights/constants are hidden, their input ports retain the real binding
  and use a dashed border with a “Hidden weight” / “Hidden constant” label.
  Hover for the source name/resource; enable weights/constants to show the edge.
- **Fit to canvas** shows the whole graph; **F** restores readable 100% zoom.
  F is ignored in editable fields and does not intercept Ctrl+F.
- **Alt+Left / Alt+Right** navigates view history. Opening another file resets it.
- Weights/constants, repeated groups and cross-token state have separate toggles.
  Changing display options preserves zoom and the selected node's screen
  position (or a surviving node nearest the viewport center when nothing is
  selected), rather than jumping back to the top of the graph.

## Logical graph definitions

The logical schema uses reusable `GraphDefinition` values. Version 2 stores
resources, operations and calls as typed nodes, with `Read`, `Write` and
`Dependency` elements as typed edges. Definition and call annotations provide
architecture metadata. For flat graphs, operation and call annotations with an
`architecture.path` attribute reconstruct an expandable display hierarchy:

```xml
<Annotation name="Embedding" role="embedding" description="Token embedding">
  <Attribute name="architecture.path" value="graph/embedding" />
  <Attribute name="architecture.formula" value="x = embedding[token]" />
</Annotation>
```

Each slash-separated path prefix becomes a display container. A common leading
segment represents the graph root; leaf names, roles, descriptions, and formulas
come from annotations, while intermediate containers use their path segments
when no annotation names them. Nodes without paths remain at the graph root.
Annotation attributes `step`, `defaultView`, `defaultCollapsed`, and `repeatGroup`
(also accepted with an `architecture.` prefix) retain their display behavior.
These containers are viewer-only: resource bindings and execution/dependency
order remain unchanged, including when nodes revisit an earlier path.
Interleaved groups may form cycles in the aggregated display graph; their
strongly connected components share a layout row without removing any edges.
This does not permit cyclic execution dependencies.

`GraphCompileBoundary` controls later compilation without changing the
architecture tree:

- `Auto` lets the compiler select a safe boundary.
- `Atomic` preserves whole-call completion semantics.
- `Transparent` preserves the architecture container but does not introduce an
  execution synchronization boundary.

GraphView reads only `LogicalGraphDefinition` logical-graph XML. It presents
each definition call as an expandable architecture component. Other XML and
JSON graph formats are not supported.

## Execution graph programs

GraphView also reads `ExecutionGraphProgram` XML. The program is a validated
collection of reusable execution graphs with one inferred root. Each graph uses
the same typed node/edge structure as the logical graph; double-click a graph
call to inspect its target graph. Graph, call, and operation `Architecture` metadata is displayed
but does not alter execution semantics. Its `architecture.path` attributes use
the same display grouping as logical annotations. Shared operation-template calls
stay inside their annotated architecture containers and remain navigable into
their target graphs. `SourceMap` metadata is retained in the source XML and ignored
for layout.

A call binding normally resolves its `resource` in the parent graph instance.
An optional `instance` attribute instead resolves the resource in that direct
child call instance:

```xml
<Binding
  port="value"
  resource="scores"
  instance="attention-1"
  access="Read" />
```

GraphView displays this resource as `attention-1/scores`. Cross-instance writes
are valid logical bindings but produce a graph diagnostic because they can
introduce surprising ordering and mutation behavior.

## Viewer regression checks

Open the viewer and run the following in its browser developer console. No
packages or server are required. The tests replace the currently displayed file
with synthetic logical/execution XML fixtures; they do not change files on disk.

```javascript
await new Promise((resolve, reject) => {
  const script = document.createElement("script");
  script.src = new URL("graph-viewer.tests.js", location.href).href;
  script.onload = resolve;
  script.onerror = () => reject(new Error("Cannot load viewer regression tests."));
  document.head.append(script);
});
await runGraphViewerTests();
```

Checks cover path grouping, interleaved access/dependency order, descriptions,
formulas, mixed annotations, identifier collisions, repeat groups, history,
language switching, invalid paths/dependencies, shared-template navigation,
execution/logical fork-and-join ranks, data-edge preservation, selection, and
mode-specific layout/history restoration. Loading checks also cover indexed
ReadWrite/instance bindings, missing-resource errors, batched progress and
responsiveness, localized phase labels, superseded loads, and embedded
production IL graphs.
