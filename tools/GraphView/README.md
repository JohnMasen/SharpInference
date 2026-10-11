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

- [Nested Region example](../../artifacts/rwkv7-tiny.region-architecture.xml)
- [Original RWKV7 graph](../../artifacts/rwkv7-tiny.layer-defined.xml)
- [Original RWKV6 graph](../../artifacts/rwkv6-tiny.layer-defined.xml)

These local artifacts are optional, git-ignored examples, not tool dependencies.
Flat `Nodes` graphs and structured `Region` / `LayerDefinition` / `Call` graphs
are supported. The former standalone `Architecture` tree is not supported.
The heading and status identify the loaded file as a **Logical graph**, an
**Execution graph**, or an **Execution graph · IL / CPU / GPU**. Files without a
declared graph kind are labeled as unspecified, not guessed to be logical graphs.

## Execution plans

The viewer accepts production `ExecutionPlan` XML
(`schema="sharp-inference"`, `version="1"`, `kind="il|cpu|gpu"`).
Generate it with the VM tool's `plan cpu|d3d12` command; `compile-plan` reloads
CPU/GPU plans without the model graph generator. Production examples:

- [Production CPU plan](../../artifacts/rwkv7-execution-plans/rwkv7-tiny.production.cpu.xml)
- [Production GPU plan](../../artifacts/rwkv7-execution-plans/rwkv7-tiny.production.gpu.xml)
- [Structured IL plan](../../artifacts/rwkv7-execution-plans/rwkv7-tiny.production.cpu.il.xml)

Production target views display explicit EntryPoints, orchestration Calls,
Functions, Kernels, Dispatches and Barriers. Double-click a call to enter its
definition. Details retain the original execution-plan XML, thread/grid
configuration, formal/actual bindings and offsets. ResourcePlan summaries show
concrete local slots, resource mappings, static views and payload capacity;
these are declarations, not measurements or allocated memory.
Resource kinds propagate through actual call bindings, so weight visibility
continues to work when reusable definitions use normalized formal slot names.

The viewer also accepts the earlier experimental `ExecutionPlan` XML format
(`schema="rwkv7-experiment"`, `version="1"`) for IL, CPU and GPU plans:

- [IL plan](../../artifacts/rwkv7-execution-plans/rwkv7-tiny.il.xml)
- [CPU plan](../../artifacts/rwkv7-execution-plans/rwkv7-tiny.cpu.xml)
- [GPU plan](../../artifacts/rwkv7-execution-plans/rwkv7-tiny.gpu.xml)

These optional, git-ignored examples can be generated with
[the experiment script](../../scripts/rwkv7_execution_plan_experiment.py).

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

## Region architecture metadata

The hierarchy and execution order come from nested Regions containing actual
operations. A Region may have one optional `Architecture` metadata element:

```xml
<Region id="memory-update" type="architecture" name="Memory Update">
  <Architecture role="state-update">
    <Description>Computes the new recurrent memory.</Description>
    <Formula>S_new = ...</Formula>
  </Architecture>
  <!-- Actual Node, Call or nested Region elements -->
</Region>
```

`Description` and `Formula` are documentation, not executable expressions.
Optional `Ports/Port` entries use `resource`, `direction` (`in` / `out`) and
`name` to label real boundary resources. The viewer validates these against
actual bindings rather than treating them as a second graph.

`defaultCollapsed="false"` expands a structural container transparently.
`repeatGroup="true"` marks a repeated group controlled by the toolbar.
`defaultView="architecture"` on the root Region's metadata selects the initial
architecture view; `computation` selects the full graph.

With `step="token"` in the computation context, cross-token state edges connect
the last writer of a `SessionState` resource to its first reader in the next
token. They do not participate in current-step layout ordering.

The viewer does not convert XML formats or change execution content.
`GraphRegion.Architecture` represents this optional metadata in the C# graph
model. XML and JSON serializers preserve it, including port resource mappings
through reusable layer definitions and call expansion. Existing graphs without
metadata remain supported. RWKV6, RWKV7 and Phi4 graph providers generate
the nested semantic Regions directly, without a separate architecture tree.

## Building semantic Regions in C#

`LogicalGraphBuilder.WithRegion` synchronously enters a Region scope. Nested
scopes infer their parent, and scoped `AddNode` calls infer their Region.
`WithRegion<T>` returns a computation's resource ID or other result.
The previous scope is restored on both normal and exceptional exits; exceptions
propagate, and mutations made before a failure are not rolled back. Callbacks
must be synchronous, and a builder must not be shared across threads.

```csharp
builder.WithRegion("root", GraphRegionTypes.Graph, "Forward", root =>
{
    var hidden = root.WithRegion(
        "projection", GraphRegionTypes.Architecture, "Projection",
        projection =>
        {
            projection.MatVec("project", "weight", "input", "hidden");
            return "hidden";
        },
        architecture: new(Description: "Projects input features."));
    root.Copy("finish", hidden, "output");
});
```

Resources must be declared before nodes that use them, as with `AddNode`.
Typed extensions cover copy, common unary/binary primitives, matrix-vector
projection, row gathering, matrix multiplication and affine projection. Matrix
transpose flags are Boolean parameters, not hand-written attribute strings.
The extensions use the existing operation contracts and precision inference;
they do not allocate resources or infer tensor shapes. An optional `regionId`
keeps them usable with explicit Region construction.

Explicit `AddRegion` and `AddNode` remain available for component composition,
custom operations, tensor views, explicit dependencies and precision overrides.
Region scopes and metadata do not reorder operations or define new execution
boundaries. Semantic `architecture` Regions are transparent to existing
stage/layer fusion boundaries.
