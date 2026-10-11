"""Generate and validate experimental execution plans, not production VM XML."""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import xml.etree.ElementTree as ET


POINTWISE = {
    "core.copy", "core.add", "core.subtract", "core.multiply", "core.divide",
    "core.maximum", "core.exp", "core.tanh", "core.sigmoid", "core.rsqrt",
    "core.square", "core.relu", "core.tensor.fill",
}
ELEMENT_SIZE = {"Byte": 1, "Float16": 2, "Float32": 4, "Int32": 4, "UInt32": 4}
COLLECTION = "00000000-0000-0000-0000-000000000000"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def tensor(element):
    value = element.find("Tensor")
    if value is None and element.get("elementType") is not None:
        return element.get("elementType"), tuple(map(int, element.get("shape").split(",")))
    require(value is not None, f"Missing tensor: {element.attrib}")
    return value.get("elementType"), tuple(int(d.get("size")) for d in value)


def byte_length(descriptor):
    return ELEMENT_SIZE[descriptor[0]] * math.prod(descriptor[1])


def bindings(node):
    return list(node.findall("Bindings/Binding"))


def nodes(definition):
    return list(definition.findall(".//Node"))


def descriptions(definition):
    return {
        item.get("id"): tensor(item)
        for item in list(definition.findall("Ports/Port"))
        + list(definition.findall("LocalResources/Resource"))
    }


def backing(definition, resource):
    declarations = {r.get("id"): r for r in definition.findall("LocalResources/Resource")}
    seen = set()
    while resource in declarations and declarations[resource].get("viewOf"):
        require(resource not in seen, "Cyclic resource view.")
        seen.add(resource)
        resource = declarations[resource].get("viewOf")
    return resource


def lower_views(original):
    """Replace safe, static reshapes with read-only frame resource declarations."""
    definition = compact_copy(original)
    sequence = nodes(definition)
    resources = {r.get("id"): r for r in definition.findall("LocalResources/Resource")}
    declarations = {r.get("id"): r for r in definition.findall("Ports/Port")
                    + definition.findall("LocalResources/Resource")}
    descriptors = descriptions(definition)
    removed = set()
    for index, node in enumerate(sequence):
        if node.get("operation") != "core.tensor.reshape":
            continue
        ports = {b.get("port"): b.get("resource") for b in bindings(node)}
        source, output = ports["input"], ports["output"]
        if any(declarations[name].get("layout", "dense") != "dense"
               for name in (source, output)):
            continue
        base = backing(definition, source)
        uses = [(i, b.get("access")) for i, other in enumerate(sequence)
                if other is not node for b in bindings(other) if b.get("resource") == output]
        if output not in resources or not uses or any(
                i <= index or access != "Read" for i, access in uses):
            continue
        last = max(i for i, _ in uses)
        if any(b.get("access") != "Read" and
               backing(definition, b.get("resource")) == base
               for other in sequence[index + 1:last + 1] for b in bindings(other)):
            continue
        require(descriptors[source][0] == descriptors[output][0] and
                byte_length(descriptors[source]) == byte_length(descriptors[output]),
                "Reshape view changes type or storage length.")
        resources[output].attrib.update({
            "viewOf": base, "byteOffset": "0", "access": "Read",
            "binding": "frame-static", "sourceNode": node.get("id"),
        })
        removed.add(node.get("id"))
    return definition, removed


def validate_view_cases():
    definition = ET.fromstring(
        '<Definition id="test"><Ports><Port id="input" elementType="Float32" shape="4"/>'
        '</Ports><LocalResources><Resource id="base" elementType="Float32" shape="4"/>'
        '<Resource id="view" elementType="Float32" shape="2,2"/>'
        '<Resource id="chain" elementType="Float32" shape="4"/>'
        '<Resource id="other" elementType="Float32" shape="4"/>'
        '</LocalResources><Region id="body"/></Definition>')
    region = definition.find("Region")

    def operation(identity, name, source, output):
        node = ET.SubElement(region, "Node", {"id": identity, "operation": name, "operationVersion": "1"})
        ports = ET.SubElement(node, "Bindings")
        ET.SubElement(ports, "Binding", {"port": "input", "resource": source, "access": "Read"})
        ET.SubElement(ports, "Binding", {"port": "output", "resource": output, "access": "Write"})
        return node

    operation("produce", "core.copy", "input", "base")
    operation("reshape", "core.tensor.reshape", "base", "view")
    operation("reshape-chain", "core.tensor.reshape", "view", "chain")
    operation("consume", "core.copy", "chain", "other")
    lowered, removed = lower_views(definition)
    require(removed == {"reshape", "reshape-chain"} and backing(lowered, "chain") == "base",
            "Static reshape chain was not flattened.")
    locals_ = {r.get("id"): tensor(r) for r in lowered.findall("LocalResources/Resource")
               if not r.get("viewOf")}
    allocate(lowered, locals_, [[("base", False, True)], [("other", False, True)],
                               [(backing(lowered, "chain"), True, False), ("other", True, False)]], set())
    maps = {m.get("resource"): m.get("slot") for m in lowered.findall("ResourcePlan/Map")}
    require(maps["base"] != maps["other"], "View base was reused before its last consumer.")
    overwritten = copy.deepcopy(definition)
    mutate = copy.deepcopy(region.find("Node"))
    mutate.set("id", "mutate-base")
    overwritten.find("Region").insert(3, mutate)
    require("reshape-chain" not in lower_views(overwritten)[1],
            "A view across base mutation must remain a materialized reshape.")
    written_view = copy.deepcopy(definition)
    writer = copy.deepcopy(region.find("Node"))
    writer.set("id", "mutate-view")
    writer.find("Bindings/Binding[@port='output']").set("resource", "view")
    written_view.find("Region").append(writer)
    require("reshape" not in lower_views(written_view)[1],
            "Writable reshape output must remain materialized.")
    return ["static-chain-flattening", "view-base-liveness",
            "base-mutation-retains-copy", "view-write-retains-copy"]


def fingerprint(root):
    normalized = copy.deepcopy(root)
    normalized.attrib.pop("planSha256", None)
    for element in normalized.iter():
        if element.text is not None and not element.text.strip():
            element.text = None
        element.tail = None
    return hashlib.sha256(ET.tostring(normalized, encoding="utf-8")).hexdigest()


def fp32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def scalar(operation, values, attributes):
    name = operation.removeprefix("core.")
    if name == "tensor.fill":
        return fp32(float(attributes["value"]))
    if name == "copy":
        return values["input"]
    if name in {"add", "subtract", "multiply", "divide", "maximum"}:
        left, right = values["left"], values["right"]
        return fp32({
            "add": lambda: left + right,
            "subtract": lambda: left - right,
            "multiply": lambda: left * right,
            "divide": lambda: left / right,
            "maximum": lambda: max(left, right),
        }[name]())
    value = values["input"]
    return fp32({
        "exp": lambda: math.exp(value),
        "tanh": lambda: math.tanh(value),
        "sigmoid": lambda: 1 / (1 + math.exp(-value)),
        "rsqrt": lambda: 1 / math.sqrt(value),
        "square": lambda: value * value,
        "relu": lambda: max(0, value),
    }[name]())


def verify_fused_schedule(operations, shape):
    """Compare whole-array steps with the proposed per-index kernel schedule."""
    count = math.prod(shape[1])
    names = {p.get("resource") for operation in operations for p in bindings(operation)}
    initial = {name: [fp32(0.25 + (i % 7) * 0.01) for i in range(count)]
               for name in names}
    sequential = copy.deepcopy(initial)
    fused = copy.deepcopy(initial)

    def run(operation, index, storage):
        ports = bindings(operation)
        attributes = {a.get("key"): a.get("value")
                      for a in operation.findall("Attributes/Attribute")}
        values = {p.get("port"): storage[p.get("resource")][index]
                  for p in ports if p.get("port") != "output"}
        output = next(p.get("resource") for p in ports if p.get("port") == "output")
        storage[output][index] = scalar(operation.get("operation"), values, attributes)

    for operation in operations:
        for index in range(count):
            run(operation, index, sequential)
    for index in range(count):
        for operation in operations:
            run(operation, index, fused)
    require(sequential == fused, "Per-index fusion changed synthetic FP32 results.")
    return count


def compact_copy(element):
    result = copy.deepcopy(element)
    for parent in result.iter():
        for child in list(parent):
            if child.tag == "Tensor":
                kind, shape = tensor(parent)
                parent.set("elementType", kind)
                parent.set("shape", ",".join(map(str, shape)))
                if child.get("layout") != "dense":
                    parent.set("layout", child.get("layout"))
                parent.remove(child)
            elif child.tag in {"Architecture", "Attributes", "DependsOn"} and (
                child.tag == "Architecture" or len(child) == 0
            ):
                parent.remove(child)
            elif child.tag == "Precision":
                require(child.attrib == {
                    "minimumArithmeticType": "Float32",
                    "minimumAccumulatorType": "Float32",
                }, "This experiment requires FP32 precision.")
                parent.remove(child)
    return result


def call_access(call, definitions):
    definition = definitions[call.get("definition")]
    by_resource = {}
    for node in nodes(definition):
        for binding in bindings(node):
            by_resource.setdefault(binding.get("resource"), []).append(binding.get("access"))
    result = []
    for binding in bindings(call):
        accesses = by_resource.get(binding.get("port"), [])
        require(accesses, f"Unused call port: {binding.attrib}")
        first = accesses[0]
        reads_input = first != "Write"
        writes = any(access != "Read" for access in accesses)
        result.append((binding.get("resource"), reads_input, writes))
    return result


def allocate(parent, resources, events, endpoints):
    """Reuse only equal descriptors with disjoint whole-event intervals."""
    intervals = {}
    written = set()
    for index, event in enumerate(events):
        for resource, reads_input, writes in event:
            if resource not in resources:
                continue
            require(not reads_input or resource in written or resource in endpoints,
                    f"Local '{resource}' read before initialization at event {index}.")
            if writes:
                written.add(resource)
            intervals.setdefault(resource, [index, index])[1] = index
    plan = ET.SubElement(parent, "ResourcePlan", {
        "allocation": "unified-manager", "reuse": "exact-shape-disjoint-events",
    })
    slots = []
    for name, descriptor in sorted(resources.items(),
                                   key=lambda pair: (intervals.get(pair[0], [-1])[0], pair[0])):
        require(name in intervals or name in endpoints, f"Unused local resource '{name}'.")
        first, last = intervals.get(name, [-1, len(events)])
        selected = None
        if name not in endpoints:
            for index, slot in enumerate(slots):
                if not slot["pinned"] and slot["descriptor"] == descriptor and slot["last"] < first:
                    selected = index
                    break
        if selected is None:
            selected = len(slots)
            slots.append({"descriptor": descriptor, "last": last, "pinned": name in endpoints})
        else:
            slots[selected]["last"] = last
        ET.SubElement(plan, "Map", {"resource": name, "slot": f"s{selected}"})
    table = ET.SubElement(plan, "Slots")
    for index, slot in enumerate(slots):
        kind, shape = slot["descriptor"]
        ET.SubElement(table, "Slot", {
            "id": f"s{index}", "elementType": kind,
            "shape": ",".join(map(str, shape)),
        })
    original = sum(byte_length(value) for value in resources.values())
    allocated = sum(byte_length(slot["descriptor"]) for slot in slots)
    return {"resources": len(resources), "slots": len(slots),
            "unreusedBytes": original, "plannedBytes": allocated}


def same_index(node, descriptors):
    if node.get("operation") not in POINTWISE:
        return None
    ports = bindings(node)
    output = [p for p in ports if p.get("port") == "output"]
    if len(output) != 1:
        return None
    shape = descriptors[output[0].get("resource")]
    if shape[0] != "Float32" or any(
        descriptors[p.get("resource")] != shape for p in ports
    ):
        return None
    require(all(p.get("resource") != output[0].get("resource")
                for p in ports if p.get("port") != "output"),
            f"Unsupported operator alias: {node.get('id')}")
    return shape


def build_units(region, descriptors, target, unit_records, view_nodes):
    body = ET.Element("Region", dict(region.attrib))
    pending = []

    def flush():
        if not pending:
            return
        unit_id = f"u{len(unit_records):03d}"
        attributes = {"id": unit_id, "kind": "Function" if target == "cpu" else "Kernel"}
        unit = ET.SubElement(body, "Unit", attributes)
        if target == "gpu":
            output = next(p for p in bindings(pending[-1]) if p.get("port") == "output")
            count = math.prod(descriptors[output.get("resource")][1])
            unit.set("threads", "64 1 1")
            unit.set("groups", f"{(count + 63) // 64} 1 1")
            unit.set("indexing", "linear-output")
            unit.set("implementation", "same-index-buffer-pipeline" if len(pending) > 1
                     else "tier-zero")
        params = {}
        for node in pending:
            for binding in bindings(node):
                name, access = binding.get("resource"), binding.get("access")
                previous = params.get(name)
                params[name] = access if previous is None or previous == access else "ReadWrite"
        require(target != "gpu" or len(params) <= 20,
                f"Kernel {unit_id} exceeds root-buffer ABI.")
        parameters = ET.SubElement(unit, "Parameters")
        for name, access in params.items():
            ET.SubElement(parameters, "Parameter", {"resource": name, "access": access})
        operations = ET.SubElement(unit, "Body")
        for node in pending:
            operation = compact_copy(node)
            operation.tag = "Operation"
            operation.set("sourceNode", node.get("id"))
            operation.set("collection", COLLECTION)
            operation.set("implementation", "serial" if node.get("operation") == "core.mat-vec"
                          and target == "gpu" else "tier-zero")
            dependencies = operation.find("DependsOn")
            if dependencies is not None:
                operation.remove(dependencies)
            operations.append(operation)
        ET.SubElement(body, "Invoke", {"unit": unit_id})
        if target == "gpu":
            writes = sorted(name for name, access in params.items() if access != "Read")
            ET.SubElement(body, "Barrier", {"kind": "uav", "resources": " ".join(writes)})
        unit_records.append(list(pending))
        pending.clear()

    for child in region:
        if child.tag == "Region":
            flush()
            body.append(build_units(child, descriptors, target, unit_records, view_nodes))
        elif child.tag == "Node":
            if child.get("id") in view_nodes:
                flush()
                continue
            shape = same_index(child, descriptors) if target == "gpu" else None
            resources = {p.get("resource") for n in pending + [child] for p in bindings(n)}
            if pending and (target == "cpu" or shape is None
                            or same_index(pending[-1], descriptors) != shape or len(resources) > 20):
                flush()
            pending.append(child)
            if shape is None:
                flush()
        elif child.tag not in {"Architecture", "Attributes"}:
            raise ValueError(f"Unexpected definition body element: {child.tag}")
    flush()
    return body


def partition_manifest(root, calls, definitions):
    manifest = ET.SubElement(root, "PartitionExample", {
        "status": "interface-only", "cutAfter": "layer.5",
        "transport": "deployment-defined", "step": "same-session-same-token",
    })
    sets = [calls[:7], calls[7:]]
    summaries = [[call_access(call, definitions) for call in group] for group in sets]
    produced_a = {name for event in summaries[0] for name, _, writes in event if writes}
    needed_b = set()
    written_b = set()
    for event in summaries[1]:
        for name, reads, writes in event:
            if reads and name not in written_b:
                needed_b.add(name)
            if writes:
                written_b.add(name)
    crossing = sorted(produced_a & needed_b)
    require(len(crossing) == 2, f"Expected hidden plus firstValue at cut: {crossing}")
    for index, group in enumerate(sets):
        shard = ET.SubElement(manifest, "Partition", {"id": f"host-{index}"})
        for call in group:
            ET.SubElement(shard, "CallRef", {"id": call.get("id")})
        states = sorted({name for call in group for name, _, _ in call_access(call, definitions)
                         if name.startswith("state.")})
        for name in states:
            ET.SubElement(shard, "OwnsState", {"resource": name})
    for name in crossing:
        ET.SubElement(manifest, "Connection", {
            "resource": name, "from": "host-0", "to": "host-1",
            "access": "Read", "ready": "producer-and-transfer-completed",
        })
    return crossing


def base_plan(source, kind, source_hash):
    root = ET.Element("ExecutionPlan", {
        "schema": "rwkv7-experiment", "version": "1", "kind": kind,
        "status": "design-only", "sourceSha256": source_hash,
    })
    ET.SubElement(root, "Defaults", {
        "order": "document-sequential", "layout": "dense", "byteOffset": "0",
        "arithmetic": "Float32", "accumulator": "Float32",
        "crossParameterWriteAlias": "forbidden", "scratchInitialContents": "unspecified",
        "sessionExecution": "exclusive", "allocation": "unified-manager",
    })
    for name in ("Identity", "Model", "Resources", "Inputs", "Outputs", "GraphState"):
        element = source.find(name)
        require(element is not None, f"Missing {name}")
        root.append(compact_copy(element))
    return root


def validate_plan(plan, source, target):
    definitions = {d.get("id"): d for d in plan.findall("Definitions/Definition")}
    source_definitions = {d.get("id"): d for d in source.findall("LayerDefinitions/LayerDefinition")}
    require(definitions.keys() == source_definitions.keys(), "Definition coverage differs.")
    for name, definition in definitions.items():
        original = nodes(source_definitions[name])
        lowered, view_nodes = lower_views(source_definitions[name]) if target else (
            source_definitions[name], set())
        expected_resources = compact_copy(lowered.find("LocalResources"))
        require(fingerprint(expected_resources) == fingerprint(definition.find("LocalResources")),
                f"Resource or static view declaration changed in {name}.")
        require(fingerprint(compact_copy(source_definitions[name].find("Ports"))) ==
                fingerprint(definition.find("Ports")), f"Formal ports changed in {name}.")
        original = [node for node in original if node.get("id") not in view_nodes]
        generated = (list(definition.findall(".//Operation")) if target else
                     list(definition.findall(".//Node")))
        require(len(original) == len(generated), f"Missing operations in {name}")
        for left, right in zip(original, generated):
            expected = compact_copy(left)
            if target:
                expected.tag = "Operation"
                expected.set("sourceNode", left.get("id"))
                expected.set("collection", COLLECTION)
                expected.set("implementation", right.get("implementation"))
                depends = expected.find("DependsOn")
                if depends is not None:
                    expected.remove(depends)
            require(fingerprint(expected) == fingerprint(right),
                    f"Operation, binding or attribute changed at {name}/{left.get('id')}")
        if target:
            units = definition.findall(".//Unit")
            invocations = definition.findall(".//Invoke")
            require(len(units) == len(invocations) and
                    [u.get("id") for u in units] == [i.get("unit") for i in invocations] and
                    len({u.get("id") for u in units}) == len(units),
                    f"Unit invocation coverage differs in {name}.")
            for unit in units:
                accesses = {}
                for parameter in unit.findall("Parameters/Parameter"):
                    accesses.setdefault(backing(definition, parameter.get("resource")), []).append(
                        parameter.get("access"))
                require(all(len(values) == 1 or all(value == "Read" for value in values)
                            for values in accesses.values()),
                        f"Writable aliased kernel/function parameters in {name}/{unit.get('id')}.")
        if target == "gpu":
            descriptors = descriptions(source_definitions[name])
            for unit in definition.findall(".//Unit"):
                operations = list(unit.findall("Body/Operation"))
                params = list(unit.findall("Parameters/Parameter"))
                require(len(params) <= 20, f"Kernel parameter capacity exceeded: {name}")
                output = next(p for p in bindings(operations[-1]) if p.get("port") == "output")
                count = math.prod(descriptors[output.get("resource")][1])
                require(unit.get("threads") == "64 1 1" and
                        unit.get("groups") == f"{(count + 63) // 64} 1 1",
                        f"Kernel launch does not cover its output: {name}/{unit.get('id')}")
                if len(operations) > 1:
                    shape = same_index(operations[0], descriptors)
                    require(shape is not None and all(same_index(op, descriptors) == shape
                                                      for op in operations),
                            "Fusion has cross-index or incompatible-shape operations.")
                    verify_fused_schedule(operations, shape)
            for region in definition.findall(".//Region"):
                children = list(region)
                for index, child in enumerate(children):
                    if child.tag == "Invoke":
                        require(index + 1 < len(children) and children[index + 1].tag == "Barrier",
                                "GPU invocation missing visibility barrier.")
                        require(index > 0 and children[index - 1].tag == "Unit" and
                                children[index - 1].get("id") == child.get("unit"),
                                "GPU invocation must follow its unit declaration.")
                        written = {p.get("resource") for p in
                                   children[index - 1].findall("Parameters/Parameter")
                                   if p.get("access") != "Read"}
                        require(set(children[index + 1].get("resources").split()) == written,
                                "GPU barrier does not cover the unit's writes.")
    expected_calls = source.findall("Region//Call")
    actual_calls = plan.findall("EntryPoints/Entry/Region//Call")
    require(len(expected_calls) == len(actual_calls) == 14, "Root calls changed.")
    for left, right in zip(expected_calls, actual_calls):
        require(fingerprint(compact_copy(left)) == fingerprint(right), "Call bindings changed.")
    counts = {name: len(nodes(definition)) for name, definition in source_definitions.items()}
    expanded = sum(counts[call.get("definition")] for call in actual_calls)
    require(expanded == 2023, f"Unexpected expanded operation count: {expanded}")
    require(len(plan.findall("GraphState/Slots/Slot")) == 36, "State coverage changed.")
    return expanded


def validate_resource_plans(plan):
    maps_checked = 0
    for owner in list(plan.findall("Definitions/Definition")) + list(plan.findall("EntryPoints/Entry")):
        resource_plan = owner.find("ResourcePlan")
        if resource_plan is None:
            continue
        slots = {slot.get("id"): tensor(slot)
            for slot in resource_plan.findall("Slots/Slot")}
        if owner.tag == "Definition":
            declarations = [r for r in owner.findall("LocalResources/Resource") if not r.get("viewOf")]
            events = [[backing(owner, p.get("resource")) for p in unit.findall("Body/Operation/Bindings/Binding")]
                      for unit in owner.findall(".//Unit")]
            descriptors_all = descriptions(owner)
            for view in owner.findall("LocalResources/Resource"):
                if not view.get("viewOf"):
                    continue
                base = view.get("viewOf")
                require(base in descriptors_all and view.get("byteOffset") == "0" and
                        view.get("access") == "Read" and view.get("binding") == "frame-static" and
                        tensor(view)[0] == descriptors_all[base][0] and
                        byte_length(tensor(view)) == byte_length(descriptors_all[base]),
                        "Invalid static reshape view.")
                require(not any(p.get("resource") == view.get("id") and p.get("access") != "Read"
                                for p in owner.findall(".//Operation/Bindings/Binding")),
                        "Read-only view is written.")
        else:
            declarations = [r for r in plan.findall("Resources/Resource") if r.get("scope") == "Local"]
            events = [[p.get("resource") for p in call.findall("Bindings/Binding")]
                      for call in owner.findall("Region//Call")]
        descriptors = {r.get("id"): tensor(r)
                       for r in declarations}
        mappings = resource_plan.findall("Map")
        require(len(mappings) == len(descriptors) and
                {m.get("resource") for m in mappings} == descriptors.keys(),
                "Incomplete or duplicate resource mapping.")
        occupants = {}
        for mapping in mappings:
            name, slot = mapping.get("resource"), mapping.get("slot")
            require(slots.get(slot) == descriptors[name], "Resource/slot tensor mismatch.")
            positions = [index for index, event in enumerate(events) if name in event]
            require(positions, f"No use for mapped resource: {name}")
            interval = (min(positions), max(positions))
            for other in occupants.get(slot, []):
                require(interval[1] < other[0] or other[1] < interval[0],
                        f"Overlapping lifetimes share slot '{slot}'.")
            occupants.setdefault(slot, []).append(interval)
            maps_checked += 1
    return maps_checked


def generate(source_path, output):
    data = source_path.read_bytes()
    source = ET.fromstring(data)
    require(source.tag == "Graph" and source.get("kind") == "LogicalGraph",
            "Expected structured logical graph XML.")
    require(not source.findall(".//View"), "Tensor views are outside this experiment.")
    definitions = {d.get("id"): d for d in source.findall("LayerDefinitions/LayerDefinition")}
    calls = source.findall("Region//Call")
    source_hash = hashlib.sha256(data).hexdigest()
    report = {"sourceSha256": source_hash,
              "validation": "structure-resource-liveness-and-synthetic-fusion-no-binary-execution",
              "viewValidationCases": validate_view_cases(),
              "plans": {}}
    output.mkdir(parents=True, exist_ok=True)
    il_hash = None
    frontend = source
    for target in (None, "cpu", "gpu"):
        label = target or "il"
        plan = base_plan(frontend, label, source_hash)
        if target:
            plan.set("target", "cpu.managed" if target == "cpu" else "direct3d12")
            plan.set("ilSha256", il_hash)
            plan.set("finalCompilerPolicy", "validate-and-emit-no-replanning")
        table = ET.SubElement(plan, "Definitions")
        summary = {"definitions": {}, "allocation": {}}
        for name, original in definitions.items():
            definition = ET.SubElement(table, "Definition", {"id": name})
            lowered, view_nodes = lower_views(original) if target else (original, set())
            for section in ("Ports", "LocalResources"):
                definition.append(compact_copy(lowered.find(section)))
            records = []
            if target:
                definition.append(build_units(lowered.find("Region"), descriptions(original),
                                              target, records, view_nodes))
                events = [[(backing(definition, p.get("resource")), p.get("access") != "Write",
                            p.get("access") != "Read") for node in unit for p in bindings(node)]
                          for unit in records]
                locals_ = {r.get("id"): tensor(r) for r in lowered.findall("LocalResources/Resource")
                           if not r.get("viewOf")}
                summary["allocation"][name] = allocate(definition, locals_, events, set())
            else:
                definition.append(compact_copy(original.find("Region")))
            summary["definitions"][name] = {
                "operations": len(nodes(original)), "units": len(records) if target else None,
                "multiStepUnits": sum(len(record) > 1 for record in records),
                "staticViews": len(view_nodes),
                "dynamicCreateViews": 0,
                "executedOperations": sum(len(record) for record in records) if target else len(nodes(original)),
            }
        entries = ET.SubElement(plan, "EntryPoints")
        entry = ET.SubElement(entries, "Entry", {"name": "forward", "step": "one-token"})
        entry.append(compact_copy(frontend.find("Region")))
        if target:
            locals_ = {r.get("id"): tensor(r) for r in frontend.findall("Resources/Resource")
                       if r.get("scope") == "Local"}
            endpoints = {r.get("id") for r in frontend.findall("Inputs/ResourceRef")
                         + frontend.findall("Outputs/ResourceRef")}
            events = [call_access(call, definitions) for call in calls]
            summary["allocation"]["entry"] = allocate(entry, locals_, events, endpoints)
        summary["crossHostResources"] = partition_manifest(plan, calls, definitions)
        summary["expandedOperations"] = validate_plan(plan, source, target)
        summary["resourceMappingsChecked"] = validate_resource_plans(plan)
        if target == "gpu":
            summary["syntheticFusionUnitsChecked"] = sum(
                d["multiStepUnits"] for d in summary["definitions"].values())
        if target:
            summary["expandedUnits"] = sum(summary["definitions"][c.get("definition")]["units"]
                                           for c in calls)
            summary["expandedStaticViews"] = sum(
                summary["definitions"][c.get("definition")]["staticViews"] for c in calls)
            summary["expandedExecutedOperations"] = sum(
                summary["definitions"][c.get("definition")]["executedOperations"] for c in calls)
            summary["dynamicCreateViews"] = 0
            summary["barriersPerToken"] = summary["expandedUnits"] if target == "gpu" else 0
        plan_hash = fingerprint(plan)
        summary["planSha256"] = plan_hash
        if target is None:
            il_hash = plan_hash
        plan.set("planSha256", plan_hash)
        ET.indent(plan, space="  ")
        ET.ElementTree(plan).write(output / f"rwkv7-tiny.{label}.xml",
                                   encoding="utf-8", xml_declaration=True)
        loaded = ET.parse(output / f"rwkv7-tiny.{label}.xml").getroot()
        require(fingerprint(loaded) == plan_hash, "Plan fingerprint changed on XML roundtrip.")
        validate_plan(loaded, source, target)
        validate_resource_plans(loaded)
        report["plans"][label] = summary
        if target is None:
            frontend = ET.Element("Graph")
            for name in ("Identity", "Model", "Resources", "Inputs", "Outputs", "GraphState"):
                frontend.append(copy.deepcopy(loaded.find(name)))
            table = ET.SubElement(frontend, "LayerDefinitions")
            for definition in loaded.findall("Definitions/Definition"):
                item = copy.deepcopy(definition)
                item.tag = "LayerDefinition"
                table.append(item)
            frontend.append(copy.deepcopy(loaded.find("EntryPoints/Entry/Region")))
            definitions = {d.get("id"): d for d in frontend.findall("LayerDefinitions/LayerDefinition")}
            calls = frontend.findall("Region//Call")
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.source, args.output)
