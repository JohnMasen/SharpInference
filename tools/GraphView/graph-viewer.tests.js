globalThis.runGraphViewerTests = async function () {
  const $ = id => document.getElementById(id);
  const results = [];
  const assert = (condition, name) => {
    if (!condition) throw new Error(`${name}: ${$("status").textContent}`);
    results.push(name);
  };
  const node = label => Array.from($("nodes").children)
    .find(item => item.textContent.includes(label));
  const operation = id => Array.from($("nodes").children)
    .find(item => item.getAttribute("aria-label")?.startsWith(`${id}: `));
  const click = element => element.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  const enter = element => element.dispatchEvent(new MouseEvent("dblclick", { bubbles: true }));

  function fixture(kind, options) {
    const execution = kind === "ExecutionGraphProgram";
    const annotationTag = execution ? "Architecture" : "Annotation";
    const graphTag = execution ? "ExecutionGraph" : "Definition";
    const callTag = execution ? "GraphCall" : "Call";
    const annotation = (path, name) => path === null ? "" :
      `<${annotationTag} name="${name}" role="projection" description="Module description">
        <Attribute name="architecture.path" value="${path}" />
        <Attribute name="architecture.formula" value="y = x" />
        ${options.repeat ? '<Attribute name="repeatGroup" value="true" />' : ""}
      </${annotationTag}>`;
    const sourceMap = id => execution ? `<SourceMap><LogicalNode id="${id}" /></SourceMap>` : "";
    const resources = (ids, owner) => ids.map((id, index) =>
      `<Resource id="${id}" name="${id}" ${execution ? "" : `owner="${owner}"`}
        kind="${index === 0 ? "Input" : index === ids.length - 1 ? "Output" : "Temporary"}">
        <Tensor elementType="Float32" layout="dense"><Dimension value="1" /></Tensor>
      </Resource>`).join("");
    const ids = options.ids || (options.crossed ? ["a", "b", "c", "d"] : ["a", "b", "c"]);
    const paths = options.paths || (options.crossed ? [null, null, null, null] :
      ["graph/left", "graph/right", "graph/left"]);
    const inputs = options.crossed ? ["input", "input", "y", "x"] :
      options.parallel ? ["input", "input", "x"] : ["input", "x", "y"];
    const outputs = options.crossed ? ["x", "y", "z", "output"] : ["x", "y", "output"];
    const nodes = ids.map((id, index) => {
      const metadata = annotation(paths[index], index === 1 ? "Right" : "Left");
      if (options.call && index === 1)
        return `<${callTag} id="${id}" ${execution ? "target" : "definition"}="template"
          ${execution ? 'compileBoundary="Atomic"' : ""}>${metadata}${sourceMap(id)}</${callTag}>`;
      return `<Operation id="${id}" operation="${options.parallel && index === 2 ? "core.add" : "core.copy"}" operationVersion="1">
        ${metadata}${sourceMap(id)}<Attributes /><Precision arithmetic="Float32" accumulator="Float32" />
      </Operation>`;
    }).join("");
    const bindings = ids.map((id, index) =>
      `<Binding id="${id}:input" node="${id}" resource="${inputs[index]}"
        port="${options.parallel && index === 2 ? "left" : "input"}"
        access="Read" readEdge="r${index}" readOrder="${index * 4}" />
      ${options.parallel && index === 2 ? `<Binding id="${id}:right" node="${id}" resource="y"
        port="right" access="Read" readEdge="right" readOrder="${index * 4 + 1}" />` : ""}
      <Binding id="${id}:output" node="${id}" resource="${outputs[index]}"
        port="output" access="Write" writeEdge="w${index}" writeOrder="${index * 4 + 2}" />`).join("");
    const dependencies = ids.map((id, index) => index ?
      `<Dependency id="d${index}" source="${ids[index - 1]}" target="${id}"
        order="${index * 4 + 3}" />` : "").join("") +
      (options.backward ? `<Dependency id="invalid" source="${ids[2]}"
        target="${ids[0]}" order="${ids.length * 4}" />` : "");
    const root = `<${graphTag} id="test" compileBoundary="Atomic">
      <${annotationTag} name="Test" /><Nodes>${resources(["input", ...outputs], "test")}${nodes}</Nodes>
      <Bindings>${bindings}</Bindings><Dependencies>${dependencies}</Dependencies></${graphTag}>`;
    const template = options.call ? `<${graphTag} id="template" compileBoundary="Atomic">
      <Nodes>${resources(["input", "output"], "template")}
        <Operation id="body" operation="core.copy" operationVersion="1">
          ${sourceMap("body")}<Attributes /><Precision arithmetic="Float32" accumulator="Float32" />
        </Operation>
      </Nodes><Bindings>
        <Binding id="body:input" node="body" resource="input"
          port="input" access="Read" readEdge="tr" readOrder="0" />
        <Binding id="body:output" node="body" resource="output"
          port="output" access="Write" writeEdge="tw" writeOrder="1" />
      </Bindings><Dependencies /></${graphTag}>` : "";
    let xml = `<Graph kind="${kind}" version="${options.version || "3"}">
      <Root ${execution ? "graph" : "definition"}="test" />
      <${execution ? "Graphs" : "Definitions"}>${root}${template}</${execution ? "Graphs" : "Definitions"}>
    </Graph>`;
    if (options.bulk || options.readWrite || options.instance || options.missingResource) {
      const documentXml = new DOMParser().parseFromString(xml, "application/xml");
      const nodes = documentXml.querySelector("Nodes"), exemplar = nodes.querySelector("Operation");
      if (options.bulk) {
        for (let index = 0; index < (options.bulkCount || 256); index++) {
          const item = exemplar.cloneNode(true);
          item.setAttribute("id", `bulk.${index}`);
          nodes.append(item);
          if (options.chain) {
            const resource = nodes.querySelector('Resource[id="output"]').cloneNode(true);
            resource.setAttribute("id", `bulk.output.${index}`);
            resource.setAttribute("kind", "Temporary");
            nodes.append(resource);
            for (const [tag, attributes] of [
              ["Binding", { id: `bulk.${index}:input`, node: `bulk.${index}`,
                resource: index ? `bulk.output.${index - 1}` : "output",
                port: "input", access: "Read", readEdge: `bulk.${index}:Read`, readOrder: 16 + index * 3 }],
              ["Binding", { id: `bulk.${index}:output`, node: `bulk.${index}`,
                resource: `bulk.output.${index}`, port: "output", access: "Write",
                writeEdge: `bulk.${index}:Write`, writeOrder: 17 + index * 3 }],
              ["Dependency", { id: `bulk.${index}:Dependency`, source: index ? `bulk.${index - 1}` : "c",
                target: `bulk.${index}`, order: 18 + index * 3 }],
            ]) {
              const edge = documentXml.createElement(tag);
              for (const [key, value] of Object.entries(attributes)) edge.setAttribute(key, value);
              documentXml.querySelector(tag === "Binding" ? "Bindings" : "Dependencies").append(edge);
            }
          }
        }
      }
      const bindings = documentXml.querySelector("Bindings");
      if (options.readWrite) {
        const binding = documentXml.createElement("Binding");
        for (const [key, value] of Object.entries({
          id: "b:state", node: "b", resource: "x", port: "state", access: "ReadWrite",
          readEdge: "state-Read", readOrder: "13", writeEdge: "state-Write", writeOrder: "14",
        })) binding.setAttribute(key, value);
        bindings.append(binding);
      }
      if (options.instance) {
        const resource = nodes.querySelector('Resource[id="output"]').cloneNode(true);
        resource.setAttribute("instance", "b");
        nodes.append(resource);
        const binding = bindings.querySelector('Binding[node="c"][access="Read"]');
        binding.setAttribute("resource", "output");
        binding.setAttribute("instance", "b");
      }
      if (options.missingResource)
        bindings.querySelector("Binding").setAttribute("resource", "missing");
      xml = new XMLSerializer().serializeToString(documentXml);
    }
    if (options.plan) xml = `<ExecutionPlan schema="sharp-inference" version="1" kind="il">${xml}</ExecutionPlan>`;
    return xml;
  }

  async function load(kind, options = {}) {
    const transfer = new DataTransfer();
    transfer.items.add(new File([fixture(kind, options)], "regression.xml", { type: "application/xml" }));
    $("layout-priority").value = options.priority || "execution";
    $("file").files = transfer.files;
    const phases = [], updates = [];
    let pulses = 0;
    const observer = new MutationObserver(() => {
      phases.push($("status").textContent);
      if (!$("load-progress").hidden)
        updates.push({ value: $("load-progress").value, max: $("load-progress").max });
    });
    observer.observe($("status"), { childList: true, subtree: true, characterData: true });
    const pulse = setInterval(() => pulses++, 0);
    $("file").dispatchEvent(new Event("change", { bubbles: true }));
    const deadline = performance.now() + 5000;
    try {
      while (!$("load-progress").hidden) {
        if (performance.now() > deadline) throw new Error("Fixture loading timed out.");
        await new Promise(resolve => setTimeout(resolve, 10));
      }
    } finally {
      observer.disconnect();
      clearInterval(pulse);
    }
    return { error: $("status").classList.contains("error"), status: $("status").textContent,
      phases, updates, pulses };
  }

  document.querySelector('[data-language="zh"]').click();
  const changePriority = value => {
    $("layout-priority").value = value;
    $("layout-priority").dispatchEvent(new Event("change", { bubbles: true }));
  };
  const position = id => {
    const matrix = operation(id).transform.baseVal.getItem(0).matrix;
    return { x: matrix.e, y: matrix.f };
  };
  const dataEdges = () => Array.from($("edges").children)
    .map(edge => edge.querySelector("title").textContent).filter(title => !title.includes("(Order)")).sort();
  for (const kind of ["LogicalGraphDefinition", "ExecutionGraphProgram"]) {
    let result = await load(kind, { version: "2" });
    assert(result.error && result.status.includes("v3"), `${kind}: XML v2 is explicitly rejected`);
    result = await load(kind);
    assert(!result.error && node("Left") && node("Right"), `${kind}: interleaved groups`);
    assert($("edges").children.length === 5, `${kind}: aggregated edges retained`);
    click(node("Left"));
    assert($("details").textContent.includes("Module description") &&
      $("details").querySelector(".formula")?.textContent === "y = x",
      `${kind}: module description and formula`);
    enter(node("Left"));
    assert(!$("status").classList.contains("error") && operation("a") && operation("c"),
      `${kind}: grouped operator navigation`);
    click(operation("c"));
    assert($("details").querySelector(".formula")?.textContent === "y = x",
      `${kind}: operation metadata retained`);
    $("home").click();
    assert(!$("status").classList.contains("error") && ["a", "b", "c"].every(operation),
      `${kind}: original dependency and access order`);
    document.querySelector('[data-language="en"]').click();
    assert(!$("status").classList.contains("error") && $("scope").value === "root",
      `${kind}: language switch preserves view`);
    document.querySelector('[data-language="zh"]').click();
    $("back").click();
    assert(!$("status").classList.contains("error") && operation("a") && operation("c"),
      `${kind}: navigation history`);

    result = await load(kind, { paths: [null, null, null] });
    assert(!result.error && ["a", "b", "c"].every(operation), `${kind}: unannotated compatibility`);
    result = await load(kind, { paths: ["graph/left", null, "graph/left"] });
    assert(!result.error && node("Left") && operation("b"), `${kind}: mixed annotations`);
    result = await load(kind, { paths: ["one/left", "two/right", "one/left"] });
    assert(!result.error && node("one") && node("two"), `${kind}: multiple path roots`);
    result = await load(kind, { paths: ["graph/parent", "graph/parent/child", "graph/other"] });
    assert(!result.error && Array.from($("scope").options).some(item => item.textContent.includes("child")),
      `${kind}: annotated parents with nested groups`);
    result = await load(kind, { ids: ["architecture:graph/left", "b", "c"] });
    assert(!result.error, `${kind}: generated ID collision safety`);
    result = await load(kind, { ids: ["operation:a", "b", "c"], paths: [null, null, null] });
    assert(!result.error && operation("operation:a"), `${kind}: node IDs remain unprefixed raw names`);
    result = await load(kind, { repeat: true });
    assert(!result.error && !$("collapse").disabled && $("collapse").checked,
      `${kind}: repeat-group metadata`);
    $("collapse").click();
    assert(!$("status").classList.contains("error") && ["a", "b", "c"].every(operation),
      `${kind}: repeat-group expansion`);
    result = await load(kind, { paths: ["graph//left", "graph/right", "graph/left"] });
    assert(result.error && result.status.includes("architecture.path"), `${kind}: invalid path rejected`);
    result = await load(kind, { backward: true });
    assert(result.error, `${kind}: cyclic execution dependencies rejected`);
    result = await load(kind, { call: true });
    assert(!result.error, `${kind}: graph-call grouping`);
    enter(node("Right"));
    assert(operation("b")?.getAttribute("aria-label") === "b: template", `${kind}: call target retained`);
    enter(operation("b"));
    assert($("scope").value === "template" && !$("status").classList.contains("error") && operation("body"),
      `${kind}: shared-template drill-down`);

    result = await load(kind, { parallel: true, paths: [null, null, null] });
    $("home").click();
    assert(!result.error && !$("layout-priority").disabled && $("layout-priority").value === "execution",
      `${kind}: default execution priority`);
    const executionPosition = position("b"), originalData = JSON.stringify(dataEdges());
    assert(position("a").y < executionPosition.y, `${kind}: execution dependencies determine ranks`);
    click(operation("a"));
    changePriority("logical");
    assert(!$("status").classList.contains("error") && position("a").y === position("b").y &&
      position("c").y > position("b").y, `${kind}: logical fork/join ranks`);
    assert(JSON.stringify(dataEdges()) === originalData &&
      !$("edges").querySelector("path[stroke-dasharray]"), `${kind}: data retained without control edges`);
    assert($("nodes").querySelectorAll(".selected").length === 1, `${kind}: selection survives mode switch`);
    document.querySelector('[data-language="en"]').click();
    assert($("layout-priority").value === "logical" &&
      $("layout-priority").selectedOptions[0].textContent === "Logic first",
      `${kind}: layout mode localization`);
    document.querySelector('[data-language="zh"]').click();
    changePriority("execution");
    assert(position("b").x === executionPosition.x && position("b").y === executionPosition.y &&
      $("edges").querySelector("path[stroke-dasharray]"), `${kind}: execution layout restored independently`);
    $("back").click();
    assert($("layout-priority").value === "logical" && position("a").y === position("b").y,
      `${kind}: history restores logical mode`);
    $("forward").click();
    assert($("layout-priority").value === "execution" && position("a").y < position("b").y,
      `${kind}: history restores execution mode`);
    result = await load(kind);
    changePriority("logical");
    assert(!result.error && !$("status").classList.contains("error") && node("Left") && node("Right"),
      `${kind}: logical mode handles cyclic display groups`);
    result = await load(kind, { crossed: true, priority: "logical" });
    $("home").click();
    assert(!result.error && position("a").y === position("b").y && position("c").y === position("d").y &&
      (position("a").x < position("b").x) === (position("d").x < position("c").x),
      `${kind}: logical rows eliminate crossed data connections`);
    result = await load(kind, { call: true, priority: "logical" });
    enter(node("Right"));
    enter(operation("b"));
    assert(!result.error && $("layout-priority").value === "logical" && $("scope").value === "template" &&
      !$("status").classList.contains("error") && operation("body"), `${kind}: logical shared-template navigation`);
    result = await load(kind, { backward: true, priority: "logical" });
    assert(result.error, `${kind}: hidden control dependencies are still validated`);
    result = await load(kind, { backward: true });
    assert(result.error && $("layout-priority").disabled, `${kind}: invalid graph disables layout switch`);
    assert($("load-progress").hidden, `${kind}: failed load clears progress`);
    result = await load(kind, { readWrite: true });
    $("home").click();
    click(operation("b"));
    assert(!result.error && $("details").textContent.includes("state") &&
      $("details").textContent.includes("ReadWrite") &&
      $("details").textContent.includes('readEdge="state-Read"') &&
      $("details").textContent.includes('writeEdge="state-Write"') &&
      $("details").textContent.includes('readOrder="13"') &&
      $("details").textContent.includes('writeOrder="14"'),
      `${kind}: one indexed ReadWrite binding retains both edge identities and orders`);
    if (kind === "ExecutionGraphProgram")
      assert($("details").textContent.includes("<SourceMap>") &&
        $("details").textContent.includes('LogicalNode id="b"'), `${kind}: source maps retained`);
    result = await load(kind, { call: true, instance: true });
    $("home").click();
    click(operation("c"));
    assert(!result.error && $("details").textContent.includes("b/output"),
      `${kind}: instance resource indexing`);
    result = await load(kind, { missingResource: true });
    assert(result.error && $("load-progress").hidden, `${kind}: missing indexed resource is reported`);
    result = await load(kind, { bulk: true });
    assert(!result.error && result.pulses > 2 && result.updates.some(item => item.value >= 64 && item.max > 64),
      `${kind}: batched conversion yields with measurable progress`);
    assert(result.phases.some(phase => phase.includes("\u6784\u5efa\u5c55\u793a\u6a21\u578b")) &&
      $("load-progress").hidden, `${kind}: progress phases and completion`);
    result = await load(kind, { plan: true });
    assert(!result.error && $("load-progress").hidden &&
      $("graph-kind").textContent.includes("IL"), `${kind}: async production IL compatibility`);
    document.querySelector('[data-language="en"]').click();
    result = await load(kind);
    assert(!result.error && result.phases.some(phase => phase.includes("Building the display model")) &&
      $("load-progress").getAttribute("aria-label") === "Loading progress",
      `${kind}: localized loading progress`);
    document.querySelector('[data-language="zh"]').click();

    const transfer = new DataTransfer();
    transfer.items.add(new File([fixture(kind, { bulk: true })], "superseded.xml",
      { type: "application/xml" }));
    $("file").files = transfer.files;
    $("file").dispatchEvent(new Event("change", { bubbles: true }));
    await new Promise(resolve => setTimeout(resolve, 10));
    result = await load(kind);
    await new Promise(resolve => setTimeout(resolve, 20));
    $("home").click();
    assert(!result.error && $("status").textContent.includes("regression.xml") &&
      !$("status").textContent.includes("superseded.xml") && $("load-progress").hidden &&
      ["a", "b", "c"].every(operation) && !operation("bulk.0"),
      `${kind}: newer file supersedes pending conversion`);

    result = await load(kind, { bulk: true, bulkCount: 640, chain: true });
    const renderPhases = [];
    let controlsStayedDisabled = true;
    const renderObserver = new MutationObserver(() => {
      renderPhases.push($("status").textContent);
      if (!$("load-progress").hidden)
        controlsStayedDisabled &&= ["home", "weights", "collapse", "states"].every(id => $(id).disabled);
    });
    renderObserver.observe($("status"), { childList: true });
    let renderTicks = 0;
    const renderTimer = setInterval(() => renderTicks++, 0);
    $("home").click();
    assert(!$("load-progress").hidden && $("home").disabled &&
      $("canvas").getAttribute("aria-busy") === "true", `${kind}: main switch reports busy immediately`);
    const waitForRender = async () => {
      const deadline = performance.now() + 10000;
      while (!$("load-progress").hidden) {
        if (performance.now() > deadline) throw new Error("Progressive rendering timed out.");
        await new Promise(resolve => setTimeout(resolve, 10));
      }
    };
    await waitForRender();
    renderObserver.disconnect();
    clearInterval(renderTimer);
    assert(!result.error && !$("status").classList.contains("error") && renderTicks > 5 && controlsStayedDisabled &&
      renderPhases.some(phase => phase.includes("\u521b\u5efa\u8282\u70b9")) &&
      renderPhases.some(phase => phase.includes("\u8ba1\u7b97\u8fde\u7ebf\u8def\u5f84")),
      `${kind}: large main view yields while creating and routing`);
    assert(operation("bulk.639") && !$("home").disabled && !$("canvas").hasAttribute("aria-busy") &&
      $("scope").value === "root", `${kind}: progressive main switch completes with controls restored`);
    $("architecture").click();
    await waitForRender();
    $("home").click();
    await new Promise(resolve => setTimeout(resolve, 10));
    result = await load(kind);
    await new Promise(resolve => setTimeout(resolve, 20));
    $("home").click();
    assert(!result.error && $("scope").value === "root" && operation("a") && !operation("bulk.0") &&
      $("load-progress").hidden && !$("canvas").hasAttribute("aria-busy"),
      `${kind}: file load cancels partial main rendering`);
  }
  return { passed: results.length, results };
};
