document.addEventListener("DOMContentLoaded", function () {
  var root = document.getElementById("gpu-compare");
  if (!root) return;

  var slotsEl = root.querySelector(".gpu-compare-slots");
  var chartsEl = root.querySelector(".gpu-compare-charts");
  var resultEl = root.querySelector(".gpu-compare-result");

  var MAX_SLOTS = 8;

  var PALETTE = [
    { line: "rgba(99,166,255,0.9)",  fill: "rgba(99,166,255,0.15)" },   // blue
    { line: "rgba(255,121,98,0.9)",  fill: "rgba(255,121,98,0.15)" },   // coral
    { line: "rgba(78,205,171,0.9)",  fill: "rgba(78,205,171,0.15)" },   // teal
    { line: "rgba(255,193,74,0.9)",  fill: "rgba(255,193,74,0.15)" },   // amber
    { line: "rgba(187,128,255,0.9)", fill: "rgba(187,128,255,0.15)" },  // purple
    { line: "rgba(255,154,191,0.9)", fill: "rgba(255,154,191,0.15)" },  // pink
    { line: "rgba(128,222,234,0.9)", fill: "rgba(128,222,234,0.15)" },  // cyan
    { line: "rgba(220,185,120,0.9)", fill: "rgba(220,185,120,0.15)" },  // sand
  ];

  function getColors() {
    return PALETTE;
  }

  var specs = [
    { key: "architecture", label: "Architecture", type: "text" },
    { key: "bf16_tflops", label: "BF16", unit: " TFLOPS", type: "higher" },
    { key: "fp8_tflops", label: "FP8", unit: " TFLOPS", type: "higher" },
    { key: "memory_gb", label: "Memory", unit: " GB", suffix_key: "memory_type", type: "higher" },
    { key: "bandwidth_gbs", label: "Bandwidth", unit: " GB/s", type: "higher" },
    { key: "tdp_w", label: "TDP", unit: "W", type: "lower" },
    { key: "price_from", label: "Price", prefix: "$", unit: "/hr", suffix_key: "price_provider", type: "lower" },
    { key: "tflops_per_dollar", label: "TFLOPS/$", type: "higher" },
  ];

  // Specs used in the radar chart (numeric only, all "higher is better" after inversion)
  var radarSpecs = [
    { key: "bf16_tflops", label: "BF16" },
    { key: "fp8_tflops", label: "FP8" },
    { key: "memory_gb", label: "Memory" },
    { key: "bandwidth_gbs", label: "Bandwidth" },
    { key: "tflops_per_dollar", label: "TFLOPS/$" },
    { key: "tdp_w", label: "Efficiency", invert: true },
    { key: "price_from", label: "Value", invert: true },
  ];

  var accelerators = [];
  var slots = [null, null];
  var charts = {};

  // ── Data loading ──

  function loadData() {
    var scope = typeof __md_scope !== "undefined" ? __md_scope : new URL("../", location);
    var url = new URL("data/accelerators.json", scope);

    fetch(url)
      .then(function (r) { return r.json(); })
      .then(function (data) {
        accelerators = data;
        render();
      });
  }

  function grouped() {
    var groups = {};
    accelerators.forEach(function (a) {
      if (!groups[a.category]) groups[a.category] = [];
      groups[a.category].push(a);
    });
    return groups;
  }

  function findGpu(id) {
    for (var i = 0; i < accelerators.length; i++) {
      if (accelerators[i].id === id) return accelerators[i];
    }
    return null;
  }

  // ── Formatting ──

  function formatValue(spec, gpu) {
    var val = gpu[spec.key];
    if (val == null) return null;

    var str = "";
    if (spec.prefix) str += spec.prefix;
    if (typeof val === "number") {
      str += val % 1 === 0 ? val.toLocaleString() : val.toFixed(2);
    } else {
      str += val;
    }
    if (spec.unit) str += spec.unit;
    if (spec.suffix_key && gpu[spec.suffix_key]) {
      str += " " + gpu[spec.suffix_key];
    }
    return str;
  }

  function gridColor() {
    var scheme = document.body.getAttribute("data-md-color-scheme");
    return scheme === "default" ? "rgba(0,0,0,0.08)" : "rgba(255,255,255,0.08)";
  }

  function textColor() {
    var scheme = document.body.getAttribute("data-md-color-scheme");
    return scheme === "default" ? "rgba(0,0,0,0.6)" : "rgba(255,255,255,0.6)";
  }

  // ── Slots UI ──

  function createSelect(index) {
    var select = document.createElement("select");
    select.setAttribute("aria-label", "Select accelerator " + (index + 1));

    var placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Select accelerator\u2026";
    select.appendChild(placeholder);

    var groups = grouped();
    Object.keys(groups).forEach(function (cat) {
      var optgroup = document.createElement("optgroup");
      optgroup.label = cat;
      groups[cat].forEach(function (a) {
        var opt = document.createElement("option");
        opt.value = a.id;
        opt.textContent = a.name + " \u2014 " + a.memory_gb + " GB";
        select.appendChild(opt);
      });
      select.appendChild(optgroup);
    });

    if (slots[index]) select.value = slots[index];

    select.addEventListener("change", function () {
      slots[index] = select.value || null;
      update();
    });

    return select;
  }

  function render() {
    slotsEl.innerHTML = "";

    slots.forEach(function (_, i) {
      var wrapper = document.createElement("div");
      wrapper.className = "gpu-compare-slot";

      wrapper.appendChild(createSelect(i));

      if (slots.length > 1) {
        var removeBtn = document.createElement("button");
        removeBtn.className = "gpu-compare-remove";
        removeBtn.textContent = "\u00d7";
        removeBtn.title = "Remove";
        removeBtn.setAttribute("aria-label", "Remove slot " + (i + 1));
        removeBtn.addEventListener("click", function () {
          slots.splice(i, 1);
          if (slots.length === 0) slots.push(null);
          render();
        });
        wrapper.appendChild(removeBtn);
      }

      slotsEl.appendChild(wrapper);
    });

    if (slots.length < MAX_SLOTS) {
      var addBtn = document.createElement("button");
      addBtn.className = "gpu-compare-add";
      addBtn.textContent = "+";
      addBtn.title = "Add accelerator";
      addBtn.setAttribute("aria-label", "Add accelerator slot");
      addBtn.addEventListener("click", function () {
        slots.push(null);
        render();
      });
      slotsEl.appendChild(addBtn);
    }

    update();
  }

  function getSelected() {
    var selected = [];
    slots.forEach(function (id) {
      if (id) {
        var gpu = findGpu(id);
        if (gpu) selected.push(gpu);
      }
    });
    return selected;
  }

  function update() {
    var selected = getSelected();
    renderCharts(selected);
    renderResult(selected);
  }

  // ── Charts ──

  function destroyCharts() {
    Object.keys(charts).forEach(function (k) {
      if (charts[k]) { charts[k].destroy(); charts[k] = null; }
    });
  }

  function ensureCanvas(id, parentEl) {
    var existing = parentEl.querySelector("#" + id);
    if (existing) return existing;
    var canvas = document.createElement("canvas");
    canvas.id = id;
    var wrapper = document.createElement("div");
    wrapper.className = "gpu-chart-container gpu-chart-" + id;
    wrapper.appendChild(canvas);
    parentEl.appendChild(wrapper);
    return canvas;
  }

  function renderCharts(gpus) {
    destroyCharts();
    chartsEl.innerHTML = "";

    if (gpus.length < 2) return;

    // Layout: radar on the left, scatter on the right, bars and efficiency below
    var topRow = document.createElement("div");
    topRow.className = "gpu-charts-row";
    chartsEl.appendChild(topRow);

    var bottomRow = document.createElement("div");
    bottomRow.className = "gpu-charts-row";
    chartsEl.appendChild(bottomRow);

    renderRadar(gpus, topRow);
    renderScatter(gpus, topRow);
    renderBars(gpus, bottomRow);
    renderEfficiency(gpus, bottomRow);
  }

  // 1. Radar chart — normalized profile
  function renderRadar(gpus, parent) {
    var canvas = ensureCanvas("chart-radar", parent);
    var colors = getColors();

    // Find max for each radar spec across selected GPUs
    var maxVals = {};
    radarSpecs.forEach(function (s) {
      var max = 0;
      gpus.forEach(function (g) {
        var v = g[s.key];
        if (v != null && v > max) max = v;
      });
      maxVals[s.key] = max || 1;
    });

    var datasets = gpus.map(function (gpu, i) {
      var data = radarSpecs.map(function (s) {
        var v = gpu[s.key];
        if (v == null) return 0;
        var normalized = v / maxVals[s.key];
        // For "lower is better" specs, invert so bigger polygon = better
        return s.invert ? (1 - normalized) * 0.8 + 0.2 : normalized;
      });

      return {
        label: gpu.name,
        data: data,
        borderColor: colors[i % colors.length].line,
        backgroundColor: colors[i % colors.length].fill,
        borderWidth: 2,
        pointRadius: 3,
        pointBackgroundColor: colors[i % colors.length].line,
      };
    });

    charts.radar = new Chart(canvas, {
      type: "radar",
      data: {
        labels: radarSpecs.map(function (s) { return s.label; }),
        datasets: datasets,
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
          legend: {
            labels: { color: textColor(), font: { family: "Space Grotesk", size: 12 } },
          },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                return ctx.dataset.label + ": " + Math.round(ctx.raw * 100) + "%";
              },
            },
          },
        },
        scales: {
          r: {
            beginAtZero: true,
            max: 1,
            ticks: { display: false, stepSize: 0.25 },
            grid: { color: gridColor() },
            angleLines: { color: gridColor() },
            pointLabels: {
              color: textColor(),
              font: { family: "Space Grotesk", size: 11 },
            },
          },
        },
      },
    });
  }

  // 2. Scatter — Price vs BF16 Performance
  function renderScatter(gpus, parent) {
    var canvas = ensureCanvas("chart-scatter", parent);
    var colors = getColors();

    var datasets = gpus
      .filter(function (g) { return g.price_from != null; })
      .map(function (gpu, i) {
        return {
          label: gpu.name,
          data: [{ x: gpu.price_from, y: gpu.bf16_tflops }],
          borderColor: colors[i % colors.length].line,
          backgroundColor: colors[i % colors.length].line,
          pointRadius: 7,
          pointHoverRadius: 9,
        };
      });

    if (datasets.length < 2) return;

    charts.scatter = new Chart(canvas, {
      type: "scatter",
      data: { datasets: datasets },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
          legend: {
            labels: { color: textColor(), font: { family: "Space Grotesk", size: 12 } },
          },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                return ctx.dataset.label + ": $" + ctx.parsed.x.toFixed(2) + "/hr, " + ctx.parsed.y.toLocaleString() + " TFLOPS";
              },
            },
          },
        },
        scales: {
          x: {
            title: {
              display: true,
              text: "Price ($/hr)",
              color: textColor(),
              font: { family: "Space Grotesk", size: 12 },
            },
            grid: { color: gridColor() },
            ticks: {
              color: textColor(),
              font: { family: "Space Grotesk" },
              callback: function (v) { return "$" + v; },
            },
          },
          y: {
            title: {
              display: true,
              text: "BF16 TFLOPS",
              color: textColor(),
              font: { family: "Space Grotesk", size: 12 },
            },
            grid: { color: gridColor() },
            ticks: {
              color: textColor(),
              font: { family: "Space Grotesk" },
            },
          },
        },
      },
    });
  }

  // 3. Horizontal bar — specs side by side
  function renderBars(gpus, parent) {
    var canvas = ensureCanvas("chart-bars", parent);
    var colors = getColors();

    var barSpecs = [
      { key: "bf16_tflops", label: "BF16 (TFLOPS)" },
      { key: "memory_gb", label: "Memory (GB)" },
      { key: "bandwidth_gbs", label: "Bandwidth (GB/s)" },
    ];

    var labels = barSpecs.map(function (s) { return s.label; });

    var datasets = gpus.map(function (gpu, i) {
      return {
        label: gpu.name,
        data: barSpecs.map(function (s) {
          // Normalize each spec to 0-100% of max across selected GPUs
          var max = 0;
          gpus.forEach(function (g) {
            if (g[s.key] != null && g[s.key] > max) max = g[s.key];
          });
          var val = gpu[s.key];
          return val != null ? (val / (max || 1)) * 100 : 0;
        }),
        backgroundColor: colors[i % colors.length].line,
        borderRadius: 3,
        // Store raw values for tooltip
        raw: barSpecs.map(function (s) { return gpu[s.key]; }),
      };
    });

    charts.bars = new Chart(canvas, {
      type: "bar",
      data: { labels: labels, datasets: datasets },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
          legend: {
            labels: { color: textColor(), font: { family: "Space Grotesk", size: 12 } },
          },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                var raw = ctx.dataset.raw[ctx.dataIndex];
                return ctx.dataset.label + ": " + (raw != null ? raw.toLocaleString() : "N/A");
              },
            },
          },
        },
        scales: {
          x: {
            display: false,
            max: 110,
          },
          y: {
            grid: { display: false },
            ticks: {
              color: textColor(),
              font: { family: "Space Grotesk", size: 12 },
            },
          },
        },
      },
    });
  }

  // 4. Efficiency ranking — TFLOPS/$ bar
  function renderEfficiency(gpus, parent) {
    var withPrice = gpus.filter(function (g) { return g.tflops_per_dollar != null; });
    if (withPrice.length < 2) return;

    // Sort by efficiency descending
    withPrice.sort(function (a, b) { return b.tflops_per_dollar - a.tflops_per_dollar; });

    var canvas = ensureCanvas("chart-efficiency", parent);
    var colors = getColors();

    var max = withPrice[0].tflops_per_dollar;

    // Map back to original index for consistent coloring
    var gpuIndex = {};
    gpus.forEach(function (g, i) { gpuIndex[g.id] = i; });

    charts.efficiency = new Chart(canvas, {
      type: "bar",
      data: {
        labels: withPrice.map(function (g) { return g.name; }),
        datasets: [{
          data: withPrice.map(function (g) { return g.tflops_per_dollar; }),
          backgroundColor: withPrice.map(function (g) {
            return colors[gpuIndex[g.id] % colors.length].line;
          }),
          borderRadius: 3,
        }],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                return ctx.parsed.x.toLocaleString() + " TFLOPS/$";
              },
            },
          },
          title: {
            display: true,
            text: "TFLOPS / $ (higher is better)",
            color: textColor(),
            font: { family: "Space Grotesk", size: 13, weight: "normal" },
            align: "start",
            padding: { bottom: 12 },
          },
        },
        scales: {
          x: {
            grid: { color: gridColor() },
            ticks: {
              color: textColor(),
              font: { family: "Space Grotesk" },
            },
          },
          y: {
            grid: { display: false },
            ticks: {
              color: textColor(),
              font: { family: "Space Grotesk", size: 12 },
            },
          },
        },
      },
    });
  }

  // ── Table / Card ──

  function renderCard(gpu) {
    resultEl.innerHTML = "";

    var card = document.createElement("div");
    card.className = "gpu-compare-card";

    var title = document.createElement("h3");
    title.textContent = gpu.name;
    card.appendChild(title);

    specs.forEach(function (spec) {
      var formatted = formatValue(spec, gpu);
      if (formatted == null) return;

      var row = document.createElement("div");
      row.className = "gpu-compare-card-row";

      var label = document.createElement("span");
      label.className = "gpu-compare-card-label";
      label.textContent = spec.label;

      var value = document.createElement("span");
      value.className = "gpu-compare-card-value";
      value.textContent = formatted;

      row.appendChild(label);
      row.appendChild(value);
      card.appendChild(row);
    });

    resultEl.appendChild(card);
  }

  function renderTable(gpus) {
    resultEl.innerHTML = "";

    var wrap = document.createElement("div");
    wrap.className = "gpu-compare-table-wrap";

    var table = document.createElement("table");
    table.className = "gpu-compare-table";

    // Header
    var thead = document.createElement("thead");
    var headerRow = document.createElement("tr");
    headerRow.appendChild(document.createElement("th"));

    gpus.forEach(function (gpu) {
      var th = document.createElement("th");
      th.className = "gpu-name-header";
      th.textContent = gpu.name;
      headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Body
    var tbody = document.createElement("tbody");

    specs.forEach(function (spec) {
      var row = document.createElement("tr");

      var labelTd = document.createElement("td");
      labelTd.className = "gpu-compare-label";
      labelTd.textContent = spec.label;
      row.appendChild(labelTd);

      var bestIdx = -1;
      var bestVal = null;
      var allSame = true;
      var firstVal = null;

      gpus.forEach(function (gpu, idx) {
        var val = gpu[spec.key];
        if (val == null || spec.type === "text") return;
        if (firstVal === null) { firstVal = val; }
        else if (val !== firstVal) { allSame = false; }
        if (bestVal === null) { bestVal = val; bestIdx = idx; }
        else if (spec.type === "higher" && val > bestVal) { bestVal = val; bestIdx = idx; }
        else if (spec.type === "lower" && val < bestVal) { bestVal = val; bestIdx = idx; }
      });

      if (allSame) bestIdx = -1;

      gpus.forEach(function (gpu, idx) {
        var td = document.createElement("td");
        var formatted = formatValue(spec, gpu);

        if (formatted == null) {
          td.className = "gpu-compare-na";
          td.textContent = "\u2014";
        } else {
          td.textContent = formatted;
          if (idx === bestIdx) td.className = "gpu-compare-best";
        }

        row.appendChild(td);
      });

      tbody.appendChild(row);
    });

    table.appendChild(tbody);
    wrap.appendChild(table);
    resultEl.appendChild(wrap);
  }

  function renderResult(selected) {
    if (selected.length === 0) {
      chartsEl.innerHTML = "";
      resultEl.innerHTML = '<p class="gpu-compare-empty">Select accelerators above to compare specs.</p>';
    } else if (selected.length === 1) {
      chartsEl.innerHTML = "";
      renderCard(selected[0]);
    } else {
      renderTable(selected);
    }
  }

  // ── Theme change observer ──

  var observer = new MutationObserver(function () {
    var selected = getSelected();
    if (selected.length >= 2) {
      renderCharts(selected);
    }
  });
  observer.observe(document.body, { attributes: true, attributeFilter: ["data-md-color-scheme"] });

  loadData();
});
