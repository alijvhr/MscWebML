const elements = {
  apiUrl: document.getElementById("apiUrl"),
  toast: document.getElementById("toast"),
  uploadForm: document.getElementById("uploadForm"),
  datasetFile: document.getElementById("datasetFile"),
  refreshDatasetsBtn: document.getElementById("refreshDatasetsBtn"),
  datasetSelect: document.getElementById("datasetSelect"),
  preprocessDatasetSelect: document.getElementById("preprocessDatasetSelect"),
  encodeDatasetSelect: document.getElementById("encodeDatasetSelect"),
  trainDatasetSelect: document.getElementById("trainDatasetSelect"),
  previewDatasetBtn: document.getElementById("previewDatasetBtn"),
  datasetPreview: document.getElementById("datasetPreview"),
  preprocessForm: document.getElementById("preprocessForm"),
  fillMissing: document.getElementById("fillMissing"),
  normalize: document.getElementById("normalize"),
  encodeForm: document.getElementById("encodeForm"),
  encodeMethod: document.getElementById("encodeMethod"),
  encodeTarget: document.getElementById("encodeTarget"),
  trainForm: document.getElementById("trainForm"),
  algorithm: document.getElementById("algorithm"),
  targetColumn: document.getElementById("targetColumn"),
  featureColumns: document.getElementById("featureColumns"),
  refreshModelsBtn: document.getElementById("refreshModelsBtn"),
  modelSelect: document.getElementById("modelSelect"),
  predictModelSelect: document.getElementById("predictModelSelect"),
  loadModelBtn: document.getElementById("loadModelBtn"),
  modelDetails: document.getElementById("modelDetails"),
  predictForm: document.getElementById("predictForm"),
  predictInput: document.getElementById("predictInput"),
  predictionOutput: document.getElementById("predictionOutput"),
};

let datasets = [];
let models = [];

function apiBase() {
  return elements.apiUrl.value.replace(/\/$/, "");
}

async function request(path, options = {}) {
  const response = await fetch(`${apiBase()}${path}`, options);
  const text = await response.text();
  let data = {};

  if (text) {
    data = JSON.parse(text);
  }

  if (!response.ok) {
    throw new Error(data.detail || "Request failed");
  }

  return data;
}

function showToast(message, type = "success") {
  elements.toast.textContent = message;
  elements.toast.className = `toast ${type}`;
  elements.toast.hidden = false;
  window.setTimeout(() => {
    elements.toast.hidden = true;
  }, 3500);
}

function selectedValue(select) {
  return select.value.trim();
}

function fillSelect(select, items, labelFn) {
  select.innerHTML = "";

  if (items.length === 0) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No items";
    select.appendChild(option);
    return;
  }

  items.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = labelFn(item);
    select.appendChild(option);
  });
}

function fillDatasetSelects() {
  const label = (dataset) => `${dataset.id} - ${dataset.original_name}`;
  fillSelect(elements.datasetSelect, datasets, label);
  fillSelect(elements.preprocessDatasetSelect, datasets, label);
  fillSelect(elements.encodeDatasetSelect, datasets, label);
  fillSelect(elements.trainDatasetSelect, datasets, label);
}

function fillModelSelects() {
  const label = (model) => `${model.id} - ${model.algorithm}`;
  fillSelect(elements.modelSelect, models, label);
  fillSelect(elements.predictModelSelect, models, label);
}

function parseColumns(value) {
  return value
    .split(",")
    .map((column) => column.trim())
    .filter(Boolean);
}

function jsonBody(data) {
  return {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  };
}

async function loadDatasets() {
  const data = await request("/datasets");
  datasets = data.datasets || [];
  fillDatasetSelects();
}

async function loadModels() {
  const data = await request("/models");
  models = data.models || [];
  fillModelSelects();
}

function renderTable(rows) {
  if (!rows || rows.length === 0) {
    return "<p>No rows to show.</p>";
  }

  const columns = Object.keys(rows[0]);
  const head = columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("");
  const body = rows
    .map((row) => {
      const cells = columns
        .map((column) => `<td>${escapeHtml(String(row[column] ?? ""))}</td>`)
        .join("");
      return `<tr>${cells}</tr>`;
    })
    .join("");

  return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

function renderModel(model) {
  const metrics = model.metrics || {};
  const metricNames = [
    ["score", "Score"],
    ["accuracy", "Accuracy"],
    ["precision", "Precision"],
    ["sensitivity", "Sensitivity"],
    ["specificity", "Specificity"],
    ["f1_score", "F1 score"],
  ];

  const metricCards = metricNames
    .map(([key, label]) => {
      const value = metrics[key] === null || metrics[key] === undefined ? "N/A" : Number(metrics[key]).toFixed(4);
      return `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`;
    })
    .join("");

  const matrix = metrics.confusion_matrix
    ? renderConfusionMatrix(metrics.confusion_matrix)
    : "<p>No confusion matrix for this model.</p>";

  elements.modelDetails.innerHTML = `
    <div class="metric-grid">${metricCards}</div>
    <div>
      <p><strong>Model:</strong> ${escapeHtml(model.id)}</p>
      <p><strong>Algorithm:</strong> ${escapeHtml(model.algorithm)}</p>
      <p><strong>Dataset:</strong> ${escapeHtml(model.dataset_id)}</p>
      <p><strong>Target:</strong> ${escapeHtml(model.target_column || "None")}</p>
    </div>
    <div>
      <h2>Confusion Matrix</h2>
      <div class="table-wrap">${matrix}</div>
    </div>
  `;
}

function renderConfusionMatrix(confusionMatrix) {
  const labels = confusionMatrix.labels || [];
  const matrix = confusionMatrix.matrix || [];
  const header = ["Actual / Predicted", ...labels]
    .map((label) => `<th>${escapeHtml(String(label))}</th>`)
    .join("");
  const body = matrix
    .map((row, index) => {
      const cells = row.map((value) => `<td>${escapeHtml(String(value))}</td>`).join("");
      return `<tr><th>${escapeHtml(String(labels[index]))}</th>${cells}</tr>`;
    })
    .join("");

  return `<table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>`;
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

elements.uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = elements.datasetFile.files[0];
  if (!file) {
    showToast("Choose a CSV file first.", "error");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  try {
    const data = await request("/datasets/upload", {
      method: "POST",
      body: formData,
    });
    showToast(`Uploaded dataset ${data.dataset_id}`);
    await loadDatasets();
  } catch (error) {
    showToast(error.message, "error");
  }
});

elements.refreshDatasetsBtn.addEventListener("click", async () => {
  try {
    await loadDatasets();
    showToast("Datasets refreshed.");
  } catch (error) {
    showToast(error.message, "error");
  }
});

elements.previewDatasetBtn.addEventListener("click", async () => {
  const datasetId = selectedValue(elements.datasetSelect);
  if (!datasetId) {
    showToast("Select a dataset first.", "error");
    return;
  }

  try {
    const data = await request(`/datasets/${datasetId}?rows=8`);
    elements.datasetPreview.innerHTML = renderTable(data.preview);
  } catch (error) {
    showToast(error.message, "error");
  }
});

elements.preprocessForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const datasetId = selectedValue(elements.preprocessDatasetSelect);

  try {
    const data = await request(
      "/preprocess",
      jsonBody({
        dataset_id: datasetId,
        fill_missing: elements.fillMissing.checked,
        normalize: elements.normalize.checked,
      })
    );
    showToast(`Created dataset ${data.new_dataset_id}`);
    await loadDatasets();
  } catch (error) {
    showToast(error.message, "error");
  }
});

elements.encodeForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const targetColumn = elements.encodeTarget.value.trim();

  try {
    const data = await request(
      "/encode",
      jsonBody({
        dataset_id: selectedValue(elements.encodeDatasetSelect),
        method: elements.encodeMethod.value,
        target_column: targetColumn || null,
      })
    );
    showToast(`Created dataset ${data.new_dataset_id}`);
    await loadDatasets();
  } catch (error) {
    showToast(error.message, "error");
  }
});

elements.trainForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const features = parseColumns(elements.featureColumns.value);
  const target = elements.targetColumn.value.trim();

  const body = {
    dataset_id: selectedValue(elements.trainDatasetSelect),
    algorithm: elements.algorithm.value,
    target_column: target || null,
    feature_columns: features.length ? features : null,
  };

  try {
    const data = await request("/train", jsonBody(body));
    showToast(`Trained model ${data.model_id}`);
    await loadModels();
    elements.modelSelect.value = data.model_id;
    renderModel(await request(`/models/${data.model_id}`));
  } catch (error) {
    showToast(error.message, "error");
  }
});

elements.refreshModelsBtn.addEventListener("click", async () => {
  try {
    await loadModels();
    showToast("Models refreshed.");
  } catch (error) {
    showToast(error.message, "error");
  }
});

elements.loadModelBtn.addEventListener("click", async () => {
  const modelId = selectedValue(elements.modelSelect);
  if (!modelId) {
    showToast("Select a model first.", "error");
    return;
  }

  try {
    const model = await request(`/models/${modelId}`);
    renderModel(model);
  } catch (error) {
    showToast(error.message, "error");
  }
});

elements.predictForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  try {
    const inputData = JSON.parse(elements.predictInput.value);
    const data = await request(
      "/predict",
      jsonBody({
        model_id: selectedValue(elements.predictModelSelect),
        input_data: inputData,
      })
    );
    elements.predictionOutput.textContent = JSON.stringify(data, null, 2);
    showToast("Prediction complete.");
  } catch (error) {
    showToast(error.message, "error");
  }
});

async function boot() {
  try {
    await loadDatasets();
    await loadModels();
  } catch (error) {
    showToast("Start the FastAPI server first.", "error");
  }
}

boot();
