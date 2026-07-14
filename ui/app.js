const getElem = id => document.getElementById(id);

let datasets = [];
let models = [];

function apiBase() {
  return getElem("api-url").value.replace(/\/$/, "");
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
  getElem("toast").textContent = message;
  getElem("toast").className = `toast ${type}`;
  getElem("toast").hidden = false;
  window.setTimeout(() => {
    getElem("toast").hidden = true;
  }, 3500);
}

function selectedValue(select) {
  return select.value.trim();
}

function fillSelect(select, items, labelFn) {
  select.replaceChildren();

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
  const elems = [
    "dataset-select",
    "preprocess-dataset-select",
    "encode-dataset-select",
    "train-dataset-select",
  ];
  for (const id of elems) {
    fillSelect(getElem(id), datasets, label);
  }
}

function fillModelSelects() {
  const label = (model) => `${model.id} - ${model.algorithm}`;
  fillSelect(getElem("model-select"), models, label);
  fillSelect(getElem("predict-model-select"), models, label);
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

function showTable(tableContainer, emptyMessage, rows) {
  tableContainer.replaceChildren();

  if (!rows || rows.length === 0) {
    tableContainer.hidden = true;
    emptyMessage.hidden = false;
    return;
  }

  tableContainer.appendChild(createTable(rows));
  tableContainer.hidden = false;
  emptyMessage.hidden = true;
}

function createTable(rows, rowHeaderLabels = null, cornerLabel = null) {
  const columns = Object.keys(rows[0]);
  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const tbody = document.createElement("tbody");
  const headerRow = document.createElement("tr");

  if (cornerLabel !== null) {
    headerRow.appendChild(createCell("th", cornerLabel));
  }

  columns.forEach((column) => {
    headerRow.appendChild(createCell("th", column));
  });

  rows.forEach((row, index) => {
    const tableRow = document.createElement("tr");

    if (rowHeaderLabels) {
      tableRow.appendChild(createCell("th", rowHeaderLabels[index]));
    }

    columns.forEach((column) => {
      tableRow.appendChild(createCell("td", row[column] ?? ""));
    });

    tbody.appendChild(tableRow);
  });

  thead.appendChild(headerRow);
  table.append(thead, tbody);
  return table;
}

function createCell(tagName, value) {
  const cell = document.createElement(tagName);
  cell.textContent = String(value);
  return cell;
}

function renderModel(model) {
  const metrics = model.metrics || {};
  document.querySelectorAll("[data-metric]").forEach((metricElem) => {
    const key = metricElem.dataset.metric;
    const value = metrics[key];
    metricElem.textContent = value === null || value === undefined ? "N/A" : Number(value).toFixed(4);
  });

  getElem("model-id").textContent = model.id;
  getElem("model-algorithm").textContent = model.algorithm;
  getElem("model-dataset-id").textContent = model.dataset_id;
  getElem("model-target-column").textContent = model.target_column || "None";

  renderConfusionMatrix(metrics.confusion_matrix);
  getElem("model-details").hidden = false;
}

function renderConfusionMatrix(confusionMatrix) {
  const container = getElem("confusion-matrix");
  const emptyMessage = getElem("confusion-matrix-empty");
  container.replaceChildren();

  if (!confusionMatrix) {
    container.hidden = true;
    emptyMessage.hidden = false;
    return;
  }

  const labels = confusionMatrix.labels || [];
  const matrix = confusionMatrix.matrix || [];
  if (labels.length === 0 || matrix.length === 0) {
    container.hidden = true;
    emptyMessage.hidden = false;
    return;
  }

  const rows = matrix.map((row) => Object.fromEntries(row.map((value, index) => [labels[index], value])));

  container.appendChild(createTable(rows, labels, "Actual / Predicted"));
  container.hidden = false;
  emptyMessage.hidden = true;
}

getElem("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = getElem("dataset-file").files[0];
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

getElem("uci-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const uciId = getElem("uci-id").value.trim();
  const name = getElem("uci-name").value.trim();

  if (!uciId && !name) {
    showToast("Enter a UCI dataset ID or name.", "error");
    return;
  }

  try {
    const data = await request(
      "/datasets/uci",
      jsonBody({
        uci_id: uciId ? Number(uciId) : null,
        name: name || null,
      })
    );
    showToast(`Imported UCI dataset ${data.dataset_id}`);
    await loadDatasets();
    getElem("dataset-select").value = data.dataset_id;
  } catch (error) {
    showToast(error.message, "error");
  }
});

getElem("refresh-datasets-btn").addEventListener("click", async () => {
  try {
    await loadDatasets();
    showToast("Datasets refreshed.");
  } catch (error) {
    showToast(error.message, "error");
  }
});

getElem("preview-dataset-btn").addEventListener("click", async () => {
  const datasetId = selectedValue(getElem("dataset-select"));
  if (!datasetId) {
    showToast("Select a dataset first.", "error");
    return;
  }

  try {
    const data = await request(`/datasets/${datasetId}?rows=8`);
    showTable(getElem("dataset-preview"), getElem("dataset-preview-empty"), data.preview);
  } catch (error) {
    showToast(error.message, "error");
  }
});

getElem("preprocess-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const datasetId = selectedValue(getElem("preprocess-dataset-select"));

  try {
    const data = await request(
      "/preprocess",
      jsonBody({
        dataset_id: datasetId,
        fill_missing: getElem("fill-missing").checked,
        normalize: getElem("normalize").checked,
      })
    );
    showToast(`Created dataset ${data.new_dataset_id}`);
    await loadDatasets();
  } catch (error) {
    showToast(error.message, "error");
  }
});

getElem("encode-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const targetColumn = getElem("encode-target").value.trim();

  try {
    const data = await request(
      "/encode",
      jsonBody({
        dataset_id: selectedValue(getElem("encode-dataset-select")),
        method: getElem("encode-method").value,
        target_column: targetColumn || null,
      })
    );
    showToast(`Created dataset ${data.new_dataset_id}`);
    await loadDatasets();
  } catch (error) {
    showToast(error.message, "error");
  }
});

getElem("train-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const features = parseColumns(getElem("feature-columns").value);
  const target = getElem("target-column").value.trim();

  const body = {
    dataset_id: selectedValue(getElem("train-dataset-select")),
    algorithm: getElem("algorithm").value,
    target_column: target || null,
    feature_columns: features.length ? features : null,
  };

  try {
    const data = await request("/train", jsonBody(body));
    showToast(`Trained model ${data.model_id}`);
    await loadModels();
    getElem("model-select").value = data.model_id;
    renderModel(await request(`/models/${data.model_id}`));
  } catch (error) {
    showToast(error.message, "error");
  }
});

getElem("refresh-models-btn").addEventListener("click", async () => {
  try {
    await loadModels();
    showToast("Models refreshed.");
  } catch (error) {
    showToast(error.message, "error");
  }
});

getElem("load-model-btn").addEventListener("click", async () => {
  const modelId = selectedValue(getElem("model-select"));
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

getElem("predict-form").addEventListener("submit", async (event) => {
  event.preventDefault();

  try {
    const inputData = JSON.parse(getElem("predict-input").value);
    const data = await request(
      "/predict",
      jsonBody({
        model_id: selectedValue(getElem("predict-model-select")),
        input_data: inputData,
      })
    );
    getElem("prediction-output").textContent = JSON.stringify(data, null, 2);
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
