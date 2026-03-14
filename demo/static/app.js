const $ = (id) => document.getElementById(id);

const MODE_LABELS = {
  voice_clone: "Voice Clone",
  custom: "CustomVoice",
  voice_design: "Voice Design",
};

const THEMES = {
  foundry: {
    label: "Foundry",
  },
  midnight: {
    label: "Midnight",
  },
  coolant: {
    label: "Coolant",
  },
  ember: {
    label: "Ember",
  },
};

const PREFS_KEY = "faster-qwen3-tts-console-prefs-v2";
const HISTORY_KEY = "faster-qwen3-tts-console-history-v2";

const state = {
  mode: "voice_clone",
  streamMode: "stream",
  xvecOnly: true,
  theme: "foundry",
  status: null,
  selectedModel: "",
  loadingModel: false,
  generating: false,
  abortController: null,
  recentJobs: [],
  activity: [],
  refFile: null,
  transcriptionRefFile: null,
  presetRefId: "",
  referencePreviewUrl: "",
  referencePreviewOwned: false,
  outputUrl: "",
  outputBlob: null,
  clientT0: null,
  firstChunkAt: null,
  firstAudioAt: null,
  lastBufferSeconds: 0,
  rawPcmSr: 24000,
  pcmQueue: [],
  rawPcmParts: [],
  chunkChain: Promise.resolve(),
  audioContext: null,
  scriptProcessor: null,
  currentRun: null,
  recorder: {
    active: false,
    stream: null,
    context: null,
    source: null,
    analyser: null,
    processor: null,
    gain: null,
    chunks: [],
    meterBuffer: null,
    meterRaf: null,
    startedAt: 0,
  },
};

document.addEventListener("DOMContentLoaded", init);
window.addEventListener("beforeunload", () => {
  if (state.referencePreviewUrl && state.referencePreviewOwned) {
    URL.revokeObjectURL(state.referencePreviewUrl);
  }
  if (state.outputUrl) {
    URL.revokeObjectURL(state.outputUrl);
  }
  if (state.abortController) {
    state.abortController.abort();
  }
  cleanupRecorder();
});

function init() {
  loadPreferences();
  state.recentJobs = loadRecentJobs();

  applyTheme();
  bindEvents();
  seedDefaults();
  renderHistory();
  logActivity("Console ready. Waiting for service status.", "success");
  refreshStatus({ initial: true });
  setInterval(() => refreshStatus({ quiet: true }), 15000);
  setInterval(updateUptimeDisplay, 1000);
}

function bindEvents() {
  $("refreshStatusBtn").addEventListener("click", () => refreshStatus());
  $("loadModelBtn").addEventListener("click", () => loadSelectedModel());
  $("recommendModelBtn").addEventListener("click", useRecommendedModel);
  $("generateBtn").addEventListener("click", generateSpeech);
  $("cancelBtn").addEventListener("click", cancelRun);
  $("downloadBtn").addEventListener("click", downloadOutput);
  $("refFileInput").addEventListener("change", onReferenceFileSelected);
  $("recordBtn").addEventListener("click", toggleRecording);
  $("transcribeBtn").addEventListener("click", transcribeReference);
  $("modelSelect").addEventListener("change", () => {
    state.selectedModel = $("modelSelect").value;
    updateModelSummary();
    renderSpeakerOptions();
    savePreferences();
  });
  $("textInput").addEventListener("input", () => {
    updateTextCounter();
    savePreferences();
  });
  $("refText").addEventListener("input", savePreferences);
  $("customInstruct").addEventListener("input", savePreferences);
  $("designInstruct").addEventListener("input", savePreferences);
  $("speakerSelect").addEventListener("change", savePreferences);
  $("languageSelect").addEventListener("change", savePreferences);

  bindRange("chunkSizeRange", "chunkSizeValue", (value) => `${value} steps`);
  bindRange("temperatureRange", "temperatureValue", (value) => Number(value).toFixed(2));
  bindRange("topKRange", "topKValue", (value) => value);
  bindRange("repetitionRange", "repetitionValue", (value) => Number(value).toFixed(2));

  document.querySelectorAll(".mode-card").forEach((button) => {
    button.addEventListener("click", () => setMode(button.dataset.mode));
  });

  document.querySelectorAll(".toggle-card").forEach((button) => {
    button.addEventListener("click", () => {
      setXvecMode(button.dataset.xvec === "true");
    });
  });

  document.querySelectorAll("[data-stream-mode]").forEach((button) => {
    button.addEventListener("click", () => {
      state.streamMode = button.dataset.streamMode;
      renderStreamMode();
      savePreferences();
    });
  });

  document.querySelectorAll("[data-theme-option]").forEach((button) => {
    button.addEventListener("click", () => {
      setTheme(button.dataset.themeOption);
    });
  });
}

function seedDefaults() {
  setMode(state.mode);
  setXvecMode(state.xvecOnly);
  renderStreamMode();
  updateTextCounter();
}

function bindRange(inputId, valueId, formatter) {
  const input = $(inputId);
  const output = $(valueId);
  const sync = () => {
    output.textContent = formatter(input.value);
    savePreferences();
  };
  input.addEventListener("input", sync);
  sync();
}

function loadPreferences() {
  try {
    const prefs = JSON.parse(localStorage.getItem(PREFS_KEY) || "{}");
    state.mode = prefs.mode || state.mode;
    state.streamMode = prefs.streamMode || state.streamMode;
    state.xvecOnly = prefs.xvecOnly ?? state.xvecOnly;
    state.theme = normalizeTheme(prefs.theme || state.theme);
    state.selectedModel = prefs.selectedModel || "";

    $("textInput").value = prefs.textInput || "";
    $("refText").value = prefs.refText || "";
    $("customInstruct").value = prefs.customInstruct || "";
    $("designInstruct").value = prefs.designInstruct || "";
    $("languageSelect").value = prefs.language || "English";
    $("chunkSizeRange").value = prefs.chunkSize || "8";
    $("temperatureRange").value = prefs.temperature || "0.9";
    $("topKRange").value = prefs.topK || "50";
    $("repetitionRange").value = prefs.repetitionPenalty || "1.05";
  } catch (error) {
    console.warn("Failed to load preferences", error);
  }
}

function savePreferences() {
  const prefs = {
    mode: state.mode,
    streamMode: state.streamMode,
    xvecOnly: state.xvecOnly,
    theme: state.theme,
    selectedModel: state.selectedModel,
    textInput: $("textInput").value,
    refText: $("refText").value,
    customInstruct: $("customInstruct").value,
    designInstruct: $("designInstruct").value,
    language: $("languageSelect").value,
    chunkSize: $("chunkSizeRange").value,
    temperature: $("temperatureRange").value,
    topK: $("topKRange").value,
    repetitionPenalty: $("repetitionRange").value,
    speaker: $("speakerSelect").value,
  };
  localStorage.setItem(PREFS_KEY, JSON.stringify(prefs));
}

function loadRecentJobs() {
  try {
    const jobs = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
    return Array.isArray(jobs) ? jobs : [];
  } catch (error) {
    console.warn("Failed to load job history", error);
    return [];
  }
}

function saveRecentJobs() {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(state.recentJobs.slice(0, 8)));
}

function normalizeTheme(theme) {
  if (theme === "light") {
    return "foundry";
  }
  if (theme === "dark") {
    return "midnight";
  }
  return Object.hasOwn(THEMES, theme) ? theme : "foundry";
}

function setTheme(theme) {
  const nextTheme = normalizeTheme(theme);
  if (state.theme === nextTheme) {
    return;
  }
  state.theme = nextTheme;
  applyTheme();
  savePreferences();
  logActivity(`Theme changed to ${THEMES[nextTheme].label}.`);
}

function applyTheme() {
  document.documentElement.dataset.theme = state.theme;
  document.querySelectorAll("[data-theme-option]").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.themeOption === state.theme);
  });
}

function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll(".mode-card").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.mode === mode);
  });
  $("cloneSection").classList.toggle("is-hidden", mode !== "voice_clone");
  $("customSection").classList.toggle("is-hidden", mode !== "custom");
  $("designSection").classList.toggle("is-hidden", mode !== "voice_design");
  updateModelSummary();
  renderSpeakerOptions();
  updateActionButtonCopy();
  savePreferences();
}

function setXvecMode(enabled) {
  state.xvecOnly = enabled;
  $("xvecButton").classList.toggle("is-active", enabled);
  $("iclButton").classList.toggle("is-active", !enabled);
  savePreferences();
}

function renderStreamMode() {
  document.querySelectorAll("[data-stream-mode]").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.streamMode === state.streamMode);
  });
  updateActionButtonCopy();
}

function updateActionButtonCopy() {
  $("generateBtn").textContent = state.streamMode === "stream" ? "Generate speech" : "Render batch audio";
}

function updateTextCounter() {
  const length = $("textInput").value.length;
  $("textCounter").textContent = `${length} chars`;
}

async function refreshStatus({ quiet = false, initial = false } = {}) {
  try {
    const response = await fetch("/status");
    if (!response.ok) {
      throw new Error(await extractError(response));
    }
    state.status = await response.json();
    renderStatus();
    renderModelOptions();
    renderPresetButtons();
    renderSpeakerOptions();
    if (initial) {
      showMessage("success", "Connected to the service. Configure a request and start generating.");
    }
  } catch (error) {
    updateServicePill("error", "Service unavailable");
    if (!quiet) {
      showMessage("error", error.message || "Failed to fetch service status.");
    }
  }
}

function renderStatus() {
  const status = state.status;
  if (!status) {
    return;
  }

  const activeModel = status.model ? shortModelName(status.model) : "None loaded";
  $("heroActiveModel").textContent = activeModel;
  $("heroQueueDepth").textContent = String(status.queue_depth || 0);
  $("heroCachedModels").textContent = String((status.cached_models || []).length);
  $("heroTranscription").textContent = transcriptionStateLabel(status);

  $("activeModelValue").textContent = activeModel;
  $("activeModelMeta").textContent = status.model_type ? describeModelType(status.model_type) : "No active checkpoint";
  $("queueDepthValue").textContent = String(status.queue_depth || 0);
  $("cachedModelsValue").textContent = String((status.cached_models || []).length);
  $("cachedModelsMeta").textContent = (status.cached_models || []).length
    ? status.cached_models.map(shortModelName).join(", ")
    : "No cached models";
  $("serviceVersionValue").textContent = status.service_version ? `v${status.service_version}` : "Unknown";
  $("transcriptionValue").textContent = transcriptionStateLabel(status);
  $("transcriptionMeta").textContent = transcriptionMeta(status);
  $("limitsValue").textContent = `${status.max_text_chars || "?"} chars`;
  $("limitsMeta").textContent = `${status.max_audio_megabytes || "?"} MB ref audio, cache size ${status.model_cache_size || "?"}`;
  updateUptimeDisplay();

  const pillState = state.loadingModel || status.loading
    ? ["busy", "Loading model"]
    : state.generating
      ? ["busy", "Generating"]
      : ["ready", status.loaded ? "Ready" : "No model loaded"];
  updateServicePill(pillState[0], pillState[1]);

  updateModelSummary();
}

function renderModelOptions() {
  const status = state.status;
  if (!status) {
    return;
  }

  const select = $("modelSelect");
  const available = status.available_models || [];
  const currentValue = state.selectedModel || status.model || recommendedModelForMode(state.mode, available);

  select.innerHTML = "";
  available.forEach((model) => {
    const option = document.createElement("option");
    option.value = model;
    option.textContent = shortModelName(model);
    select.appendChild(option);
  });

  state.selectedModel = available.includes(currentValue) ? currentValue : (available[0] || "");
  select.value = state.selectedModel;
  $("recommendModelBtn").textContent = `Use ${shortModelName(recommendedModelForMode(state.mode, available) || state.selectedModel || "recommended")}`;
}

function renderPresetButtons() {
  const grid = $("presetGrid");
  grid.innerHTML = "";
  const presets = state.status?.preset_refs || [];

  if (!presets.length) {
    const empty = document.createElement("span");
    empty.className = "inline-status";
    empty.textContent = "No preset references available.";
    grid.appendChild(empty);
    return;
  }

  presets.forEach((preset) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "preset-btn";
    button.textContent = preset.label;
    button.classList.toggle("is-active", preset.id === state.presetRefId);
    button.addEventListener("click", () => {
      if (state.presetRefId === preset.id) {
        resetReferenceSelection();
        logActivity(`Preset ${preset.label} cleared.`, "warning");
        return;
      }
      loadPresetReference(preset.id);
    });
    grid.appendChild(button);
  });
}

function renderSpeakerOptions() {
  const select = $("speakerSelect");
  select.innerHTML = "";

  if (state.mode !== "custom") {
    return;
  }

  const selectedModelKind = modelKind(state.selectedModel);
  const activeMatches = state.status?.model === state.selectedModel;
  const speakers = activeMatches ? state.status?.speakers || [] : [];

  if (selectedModelKind !== "custom") {
    appendSpeakerOption(select, "", "Select a CustomVoice model first");
    select.value = "";
    return;
  }

  if (!activeMatches) {
    appendSpeakerOption(select, "", "Load the selected CustomVoice model to list speakers");
    select.value = "";
    return;
  }

  if (!speakers.length) {
    appendSpeakerOption(select, "", "No speakers reported by the active model");
    select.value = "";
    return;
  }

  appendSpeakerOption(select, "", "Choose a speaker");
  speakers.forEach((speaker) => appendSpeakerOption(select, speaker, speaker));

  const prefs = loadPreferenceSnapshot();
  if (prefs.speaker && speakers.includes(prefs.speaker)) {
    select.value = prefs.speaker;
  } else if (select.options.length > 1) {
    select.selectedIndex = 1;
  }
}

function appendSpeakerOption(select, value, label) {
  const option = document.createElement("option");
  option.value = value;
  option.textContent = label;
  select.appendChild(option);
}

function loadPreferenceSnapshot() {
  try {
    return JSON.parse(localStorage.getItem(PREFS_KEY) || "{}");
  } catch {
    return {};
  }
}

function recommendedModelForMode(mode, availableModels) {
  const preferred = {
    voice_clone: [
      "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
      "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    ],
    custom: [
      "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
      "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    ],
    voice_design: [
      "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    ],
  }[mode] || [];

  for (const model of preferred) {
    if (availableModels.includes(model)) {
      return model;
    }
  }

  return availableModels.find((model) => modelKind(model) === mode) || availableModels[0] || "";
}

function useRecommendedModel() {
  const recommended = recommendedModelForMode(state.mode, state.status?.available_models || []);
  if (!recommended) {
    showMessage("warning", "No compatible model is available for the current mode.");
    return;
  }

  state.selectedModel = recommended;
  $("modelSelect").value = recommended;
  updateModelSummary();
  renderSpeakerOptions();
  savePreferences();
}

function updateModelSummary() {
  const summary = $("modelSummaryText");
  const chip = $("modeCompatibility");

  if (!state.selectedModel) {
    summary.textContent = "No model selected.";
    chip.textContent = "Select a model";
    chip.className = "compat-chip";
    return;
  }

  const compatible = modelKind(state.selectedModel) === state.mode;
  const active = state.status?.model;
  const activeText = active ? `Active: ${shortModelName(active)}.` : "No model active.";
  const autoLoadText = active === state.selectedModel
    ? "This checkpoint is already active."
    : "The app will load this checkpoint automatically before generation.";

  summary.textContent = compatible
    ? `${shortModelName(state.selectedModel)} matches ${MODE_LABELS[state.mode]}. ${activeText} ${autoLoadText}`
    : `${shortModelName(state.selectedModel)} does not support ${MODE_LABELS[state.mode]}. Switch to a compatible model before generating.`;

  chip.className = `compat-chip ${compatible ? "is-good" : "is-warn"}`;
  chip.textContent = compatible ? "Model compatible" : "Model mismatch";
}

async function loadSelectedModel({ silent = false } = {}) {
  if (!state.selectedModel) {
    showMessage("warning", "Choose a model before loading.");
    return false;
  }
  if (state.loadingModel) {
    return false;
  }

  state.loadingModel = true;
  updateControls();
  updateServicePill("busy", "Loading model");
  logActivity(`Loading ${shortModelName(state.selectedModel)}...`);

  try {
    const formData = new FormData();
    formData.append("model_id", state.selectedModel);
    const response = await fetch("/load", {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      throw new Error(await extractError(response));
    }
    await response.json();
    await refreshStatus({ quiet: true });
    renderSpeakerOptions();
    logActivity(`Model ready: ${shortModelName(state.selectedModel)}.`, "success");
    if (!silent) {
      showMessage("success", `Loaded ${shortModelName(state.selectedModel)}.`);
    }
    return true;
  } catch (error) {
    logActivity(`Model load failed: ${error.message}`, "error");
    showMessage("error", error.message || "Failed to load model.");
    return false;
  } finally {
    state.loadingModel = false;
    updateControls();
    renderStatus();
  }
}

function updateControls() {
  const busy = state.loadingModel || state.generating;
  $("generateBtn").disabled = busy;
  $("loadModelBtn").disabled = busy;
  $("recommendModelBtn").disabled = busy;
  $("refreshStatusBtn").disabled = busy;
  $("cancelBtn").disabled = !state.generating;
  $("downloadBtn").disabled = !state.outputBlob;
  $("recordBtn").disabled = state.loadingModel || state.generating;
  $("transcribeBtn").disabled = state.loadingModel || state.generating;

  [
    "modelSelect",
    "textInput",
    "refFileInput",
    "refText",
    "speakerSelect",
    "customInstruct",
    "designInstruct",
    "languageSelect",
    "chunkSizeRange",
    "temperatureRange",
    "topKRange",
    "repetitionRange",
  ].forEach((id) => {
    const element = $(id);
    if (element) {
      element.disabled = busy;
    }
  });

  document.querySelectorAll(".mode-card, .toggle-card, [data-stream-mode], .preset-btn").forEach((button) => {
    button.disabled = busy;
  });
}

async function onReferenceFileSelected(event) {
  const [file] = event.target.files || [];
  if (!file) {
    return;
  }
  clearPresetSelection();
  setReferenceFile(file, file.name, "Uploaded reference clip");
  event.target.value = "";
  logActivity(`Reference loaded from file: ${file.name}.`, "success");
}

function setReferenceFile(file, name, meta) {
  state.refFile = file;
  state.transcriptionRefFile = file;
  const url = URL.createObjectURL(file);
  setReferencePreview(url, name, meta, true);
  $("referenceSourceLabel").textContent = meta;
}

function setReferencePreview(url, name, meta, owned) {
  if (state.referencePreviewUrl && state.referencePreviewOwned) {
    URL.revokeObjectURL(state.referencePreviewUrl);
  }
  state.referencePreviewUrl = url;
  state.referencePreviewOwned = owned;

  $("referencePreview").src = url;
  $("referenceName").textContent = name;
  $("referenceMeta").textContent = meta;
  $("referencePreviewWrap").classList.remove("is-hidden");
}

function clearPresetSelection() {
  state.presetRefId = "";
  renderPresetButtons();
  savePreferences();
}

function resetReferenceSelection() {
  if (state.referencePreviewUrl && state.referencePreviewOwned) {
    URL.revokeObjectURL(state.referencePreviewUrl);
  }
  state.refFile = null;
  state.transcriptionRefFile = null;
  state.presetRefId = "";
  state.referencePreviewUrl = "";
  state.referencePreviewOwned = false;
  $("refFileInput").value = "";
  $("referencePreview").pause();
  $("referencePreview").removeAttribute("src");
  $("referencePreview").load();
  $("referencePreviewWrap").classList.add("is-hidden");
  $("referenceName").textContent = "No file selected";
  $("referenceMeta").textContent = "Attach a short WAV reference to enable voice cloning.";
  $("referenceSourceLabel").textContent = "No reference loaded";
  renderPresetButtons();
  savePreferences();
}

async function loadPresetReference(presetId) {
  try {
    const response = await fetch(`/preset_ref/${encodeURIComponent(presetId)}`);
    if (!response.ok) {
      throw new Error(await extractError(response));
    }
    const preset = await response.json();
    const bytes = base64ToBytes(preset.audio_b64);
    const blob = new Blob([bytes], { type: "audio/wav" });
    const file = new File([blob], preset.filename || `${preset.id}.wav`, { type: "audio/wav" });

    state.presetRefId = preset.id;
    state.refFile = null;
    state.transcriptionRefFile = file;
    $("refFileInput").value = "";
    if (!$("refText").value.trim() || preset.ref_text) {
      $("refText").value = preset.ref_text || "";
    }
    setReferencePreview(URL.createObjectURL(blob), preset.label, "Preset reference voice", true);
    $("referenceSourceLabel").textContent = `Preset: ${preset.label}`;
    renderPresetButtons();
    savePreferences();
    logActivity(`Preset reference selected: ${preset.label}.`, "success");
  } catch (error) {
    showMessage("error", error.message || "Failed to load preset reference.");
  }
}

async function transcribeReference() {
  const sourceFile = state.refFile || state.transcriptionRefFile;
  if (!sourceFile) {
    showMessage("warning", "Attach, record, or select a reference clip before requesting transcription.");
    return;
  }

  $("transcribeBtn").disabled = true;
  const originalLabel = $("transcribeBtn").textContent;
  $("transcribeBtn").textContent = "Transcribing...";
  logActivity("Running reference transcription...");

  try {
    const formData = new FormData();
    formData.append("audio", sourceFile);
    const response = await fetch("/transcribe", {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      throw new Error(await extractError(response));
    }
    const payload = await response.json();
    $("refText").value = payload.text || "";
    savePreferences();
    await refreshStatus({ quiet: true });
    showMessage("success", "Reference transcript updated.");
    logActivity("Reference transcription completed.", "success");
  } catch (error) {
    showMessage("error", error.message || "Reference transcription failed.");
    logActivity(`Transcription failed: ${error.message}`, "error");
  } finally {
    $("transcribeBtn").disabled = state.loadingModel || state.generating;
    $("transcribeBtn").textContent = originalLabel;
  }
}

async function toggleRecording() {
  if (state.recorder.active) {
    await stopRecording();
  } else {
    await startRecording();
  }
}

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });

    const context = new (window.AudioContext || window.webkitAudioContext)({
      latencyHint: "interactive",
    });
    if (context.state === "suspended") {
      await context.resume();
    }

    const source = context.createMediaStreamSource(stream);
    const analyser = context.createAnalyser();
    analyser.fftSize = 2048;
    const gain = context.createGain();
    gain.gain.value = 0.0001;
    const processor = context.createScriptProcessor(4096, 1, 1);
    const chunks = [];

    processor.onaudioprocess = (event) => {
      if (!state.recorder.active) {
        return;
      }
      const input = event.inputBuffer.getChannelData(0);
      chunks.push(new Float32Array(input));
    };

    source.connect(analyser);
    analyser.connect(processor);
    processor.connect(gain);
    gain.connect(context.destination);

    state.recorder = {
      active: true,
      stream,
      context,
      source,
      analyser,
      processor,
      gain,
      chunks,
      meterBuffer: new Uint8Array(analyser.fftSize),
      meterRaf: null,
      startedAt: performance.now(),
    };

    $("recordBtn").textContent = "Stop recording";
    $("recordStrip").classList.add("is-visible");
    $("recordState").textContent = "Recording";
    $("recordTimer").textContent = "0:00";
    logActivity("Reference recording started.");
    tickRecorderMeter();
  } catch (error) {
    showMessage("error", `Microphone failed: ${error.message}`);
    cleanupRecorder();
  }
}

function tickRecorderMeter() {
  if (!state.recorder.active || !state.recorder.analyser) {
    return;
  }

  const { analyser, meterBuffer, startedAt } = state.recorder;
  analyser.getByteTimeDomainData(meterBuffer);

  let peak = 0;
  for (let i = 0; i < meterBuffer.length; i += 1) {
    peak = Math.max(peak, Math.abs(meterBuffer[i] - 128));
  }

  const normalized = Math.min(1, peak / 64);
  $("recordMeterFill").style.width = `${Math.round(normalized * 100)}%`;
  $("recordTimer").textContent = formatDuration((performance.now() - startedAt) / 1000);

  state.recorder.meterRaf = requestAnimationFrame(tickRecorderMeter);
}

async function stopRecording() {
  if (!state.recorder.active) {
    return;
  }

  const recorder = { ...state.recorder };
  state.recorder.active = false;
  if (state.recorder.meterRaf) {
    cancelAnimationFrame(state.recorder.meterRaf);
  }

  try {
    const totalSamples = recorder.chunks.reduce((sum, chunk) => sum + chunk.length, 0);
    if (!totalSamples) {
      throw new Error("No audio captured.");
    }

    const buffer = recorder.context.createBuffer(1, totalSamples, recorder.context.sampleRate);
    const channel = buffer.getChannelData(0);
    let offset = 0;
    recorder.chunks.forEach((chunk) => {
      channel.set(chunk, offset);
      offset += chunk.length;
    });

    let output = buffer;
    if (buffer.sampleRate !== 24000) {
      const offline = new OfflineAudioContext(1, Math.ceil(buffer.duration * 24000), 24000);
      const source = offline.createBufferSource();
      source.buffer = buffer;
      source.connect(offline.destination);
      source.start(0);
      output = await offline.startRendering();
    }

    const wavBlob = bufferToWav(output);
    const file = new File([wavBlob], `recorded-${Date.now()}.wav`, { type: "audio/wav" });
    clearPresetSelection();
    setReferenceFile(file, file.name, "Recorded reference clip");
    $("referenceSourceLabel").textContent = "Recorded reference clip";
    showMessage("success", "Reference recording captured.");
    logActivity("Reference recording saved.", "success");
  } catch (error) {
    showMessage("error", error.message || "Recording failed.");
    logActivity(`Recording failed: ${error.message}`, "error");
  } finally {
    cleanupRecorder();
  }
}

function cleanupRecorder() {
  const recorder = state.recorder;
  if (recorder.meterRaf) {
    cancelAnimationFrame(recorder.meterRaf);
  }
  if (recorder.stream) {
    recorder.stream.getTracks().forEach((track) => track.stop());
  }
  recorder.source?.disconnect?.();
  recorder.analyser?.disconnect?.();
  recorder.processor?.disconnect?.();
  recorder.gain?.disconnect?.();
  recorder.context?.close?.().catch(() => {});

  state.recorder = {
    active: false,
    stream: null,
    context: null,
    source: null,
    analyser: null,
    processor: null,
    gain: null,
    chunks: [],
    meterBuffer: null,
    meterRaf: null,
    startedAt: 0,
  };

  $("recordBtn").textContent = "Record clip";
  $("recordState").textContent = "Recorder idle";
  $("recordTimer").textContent = "0:00";
  $("recordMeterFill").style.width = "0%";
  $("recordStrip").classList.remove("is-visible");
}

function bufferToWav(buffer) {
  const samples = buffer.getChannelData(0);
  const sampleRate = buffer.sampleRate;
  const arrayBuffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(arrayBuffer);

  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    offset += 2;
  }

  return new Blob([arrayBuffer], { type: "audio/wav" });
}

function writeAscii(view, offset, text) {
  for (let i = 0; i < text.length; i += 1) {
    view.setUint8(offset + i, text.charCodeAt(i));
  }
}

async function generateSpeech() {
  if (state.loadingModel || state.generating) {
    return;
  }

  const text = $("textInput").value.trim();
  if (!text) {
    showMessage("warning", "Add some text before starting generation.");
    return;
  }

  if (state.mode === "voice_clone" && !state.refFile && !state.presetRefId) {
    showMessage("warning", "Voice clone mode needs a reference clip or preset voice.");
    return;
  }

  if (state.mode === "custom" && !$("speakerSelect").value) {
    showMessage("warning", "Choose a speaker ID for CustomVoice mode.");
    return;
  }

  if (state.mode === "voice_design" && !$("designInstruct").value.trim()) {
    showMessage("warning", "Voice Design mode needs an instruction prompt.");
    return;
  }

  if (modelKind(state.selectedModel) !== state.mode) {
    const recommended = recommendedModelForMode(state.mode, state.status?.available_models || []);
    if (!recommended) {
      showMessage("error", "No compatible model is available for the selected mode.");
      return;
    }
    state.selectedModel = recommended;
    $("modelSelect").value = recommended;
    updateModelSummary();
    logActivity(`Switched target model to ${shortModelName(recommended)} for ${MODE_LABELS[state.mode]}.`, "warning");
  }

  const loaded = await ensureSelectedModelLoaded();
  if (!loaded) {
    return;
  }

  resetRunSurface();
  state.generating = true;
  state.abortController = new AbortController();
  state.currentRun = {
    startedAt: Date.now(),
    mode: state.mode,
    model: state.selectedModel,
    streaming: state.streamMode === "stream",
    text: text.slice(0, 180),
    language: $("languageSelect").value,
    speaker: $("speakerSelect").value,
    xvecOnly: state.xvecOnly,
  };
  updateControls();
  setRunBadge("Starting", "is-loading");
  logActivity(`Starting ${MODE_LABELS[state.mode]} run on ${shortModelName(state.selectedModel)}.`);

  try {
    const formData = buildGenerationForm();
    if (state.streamMode === "stream") {
      await runStream(formData, state.abortController.signal);
    } else {
      await runBatch(formData, state.abortController.signal);
    }
  } catch (error) {
    if (error.name === "AbortError") {
      handleCancelledRun();
    } else {
      showMessage("error", error.message || "Generation failed.");
      setRunBadge("Error", "is-error");
      logActivity(`Generation failed: ${error.message}`, "error");
    }
  } finally {
    state.generating = false;
    state.abortController = null;
    updateControls();
    await refreshStatus({ quiet: true });
  }
}

function buildGenerationForm() {
  const formData = new FormData();
  formData.append("text", $("textInput").value.trim());
  formData.append("mode", state.mode);
  formData.append("language", $("languageSelect").value);
  formData.append("temperature", $("temperatureRange").value);
  formData.append("top_k", $("topKRange").value);
  formData.append("repetition_penalty", $("repetitionRange").value);

  if (state.mode === "voice_clone") {
    if (state.presetRefId) {
      formData.append("ref_preset", state.presetRefId);
    } else if (state.refFile) {
      formData.append("ref_audio", state.refFile);
    }
    formData.append("ref_text", $("refText").value.trim());
    formData.append("xvec_only", state.xvecOnly ? "true" : "false");
  } else if (state.mode === "custom") {
    formData.append("speaker", $("speakerSelect").value);
    formData.append("instruct", $("customInstruct").value.trim());
  } else if (state.mode === "voice_design") {
    formData.append("instruct", $("designInstruct").value.trim());
  }

  if (state.streamMode === "stream") {
    formData.append("chunk_size", $("chunkSizeRange").value);
  }

  return formData;
}

async function ensureSelectedModelLoaded() {
  if (state.status?.loaded && state.status.model === state.selectedModel && !state.status.loading) {
    return true;
  }
  return loadSelectedModel({ silent: true });
}

async function runStream(formData, signal) {
  await initPlayback(24000);
  state.clientT0 = performance.now();
  state.firstChunkAt = null;
  state.firstAudioAt = null;
  state.lastBufferSeconds = 0;

  const response = await fetch("/generate/stream", {
    method: "POST",
    body: formData,
    signal,
  });
  if (!response.ok) {
    throw new Error(await extractError(response));
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalPayload = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();

    for (const line of lines) {
      if (!line.startsWith("data: ")) {
        continue;
      }

      const payload = JSON.parse(line.slice(6));

      if (payload.type === "queued") {
        const ahead = payload.position === 1 ? "1 request ahead of you" : `${payload.position} requests ahead of you`;
        $("queueBanner").hidden = false;
        $("queueBanner").textContent = `Waiting for the generation lock: ${ahead}.`;
        setRunBadge("Queued", "is-loading");
      }

      if (payload.type === "chunk") {
        $("queueBanner").hidden = true;
        setRunBadge("Streaming", "is-active");
        updateMetricsFromPayload(payload, true);
        enqueueChunk(payload.audio_b64);
        $("resultSummary").textContent = `Streaming ${payload.total_audio_s?.toFixed?.(1) || payload.total_audio_s || 0}s of audio. Buffering live playback while more chunks arrive.`;
      }

      if (payload.type === "done") {
        finalPayload = payload;
        await state.chunkChain;
        const blob = buildFinalWav();
        if (blob) {
          setOutputBlob(blob);
        }
        finishRun(payload, true);
      }

      if (payload.type === "error") {
        throw new Error(payload.message || "Streaming request failed.");
      }
    }
  }

  if (!finalPayload) {
    throw new Error("Streaming ended before the server sent a completion event.");
  }
}

async function runBatch(formData, signal) {
  setRunBadge("Synthesizing", "is-active");
  const response = await fetch("/generate", {
    method: "POST",
    body: formData,
    signal,
  });
  if (!response.ok) {
    throw new Error(await extractError(response));
  }

  const payload = await response.json();
  const metrics = payload.metrics || {};
  updateMetricsFromPayload({
    total_audio_s: metrics.audio_duration_s,
    rtf: metrics.rtf,
    ttfa_ms: metrics.total_ms,
  }, false);

  const bytes = base64ToBytes(payload.audio_b64);
  setOutputBlob(new Blob([bytes], { type: "audio/wav" }));
  $("audioPlayer").play().catch(() => {});
  finishRun({
    total_audio_s: metrics.audio_duration_s,
    rtf: metrics.rtf,
    total_ms: metrics.total_ms,
    ttfa_ms: metrics.total_ms,
  }, false);
}

function updateMetricsFromPayload(payload, streaming) {
  if (payload.ttfa_ms != null) {
    $("metricServerTtfa").textContent = `${Math.round(payload.ttfa_ms)} ms`;
  }
  if (payload.rtf != null) {
    $("metricRtf").textContent = `${Number(payload.rtf).toFixed(2)}x`;
  }
  if (payload.total_audio_s != null) {
    $("metricDuration").textContent = `${Number(payload.total_audio_s).toFixed(2)} s`;
  }
  if (payload.voice_clone_ms != null && payload.voice_clone_ms > 0) {
    $("metricClonePrep").textContent = `${Math.round(payload.voice_clone_ms)} ms`;
  } else if (!streaming) {
    $("metricClonePrep").textContent = "n/a";
  }
}

function finishRun(payload, streaming) {
  setRunBadge("Complete", "is-done");
  $("queueBanner").hidden = true;
  const duration = payload.total_audio_s != null ? `${Number(payload.total_audio_s).toFixed(2)}s` : "unknown duration";
  const rtf = payload.rtf != null ? `${Number(payload.rtf).toFixed(2)}x` : "n/a";
  const headline = streaming
    ? `Streaming run completed. Produced ${duration} of audio at ${rtf} real-time factor.`
    : `Batch render completed. Produced ${duration} of audio at ${rtf} real-time factor.`;
  $("resultSummary").textContent = headline;
  showMessage("success", headline);
  logActivity(headline, "success");

  const historyEntry = {
    id: `${Date.now()}`,
    createdAt: Date.now(),
    mode: state.currentRun?.mode || state.mode,
    model: state.currentRun?.model || state.selectedModel,
    streaming,
    text: state.currentRun?.text || $("textInput").value.trim().slice(0, 180),
    language: state.currentRun?.language || $("languageSelect").value,
    speaker: state.currentRun?.speaker || "",
    xvecOnly: state.currentRun?.xvecOnly ?? state.xvecOnly,
    refText: $("refText").value,
    instruct: state.mode === "custom" ? $("customInstruct").value : $("designInstruct").value,
    chunkSize: $("chunkSizeRange").value,
    temperature: $("temperatureRange").value,
    topK: $("topKRange").value,
    repetitionPenalty: $("repetitionRange").value,
    metrics: {
      audioDuration: payload.total_audio_s || null,
      rtf: payload.rtf || null,
      ttfa: payload.ttfa_ms || null,
      totalMs: payload.total_ms || null,
    },
  };

  state.recentJobs.unshift(historyEntry);
  state.recentJobs = state.recentJobs.slice(0, 8);
  saveRecentJobs();
  renderHistory();
}

function handleCancelledRun() {
  const partial = buildFinalWav();
  if (partial) {
    setOutputBlob(partial);
    $("resultSummary").textContent = "Run cancelled. Partial audio was preserved for review.";
  } else {
    $("resultSummary").textContent = "Run cancelled before any audio was produced.";
  }
  setRunBadge("Cancelled", "is-error");
  $("queueBanner").hidden = true;
  showMessage("warning", "Generation cancelled.");
  logActivity("Generation cancelled by operator.", "warning");
}

function cancelRun() {
  if (!state.generating || !state.abortController) {
    return;
  }
  state.abortController.abort();
  setRunBadge("Cancelling", "is-loading");
  showMessage("warning", "Cancellation requested. Waiting for the stream to close cleanly...");
}

function resetRunSurface() {
  hideMessage();
  clearOutputBlob();
  resetPlaybackQueue();
  $("metricServerTtfa").textContent = "—";
  $("metricClientTtfa").textContent = "—";
  $("metricRtf").textContent = "—";
  $("metricDuration").textContent = "—";
  $("metricBuffer").textContent = "—";
  $("metricClonePrep").textContent = "—";
  $("resultSummary").textContent = "Generated audio and run summary will appear here.";
  $("queueBanner").hidden = true;
}

async function initPlayback(sampleRate) {
  state.rawPcmSr = sampleRate || 24000;
  state.pcmQueue = [];
  state.rawPcmParts = [];
  state.chunkChain = Promise.resolve();
  state.lastBufferSeconds = 0;

  if (!state.audioContext) {
    state.audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: state.rawPcmSr });
    state.scriptProcessor = state.audioContext.createScriptProcessor(512, 0, 1);
    state.scriptProcessor.onaudioprocess = onPlaybackProcess;
    state.scriptProcessor.connect(state.audioContext.destination);
  }

  if (state.audioContext.state === "suspended") {
    await state.audioContext.resume();
  }
}

function onPlaybackProcess(event) {
  const output = event.outputBuffer.getChannelData(0);
  let index = 0;
  let wroteAudio = false;

  while (index < output.length) {
    if (!state.pcmQueue.length) {
      output.fill(0, index);
      break;
    }

    const segment = state.pcmQueue[0];
    const remaining = segment.data.length - segment.position;
    const available = Math.min(output.length - index, remaining);
    output.set(segment.data.subarray(segment.position, segment.position + available), index);
    segment.position += available;
    index += available;
    wroteAudio = true;

    if (segment.position >= segment.data.length) {
      state.pcmQueue.shift();
    }
  }

  if (wroteAudio && state.firstAudioAt == null) {
    state.firstAudioAt = performance.now();
    updateClientMetrics();
  }
}

function enqueueChunk(audioBase64) {
  state.chunkChain = state.chunkChain.then(() => {
    const bytes = base64ToBytes(audioBase64);
    const parsed = parseWav(bytes);
    if (!parsed) {
      return;
    }

    state.rawPcmParts.push(parsed.rawPcm);
    state.pcmQueue.push({ data: parsed.pcm, position: 0 });

    if (state.firstChunkAt == null) {
      state.firstChunkAt = performance.now();
    }

    state.lastBufferSeconds = state.pcmQueue.reduce((sum, segment) => {
      return sum + (segment.data.length - segment.position);
    }, 0) / state.rawPcmSr;

    updateClientMetrics();
  });
}

function parseWav(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let offset = 12;

  while (offset + 8 <= bytes.length) {
    const chunkId = String.fromCharCode(bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]);
    const chunkSize = view.getUint32(offset + 4, true);
    if (chunkId === "data") {
      const rawPcm = bytes.slice(offset + 8, offset + 8 + chunkSize);
      const int16 = new Int16Array(rawPcm.buffer, rawPcm.byteOffset, rawPcm.byteLength / 2);
      const pcm = new Float32Array(int16.length);
      for (let i = 0; i < int16.length; i += 1) {
        pcm[i] = int16[i] / 32768;
      }
      return {
        rawPcm,
        pcm,
      };
    }
    offset += 8 + chunkSize;
  }

  return null;
}

function buildFinalWav() {
  if (!state.rawPcmParts.length) {
    return null;
  }

  const totalPcmBytes = state.rawPcmParts.reduce((sum, part) => sum + part.length, 0);
  const arrayBuffer = new ArrayBuffer(44 + totalPcmBytes);
  const view = new DataView(arrayBuffer);
  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + totalPcmBytes, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, state.rawPcmSr, true);
  view.setUint32(28, state.rawPcmSr * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, totalPcmBytes, true);

  const output = new Uint8Array(arrayBuffer, 44);
  let offset = 0;
  state.rawPcmParts.forEach((part) => {
    output.set(part, offset);
    offset += part.length;
  });

  return new Blob([arrayBuffer], { type: "audio/wav" });
}

function resetPlaybackQueue() {
  state.pcmQueue = [];
  state.rawPcmParts = [];
  state.chunkChain = Promise.resolve();
  state.firstChunkAt = null;
  state.firstAudioAt = null;
  state.lastBufferSeconds = 0;
  updateClientMetrics();
}

function clearOutputBlob() {
  if (state.outputUrl) {
    URL.revokeObjectURL(state.outputUrl);
  }
  state.outputUrl = "";
  state.outputBlob = null;
  $("audioPlayer").pause();
  $("audioPlayer").removeAttribute("src");
  $("audioPlayer").load();
  $("downloadBtn").disabled = true;
}

function setOutputBlob(blob) {
  clearOutputBlob();
  state.outputBlob = blob;
  state.outputUrl = URL.createObjectURL(blob);
  $("audioPlayer").src = state.outputUrl;
  $("downloadBtn").disabled = false;
}

function downloadOutput() {
  if (!state.outputBlob) {
    return;
  }
  const anchor = document.createElement("a");
  anchor.href = state.outputUrl;
  anchor.download = `faster-qwen3-tts-${Date.now()}.wav`;
  anchor.click();
}

function updateClientMetrics() {
  if (state.clientT0 != null) {
    const firstAudible = state.firstAudioAt ?? state.firstChunkAt;
    if (firstAudible != null) {
      $("metricClientTtfa").textContent = `${Math.round(firstAudible - state.clientT0)} ms`;
    }
  }

  if (state.lastBufferSeconds != null) {
    $("metricBuffer").textContent = `${state.lastBufferSeconds.toFixed(2)} s`;
  }
}

function showMessage(type, text) {
  const bar = $("messageBar");
  bar.hidden = false;
  bar.className = `message-bar is-${type}`;
  bar.textContent = text;
}

function hideMessage() {
  const bar = $("messageBar");
  bar.hidden = true;
  bar.className = "message-bar";
  bar.textContent = "";
}

function setRunBadge(text, variant) {
  const badge = $("runBadge");
  badge.textContent = text;
  badge.className = `run-badge ${variant || ""}`.trim();
}

function updateServicePill(kind, label) {
  const pill = $("serviceHealthPill");
  pill.className = `service-pill is-${kind}`;
  $("serviceHealthText").textContent = label;
}

function transcriptionStateLabel(status) {
  if (status.transcription_loading) {
    return "Loading";
  }
  if (status.transcription_available) {
    return "Ready";
  }
  if (status.transcription_error) {
    return "Error";
  }
  return "On demand";
}

function transcriptionMeta(status) {
  if (status.transcription_error) {
    return status.transcription_error;
  }
  if (status.transcription_loading) {
    return "Loading the transcription model";
  }
  if (status.transcription_available) {
    return "Reference audio transcription is available";
  }
  return "Loads when first requested";
}

function describeModelType(modelType) {
  if (modelType === "custom_voice") {
    return "CustomVoice checkpoint";
  }
  if (modelType === "voice_design") {
    return "VoiceDesign checkpoint";
  }
  return "Base voice clone checkpoint";
}

function modelKind(model) {
  if (!model) {
    return "";
  }
  if (model.includes("CustomVoice")) {
    return "custom";
  }
  if (model.includes("VoiceDesign")) {
    return "voice_design";
  }
  return "voice_clone";
}

function shortModelName(model) {
  return model ? model.replace("Qwen/", "") : "";
}

function logActivity(message, tone = "info") {
  state.activity.unshift({
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    time: new Date(),
    message,
    tone,
  });
  state.activity = state.activity.slice(0, 12);
  renderActivity();
}

function renderActivity() {
  const list = $("activityLog");
  list.innerHTML = "";

  state.activity.forEach((entry) => {
    const item = document.createElement("li");
    item.className = "activity-item";
    item.dataset.tone = entry.tone;

    const time = document.createElement("span");
    time.className = "activity-item__time";
    time.textContent = entry.time.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });

    const body = document.createElement("div");
    body.className = "activity-item__body";
    body.textContent = entry.message;

    item.append(time, body);
    list.appendChild(item);
  });
}

function renderHistory() {
  const container = $("recentJobs");
  container.innerHTML = "";

  if (!state.recentJobs.length) {
    const empty = document.createElement("div");
    empty.className = "history-card";
    empty.textContent = "No runs yet. Completed jobs will appear here with a one-click restore action.";
    container.appendChild(empty);
    return;
  }

  state.recentJobs.forEach((job, index) => {
    const card = document.createElement("article");
    card.className = "history-card";

    const top = document.createElement("div");
    top.className = "history-card__top";

    const title = document.createElement("strong");
    title.textContent = `${MODE_LABELS[job.mode] || job.mode} - ${shortModelName(job.model)}`;

    const time = document.createElement("span");
    time.className = "inline-status";
    time.textContent = new Date(job.createdAt).toLocaleString([], {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });

    top.append(title, time);

    const meta = document.createElement("div");
    meta.className = "history-card__meta";
    meta.innerHTML = `
      <span>${job.streaming ? "Streaming" : "Batch"}</span>
      <span>${job.metrics?.audioDuration ? `${Number(job.metrics.audioDuration).toFixed(2)}s audio` : "No audio metric"}</span>
      <span>${job.metrics?.rtf ? `${Number(job.metrics.rtf).toFixed(2)}x RTF` : "RTF unavailable"}</span>
    `;

    const copy = document.createElement("p");
    copy.className = "history-card__copy";
    copy.textContent = job.text;

    const action = document.createElement("button");
    action.type = "button";
    action.className = "history-card__action";
    action.textContent = "Restore settings";
    action.addEventListener("click", () => restoreJob(index));

    card.append(top, meta, copy, action);
    container.appendChild(card);
  });
}

function restoreJob(index) {
  const job = state.recentJobs[index];
  if (!job) {
    return;
  }

  setMode(job.mode);
  state.streamMode = job.streaming ? "stream" : "batch";
  renderStreamMode();
  if (job.model && state.status?.available_models?.includes(job.model)) {
    state.selectedModel = job.model;
    $("modelSelect").value = job.model;
  }

  $("textInput").value = job.text || "";
  $("languageSelect").value = job.language || "English";
  $("chunkSizeRange").value = job.chunkSize || $("chunkSizeRange").value;
  $("temperatureRange").value = job.temperature || $("temperatureRange").value;
  $("topKRange").value = job.topK || $("topKRange").value;
  $("repetitionRange").value = job.repetitionPenalty || $("repetitionRange").value;
  if (job.mode === "custom") {
    $("customInstruct").value = job.instruct || "";
  }
  if (job.mode === "voice_design") {
    $("designInstruct").value = job.instruct || "";
  }
  if (job.mode === "voice_clone") {
    state.xvecOnly = job.xvecOnly ?? true;
    setXvecMode(state.xvecOnly);
    $("refText").value = job.refText || "";
  }

  if (job.speaker) {
    $("speakerSelect").value = job.speaker;
  }

  ["chunkSizeRange", "temperatureRange", "topKRange", "repetitionRange"].forEach((id) => {
    $(id).dispatchEvent(new Event("input"));
  });
  updateTextCounter();
  updateModelSummary();
  savePreferences();
  showMessage("success", "Job settings restored. Reattach a reference clip if you are using voice clone mode.");
  logActivity("Previous job restored into the composer.", "success");
}

function updateUptimeDisplay() {
  if (!state.status?.started_at) {
    $("serviceUptimeValue").textContent = "Uptime unavailable";
    return;
  }

  const seconds = Math.max(0, Math.floor(Date.now() / 1000 - state.status.started_at));
  $("serviceUptimeValue").textContent = `Uptime ${formatDuration(seconds)}`;
}

function formatDuration(totalSeconds) {
  const rounded = Math.max(0, Math.floor(totalSeconds));
  const minutes = Math.floor(rounded / 60);
  const seconds = rounded % 60;
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

async function extractError(response) {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    const payload = await response.json();
    return payload.detail || payload.message || "Request failed.";
  }
  return response.text();
}

function base64ToBytes(value) {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}
