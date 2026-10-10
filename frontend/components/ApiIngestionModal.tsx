"use client";

import { useState } from "react";
import {
  Terminal,
  Code,
  Send,
  Copy,
  Check,
  Zap,
  Globe,
  Database,
  Layers,
  Activity,
  X,
  Play,
  Cpu,
  RefreshCw,
  Clock,
  ShieldCheck,
  ArrowRight,
  FileCode,
  CheckCircle2,
  AlertCircle
} from "lucide-react";
import { HealthStatus, ServerConfig } from "../app/hooks/useMonitoring";

interface ApiIngestionModalProps {
  isOpen: boolean;
  onClose: () => void;
  health: HealthStatus | null;
  config: ServerConfig | null;
  sendPredict: (features: Record<string, number>, label?: number) => Promise<{ success: boolean; data?: any; error?: string; latencyMs?: number }>;
  sendDelayedLabels: (labels: Array<{ prediction_id: string; label: number }>) => Promise<{ success: boolean; data?: any; error?: string }>;
}

export function ApiIngestionModal({
  isOpen,
  onClose,
  health,
  config,
  sendPredict,
  sendDelayedLabels,
}: ApiIngestionModalProps) {
  const [activeTab, setActiveTab] = useState<"tester" | "snippets" | "buffer">("tester");
  const [snippetLanguage, setSnippetLanguage] = useState<"python" | "node" | "curl">("python");
  const [copiedCode, setCopiedCode] = useState(false);

  // Predict tester state
  const defaultFeatures = JSON.stringify(
    {
      "FIT101": 2.45,
      "LIT101": 505.2,
      "MV101": 1.0,
      "P101": 2.0,
      "P102": 1.0,
      "AIT201": 260.1,
      "AIT202": 8.3,
      "FIT201": 2.41,
      "MV201": 2.0,
      "P201": 2.0,
      "P203": 1.0,
      "FIT301": 2.18,
      "LIT301": 950.4,
      "MV301": 1.0,
      "DPIT301": 19.8,
      "P402": 1.0
    },
    null,
    2
  );

  const [jsonPayload, setJsonPayload] = useState<string>(defaultFeatures);
  const [includeLabel, setIncludeLabel] = useState<boolean>(false);
  const [sampleLabel, setSampleLabel] = useState<number>(0);
  const [isSubmittingPredict, setIsSubmittingPredict] = useState<boolean>(false);
  const [predictResult, setPredictResult] = useState<{ success: boolean; data?: any; error?: string; latencyMs?: number } | null>(null);

  // Delayed label state
  const [delayedPredId, setDelayedPredId] = useState<string>("");
  const [delayedLabelVal, setDelayedLabelVal] = useState<number>(0);
  const [isSubmittingLabel, setIsSubmittingLabel] = useState<boolean>(false);
  const [labelResult, setLabelResult] = useState<{ success: boolean; data?: any; error?: string } | null>(null);

  // Session history log
  const [apiLog, setApiLog] = useState<Array<{ id: string; timestamp: string; latencyMs: number; status: string; predId?: string; predVal?: number }>>([]);

  if (!isOpen) return null;

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedCode(true);
    setTimeout(() => setCopiedCode(false), 2000);
  };

  const handleSendPredict = async () => {
    setIsSubmittingPredict(true);
    setPredictResult(null);
    try {
      const parsedFeatures = JSON.parse(jsonPayload);
      const res = await sendPredict(parsedFeatures, includeLabel ? sampleLabel : undefined);
      setPredictResult(res);

      if (res.success && res.data) {
        if (res.data.prediction_id) {
          setDelayedPredId(res.data.prediction_id);
        }
        setApiLog((prev) => [
          {
            id: `req_${Date.now().toString().slice(-6)}`,
            timestamp: new Date().toLocaleTimeString(),
            latencyMs: res.latencyMs || 0,
            status: "200 OK",
            predId: res.data.prediction_id,
            predVal: res.data.prediction,
          },
          ...prev.slice(0, 19),
        ]);
      } else {
        setApiLog((prev) => [
          {
            id: `req_${Date.now().toString().slice(-6)}`,
            timestamp: new Date().toLocaleTimeString(),
            latencyMs: res.latencyMs || 0,
            status: "Error",
          },
          ...prev.slice(0, 19),
        ]);
      }
    } catch (err: any) {
      setPredictResult({ success: false, error: `Invalid JSON syntax: ${err.message}` });
    } finally {
      setIsSubmittingPredict(false);
    }
  };

  const handleSendDelayedLabel = async () => {
    if (!delayedPredId.trim()) {
      setLabelResult({ success: false, error: "Please provide a valid prediction_id" });
      return;
    }
    setIsSubmittingLabel(true);
    setLabelResult(null);
    const res = await sendDelayedLabels([{ prediction_id: delayedPredId.trim(), label: delayedLabelVal }]);
    setLabelResult(res);
    setIsSubmittingLabel(false);
  };

  const getPythonSnippet = () => `import requests

# Live Model Decay Radar API Endpoint
URL = "http://127.0.0.1:8000/predict"

# 1. Send real-time model inference payload
payload = {
    "features": {
        "FIT101": 2.45,
        "LIT101": 505.2,
        "DPIT301": 19.8,
        "P402": 1.0
    },
    "label": None  # Optional: send ground truth if known immediately
}

response = requests.post(URL, json=payload)
data = response.json()

print(f"Prediction ID: {data['prediction_id']}")
print(f"Prediction: {data['prediction']} (Confidence: {data['confidence']:.2f})")
print(f"Model Status: {data['model_status']} | Buffer: {data['batch_buffered']}/500")

# 2. Later: Send delayed ground truth label when verified
delayed_url = "http://127.0.0.1:8000/labels/delayed"
requests.post(delayed_url, json={
    "labels": [
        {"prediction_id": data["prediction_id"], "label": 0}
    ]
})`;

  const getNodeSnippet = () => `// Live Model Decay Radar Node.js Client
const API_BASE = "http://127.0.0.1:8000";

async function predictSample() {
  const response = await fetch(\`\${API_BASE}/predict\`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      features: {
        FIT101: 2.45,
        LIT101: 505.2,
        DPIT301: 19.8,
        P402: 1.0
      }
    })
  });

  const result = await response.json();
  console.log("Prediction Result:", result);

  // Submit delayed ground truth label asynchronously
  await fetch(\`\${API_BASE}/labels/delayed\`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      labels: [{ prediction_id: result.prediction_id, label: 0 }]
    })
  });
}

predictSample();`;

  const getCurlSnippet = () => `# 1. POST Real-time Prediction Sample
curl -X POST "http://127.0.0.1:8000/predict" \\
  -H "Content-Type: application/json" \\
  -d '{
    "features": {
      "FIT101": 2.45,
      "LIT101": 505.2,
      "DPIT301": 19.8,
      "P402": 1.0
    }
  }'

# 2. POST Delayed Ground Truth Label
curl -X POST "http://127.0.0.1:8000/labels/delayed" \\
  -H "Content-Type: application/json" \\
  -d '{
    "labels": [
      {
        "prediction_id": "pred_a1b2c3d4e5",
        "label": 0
      }
    ]
  }'`;

  const currentSnippet = snippetLanguage === "python" ? getPythonSnippet() : snippetLanguage === "node" ? getNodeSnippet() : getCurlSnippet();

  const bufferPct = health ? Math.min(100, Math.round((health.buffer_size / health.window_size) * 100)) : 0;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md p-4 sm:p-6 overflow-y-auto">
      <div className="relative w-full max-w-5xl bg-zinc-950 border border-zinc-800 rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
        
        {/* Header Bar */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-800 bg-zinc-900/60">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-indigo-500/20 to-rose-500/20 border border-indigo-500/30 flex items-center justify-center text-indigo-400">
              <Globe className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-white flex items-center gap-2">
                API Ingestion & Integration Hub
                <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 font-semibold">
                  LIVE REST API
                </span>
              </h2>
              <p className="text-xs text-zinc-400">Stream live predictions & ground truth labels from your microservices into Model Decay Radar</p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-2 text-zinc-400 hover:text-white rounded-lg hover:bg-zinc-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Live System Status Sub-bar */}
        <div className="bg-zinc-900/40 px-6 py-2.5 border-b border-zinc-800/80 flex flex-wrap items-center justify-between text-xs font-mono gap-3">
          <div className="flex items-center gap-4 text-zinc-400">
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
              API Base: <code className="text-zinc-200 bg-zinc-900 px-1.5 py-0.5 rounded border border-zinc-800">http://127.0.0.1:8000</code>
            </span>
            <span className="hidden sm:inline text-zinc-600">|</span>
            <span className="hidden sm:flex items-center gap-1 text-zinc-400">
              Active Classifier: <span className="text-indigo-400 font-semibold">{config?.active_classifier || "Random Forest"}</span>
            </span>
          </div>

          <div className="flex items-center gap-3">
            <span className="text-zinc-400">Buffer Window:</span>
            <div className="w-28 bg-zinc-900 h-2 rounded-full overflow-hidden border border-zinc-800">
              <div className="bg-gradient-to-r from-indigo-500 to-rose-500 h-full transition-all duration-500" style={{ width: `${bufferPct}%` }}></div>
            </div>
            <span className="text-zinc-200 font-bold">{health?.buffer_size ?? 0} / {health?.window_size ?? 500}</span>
          </div>
        </div>

        {/* Top Navigation Tabs */}
        <div className="flex border-b border-zinc-800 bg-zinc-900/30 px-6 pt-3 gap-2">
          <button
            onClick={() => setActiveTab("tester")}
            className={`flex items-center gap-2 px-4 py-2.5 rounded-t-xl text-xs font-semibold font-mono transition-all cursor-pointer border-t border-x ${
              activeTab === "tester"
                ? "bg-zinc-950 text-white border-zinc-800 border-b-transparent shadow-sm"
                : "text-zinc-400 hover:text-zinc-200 border-transparent hover:bg-zinc-900/40"
            }`}
          >
            <Send className="w-3.5 h-3.5 text-rose-400" />
            Interactive REST Playground
          </button>

          <button
            onClick={() => setActiveTab("snippets")}
            className={`flex items-center gap-2 px-4 py-2.5 rounded-t-xl text-xs font-semibold font-mono transition-all cursor-pointer border-t border-x ${
              activeTab === "snippets"
                ? "bg-zinc-950 text-white border-zinc-800 border-b-transparent shadow-sm"
                : "text-zinc-400 hover:text-zinc-200 border-transparent hover:bg-zinc-900/40"
            }`}
          >
            <Code className="w-3.5 h-3.5 text-indigo-400" />
            Integration SDK & Code Snippets
          </button>

          <button
            onClick={() => setActiveTab("buffer")}
            className={`flex items-center gap-2 px-4 py-2.5 rounded-t-xl text-xs font-semibold font-mono transition-all cursor-pointer border-t border-x ${
              activeTab === "buffer"
                ? "bg-zinc-950 text-white border-zinc-800 border-b-transparent shadow-sm"
                : "text-zinc-400 hover:text-zinc-200 border-transparent hover:bg-zinc-900/40"
            }`}
          >
            <Activity className="w-3.5 h-3.5 text-emerald-400" />
            Live Buffer & Ingestion Telemetry
          </button>
        </div>

        {/* Body Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">

          {/* TAB 1: REST Playground */}
          {activeTab === "tester" && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              
              {/* Left Column: Request Form */}
              <div className="space-y-5">
                <div className="bg-zinc-900/50 border border-zinc-800/80 rounded-xl p-4 space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">POST</span>
                      <code className="text-xs font-mono text-zinc-200">/predict</code>
                    </div>
                    <button
                      onClick={() => setJsonPayload(defaultFeatures)}
                      className="text-[11px] font-mono text-indigo-400 hover:text-indigo-300 flex items-center gap-1 cursor-pointer"
                    >
                      <RefreshCw className="w-3 h-3" /> Reset Template
                    </button>
                  </div>

                  <div>
                    <label className="block text-xs font-semibold text-zinc-300 mb-1.5">Feature Payload (JSON dict)</label>
                    <textarea
                      value={jsonPayload}
                      onChange={(e) => setJsonPayload(e.target.value)}
                      rows={10}
                      className="w-full bg-zinc-950 font-mono text-xs text-emerald-300 p-3 rounded-lg border border-zinc-800 focus:outline-none focus:border-indigo-500/50 transition-colors"
                      placeholder='{ "FIT101": 2.45, ... }'
                    />
                  </div>

                  <div className="flex items-center justify-between pt-1">
                    <label className="flex items-center gap-2 text-xs text-zinc-300 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={includeLabel}
                        onChange={(e) => setIncludeLabel(e.target.checked)}
                        className="rounded border-zinc-800 bg-zinc-950 text-indigo-600 focus:ring-0"
                      />
                      Include immediate ground truth label
                    </label>

                    {includeLabel && (
                      <select
                        value={sampleLabel}
                        onChange={(e) => setSampleLabel(Number(e.target.value))}
                        className="bg-zinc-950 text-xs font-mono text-zinc-200 border border-zinc-800 rounded px-2 py-1"
                      >
                        <option value={0}>Label: 0 (Normal)</option>
                        <option value={1}>Label: 1 (Anomaly)</option>
                      </select>
                    )}
                  </div>

                  <button
                    onClick={handleSendPredict}
                    disabled={isSubmittingPredict}
                    className="w-full py-2.5 rounded-lg bg-gradient-to-r from-rose-600 to-indigo-600 hover:from-rose-500 hover:to-indigo-500 text-white font-semibold text-xs font-mono flex items-center justify-center gap-2 shadow-lg shadow-rose-950/40 transition-all cursor-pointer disabled:opacity-50"
                  >
                    {isSubmittingPredict ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin" /> Ingesting Request...
                      </>
                    ) : (
                      <>
                        <Send className="w-4 h-4" /> Send Real-Time Prediction Sample
                      </>
                    )}
                  </button>
                </div>

                {/* Section: Delayed Ground Truth */}
                <div className="bg-zinc-900/50 border border-zinc-800/80 rounded-xl p-4 space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-amber-500/20 text-amber-400 border border-amber-500/30">POST</span>
                      <code className="text-xs font-mono text-zinc-200">/labels/delayed</code>
                    </div>
                    <span className="text-[10px] text-zinc-500 font-mono">Supervised Evaluation</span>
                  </div>

                  <p className="text-[11px] text-zinc-400 leading-relaxed">
                    Submit ground truth outcome for previous prediction IDs to trigger supervised F1 & accuracy monitoring.
                  </p>

                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                    <div className="sm:col-span-2">
                      <input
                        type="text"
                        value={delayedPredId}
                        onChange={(e) => setDelayedPredId(e.target.value)}
                        placeholder="prediction_id (e.g. pred_a1b2c3...)"
                        className="w-full bg-zinc-950 font-mono text-xs text-zinc-200 p-2 rounded-lg border border-zinc-800 focus:outline-none focus:border-amber-500/50"
                      />
                    </div>
                    <div>
                      <select
                        value={delayedLabelVal}
                        onChange={(e) => setDelayedLabelVal(Number(e.target.value))}
                        className="w-full bg-zinc-950 font-mono text-xs text-zinc-200 p-2 rounded-lg border border-zinc-800"
                      >
                        <option value={0}>Label: 0</option>
                        <option value={1}>Label: 1</option>
                      </select>
                    </div>
                  </div>

                  <button
                    onClick={handleSendDelayedLabel}
                    disabled={isSubmittingLabel}
                    className="w-full py-2 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-amber-300 font-semibold text-xs font-mono flex items-center justify-center gap-2 transition-all cursor-pointer disabled:opacity-50"
                  >
                    {isSubmittingLabel ? "Submitting Label..." : "Submit Delayed Ground Truth"}
                  </button>

                  {labelResult && (
                    <div className={`p-2.5 rounded-lg text-xs font-mono flex items-center gap-2 ${labelResult.success ? "bg-emerald-950/40 text-emerald-400 border border-emerald-500/30" : "bg-rose-950/40 text-rose-400 border border-rose-500/30"}`}>
                      {labelResult.success ? <CheckCircle2 className="w-4 h-4 shrink-0" /> : <AlertCircle className="w-4 h-4 shrink-0" />}
                      <span>{labelResult.success ? labelResult.data?.message : labelResult.error}</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Right Column: Live Response & Latency Inspector */}
              <div className="space-y-4">
                <div className="bg-zinc-950 border border-zinc-800 rounded-xl p-4 h-full flex flex-col justify-between">
                  <div>
                    <div className="flex items-center justify-between border-b border-zinc-800/80 pb-3 mb-3">
                      <h3 className="text-xs font-bold font-mono text-zinc-300 uppercase tracking-wider flex items-center gap-2">
                        <Terminal className="w-4 h-4 text-indigo-400" />
                        Live API Response Console
                      </h3>
                      {predictResult && (
                        <div className="flex items-center gap-2 text-[10px] font-mono">
                          <span className={`px-2 py-0.5 rounded font-bold ${predictResult.success ? "bg-emerald-500/20 text-emerald-400" : "bg-rose-500/20 text-rose-400"}`}>
                            {predictResult.success ? "200 OK" : "ERROR"}
                          </span>
                          {predictResult.latencyMs !== undefined && (
                            <span className="text-zinc-400 flex items-center gap-1">
                              <Clock className="w-3 h-3 text-amber-400" /> {predictResult.latencyMs}ms
                            </span>
                          )}
                        </div>
                      )}
                    </div>

                    {!predictResult ? (
                      <div className="flex flex-col items-center justify-center h-64 text-zinc-600 text-center p-6 border border-dashed border-zinc-800 rounded-lg">
                        <Send className="w-8 h-8 mb-2 text-zinc-700" />
                        <p className="text-xs font-semibold text-zinc-400">No Request Executed Yet</p>
                        <p className="text-[10px] text-zinc-600 mt-1 max-w-[220px]">
                          Click &quot;Send Real-Time Prediction Sample&quot; to test model inference and examine latency metrics.
                        </p>
                      </div>
                    ) : predictResult.success ? (
                      <div className="space-y-4">
                        <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                          <div className="bg-zinc-900/60 p-2.5 rounded-lg border border-zinc-800">
                            <span className="text-zinc-500 text-[10px] block">PREDICTION ID</span>
                            <span className="text-rose-400 font-bold truncate block">{predictResult.data?.prediction_id}</span>
                          </div>
                          <div className="bg-zinc-900/60 p-2.5 rounded-lg border border-zinc-800">
                            <span className="text-zinc-500 text-[10px] block">PREDICTION VALUE</span>
                            <span className="text-emerald-400 font-bold text-sm block">{predictResult.data?.prediction}</span>
                          </div>
                          <div className="bg-zinc-900/60 p-2.5 rounded-lg border border-zinc-800">
                            <span className="text-zinc-500 text-[10px] block">CONFIDENCE</span>
                            <span className="text-indigo-400 font-bold block">{((predictResult.data?.confidence || 0) * 100).toFixed(1)}%</span>
                          </div>
                          <div className="bg-zinc-900/60 p-2.5 rounded-lg border border-zinc-800">
                            <span className="text-zinc-500 text-[10px] block">MODEL STATUS</span>
                            <span className="text-zinc-200 font-bold block">{predictResult.data?.model_status}</span>
                          </div>
                        </div>

                        <div>
                          <span className="text-[10px] font-mono text-zinc-400 block mb-1">RAW JSON RESPONSE</span>
                          <pre className="bg-zinc-900/90 text-emerald-300 font-mono text-[11px] p-3 rounded-lg border border-zinc-800/80 overflow-x-auto max-h-48">
                            {JSON.stringify(predictResult.data, null, 2)}
                          </pre>
                        </div>
                      </div>
                    ) : (
                      <div className="bg-rose-950/30 border border-rose-500/30 p-4 rounded-lg text-rose-400 text-xs font-mono space-y-2">
                        <div className="font-bold flex items-center gap-2">
                          <AlertCircle className="w-4 h-4" /> Request Ingestion Error
                        </div>
                        <p className="text-zinc-300 leading-relaxed">{predictResult.error}</p>
                      </div>
                    )}
                  </div>

                  <div className="pt-4 border-t border-zinc-900 mt-4 text-[10px] text-zinc-500 font-mono flex justify-between items-center">
                    <span>Non-blocking background buffering active</span>
                    <span>RadarOrchestrator v1.0</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 2: Code Snippets */}
          {activeTab === "snippets" && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 bg-zinc-900/80 border border-zinc-800 p-1 rounded-lg">
                  <button
                    onClick={() => setSnippetLanguage("python")}
                    className={`px-3 py-1.5 rounded text-xs font-mono font-semibold transition-all cursor-pointer ${
                      snippetLanguage === "python" ? "bg-indigo-600 text-white" : "text-zinc-400 hover:text-zinc-200"
                    }`}
                  >
                    Python (requests)
                  </button>
                  <button
                    onClick={() => setSnippetLanguage("node")}
                    className={`px-3 py-1.5 rounded text-xs font-mono font-semibold transition-all cursor-pointer ${
                      snippetLanguage === "node" ? "bg-indigo-600 text-white" : "text-zinc-400 hover:text-zinc-200"
                    }`}
                  >
                    Node.js (fetch)
                  </button>
                  <button
                    onClick={() => setSnippetLanguage("curl")}
                    className={`px-3 py-1.5 rounded text-xs font-mono font-semibold transition-all cursor-pointer ${
                      snippetLanguage === "curl" ? "bg-indigo-600 text-white" : "text-zinc-400 hover:text-zinc-200"
                    }`}
                  >
                    cURL CLI
                  </button>
                </div>

                <button
                  onClick={() => handleCopy(currentSnippet)}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-zinc-900 border border-zinc-800 hover:bg-zinc-800 text-zinc-300 hover:text-white text-xs font-mono transition-all cursor-pointer"
                >
                  {copiedCode ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                  {copiedCode ? "Copied!" : "Copy Code"}
                </button>
              </div>

              <div className="relative bg-zinc-950 border border-zinc-800 rounded-xl p-4 overflow-x-auto shadow-inner">
                <pre className="text-xs font-mono text-zinc-300 leading-relaxed">
                  {currentSnippet}
                </pre>
              </div>

              <div className="bg-zinc-900/40 border border-zinc-800/80 p-4 rounded-xl text-xs text-zinc-400 space-y-2">
                <h4 className="font-bold text-zinc-200 flex items-center gap-2">
                  <ShieldCheck className="w-4 h-4 text-emerald-400" /> Pipeline Behavior Notes
                </h4>
                <ul className="list-disc list-inside space-y-1 text-zinc-400 leading-relaxed font-sans">
                  <li><strong className="text-zinc-200">Instant Response:</strong> The <code className="font-mono text-rose-400 text-[11px]">/predict</code> endpoint returns predictions within milliseconds and does not block client execution.</li>
                  <li><strong className="text-zinc-200">Window Trigger:</strong> Once buffered samples reach your configured window size (default 500 samples), a 7-layer drift cycle fires in a background thread.</li>
                  <li><strong className="text-zinc-200">Delayed Labels:</strong> Ground truth labels submitted via <code className="font-mono text-amber-400 text-[11px]">/labels/delayed</code> are matched by <code className="font-mono text-zinc-300 text-[11px]">prediction_id</code> to evaluate supervised accuracy & F1 score.</li>
                </ul>
              </div>
            </div>
          )}

          {/* TAB 3: Buffer & Session Log */}
          {activeTab === "buffer" && (
            <div className="space-y-6">
              
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div className="bg-zinc-900/50 border border-zinc-800 p-4 rounded-xl">
                  <span className="text-zinc-500 text-xs font-mono block">BUFFERED SAMPLES</span>
                  <span className="text-2xl font-bold font-mono text-white mt-1 block">
                    {health?.buffer_size ?? 0} <span className="text-xs text-zinc-500 font-normal">/ {health?.window_size ?? 500}</span>
                  </span>
                  <div className="w-full bg-zinc-950 h-1.5 rounded-full overflow-hidden mt-3 border border-zinc-800">
                    <div className="bg-indigo-500 h-full transition-all duration-500" style={{ width: `${bufferPct}%` }} />
                  </div>
                </div>

                <div className="bg-zinc-900/50 border border-zinc-800 p-4 rounded-xl">
                  <span className="text-zinc-500 text-xs font-mono block">COMPLETED CYCLES</span>
                  <span className="text-2xl font-bold font-mono text-emerald-400 mt-1 block">
                    {health?.monitoring_cycles_completed ?? 0}
                  </span>
                  <span className="text-[10px] text-zinc-500 mt-2 block">Triggered every full window</span>
                </div>

                <div className="bg-zinc-900/50 border border-zinc-800 p-4 rounded-xl">
                  <span className="text-zinc-500 text-xs font-mono block">INTERACTIVE LOG COUNT</span>
                  <span className="text-2xl font-bold font-mono text-rose-400 mt-1 block">
                    {apiLog.length}
                  </span>
                  <span className="text-[10px] text-zinc-500 mt-2 block">Sent in current session</span>
                </div>
              </div>

              <div className="bg-zinc-950 border border-zinc-800 rounded-xl p-4">
                <h3 className="text-xs font-bold font-mono text-zinc-300 mb-3 flex items-center gap-2">
                  <Activity className="w-4 h-4 text-emerald-400" />
                  Interactive Request Stream Log
                </h3>

                {apiLog.length === 0 ? (
                  <div className="text-center py-8 text-zinc-600 text-xs font-mono">
                    No requests logged yet. Use the Interactive REST Playground tab to submit test samples.
                  </div>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs font-mono">
                      <thead className="border-b border-zinc-800 text-zinc-500">
                        <tr>
                          <th className="py-2 px-3">Req ID</th>
                          <th className="py-2 px-3">Timestamp</th>
                          <th className="py-2 px-3">Status</th>
                          <th className="py-2 px-3">Latency</th>
                          <th className="py-2 px-3">Prediction ID</th>
                          <th className="py-2 px-3">Value</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-zinc-900 text-zinc-300">
                        {apiLog.map((log) => (
                          <tr key={log.id} className="hover:bg-zinc-900/40">
                            <td className="py-2 px-3 text-zinc-500">{log.id}</td>
                            <td className="py-2 px-3 text-zinc-400">{log.timestamp}</td>
                            <td className="py-2 px-3">
                              <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${log.status.includes("OK") ? "bg-emerald-500/20 text-emerald-400" : "bg-rose-500/20 text-rose-400"}`}>
                                {log.status}
                              </span>
                            </td>
                            <td className="py-2 px-3 text-amber-400">{log.latencyMs}ms</td>
                            <td className="py-2 px-3 text-rose-400 font-semibold">{log.predId || "N/A"}</td>
                            <td className="py-2 px-3 text-emerald-400 font-bold">{log.predVal !== undefined ? log.predVal : "N/A"}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>

            </div>
          )}

        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-zinc-800 bg-zinc-900/60 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-200 text-xs font-mono font-semibold transition-all cursor-pointer"
          >
            Close Window
          </button>
        </div>

      </div>
    </div>
  );
}
