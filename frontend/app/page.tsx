"use client";

import { useMonitoring, MonitoringResult } from "./hooks/useMonitoring";
import { useState, useEffect, useRef } from "react";
import Link from "next/link";
import {
  Activity,
  ShieldAlert,
  CheckCircle,
  AlertTriangle,
  RefreshCw,
  Play,
  BarChart2,
  Cpu,
  Server,
  Layers,
  TrendingUp,
  Clock,
  Check,
  ChevronRight,
  Database,
  Upload,
  History,
  FileText
} from "lucide-react";

export default function Dashboard() {
  const {
    latest,
    history,
    health,
    config,
    isConnected,
    isLoading,
    error,
    simulateMessage,
    simulateDrift,
    resetDashboard,
    updateConfig,
    triggerManualRetrain,
    setOperatingMode,
    uploadDataset,
    replayStep,
    fetchRegistryHistory,
    rollbackModel,
    refetch,
  } = useMonitoring();

  const [isSimulating, setIsSimulating] = useState(false);
  const [isResetting, setIsResetting] = useState(false);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  // Dynamic configuration and observability states
  const [retrainMsg, setRetrainMsg] = useState<string | null>(null);
  const [isRetraining, setIsRetraining] = useState(false);
  const [activeAttributionTab, setActiveAttributionTab] = useState<"shap" | "ks">("shap");

  // Registry state
  const [isRegistryOpen, setIsRegistryOpen] = useState(false);
  const [registryHistory, setRegistryHistory] = useState<any[]>([]);
  const [isRollingBack, setIsRollingBack] = useState(false);

  // Validation gate state
  const [showValidationGate, setShowValidationGate] = useState(false);
  const previousValidationStatus = useRef<string | undefined>(undefined);
  const previousValidationCycle = useRef<number | undefined>(undefined);

  useEffect(() => {
    if (latest?.validation_status && latest.validation_status !== "none") {
      if (
        latest.validation_status !== previousValidationStatus.current ||
        latest.cycles !== previousValidationCycle.current
      ) {
        setShowValidationGate(true);
        previousValidationStatus.current = latest.validation_status;
        previousValidationCycle.current = latest.cycles;
      }
    }
  }, [latest?.validation_status, latest?.cycles]);

  // Upload state
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  const handleRetrain = async () => {
    setIsRetraining(true);
    setRetrainMsg("🔄 Retraining requested...");
    const msg = await triggerManualRetrain();
    setRetrainMsg(msg);
    setIsRetraining(false);
    setTimeout(() => setRetrainMsg(null), 5000);
  };

  const orchestratorReady = health?.orchestrator_ready ?? false;

  const handleSimulate = async () => {
    setIsSimulating(true);
    await simulateDrift();
    setIsSimulating(false);
  };

  const handleReset = async () => {
    if (confirm("Are you sure you want to reset all monitoring history and the prediction buffer?")) {
      setIsResetting(true);
      await resetDashboard();
      setIsResetting(false);
    }
  };

  const handleOpenRegistry = async () => {
    const data = await fetchRegistryHistory();
    setRegistryHistory(data);
    setIsRegistryOpen(true);
  };

  const handleRollback = async (versionId: string) => {
    setIsRollingBack(true);
    const msg = await rollbackModel(versionId);
    alert(msg);
    setIsRollingBack(false);
    await handleOpenRegistry();
  };

  const handleUpload = async () => {
    if (!uploadFile) return;
    setIsUploading(true);
    const msg = await uploadDataset(uploadFile);
    alert(msg);
    setIsUploading(false);
    setUploadFile(null);
  };

  const handleReplayBatch = async () => {
    setIsSimulating(true);
    await replayStep();
    setIsSimulating(false);
  };

  // Color mapping helpers based on status
  const getStatusColor = (status?: string) => {
    switch (status) {
      case "Healthy":
        return "text-emerald-400 border-emerald-500/30 bg-emerald-950/20";
      case "Warning":
        return "text-amber-400 border-amber-500/30 bg-amber-950/20";
      case "Critical":
        return "text-rose-400 border-rose-500/30 bg-rose-950/20";
      default:
        return "text-zinc-400 border-zinc-500/30 bg-zinc-950/20";
    }
  };

  const getAlertBadge = (level?: string) => {
    switch (level) {
      case "none":
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-950/40 text-emerald-400 border border-emerald-500/30">
            <Check className="w-3.5 h-3.5" /> Nominal
          </span>
        );
      case "warning":
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-amber-950/40 text-amber-400 border border-amber-500/30 animate-pulse">
            <AlertTriangle className="w-3.5 h-3.5" /> Warning
          </span>
        );
      case "critical":
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-rose-950/40 text-rose-400 border border-rose-500/30 animate-pulse">
            <ShieldAlert className="w-3.5 h-3.5" /> Critical
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-zinc-900 text-zinc-400 border border-zinc-700/50">
            Inactive
          </span>
        );
    }
  };

  // Render SVG Line Chart for Reconstruction Error
  const renderLineChart = () => {
    if (history.length === 0) {
      return (
        <div className="flex flex-col items-center justify-center h-64 text-zinc-500 border border-dashed border-zinc-800 rounded-2xl bg-zinc-950/30">
          <TrendingUp className="w-10 h-10 mb-2 text-zinc-700" />
          <p>No historical monitoring batches available.</p>
          <p className="text-xs text-zinc-600 mt-1">Start simulated ingestion to populate charts.</p>
        </div>
      );
    }

    const svgWidth = 600;
    const svgHeight = 260;
    const padding = { top: 20, right: 30, bottom: 40, left: 50 };
    const chartWidth = svgWidth - padding.left - padding.right;
    const chartHeight = svgHeight - padding.top - padding.bottom;

    // Find min/max values for scaling
    const errors = history.map((d) => d.mean_reconstruction_error);
    const thresholds = history.map((d) => d.dynamic_threshold);
    const maxVal = Math.max(...errors, ...thresholds, 0.01) * 1.2;
    const minVal = 0; // standard baseline is 0 for reconstruction errors

    const points = history.map((d, index) => {
      const x = padding.left + (index / Math.max(1, history.length - 1)) * chartWidth;
      const yErr = padding.top + chartHeight - ((d.mean_reconstruction_error - minVal) / (maxVal - minVal)) * chartHeight;
      const yThresh = padding.top + chartHeight - ((d.dynamic_threshold - minVal) / (maxVal - minVal)) * chartHeight;
      return { x, yErr, yThresh, data: d, index };
    });

    // Create line path for reconstruction error
    const errorLinePath = points.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.yErr}`).join(" ");
    
    // Create line path for threshold
    const threshLinePath = points.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.yThresh}`).join(" ");

    // Create gradient fill area path
    const areaPath = points.length > 0 
      ? `${errorLinePath} L ${points[points.length - 1].x} ${padding.top + chartHeight} L ${points[0].x} ${padding.top + chartHeight} Z`
      : "";

    // Generate grid lines
    const gridLines = [];
    const numGridLines = 4;
    for (let i = 0; i <= numGridLines; i++) {
      const yVal = minVal + (i / numGridLines) * (maxVal - minVal);
      const yPos = padding.top + chartHeight - (i / numGridLines) * chartHeight;
      gridLines.push(
        <g key={`grid-${i}`}>
          <line
            x1={padding.left}
            y1={yPos}
            x2={svgWidth - padding.right}
            y2={yPos}
            stroke="#1e293b"
            strokeWidth="1"
            strokeDasharray="4 4"
          />
          <text
            x={padding.left - 10}
            y={yPos + 4}
            fill="#64748b"
            fontSize="10"
            textAnchor="end"
            className="font-mono"
          >
            {yVal.toFixed(3)}
          </text>
        </g>
      );
    }

    return (
      <div className="relative bg-zinc-950/40 border border-zinc-900 rounded-2xl p-5 backdrop-blur-xl">
        <div className="flex justify-between items-center mb-4">
          <div className="flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-indigo-400" />
            <h3 className="text-sm font-semibold text-zinc-200">Reconstruction Error vs Dynamic Threshold</h3>
          </div>
          <div className="flex gap-4 text-xs">
            <span className="flex items-center gap-1.5 text-zinc-400">
              <span className="w-3 h-0.5 bg-indigo-500 inline-block"></span> Error
            </span>
            <span className="flex items-center gap-1.5 text-zinc-400">
              <span className="w-3 h-0.5 bg-rose-500 stroke-dasharray-[2_2] inline-block"></span> Threshold
            </span>
          </div>
        </div>

        <div className="w-full overflow-hidden">
          <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} className="w-full h-auto overflow-visible">
            <defs>
              <linearGradient id="chartGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#6366f1" stopOpacity="0.25" />
                <stop offset="100%" stopColor="#6366f1" stopOpacity="0.0" />
              </linearGradient>
            </defs>

            {/* Grid Lines */}
            {gridLines}

            {/* X-Axis labels (Batch IDs) */}
            {points.filter((_, idx) => idx % Math.max(1, Math.floor(points.length / 5)) === 0 || idx === points.length - 1).map((p, idx) => (
              <text
                key={`x-label-${idx}`}
                x={p.x}
                y={svgHeight - padding.bottom + 20}
                fill="#64748b"
                fontSize="10"
                textAnchor="middle"
                className="font-mono"
              >
                B{p.data.batch_id}
              </text>
            ))}

            {/* Gradient Area Fill */}
            {points.length > 0 && (
              <path d={areaPath} fill="url(#chartGradient)" />
            )}

            {/* Error Line */}
            {points.length > 0 && (
              <path
                d={errorLinePath}
                fill="none"
                stroke="#6366f1"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            )}

            {/* Threshold Line */}
            {points.length > 0 && (
              <path
                d={threshLinePath}
                fill="none"
                stroke="#f43f5e"
                strokeWidth="1.5"
                strokeDasharray="4 4"
                strokeLinecap="round"
              />
            )}

            {/* Interactive Circles & Tooltips */}
            {points.map((p) => {
              const isDrift = p.data.drift_confirmed;
              const isHovered = hoveredIndex === p.index;
              return (
                <g key={`point-${p.index}`}>
                  {/* Outer pulse for drift confirmed batches */}
                  {isDrift && (
                    <circle
                      cx={p.x}
                      cy={p.yErr}
                      r={isHovered ? "10" : "6"}
                      fill="none"
                      stroke="#f43f5e"
                      strokeWidth="1.5"
                      className="animate-ping origin-center"
                      style={{ transformOrigin: `${p.x}px ${p.yErr}px` }}
                    />
                  )}

                  {/* Intersecting vertical guide line on hover */}
                  {isHovered && (
                    <line
                      x1={p.x}
                      y1={padding.top}
                      x2={p.x}
                      y2={svgHeight - padding.bottom}
                      stroke="#475569"
                      strokeWidth="1"
                      strokeDasharray="2 2"
                    />
                  )}

                  {/* Main data point circle */}
                  <circle
                    cx={p.x}
                    cy={p.yErr}
                    r={isDrift ? "4.5" : isHovered ? "5" : "3.5"}
                    fill={isDrift ? "#ef4444" : "#6366f1"}
                    stroke={isDrift ? "#fee2e2" : "#312e81"}
                    strokeWidth={isHovered ? "2" : "1"}
                    className="transition-all cursor-pointer"
                    onMouseEnter={() => setHoveredIndex(p.index)}
                    onMouseLeave={() => setHoveredIndex(null)}
                  />

                  {/* Custom tooltip displaying batch information */}
                  {isHovered && (
                    <g className="pointer-events-none transition-all">
                      <rect
                        x={p.x > svgWidth / 2 ? p.x - 145 : p.x + 5}
                        y={p.yErr - 45}
                        width="140"
                        height="55"
                        rx="6"
                        fill="#09090b"
                        stroke="#27272a"
                        strokeWidth="1"
                      />
                      <text
                        x={p.x > svgWidth / 2 ? p.x - 137 : p.x + 13}
                        y={p.yErr - 30}
                        fill="#f4f4f5"
                        fontSize="10"
                        fontWeight="semibold"
                      >
                        Batch {p.data.batch_id} {isDrift ? "⚠️ DRIFT" : ""}
                      </text>
                      <text
                        x={p.x > svgWidth / 2 ? p.x - 137 : p.x + 13}
                        y={p.yErr - 18}
                        fill="#a1a1aa"
                        fontSize="9"
                        className="font-mono"
                      >
                        Error: {p.data.mean_reconstruction_error.toFixed(4)}
                      </text>
                      <text
                        x={p.x > svgWidth / 2 ? p.x - 137 : p.x + 13}
                        y={p.yErr - 6}
                        fill="#f43f5e"
                        fontSize="9"
                        className="font-mono"
                      >
                        Thresh: {p.data.dynamic_threshold.toFixed(4)}
                      </text>
                    </g>
                  )}
                </g>
              );
            })}
          </svg>
        </div>
      </div>
    );
  };

  // Render Feature Attribution: Toggle between SHAP (Model-centric) and KS-test (Data-centric)
  const renderAttributionChart = () => {
    const isDrift = latest?.drift_confirmed;
    const shapFeatures = latest?.top_drift_features || [];
    const ksFeatures = latest?.feature_ks_results || [];

    const hasData = activeAttributionTab === "shap" 
      ? (isDrift && shapFeatures.length > 0)
      : (ksFeatures.length > 0);

    if (!latest) {
      return (
        <div className="flex flex-col items-center justify-center h-80 text-zinc-500 border border-dashed border-zinc-800 rounded-2xl bg-zinc-950/30 p-6 text-center">
          <BarChart2 className="w-10 h-10 mb-2 text-zinc-700" />
          <p className="font-semibold text-zinc-400">Awaiting Data</p>
        </div>
      );
    }

    const currentFeatures = activeAttributionTab === "shap" ? shapFeatures : ksFeatures;
    const maxVal = currentFeatures.length > 0 ? Math.max(...currentFeatures.map((f) => f.importance), 0.00001) : 1;

    return (
      <div className="bg-zinc-950/40 border border-zinc-900 rounded-2xl p-5 backdrop-blur-xl h-full flex flex-col justify-between">
        <div>
          {/* Header & Tabs */}
          <div className="flex flex-col gap-3 mb-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <BarChart2 className="w-5 h-5 text-indigo-400" />
                <h3 className="text-sm font-semibold text-zinc-200">Feature Shift Attribution</h3>
              </div>
              <span className="text-[10px] text-zinc-500 font-mono">Batch {latest.batch_id}</span>
            </div>

            {/* Toggle tabs */}
            <div className="grid grid-cols-2 bg-zinc-900/60 p-0.5 rounded-lg border border-zinc-800/40">
              <button
                onClick={() => setActiveAttributionTab("shap")}
                className={`py-1.5 rounded-md text-[10px] font-mono font-semibold transition-all cursor-pointer ${
                  activeAttributionTab === "shap" 
                    ? "bg-zinc-800 text-white shadow-sm" 
                    : "text-zinc-500 hover:text-zinc-300"
                }`}
              >
                Model sensitivity (SHAP)
              </button>
              <button
                onClick={() => setActiveAttributionTab("ks")}
                className={`py-1.5 rounded-md text-[10px] font-mono font-semibold transition-all cursor-pointer ${
                  activeAttributionTab === "ks" 
                    ? "bg-zinc-800 text-white shadow-sm" 
                    : "text-zinc-500 hover:text-zinc-300"
                }`}
              >
                Distribution Shift (KS-Test)
              </button>
            </div>
          </div>

          {!hasData ? (
            <div className="flex flex-col items-center justify-center h-48 text-zinc-500 text-center p-4">
              <BarChart2 className="w-8 h-8 mb-2 text-zinc-700" />
              <p className="text-xs font-semibold text-zinc-400">
                {activeAttributionTab === "shap" ? "SHAP Explanations Inactive" : "No KS-Test Data"}
              </p>
              <p className="text-[10px] text-zinc-600 mt-1 max-w-[200px]">
                {activeAttributionTab === "shap" 
                  ? "SHAP runs strictly when data drift is confirmed (p < 0.01) to attribute model loss sensitivity."
                  : "Start simulated ingestion to evaluate feature-level Kolmogorov-Smirnov tests."}
              </p>
            </div>
          ) : (
            <div className="space-y-2.5">
              {currentFeatures.slice(0, 7).map((item, idx) => {
                const widthPct = (item.importance / maxVal) * 100;
                const isTargetSensor = ["LIT101", "DPIT301", "P402"].includes(item.feature);
                
                const pVal = "p_value" in item ? (item as any).p_value : null;
                const displayVal = activeAttributionTab === "shap" 
                  ? item.importance.toFixed(5)
                  : `stat: ${item.importance.toFixed(3)} (p: ${pVal !== null ? pVal.toExponential(1) : "N/A"})`;

                return (
                  <div key={idx} className="space-y-1">
                    <div className="flex justify-between text-[10px] font-mono">
                      <span className={`font-semibold ${isTargetSensor ? "text-rose-400" : "text-zinc-300"}`}>
                        {item.feature} {isTargetSensor ? "🔥" : ""}
                      </span>
                      <span className="text-zinc-500">{displayVal}</span>
                    </div>
                    <div className="w-full h-2 bg-zinc-900 rounded-full overflow-hidden border border-zinc-800/30">
                      <div
                        className={`h-full rounded-full transition-all duration-1000 ${
                          isTargetSensor
                            ? "bg-gradient-to-r from-rose-600 to-rose-400 shadow-[0_0_8px_rgba(244,63,94,0.3)]"
                            : "bg-gradient-to-r from-indigo-600 to-indigo-400"
                        }`}
                        style={{ width: `${widthPct}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        <div className="text-[9px] text-zinc-500 border-t border-zinc-900 pt-3.5 mt-4 flex items-center justify-between font-mono">
          <span>Injected: LIT101, DPIT301, P402</span>
          <span>
            {activeAttributionTab === "shap" ? "Method: VAE KernelExplainer" : "Method: Scipy 2-sample KS test"}
          </span>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col selection:bg-rose-500/30 selection:text-rose-200 relative overflow-hidden">
      
      {/* Background glow effects */}
      <div className="hero-glow top-[10%] left-[20%]"></div>
      <div className="hero-glow top-[50%] right-[10%] bg-[radial-gradient(circle,rgba(225,29,72,0.08)_0%,rgba(13,15,18,0)_70%)]"></div>

      {/* Sleek Navigation Bar */}
      <header className="border-b border-white/5 bg-black/40 backdrop-blur-xl sticky top-0 z-50 px-6 py-4 flex flex-col md:flex-row justify-between items-center gap-4">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-3">
            <div className="relative flex items-center justify-center w-8 h-8 rounded-lg bg-zinc-900 border border-zinc-800">
              <span className={`absolute inline-flex h-2.5 w-2.5 rounded-full opacity-75 ${isConnected ? "bg-emerald-400 animate-ping" : "bg-rose-400 animate-pulse"}`}></span>
              <span className={`relative inline-flex rounded-full h-2 w-2 ${isConnected ? "bg-emerald-500" : "bg-rose-500"}`}></span>
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight text-white flex items-center gap-2">
                MODEL DECAY RADAR
                <span className="text-[9px] uppercase font-mono px-1.5 py-0.5 rounded border border-zinc-800 text-zinc-400 bg-zinc-900/50">OSS</span>
              </h1>
            </div>
          </div>
          
          <nav className="hidden md:flex gap-6 text-sm font-medium text-zinc-400">
            <span className="text-white">Dashboard</span>
            <span className="hover:text-white cursor-pointer transition-colors" onClick={handleOpenRegistry}>Model Registry</span>
            <Link href="/docs" className="hover:text-white transition-colors">Documentation</Link>
          </nav>
        </div>

        <div className="flex items-center gap-3">
          <p className="text-[10px] text-zinc-500 font-mono hidden lg:block mr-4">
            {isConnected
              ? orchestratorReady
                ? "CONNECTED · READY"
                : health?.setup_progress === "training_ae"
                  ? "TRAINING AE..."
                  : health?.setup_progress === "training_rnn"
                    ? "TRAINING RNN..."
                    : health?.setup_progress === "failed"
                      ? "SETUP FAILED"
                      : "INITIALISING..."
              : "DISCONNECTED"}
          </p>

          {/* Operating Mode Toggle */}
          <div className="flex items-center bg-zinc-900/80 border border-zinc-800/80 rounded-lg p-1">
            <button
              onClick={() => setOperatingMode("demo")}
              className={`px-3 py-1.5 text-xs font-mono font-medium rounded-md transition-all ${
                (!config?.operating_mode || config.operating_mode === "demo")
                  ? "bg-zinc-800 text-white shadow-sm"
                  : "text-zinc-500 hover:text-zinc-300"
              }`}
            >
              Demo
            </button>
            <button
              onClick={() => setOperatingMode("real_data")}
              className={`px-3 py-1.5 text-xs font-mono font-medium rounded-md transition-all ${
                config?.operating_mode === "real_data"
                  ? "bg-zinc-800 text-white shadow-sm"
                  : "text-zinc-500 hover:text-zinc-300"
              }`}
            >
              Production
            </button>
          </div>

          <button
            onClick={refetch}
            className="p-2 border border-zinc-800 rounded-lg hover:bg-zinc-800 text-zinc-400 hover:text-white transition-all cursor-pointer"
            title="Force refresh"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
          
          <button
            onClick={handleReset}
            disabled={isResetting}
            className="p-2 border border-rose-500/20 rounded-lg bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 transition-all cursor-pointer disabled:opacity-50"
            title="Reset DB"
          >
            <Database className="w-4 h-4" />
          </button>

          {config?.operating_mode === "real_data" ? (
            <div className="flex items-center gap-2">
              <label className="flex items-center gap-2 px-3 py-2 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-white transition-all font-semibold text-xs cursor-pointer">
                <Upload className="w-3.5 h-3.5" />
                {uploadFile ? uploadFile.name : "CSV"}
                <input
                  type="file"
                  accept=".csv"
                  className="hidden"
                  onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                />
              </label>
              {uploadFile && (
                <button
                  onClick={handleUpload}
                  disabled={isUploading}
                  className="px-3 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-xs cursor-pointer disabled:opacity-50"
                >
                  {isUploading ? "..." : "Up"}
                </button>
              )}
              <button
                onClick={handleReplayBatch}
                disabled={isSimulating || !isConnected || !orchestratorReady}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-rose-600 hover:bg-rose-500 text-white shadow-lg shadow-rose-900/50 transition-all font-semibold text-xs cursor-pointer disabled:opacity-50"
              >
                <Play className="w-3.5 h-3.5 fill-current" />
                Replay Batch
              </button>
            </div>
          ) : (
            <button
              onClick={handleSimulate}
              disabled={isSimulating || !isConnected || !orchestratorReady}
              className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-rose-600 hover:bg-rose-500 text-white shadow-lg shadow-rose-900/50 transition-all font-semibold text-xs cursor-pointer disabled:opacity-50"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              {isSimulating ? "Injecting..." : !orchestratorReady ? "Training…" : "Simulate Drift"}
            </button>
          )}
        </div>
      </header>

      {/* Main dashboard content */}
      <main className="flex-1 p-6 space-y-8 max-w-[1400px] mx-auto w-full z-10">
        
        {/* Hero Section Header */}
        <div className="text-center md:text-left py-8 md:py-12 px-4 max-w-3xl">
          <h2 className="text-sm font-semibold uppercase tracking-widest text-zinc-500 mb-3">Model Quality & Observability</h2>
          <h1 className="text-4xl md:text-5xl font-bold tracking-tight text-white mb-4 leading-tight">
            Monitor and test your <span className="text-gradient">predictive ML systems</span>
          </h1>
          <p className="text-lg text-zinc-400 leading-relaxed">
            Track data drift, model performance, and data quality in real-time. Detect issues before they impact your application.
          </p>
        </div>

        {error && (
          <div className="glass-card bg-rose-950/20 border-rose-500/30 p-4 flex items-center gap-3 text-rose-400 text-sm">
            <ShieldAlert className="w-5 h-5 flex-shrink-0" />
            <div>
              <span className="font-semibold">Backend Error:</span> {error}
            </div>
          </div>
        )}

        {isConnected && !orchestratorReady && health?.setup_progress !== "failed" && (
          <div className="glass-card bg-zinc-900/50 border-zinc-700/50 p-4 flex items-center gap-3 text-zinc-300 text-sm animate-pulse">
            <RefreshCw className="w-5 h-5 flex-shrink-0 animate-spin text-zinc-400" />
            <div>
              <span className="font-semibold">Initialising:</span> Models are training on reference data. This takes ~1-2 minutes.
            </div>
          </div>
        )}

        {health?.setup_progress === "failed" && (
          <div className="glass-card bg-rose-950/20 border-rose-500/30 p-4 flex items-center gap-3 text-rose-400 text-sm">
            <ShieldAlert className="w-5 h-5 flex-shrink-0" />
            <div><span className="font-semibold">Setup Failed:</span> {health.setup_error}</div>
          </div>
        )}

        {isLoading && !latest ? (
          <div className="flex flex-col items-center justify-center py-32 text-zinc-500 gap-4">
            <RefreshCw className="w-8 h-8 animate-spin text-rose-400" />
            <p className="font-mono text-xs uppercase tracking-widest">Awaiting telemetry...</p>
          </div>
        ) : (
          <>
            {/* Top Row: Metric Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
              
              {/* Card 1: Model Health Score (MHS) */}
              <div className="glass-card p-6 flex flex-col justify-between hover:bg-white/[0.04] transition-all group relative overflow-hidden">
                <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                  <Cpu className="w-16 h-16 text-white" />
                </div>
                <div className="flex justify-between items-start text-zinc-400 relative z-10">
                  <span className="text-xs font-semibold uppercase tracking-wider">Model Health Score</span>
                </div>
                <div className="my-6 relative z-10">
                  <div className="text-4xl font-bold tracking-tight text-white font-mono">
                    {latest?.mhs ? `${(latest.mhs * 100).toFixed(1)}%` : "N/A"}
                  </div>
                  <div className="flex items-center gap-2 mt-3">
                    <span className={`inline-flex items-center px-2.5 py-1 rounded-md text-[10px] font-bold uppercase tracking-wider ${getStatusColor(latest?.mhs_status)}`}>
                      {latest?.mhs_status || "Pending"}
                    </span>
                  </div>
                </div>
                <div className="w-full bg-zinc-900/80 h-1.5 rounded-full overflow-hidden relative z-10">
                  <div
                    className={`h-full transition-all duration-1000 ${
                      latest?.mhs_status === "Healthy"
                        ? "bg-emerald-500"
                        : latest?.mhs_status === "Warning"
                        ? "bg-amber-500"
                        : "bg-rose-500"
                    }`}
                    style={{ width: latest?.mhs ? `${latest.mhs * 100}%` : "0%" }}
                  />
                </div>
              </div>

              {/* Card 2: Drift Confirmation */}
              <div className="glass-card p-6 flex flex-col justify-between hover:bg-white/[0.04] transition-all">
                <div className="flex justify-between items-start text-zinc-400">
                  <span className="text-xs font-semibold uppercase tracking-wider">Data Drift Status</span>
                  <Activity className="w-4 h-4 text-zinc-500" />
                </div>
                <div className="my-6">
                  <div className={`text-2xl md:text-3xl font-bold tracking-tight ${latest?.drift_confirmed ? "text-rose-400" : "text-white"}`}>
                    {latest?.drift_confirmed ? "Drift Detected" : latest ? "No Drift" : "N/A"}
                  </div>
                  <div className="text-[11px] text-zinc-500 font-mono mt-2">
                    p-value: <span className="text-zinc-300">{latest?.p_value !== undefined ? latest.p_value.toFixed(6) : "N/A"}</span>
                  </div>
                </div>
                <div className="text-[10px] text-zinc-600 font-medium">
                  Permutation tests running on background batches
                </div>
              </div>

              {/* Card 3: Alert Level */}
              <div className="glass-card p-6 flex flex-col justify-between hover:bg-white/[0.04] transition-all">
                <div className="flex justify-between items-start text-zinc-400">
                  <span className="text-xs font-semibold uppercase tracking-wider">System Alerts</span>
                  <ShieldAlert className="w-4 h-4 text-zinc-500" />
                </div>
                <div className="my-6">
                  <div className="mb-3">
                    {getAlertBadge(latest?.alert_level)}
                  </div>
                  <div className="text-sm text-zinc-300 line-clamp-2 leading-snug">
                    {latest?.alert_message || "Awaiting inputs..."}
                  </div>
                </div>
                <div className="text-[10px] text-zinc-600 font-medium">
                  Retrained: {latest?.retraining_triggered ? <span className="text-amber-400 font-bold">YES</span> : "NO"}
                </div>
              </div>

              {/* Card 4: F1 Score / Performance */}
              <div className="glass-card p-6 flex flex-col justify-between hover:bg-white/[0.04] transition-all">
                <div className="flex justify-between items-start text-zinc-400">
                  <span className="text-xs font-semibold uppercase tracking-wider">Batch F1-Score</span>
                  <Layers className="w-4 h-4 text-zinc-500" />
                </div>
                <div className="my-6">
                  <div className="text-3xl font-bold tracking-tight text-white font-mono">
                    {latest?.f1_score !== undefined ? `${(latest.f1_score * 100).toFixed(1)}%` : "N/A"}
                  </div>
                  <div className="text-[11px] text-zinc-500 font-mono mt-2 flex gap-3">
                    <span>Acc: <span className="text-zinc-300">{latest?.accuracy !== undefined ? (latest.accuracy*100).toFixed(1) : "-"}%</span></span>
                    <span>Rec: <span className="text-zinc-300">{latest?.recall !== undefined ? (latest.recall*100).toFixed(1) : "-"}%</span></span>
                  </div>
                </div>
                <div className="text-[10px] text-zinc-600 font-medium">
                  Active Model: {config?.active_classifier || "RF"}
                </div>
              </div>

            </div>

            {/* Middle Row: Visual Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2 glass-card p-2">
                {renderLineChart()}
              </div>
              <div className="lg:col-span-1 glass-card p-2">
                {renderAttributionChart()}
              </div>
            </div>

            {/* Config & Table Row */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              
              {/* Left Config Panel */}
              <div className="lg:col-span-3 glass-card p-6 space-y-6 flex flex-col">
                <div className="mb-2">
                  <h3 className="text-sm font-bold text-white uppercase tracking-wider">Settings</h3>
                  <p className="text-[11px] text-zinc-500 mt-1">Adjust monitoring parameters</p>
                </div>
                
                <div className="space-y-2 flex-1">
                  <label className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider block">Classifier Model</label>
                  <select
                    value={config?.active_classifier || "Random Forest"}
                    onChange={async (e) => {
                      await updateConfig({ active_classifier: e.target.value });
                    }}
                    className="w-full bg-zinc-900/50 border border-zinc-800 rounded-lg px-3 py-2.5 text-xs text-white focus:outline-none focus:border-rose-500 transition-colors cursor-pointer"
                  >
                    {config?.available_classifiers.map((name) => (
                      <option key={name} value={name}>{name}</option>
                    ))}
                  </select>
                </div>

                <div className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex justify-between text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">
                      <span>Buffer Window</span>
                      <span className="text-white">{config?.window_size || 500}</span>
                    </div>
                    <input
                      type="range"
                      min="50"
                      max="1500"
                      step="50"
                      value={config?.window_size || 500}
                      onChange={async (e) => {
                        await updateConfig({ window_size: parseInt(e.target.value) });
                      }}
                      className="w-full h-1 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-rose-500"
                    />
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">
                      <span>Drift Alpha</span>
                      <span className="text-white">{config?.p_value_threshold || 0.01}</span>
                    </div>
                    <input
                      type="range"
                      min="0.001"
                      max="0.1"
                      step="0.001"
                      value={config?.p_value_threshold || 0.01}
                      onChange={async (e) => {
                        await updateConfig({ p_value_threshold: parseFloat(e.target.value) });
                      }}
                      className="w-full h-1 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-rose-500"
                    />
                  </div>
                </div>

                <div className="pt-4 border-t border-zinc-800/50">
                  <button
                    onClick={handleRetrain}
                    disabled={isRetraining || !orchestratorReady}
                    className="w-full flex items-center justify-center gap-2 px-4 py-3 border border-amber-500/20 rounded-xl bg-amber-500/10 hover:bg-amber-500/20 text-amber-400 transition-all text-xs font-bold uppercase tracking-wider cursor-pointer disabled:opacity-50"
                  >
                    <RefreshCw className={`w-3.5 h-3.5 ${isRetraining ? "animate-spin" : ""}`} />
                    {isRetraining ? "Retraining..." : "Force Retrain"}
                  </button>
                  {retrainMsg && (
                    <span className="text-[10px] text-amber-400 text-center block mt-2 font-medium">
                      {retrainMsg}
                    </span>
                  )}
                </div>
              </div>

              {/* Right Table Panel */}
              <div className="lg:col-span-9 glass-card p-6 overflow-hidden flex flex-col">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-sm font-bold text-white uppercase tracking-wider">History Logs</h3>
                  <button onClick={handleOpenRegistry} className="text-xs font-semibold text-rose-400 hover:text-rose-300 transition-colors flex items-center gap-1">
                    View Registry <ChevronRight className="w-3 h-3" />
                  </button>
                </div>

                {history.length === 0 ? (
                  <div className="flex-1 flex items-center justify-center text-zinc-600 text-sm">
                    No records to display.
                  </div>
                ) : (
                  <div className="overflow-x-auto overflow-y-auto max-h-[350px] w-full pr-2">
                    <table className="w-full border-collapse text-left text-xs font-medium">
                      <thead className="sticky top-0 bg-[#0d0f12]/90 backdrop-blur z-10">
                        <tr className="border-b border-zinc-800 text-zinc-500 pb-2 uppercase tracking-wider text-[10px]">
                          <th className="py-3 px-3">Batch ID</th>
                          <th className="py-3 px-3 text-right">MHS</th>
                          <th className="py-3 px-3 text-right">Accuracy</th>
                          <th className="py-3 px-3 text-right">F1-Score</th>
                          <th className="py-3 px-3 text-right">Uncertainty</th>
                          <th className="py-3 px-3 text-center">Drift Detected</th>
                          <th className="py-3 px-3">Status Message</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-zinc-900/60 text-zinc-400">
                        {[...history].reverse().map((item, idx) => (
                          <tr
                            key={idx}
                            className={`hover:bg-white/[0.02] transition-colors ${
                              item.drift_confirmed ? "bg-rose-950/10" : ""
                            }`}
                          >
                            <td className="py-3 px-3 text-white font-mono">B{item.batch_id}</td>
                            <td className="py-3 px-3 text-right text-white font-mono">
                              {(item.mhs * 100).toFixed(1)}%
                            </td>
                            <td className="py-3 px-3 text-right text-zinc-300 font-mono">
                              {(item.accuracy * 100).toFixed(1)}%
                            </td>
                            <td className="py-3 px-3 text-right text-zinc-300 font-mono">
                              {item.f1_score !== undefined ? `${(item.f1_score * 100).toFixed(1)}%` : "-"}
                            </td>
                            <td className="py-3 px-3 text-right text-zinc-500 font-mono">
                              {(item.mean_uncertainty * 100).toFixed(1)}%
                            </td>
                            <td className="py-3 px-3 text-center">
                              {item.drift_confirmed ? (
                                <span className="inline-block bg-rose-500/20 text-rose-400 px-2 py-0.5 rounded text-[10px] font-bold tracking-wider uppercase">Drift</span>
                              ) : (
                                <span className="text-zinc-600 text-[10px] font-bold tracking-wider uppercase">-</span>
                              )}
                            </td>
                            <td className="py-3 px-3 max-w-[200px] truncate text-[11px] text-zinc-500" title={item.alert_message}>
                              {item.alert_message}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </div>
          </>
        )}
      </main>

      {/* Modals remain structurally similar but updated with dark theme classes */}
      {/* Registry Modal */}
      {isRegistryOpen && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-md p-4">
          <div className="bg-[#0d0f12] border border-zinc-800 rounded-2xl w-full max-w-4xl shadow-2xl overflow-hidden flex flex-col max-h-[85vh]">
            <div className="flex items-center justify-between p-6 border-b border-zinc-800 bg-white/[0.02]">
              <div className="flex items-center gap-3">
                <History className="w-5 h-5 text-rose-400" />
                <h2 className="text-lg font-bold text-white">Model Registry</h2>
              </div>
              <button 
                onClick={() => setIsRegistryOpen(false)}
                className="text-zinc-500 hover:text-white transition-colors"
              >
                ✕
              </button>
            </div>
            
            <div className="p-6 overflow-y-auto flex-1">
              <table className="w-full border-collapse text-left text-xs">
                <thead>
                  <tr className="border-b border-zinc-800 text-zinc-500 pb-2 uppercase text-[10px] tracking-wider font-semibold">
                    <th className="py-3 px-3">Version ID</th>
                    <th className="py-3 px-3">Type</th>
                    <th className="py-3 px-3">Timestamp</th>
                    <th className="py-3 px-3 text-right">Metrics</th>
                    <th className="py-3 px-3 text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-900 text-zinc-300">
                  {registryHistory.length === 0 ? (
                    <tr>
                      <td colSpan={5} className="py-10 text-center text-zinc-600">No models in registry.</td>
                    </tr>
                  ) : (
                    registryHistory.map((model, idx) => (
                      <tr key={idx} className="hover:bg-white/[0.02] transition-colors">
                        <td className="py-4 px-3 font-mono font-semibold text-white">
                          {model.version_id}
                          {idx === registryHistory.length - 1 && " (Active)"}
                        </td>
                        <td className="py-4 px-3 text-zinc-500 uppercase text-[10px] tracking-wider font-bold">
                          {model.model_type}
                        </td>
                        <td className="py-4 px-3 text-zinc-500 font-mono">
                          {new Date(model.timestamp).toLocaleString()}
                        </td>
                        <td className="py-4 px-3 text-right text-zinc-300 font-mono">
                          {model.metrics?.accuracy ? `Acc: ${(model.metrics.accuracy * 100).toFixed(1)}%` : "-"}
                        </td>
                        <td className="py-4 px-3 text-right">
                          <button
                            onClick={() => handleRollback(model.version_id)}
                            disabled={isRollingBack}
                            className="px-4 py-1.5 bg-zinc-800 hover:bg-zinc-700 text-white rounded-lg text-[10px] uppercase font-bold tracking-wider transition-colors disabled:opacity-50"
                          >
                            Rollback
                          </button>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Validation Gate Modal */}
      {showValidationGate && latest?.validation_metrics && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md">
          <div className="bg-[#0d0f12] border border-zinc-800 rounded-2xl shadow-2xl w-[600px] overflow-hidden flex flex-col">
            <div className="px-6 py-5 border-b border-zinc-800 flex justify-between items-center bg-white/[0.02]">
              <h2 className="text-lg font-bold text-white flex items-center gap-2">
                <CheckCircle className={`w-5 h-5 ${latest?.validation_status === "promoted" ? "text-emerald-500" : "text-rose-500"}`} />
                Validation Gate Result
              </h2>
              <button
                className="text-zinc-500 hover:text-white transition-colors"
                onClick={() => setShowValidationGate(false)}
              >
                ✕
              </button>
            </div>
            
            <div className="p-6">
              <div className={`mb-6 p-5 rounded-xl border ${
                latest?.validation_status === "promoted" 
                  ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-400" 
                  : "bg-rose-500/10 border-rose-500/30 text-rose-400"
              }`}>
                <h3 className="font-bold text-base mb-1">
                  {latest?.validation_status === "promoted" ? "Model Promoted" : "Model Rejected"}
                </h3>
                <p className="text-sm opacity-80 leading-relaxed">
                  {latest?.validation_status === "promoted" 
                    ? "The candidate model outperformed the baseline safely and has been promoted to production."
                    : "The candidate model failed to meet safety margins. Active baseline remains in production."}
                </p>
              </div>

              <div className="grid grid-cols-3 gap-4 mb-2 border-b border-zinc-800 pb-2">
                <div className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider text-center">Metric</div>
                <div className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider text-center">Baseline</div>
                <div className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider text-center">Candidate</div>
              </div>

              <div className="space-y-2 mt-4">
                <div className="grid grid-cols-3 gap-4 p-3 bg-white/[0.02] rounded-lg">
                  <div className="text-xs font-semibold text-zinc-400 flex items-center justify-center">Accuracy</div>
                  <div className="text-sm font-mono text-center text-zinc-300">
                    {(latest.validation_metrics.active_accuracy * 100).toFixed(2)}%
                  </div>
                  <div className={`text-sm font-mono text-center font-bold ${
                    latest.validation_metrics.candidate_accuracy >= latest.validation_metrics.active_accuracy ? "text-emerald-400" : "text-rose-400"
                  }`}>
                    {(latest.validation_metrics.candidate_accuracy * 100).toFixed(2)}%
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-4 p-3 bg-white/[0.02] rounded-lg">
                  <div className="text-xs font-semibold text-zinc-400 flex items-center justify-center">F1-Score</div>
                  <div className="text-sm font-mono text-center text-zinc-300">
                    {(latest.validation_metrics.active_f1 * 100).toFixed(2)}%
                  </div>
                  <div className={`text-sm font-mono text-center font-bold ${
                    latest.validation_metrics.candidate_f1 >= latest.validation_metrics.active_f1 - 0.02 ? "text-emerald-400" : "text-rose-400"
                  }`}>
                    {(latest.validation_metrics.candidate_f1 * 100).toFixed(2)}%
                  </div>
                </div>
              </div>

              <div className="mt-8 flex justify-end">
                <button
                  className="px-6 py-2.5 bg-zinc-800 hover:bg-zinc-700 text-white rounded-lg font-bold text-xs uppercase tracking-wider transition-colors"
                  onClick={() => setShowValidationGate(false)}
                >
                  Dismiss
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
