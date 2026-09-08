"use client";

import Link from "next/link";
import { useState, useEffect } from "react";
import {
  ChevronRight,
  BookOpen,
  Terminal,
  Activity,
  Layers,
  FolderOpen,
  Database,
  ArrowLeft,
  GitBranch
} from "lucide-react";

export default function DocsPage() {
  const [activeSection, setActiveSection] = useState("introduction");

  // Intersection observer to highlight active section in sidebar based on scroll
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setActiveSection(entry.target.id);
          }
        });
      },
      { rootMargin: "-100px 0px -60% 0px" }
    );

    const sections = document.querySelectorAll("section[id]");
    sections.forEach((section) => observer.observe(section));

    return () => {
      sections.forEach((section) => observer.unobserve(section));
    };
  }, []);

  const scrollTo = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      const top = element.getBoundingClientRect().top + window.scrollY - 100;
      window.scrollTo({ top, behavior: "smooth" });
    }
  };

  const navItems = [
    {
      group: "Get Started",
      items: [
        { id: "introduction", title: "Introduction", icon: <BookOpen className="w-4 h-4" /> },
        { id: "quickstart", title: "Quickstart", icon: <Terminal className="w-4 h-4" /> },
      ]
    },
    {
      group: "Deep Dive",
      items: [
        { id: "architecture", title: "System Architecture", icon: <Activity className="w-4 h-4" /> },
        { id: "pipeline", title: "7-Layer Pipeline", icon: <Layers className="w-4 h-4" /> },
        { id: "dataset", title: "Synthetic Dataset", icon: <Database className="w-4 h-4" /> },
      ]
    },
    {
      group: "Reference",
      items: [
        { id: "structure", title: "Folder Structure", icon: <FolderOpen className="w-4 h-4" /> },
      ]
    }
  ];

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col selection:bg-rose-500/30 selection:text-rose-200">
      
      {/* Sleek Navigation Bar */}
      <header className="border-b border-white/5 bg-black/40 backdrop-blur-xl sticky top-0 z-50 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-6">
          <Link href="/" className="flex items-center gap-3 group">
            <div className="relative flex items-center justify-center w-8 h-8 rounded-lg bg-zinc-900 border border-zinc-800 group-hover:border-rose-500/50 transition-colors">
              <span className="relative inline-flex rounded-full h-2 w-2 bg-rose-500"></span>
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight text-white flex items-center gap-2">
                MODEL DECAY RADAR
                <span className="text-[9px] uppercase font-mono px-1.5 py-0.5 rounded border border-zinc-800 text-zinc-400 bg-zinc-900/50">Docs</span>
              </h1>
            </div>
          </Link>
        </div>

        <div className="flex items-center gap-4 text-sm font-medium">
          <Link href="/" className="flex items-center gap-2 text-zinc-400 hover:text-white transition-colors">
            <ArrowLeft className="w-4 h-4" /> Back to Dashboard
          </Link>
          <a href="https://github.com/vijay/Model-Decay-Radar" target="_blank" rel="noreferrer" className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-zinc-900 border border-zinc-800 hover:bg-zinc-800 text-zinc-300 hover:text-white transition-colors">
            <GitBranch className="w-4 h-4" /> GitHub
          </a>
        </div>
      </header>

      <div className="flex-1 flex max-w-[1400px] w-full mx-auto">
        
        {/* Left Sidebar */}
        <aside className="hidden md:block w-64 border-r border-white/5 bg-black/20 p-6 sticky top-[73px] h-[calc(100vh-73px)] overflow-y-auto">
          {navItems.map((group, i) => (
            <div key={i} className="mb-8">
              <h4 className="text-xs font-bold text-zinc-500 uppercase tracking-wider mb-3">{group.group}</h4>
              <ul className="space-y-1">
                {group.items.map((item) => (
                  <li key={item.id}>
                    <button
                      onClick={() => scrollTo(item.id)}
                      className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-all ${
                        activeSection === item.id 
                          ? "bg-rose-500/10 text-rose-400 font-semibold" 
                          : "text-zinc-400 hover:bg-white/5 hover:text-zinc-200"
                      }`}
                    >
                      {item.icon}
                      {item.title}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </aside>

        {/* Main Content Area */}
        <main className="flex-1 p-8 md:p-12 lg:p-16 overflow-y-auto max-w-4xl">
          <div className="space-y-24 pb-32">
            
            {/* Introduction */}
            <section id="introduction" className="scroll-mt-24 space-y-6">
              <div>
                <h1 className="text-4xl font-bold tracking-tight text-white mb-4">What is Model Decay Radar?</h1>
                <p className="text-lg text-zinc-400 leading-relaxed">
                  Real-time ML model monitoring demo inspired by industrial IoT (SWaT-style water treatment sensors). Simulates a deployed model receiving live sensor data, detects when the data distribution drifts, explains which sensors changed, and triggers alerts/retraining — all visible on a live dashboard.
                </p>
              </div>

              <div className="glass-card bg-amber-500/10 border-amber-500/20 p-5 rounded-xl">
                <h4 className="text-amber-400 font-semibold mb-2">Notice</h4>
                <p className="text-sm text-zinc-300">
                  This is <strong>not</strong> a full production MLOps platform. It is an MVP that wires together drift detection, uncertainty, SHAP, and alerting into one runnable demo.
                </p>
              </div>

              <div className="space-y-4">
                <h2 className="text-2xl font-bold text-white border-b border-zinc-800 pb-2">High-Level Idea</h2>
                <p className="text-zinc-300 leading-relaxed">
                  Imagine you deployed an ML model on 51 industrial sensors (flow, level, pressure, actuators, etc.). Over time, the plant behavior changes — sensors drift, valves wear, processes shift. Your model may silently get worse.
                </p>
                <p className="text-zinc-300 leading-relaxed">
                  <strong>Model Decay Radar</strong> watches incoming data in the background and answers:
                </p>
                <ul className="list-disc list-inside space-y-2 text-zinc-300 ml-4">
                  <li>Is the model still healthy?</li>
                  <li>Has data drift occurred (statistically)?</li>
                  <li>Which sensors caused the drift?</li>
                  <li>Should we alert or retrain?</li>
                </ul>
                <p className="text-zinc-300 leading-relaxed mt-4">
                  The user/IoT device only calls <code className="bg-zinc-800 px-1.5 py-0.5 rounded text-rose-400 font-mono text-sm">POST /predict</code> and gets a prediction back. Monitoring happens invisibly in the background.
                </p>
              </div>
            </section>

            {/* Quickstart */}
            <section id="quickstart" className="scroll-mt-24 space-y-6">
              <h2 className="text-3xl font-bold tracking-tight text-white mb-6">Quickstart</h2>
              
              <div className="space-y-4">
                <h3 className="text-xl font-bold text-white">1. Install Dependencies</h3>
                <div className="glass-card bg-zinc-950 p-4 rounded-xl border border-zinc-800 overflow-x-auto">
                  <pre className="text-sm font-mono text-zinc-300">
                    <span className="text-zinc-500"># Python Dependencies</span><br/>
                    pip install -r requirements.txt<br/><br/>
                    <span className="text-zinc-500"># Next.js Frontend Dependencies</span><br/>
                    cd frontend<br/>
                    npm install
                  </pre>
                </div>
              </div>

              <div className="space-y-4">
                <h3 className="text-xl font-bold text-white">2. Run the Services</h3>
                <p className="text-zinc-300">Simply run the startup script (Bash / Git Bash / Linux / macOS):</p>
                <div className="glass-card bg-zinc-950 p-4 rounded-xl border border-zinc-800 overflow-x-auto">
                  <pre className="text-sm font-mono text-zinc-300">
                    bash run.sh
                  </pre>
                </div>
                <p className="text-zinc-300">
                  Wait until the backend shows orchestrator readiness (takes ~1 minute for background training), then open the dashboard at <a href="http://localhost:3000" className="text-rose-400 hover:underline">http://localhost:3000</a>.
                </p>
                <p className="text-zinc-300 mt-4">To stop the services:</p>
                <div className="glass-card bg-zinc-950 p-4 rounded-xl border border-zinc-800 overflow-x-auto">
                  <pre className="text-sm font-mono text-zinc-300">
                    bash stop.sh
                  </pre>
                </div>
              </div>
            </section>

            {/* System Architecture */}
            <section id="architecture" className="scroll-mt-24 space-y-6">
              <h2 className="text-3xl font-bold tracking-tight text-white mb-6">System Architecture</h2>
              <p className="text-zinc-300 leading-relaxed mb-4">
                The platform is separated into a lightweight prediction API and a heavy background orchestrator. Data is buffered until a window limit is reached, then piped through a multi-stage monitoring engine.
              </p>
              
              <div className="glass-card bg-zinc-900/50 p-6 rounded-xl border border-zinc-800 overflow-x-auto">
                <pre className="text-xs sm:text-sm font-mono text-emerald-400 leading-relaxed">
{`┌─────────────┐     POST /predict      ┌─────────────────────────────────────────┐
│ Client/IoT  │ ─────────────────────► │ FastAPI Server (port 8000)              │
│ Simulation  │                        │  • Active Production Classifier (RF/GB) │
│ Engine      │                        │  • Raw Sample Buffer (500 samples)      │
└─────────────┘                        │  • Delayed Ground Truth Tracker         │
                                       └────────────────────┬────────────────────┘
                                                            │ every 500 samples
                                                            ▼
                                       ┌─────────────────────────────────────────┐
                                       │ RadarOrchestrator Pipeline (7 Layers)   │
                                       └────────────────────┬────────────────────┘
                                                            │
                                                            ▼
                                       ┌─────────────────────────────────────────┐
                                       │ Validation Gate & SQLite Model Registry │
                                       │  • Compares Candidate vs Active Model   │
                                       │  • Version Tracking (v1, v2, v3)        │
                                       │  • Post-Promotion Reference Update Gate │
                                       └────────────────────┬────────────────────┘
                                                            │ MonitoringResult
                                                            ▼
┌─────────────┐     polls every 3s     ┌─────────────────────────────────────────┐
│ Next.js     │ ◄───────────────────── │ In-memory Monitoring History Log        │
│ Dashboard   │   /monitoring/*        │ (last 200 results)                      │
└─────────────┘                        └─────────────────────────────────────────┘`}
                </pre>
              </div>
            </section>

            {/* 7-Layer Pipeline */}
            <section id="pipeline" className="scroll-mt-24 space-y-6">
              <h2 className="text-3xl font-bold tracking-tight text-white mb-6">The 7-Layer Monitoring Pipeline</h2>
              <p className="text-zinc-300 leading-relaxed mb-6">
                Implemented in <code className="text-rose-400">pipeline/orchestrator.py</code>. Runs every time 500 samples fill the buffer.
              </p>

              <div className="grid gap-4">
                {[
                  { title: "Layer 1 — Data Ingestion", desc: "Incoming features are scaled with a MinMaxScaler fit on the reference window. Samples are buffered until the window is full, triggering the background cycle." },
                  { title: "Layer 2 — Class Imbalance", desc: "Borderline SMOTE applies minority class oversampling if anomalies are detected, stabilizing downstream training." },
                  { title: "Layer 3 — Distribution Drift (VAE)", desc: "A Variational Autoencoder calculates reconstruction errors. A 1D-KDE models the error distribution, and a Permutation Test computes a p-value to strictly confirm drift." },
                  { title: "Layer 4 — MC Dropout Uncertainty", desc: "An LSTM/RNN ensemble with Monte Carlo Dropout runs 50 forward passes per sample to quantify predictive epistemic uncertainty during the drift phase." },
                  { title: "Layer 5 — Root Cause Attribution", desc: "If drift occurs, the VAE bottleneck gradients are passed to a SHAP KernelExplainer to pinpoint the exact sensors causing the model to fail, paired with KS-Tests." },
                  { title: "Layer 6 — Health Scoring", desc: "The Model Health Score (MHS) is calculated as an inverse weighted penalty of KL divergence, uncertainty spikes, and F1 degradation." },
                  { title: "Layer 7 — Automated Retraining", desc: "If MHS drops below 85% and drift is confirmed, a Candidate Model is trained on the combined historical and drifted window." }
                ].map((layer, idx) => (
                  <div key={idx} className="glass-card p-5 border border-zinc-800 rounded-xl hover:border-zinc-700 transition-colors">
                    <h4 className="text-white font-bold mb-2 flex items-center gap-2">
                      <span className="flex items-center justify-center w-6 h-6 rounded bg-rose-500/20 text-rose-400 text-xs font-mono">{idx + 1}</span>
                      {layer.title}
                    </h4>
                    <p className="text-sm text-zinc-400 pl-8">{layer.desc}</p>
                  </div>
                ))}
              </div>
            </section>

            {/* Synthetic Dataset */}
            <section id="dataset" className="scroll-mt-24 space-y-6">
              <h2 className="text-3xl font-bold tracking-tight text-white mb-6">The Synthetic Dataset</h2>
              <p className="text-zinc-300 leading-relaxed mb-4">
                The project ships with <code className="text-rose-400">data/synthetic_swat.csv</code> (38,000 rows, 51 sensors).
              </p>
              
              <div className="overflow-hidden rounded-xl border border-zinc-800">
                <table className="w-full text-left text-sm text-zinc-300">
                  <thead className="bg-zinc-900 border-b border-zinc-800 text-zinc-400">
                    <tr>
                      <th className="px-4 py-3 font-semibold uppercase tracking-wider text-xs">Window</th>
                      <th className="px-4 py-3 font-semibold uppercase tracking-wider text-xs">Rows</th>
                      <th className="px-4 py-3 font-semibold uppercase tracking-wider text-xs">Meaning</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-800/50">
                    <tr>
                      <td className="px-4 py-3 font-mono text-white">reference</td>
                      <td className="px-4 py-3 text-zinc-400">20,000</td>
                      <td className="px-4 py-3">Historical baseline — used to train AE/RNN and fit scaler</td>
                    </tr>
                    <tr>
                      <td className="px-4 py-3 font-mono text-white">current_stable</td>
                      <td className="px-4 py-3 text-zinc-400">10,000</td>
                      <td className="px-4 py-3">Recent production data, still similar to reference</td>
                    </tr>
                    <tr>
                      <td className="px-4 py-3 font-mono text-white">current_drift</td>
                      <td className="px-4 py-3 text-zinc-400">8,000</td>
                      <td className="px-4 py-3">Drift injected into 3 sensors: LIT101, DPIT301, P402 (ramp + frequency change + noise)</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </section>

            {/* Folder Structure */}
            <section id="structure" className="scroll-mt-24 space-y-6">
              <h2 className="text-3xl font-bold tracking-tight text-white mb-6">Folder Structure</h2>
              
              <div className="glass-card p-6 rounded-xl border border-zinc-800">
                <ul className="space-y-3 font-mono text-sm">
                  <li className="flex gap-4"><span className="text-rose-400 w-48 shrink-0">api/server.py</span><span className="text-zinc-400">FastAPI server — prediction endpoint, buffer, triggers</span></li>
                  <li className="flex gap-4"><span className="text-rose-400 w-48 shrink-0">frontend/</span><span className="text-zinc-400">Next.js TypeScript web application (modern UI dashboard)</span></li>
                  <li className="flex gap-4"><span className="text-rose-400 w-48 shrink-0">pipeline/orchestrator.py</span><span className="text-zinc-400">Core 7-layer monitoring pipeline</span></li>
                  <li className="flex gap-4"><span className="text-rose-400 w-48 shrink-0">src/autoencoder.py</span><span className="text-zinc-400">Variational autoencoder for drift detection</span></li>
                  <li className="flex gap-4"><span className="text-rose-400 w-48 shrink-0">src/uncertainty.py</span><span className="text-zinc-400">3-member RNN ensemble with MC Dropout</span></li>
                  <li className="flex gap-4"><span className="text-rose-400 w-48 shrink-0">src/data_pipeline.py</span><span className="text-zinc-400">MinMax scaling from reference window</span></li>
                  <li className="flex gap-4"><span className="text-rose-400 w-48 shrink-0">run.sh / stop.sh</span><span className="text-zinc-400">Start/stop both backend and frontend services</span></li>
                </ul>
              </div>
            </section>

          </div>
        </main>
      </div>
    </div>
  );
}
