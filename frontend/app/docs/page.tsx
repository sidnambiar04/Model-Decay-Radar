"use client";

import Link from "next/link";
import { useState, useEffect } from "react";
import {
  BookOpen,
  Terminal,
  Activity,
  Layers,
  FolderOpen,
  Database,
  ArrowLeft,
  GitBranch,
  Cpu,
  Brain,
  ShieldCheck,
  Zap,
  Network
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
        { id: "features", title: "Core Features", icon: <Zap className="w-4 h-4" /> },
        { id: "quickstart", title: "Quickstart", icon: <Terminal className="w-4 h-4" /> },
      ]
    },
    {
      group: "Deep Dive",
      items: [
        { id: "models", title: "ML Models & Algorithms", icon: <Brain className="w-4 h-4" /> },
        { id: "architecture", title: "System Architecture", icon: <Network className="w-4 h-4" /> },
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
          <Link href="/dashboard" className="flex items-center gap-2 text-zinc-400 hover:text-white transition-colors">
            <ArrowLeft className="w-4 h-4" /> Back to Dashboard
          </Link>
          <a href="https://github.com/sidnambiar04/Model-Decay-Radar" target="_blank" rel="noreferrer" className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-zinc-900 border border-zinc-800 hover:bg-zinc-800 text-zinc-300 hover:text-white transition-colors">
            <GitBranch className="w-4 h-4" /> GitHub
          </a>
        </div>
      </header>

      <div className="flex-1 flex max-w-[1400px] w-full mx-auto relative">

        {/* Left Sidebar */}
        <aside className="hidden md:block w-72 border-r border-white/5 bg-black/20 p-8 sticky top-[73px] h-[calc(100vh-73px)] overflow-y-auto">
          {navItems.map((group, i) => (
            <div key={i} className="mb-10">
              <h4 className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest mb-4">{group.group}</h4>
              <ul className="space-y-1.5">
                {group.items.map((item) => (
                  <li key={item.id}>
                    <button
                      onClick={() => scrollTo(item.id)}
                      className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-lg text-sm transition-all duration-200 ${activeSection === item.id
                          ? "bg-gradient-to-r from-rose-500/10 to-transparent text-rose-400 font-semibold border-l-2 border-rose-500"
                          : "text-zinc-400 hover:bg-white/5 hover:text-zinc-200 border-l-2 border-transparent"
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
        <main className="flex-1 p-8 md:p-12 lg:p-20 overflow-y-auto max-w-5xl">
          <div className="space-y-32 pb-32">

            {/* Introduction */}
            <section id="introduction" className="scroll-mt-32 space-y-8">
              <div>
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs font-semibold tracking-widest uppercase mb-6">
                  Overview
                </div>
                <h1 className="text-4xl md:text-5xl font-bold tracking-tight text-white mb-6 leading-tight">
                  What is Model Decay Radar?
                </h1>
                <p className="text-xl text-zinc-400 leading-relaxed max-w-3xl">
                  Model Decay Radar is a real-time ML model monitoring orchestrator inspired by industrial IoT systems. It watches live data streams in the background, autonomously tests for structural distribution drift, explains the root causes of anomalies using SHAP, and automatically triggers candidate retraining.
                </p>
              </div>

              <div className="glass-card bg-amber-500/5 border-amber-500/20 p-6 rounded-2xl flex gap-4 items-start">
                <ShieldCheck className="w-6 h-6 text-amber-400 shrink-0 mt-1" />
                <div>
                  <h4 className="text-amber-400 font-bold mb-2">Scope & Limitations</h4>
                  <p className="text-zinc-300 leading-relaxed">
                    This is an MVP demonstrating a complex MLOps pipeline. It wires together drift detection, epistemic uncertainty quantification, local explanations (SHAP), and background orchestration into a single runnable demo.
                  </p>
                </div>
              </div>
            </section>

            {/* Core Features */}
            <section id="features" className="scroll-mt-32 space-y-8">
              <h2 className="text-3xl font-bold tracking-tight text-white">Core Features</h2>

              <div className="grid md:grid-cols-2 gap-6">
                <div className="glass-card p-8 rounded-2xl border border-zinc-800 hover:border-zinc-700 transition-colors relative overflow-hidden group">
                  <div className="absolute inset-0 bg-gradient-to-br from-rose-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                  <Activity className="w-8 h-8 text-rose-400 mb-6" />
                  <h3 className="text-xl font-bold text-white mb-3">Unsupervised Drift Detection</h3>
                  <p className="text-zinc-400 leading-relaxed">
                    Detects structural changes in incoming data distributions without relying on labels, using deep generative models to catch anomalies before ground truth arrives.
                  </p>
                </div>

                <div className="glass-card p-8 rounded-2xl border border-zinc-800 hover:border-zinc-700 transition-colors relative overflow-hidden group">
                  <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                  <Brain className="w-8 h-8 text-indigo-400 mb-6" />
                  <h3 className="text-xl font-bold text-white mb-3">Epistemic Uncertainty</h3>
                  <p className="text-zinc-400 leading-relaxed">
                    Quantifies what the model "doesn't know" by passing samples through multiple stochastic forward passes, flagging out-of-distribution inputs.
                  </p>
                </div>

                <div className="glass-card p-8 rounded-2xl border border-zinc-800 hover:border-zinc-700 transition-colors relative overflow-hidden group">
                  <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                  <Zap className="w-8 h-8 text-emerald-400 mb-6" />
                  <h3 className="text-xl font-bold text-white mb-3">Root Cause Attribution</h3>
                  <p className="text-zinc-400 leading-relaxed">
                    Explains exactly *why* a model is failing by isolating the specific features (sensors) that drifted using game-theoretic Shapley values.
                  </p>
                </div>

                <div className="glass-card p-8 rounded-2xl border border-zinc-800 hover:border-zinc-700 transition-colors relative overflow-hidden group">
                  <div className="absolute inset-0 bg-gradient-to-br from-amber-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                  <Cpu className="w-8 h-8 text-amber-400 mb-6" />
                  <h3 className="text-xl font-bold text-white mb-3">Zero-Blocking Architecture</h3>
                  <p className="text-zinc-400 leading-relaxed">
                    The entire 7-layer pipeline runs entirely asynchronously. The `/predict` endpoint maintains ultra-low latency while the orchestrator works in the background.
                  </p>
                </div>
              </div>
            </section>

            {/* Quickstart */}
            <section id="quickstart" className="scroll-mt-32 space-y-8">
              <h2 className="text-3xl font-bold tracking-tight text-white">Quickstart</h2>

              <div className="grid gap-6">
                <div className="glass-card bg-[#0a0d10] p-6 rounded-2xl border border-zinc-800">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-bold text-white">1. Install Dependencies</h3>
                  </div>
                  <pre className="text-sm font-mono text-zinc-400 leading-loose">
                    <span className="text-zinc-600"># Python Backend</span><br />
                    <span className="text-rose-400">pip</span> install -r requirements.txt<br /><br />
                    <span className="text-zinc-600"># Next.js Frontend</span><br />
                    <span className="text-rose-400">cd</span> frontend<br />
                    <span className="text-rose-400">npm</span> install
                  </pre>
                </div>

                <div className="glass-card bg-[#0a0d10] p-6 rounded-2xl border border-zinc-800">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-bold text-white">2. Launch Services</h3>
                  </div>
                  <pre className="text-sm font-mono text-zinc-400 leading-loose">
                    <span className="text-zinc-600"># Start both servers (FastAPI + Next.js)</span><br />
                    <span className="text-rose-400">bash</span> run.sh
                  </pre>
                </div>
              </div>
            </section>

            {/* ML Models & Algorithms */}
            <section id="models" className="scroll-mt-32 space-y-8">
              <h2 className="text-3xl font-bold tracking-tight text-white mb-8">ML Models & Algorithms</h2>

              <div className="space-y-8">
                {/* VAE */}
                <div className="glass-card p-8 rounded-2xl border border-zinc-800 hover:border-zinc-700 transition-all shadow-xl">
                  <div className="flex items-center gap-4 mb-6">
                    <div className="w-12 h-12 rounded-xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center">
                      <Network className="w-6 h-6 text-indigo-400" />
                    </div>
                    <div>
                      <h3 className="text-2xl font-bold text-white">Variational Autoencoder (VAE)</h3>
                      <p className="text-zinc-500 font-mono text-xs uppercase tracking-widest mt-1">Drift Detection Engine</p>
                    </div>
                  </div>
                  <p className="text-zinc-300 leading-relaxed mb-6">
                    A deep neural network trained to reconstruct the historical "healthy" data distribution. When new data arrives, the VAE attempts to reconstruct it. If the data has drifted from the historical distribution, the VAE will fail to reconstruct it accurately, resulting in a spiked <strong>Reconstruction Error (MSE)</strong>.
                  </p>
                  <div className="bg-[#0a0d10] border border-zinc-800 rounded-xl p-4">
                    <code className="text-sm font-mono text-indigo-300">
                      loss = reconstruction_loss(x, x_hat) + KL_divergence(q(z|x) || p(z))
                    </code>
                  </div>
                </div>

                {/* MC Dropout */}
                <div className="glass-card p-8 rounded-2xl border border-zinc-800 hover:border-zinc-700 transition-all shadow-xl">
                  <div className="flex items-center gap-4 mb-6">
                    <div className="w-12 h-12 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center">
                      <Layers className="w-6 h-6 text-emerald-400" />
                    </div>
                    <div>
                      <h3 className="text-2xl font-bold text-white">Monte Carlo (MC) Dropout</h3>
                      <p className="text-zinc-500 font-mono text-xs uppercase tracking-widest mt-1">Uncertainty Quantification</p>
                    </div>
                  </div>
                  <p className="text-zinc-300 leading-relaxed mb-6">
                    Traditional neural networks output a single prediction, masking their confidence. By leaving Dropout layers <strong>active during inference</strong>, we run 50 stochastic forward passes for a single input. The variance across these passes quantifies the model's <em>epistemic uncertainty</em> (what it doesn't know).
                  </p>
                </div>

                {/* SHAP */}
                <div className="glass-card p-8 rounded-2xl border border-zinc-800 hover:border-zinc-700 transition-all shadow-xl">
                  <div className="flex items-center gap-4 mb-6">
                    <div className="w-12 h-12 rounded-xl bg-rose-500/10 border border-rose-500/20 flex items-center justify-center">
                      <Brain className="w-6 h-6 text-rose-400" />
                    </div>
                    <div>
                      <h3 className="text-2xl font-bold text-white">SHAP KernelExplainer</h3>
                      <p className="text-zinc-500 font-mono text-xs uppercase tracking-widest mt-1">Root Cause Attribution</p>
                    </div>
                  </div>
                  <p className="text-zinc-300 leading-relaxed mb-6">
                    When the VAE detects a drift, we isolate the specific samples causing the highest error. We pass these samples into a <code>shap.KernelExplainer</code>. By treating the features as players in a cooperative game, SHAP calculates the exact marginal contribution of each sensor to the drift.
                  </p>
                </div>
              </div>
            </section>

            {/* System Architecture */}
            <section id="architecture" className="scroll-mt-32 space-y-8">
              <h2 className="text-3xl font-bold tracking-tight text-white mb-6">System Architecture</h2>
              <p className="text-zinc-300 leading-relaxed mb-6">
                The platform is separated into a lightweight prediction API and a heavy background orchestrator. Data is buffered until a window limit is reached, then piped through a multi-stage monitoring engine.
              </p>

              <div className="glass-card bg-[#0a0d10] p-8 rounded-2xl border border-zinc-800 overflow-x-auto shadow-2xl">
                <pre className="text-xs sm:text-sm font-mono text-emerald-400/90 leading-relaxed">
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
                                       │ Validation Gate & SQLite Registry       │
                                       │  • Compares Candidate vs Active Model   │
                                       │  • Version Tracking (v1, v2, v3)        │
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
            <section id="pipeline" className="scroll-mt-32 space-y-8">
              <h2 className="text-3xl font-bold tracking-tight text-white mb-6">The 7-Layer Monitoring Pipeline</h2>

              <div className="relative border-l border-zinc-800 ml-4 md:ml-6 space-y-12 pb-8">
                {[
                  { title: "Data Ingestion", desc: "Incoming features are scaled with a MinMaxScaler fit on the reference window. Samples are buffered until the window is full, triggering the background cycle." },
                  { title: "Class Imbalance", desc: "Borderline SMOTE applies minority class oversampling if anomalies are detected, stabilizing downstream training." },
                  { title: "Distribution Drift (VAE)", desc: "A Variational Autoencoder calculates reconstruction errors. A 1D-KDE models the error distribution, and a Permutation Test computes a p-value to strictly confirm drift." },
                  { title: "MC Dropout Uncertainty", desc: "An LSTM/RNN ensemble with Monte Carlo Dropout runs 50 forward passes per sample to quantify predictive epistemic uncertainty during the drift phase." },
                  { title: "Root Cause Attribution", desc: "If drift occurs, the VAE bottleneck gradients are passed to a SHAP KernelExplainer to pinpoint the exact sensors causing the model to fail, paired with KS-Tests." },
                  { title: "Health Scoring", desc: "The Model Health Score (MHS) is calculated as an inverse weighted penalty of KL divergence, uncertainty spikes, and F1 degradation." },
                  { title: "Automated Retraining", desc: "If MHS drops below 85% and drift is confirmed, a Candidate Model is trained on the combined historical and drifted window." }
                ].map((layer, idx) => (
                  <div key={idx} className="relative pl-10 md:pl-12 group">
                    <div className="absolute -left-[17px] top-1 w-8 h-8 rounded-full bg-zinc-900 border-2 border-zinc-800 group-hover:border-rose-500 flex items-center justify-center transition-colors">
                      <span className="text-xs font-mono font-bold text-zinc-500 group-hover:text-rose-400">{idx + 1}</span>
                    </div>
                    <div className="glass-card p-6 rounded-xl border border-zinc-800/50 hover:border-zinc-700 transition-all">
                      <h4 className="text-lg text-white font-bold mb-2">{layer.title}</h4>
                      <p className="text-sm text-zinc-400 leading-relaxed">{layer.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </section>

            {/* Synthetic Dataset */}
            <section id="dataset" className="scroll-mt-32 space-y-8">
              <h2 className="text-3xl font-bold tracking-tight text-white mb-6">The Synthetic Dataset</h2>
              <p className="text-zinc-300 leading-relaxed mb-4">
                The project ships with a synthetic dataset modeled after SWaT (<code className="bg-zinc-800 px-2 py-1 rounded text-rose-400 font-mono text-sm">data/synthetic_swat.csv</code>), containing 38,000 rows across 51 sensor features.
              </p>

              <div className="overflow-hidden rounded-xl border border-zinc-800 shadow-xl">
                <table className="w-full text-left text-sm text-zinc-300">
                  <thead className="bg-zinc-900/80 backdrop-blur border-b border-zinc-800 text-zinc-400">
                    <tr>
                      <th className="px-6 py-4 font-semibold uppercase tracking-widest text-xs">Window Name</th>
                      <th className="px-6 py-4 font-semibold uppercase tracking-widest text-xs">Rows</th>
                      <th className="px-6 py-4 font-semibold uppercase tracking-widest text-xs">Description</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-800/50 bg-[#0a0d10]/50">
                    <tr className="hover:bg-white/5 transition-colors">
                      <td className="px-6 py-4 font-mono text-white">reference</td>
                      <td className="px-6 py-4 text-zinc-400">20,000</td>
                      <td className="px-6 py-4 leading-relaxed">Historical baseline — used to train the VAE, RNN, and fit the data scaler.</td>
                    </tr>
                    <tr className="hover:bg-white/5 transition-colors">
                      <td className="px-6 py-4 font-mono text-white">current_stable</td>
                      <td className="px-6 py-4 text-zinc-400">10,000</td>
                      <td className="px-6 py-4 leading-relaxed">Recent production data. Statistically similar to the reference window.</td>
                    </tr>
                    <tr className="hover:bg-white/5 transition-colors">
                      <td className="px-6 py-4 font-mono text-rose-400">current_drift</td>
                      <td className="px-6 py-4 text-zinc-400">8,000</td>
                      <td className="px-6 py-4 leading-relaxed">Anomalous data injected into 3 specific sensors (LIT101, DPIT301, P402) using ramps, frequency shifts, and noise.</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </section>

          </div>
        </main>
      </div>
    </div>
  );
}
