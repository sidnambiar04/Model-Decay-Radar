"use client";

import Link from "next/link";
import { useState, useEffect } from "react";
import {
  ArrowRight,
  Activity,
  Terminal,
  Layers,
  ShieldAlert,
  BrainCircuit,
  Settings2,
  Workflow
} from "lucide-react";
import CustomCursor from "@/components/CustomCursor";

export default function LandingPage() {
  const [terminalStep, setTerminalStep] = useState(0);

  useEffect(() => {
    // Simple sequence for the terminal typing effect
    const timer1 = setTimeout(() => setTerminalStep(1), 1000);
    const timer2 = setTimeout(() => setTerminalStep(2), 2500);
    const timer3 = setTimeout(() => setTerminalStep(3), 4000);
    const timer4 = setTimeout(() => setTerminalStep(4), 5500);

    // Loop the terminal effect
    const interval = setInterval(() => {
      setTerminalStep(0);
      setTimeout(() => setTerminalStep(1), 1000);
      setTimeout(() => setTerminalStep(2), 2500);
      setTimeout(() => setTerminalStep(3), 4000);
      setTimeout(() => setTerminalStep(4), 5500);
    }, 9000);

    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);
      clearTimeout(timer4);
      clearInterval(interval);
    };
  }, []);

  return (
    <div className="min-h-screen bg-background text-foreground selection:bg-rose-500/30 selection:text-rose-200">
      <CustomCursor />

      {/* Navigation */}
      <nav className="fixed top-0 inset-x-0 z-50 bg-black/40 backdrop-blur-xl border-b border-white/5">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="relative flex items-center justify-center w-8 h-8 rounded-lg bg-zinc-900 border border-zinc-800">
              <span className="relative inline-flex rounded-full h-2 w-2 bg-rose-500"></span>
            </div>
            <span className="font-bold tracking-tight text-white">MODEL DECAY RADAR</span>
          </div>

          <div className="hidden md:flex items-center gap-6 text-sm font-medium text-zinc-400">
            <a href="#how-it-works" className="hover:text-white transition-colors">How it Works</a>
            <a href="#features" className="hover:text-white transition-colors">Features</a>
            <Link href="/docs" className="hover:text-white transition-colors">Documentation</Link>
          </div>

          <div className="flex items-center gap-4">
            <Link href="/dashboard" className="hidden md:flex items-center gap-2 text-sm font-semibold text-white px-4 py-2 rounded-lg bg-zinc-900 border border-zinc-800 hover:bg-zinc-800 transition-colors">
              Dashboard
            </Link>
            <Link href="/dashboard" className="flex items-center gap-2 text-sm font-semibold text-white px-4 py-2 rounded-lg bg-gradient-to-r from-rose-600 to-rose-500 hover:from-rose-500 hover:to-rose-400 transition-all shadow-[0_0_20px_rgba(244,63,94,0.3)] hover:shadow-[0_0_25px_rgba(244,63,94,0.5)]">
              Launch <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <main className="min-h-screen pt-24 pb-16 px-6 relative overflow-hidden flex flex-col justify-center">
        
        {/* Glow & Grid Effects */}
        <div className="absolute inset-0 grid-bg opacity-30 pointer-events-none" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-rose-500/10 blur-[120px] rounded-full pointer-events-none" />
        <div className="absolute top-0 inset-x-0 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />
        
        <div className="max-w-7xl mx-auto relative z-10 w-full flex flex-col lg:flex-row items-center gap-12 lg:gap-4 -mt-16">
          <div className="text-center lg:text-left flex-1">
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs font-semibold tracking-widest uppercase mb-8">
              <span className="w-1.5 h-1.5 rounded-full bg-rose-500 animate-pulse" />
              Active Monitoring Pipeline
            </div>

            <h1 className="text-5xl md:text-6xl lg:text-7xl font-bold tracking-tight text-white mb-8 leading-[1.1]">
              Detect Model Decay <br />
              <span className="bg-gradient-to-r from-rose-400 via-fuchsia-400 to-indigo-400 bg-clip-text text-transparent">
                Before It Hits Production
              </span>
            </h1>

            <p className="text-xl text-zinc-400 max-w-2xl mx-auto lg:mx-0 mb-12 leading-relaxed">
              The first 7-layer autonomous orchestrator for ML monitoring. Watches live streams, strictly tests for drift, explains root causes with SHAP, and automatically triggers candidate retraining.
            </p>

            <div className="flex flex-col sm:flex-row items-center justify-center lg:justify-start gap-4 mb-16">
              <Link href="/dashboard" className="w-full sm:w-auto flex items-center justify-center gap-2 text-base font-semibold text-white px-8 py-4 rounded-xl bg-gradient-to-r from-rose-600 to-rose-500 hover:from-rose-500 hover:to-rose-400 transition-all shadow-[0_0_30px_rgba(244,63,94,0.3)] hover:shadow-[0_0_40px_rgba(244,63,94,0.5)] hover:-translate-y-0.5">
                Launch Dashboard <ArrowRight className="w-5 h-5" />
              </Link>
              <Link href="/docs" className="w-full sm:w-auto flex items-center justify-center gap-2 text-base font-semibold text-white px-8 py-4 rounded-xl bg-zinc-900 border border-zinc-800 hover:bg-zinc-800 transition-all hover:-translate-y-0.5">
                Read Documentation <Terminal className="w-5 h-5" />
              </Link>
            </div>

            <div className="flex items-center justify-center lg:justify-start gap-6 font-mono text-xs text-zinc-500">
              <span className="flex items-center gap-2">
                <div className="w-4 h-4 rounded bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">✓</div>
                v2.0 OSS
              </span>
              <span className="flex items-center gap-2">
                <div className="w-4 h-4 rounded bg-rose-500/10 border border-rose-500/20 flex items-center justify-center text-rose-400">⚡</div>
                Low Latency
              </span>
              <span className="flex items-center gap-2">
                <div className="w-4 h-4 rounded bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400">⎈</div>
                FastAPI + Next.js
              </span>
            </div>
          </div>

          {/* Arize-style Network Animation */}
          <div className="flex-1 w-full flex justify-center lg:justify-end">
            <div className="relative w-[350px] h-[350px] md:w-[450px] md:h-[450px] lg:w-[600px] lg:h-[600px] xl:w-[700px] xl:h-[700px] pointer-events-none perspective-[1000px] lg:-mt-24 xl:-mt-32">
              
              <svg className="absolute inset-0 w-full h-full [transform:rotateX(10deg)_rotateZ(-5deg)]" viewBox="0 0 500 500">
                <defs>
                  <filter id="glow-strong">
                    <feGaussianBlur stdDeviation="4" result="coloredBlur" />
                    <feMerge>
                      <feMergeNode in="coloredBlur" />
                      <feMergeNode in="coloredBlur" />
                      <feMergeNode in="SourceGraphic" />
                    </feMerge>
                  </filter>
                  <filter id="glow-light">
                    <feGaussianBlur stdDeviation="2" result="coloredBlur" />
                    <feMerge>
                      <feMergeNode in="coloredBlur" />
                      <feMergeNode in="SourceGraphic" />
                    </feMerge>
                  </filter>
                  <linearGradient id="trace-grad-1" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#f43f5e" />
                    <stop offset="100%" stopColor="#6366f1" />
                  </linearGradient>
                  <linearGradient id="trace-grad-2" x1="100%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" stopColor="#10b981" />
                    <stop offset="100%" stopColor="#3b82f6" />
                  </linearGradient>
                </defs>

                {/* Background faint paths */}
                <g stroke="rgba(255,255,255,0.06)" strokeWidth="1.5" fill="none">
                  <path d="M 50 250 Q 150 100 250 250 T 450 250" />
                  <path d="M 100 400 Q 250 300 250 250 T 400 100" />
                  <path d="M 150 150 Q 250 250 350 150" />
                  <path d="M 150 350 Q 250 250 350 350" />
                  <circle cx="250" cy="250" r="100" strokeDasharray="4 8" />
                  <circle cx="250" cy="250" r="150" strokeDasharray="2 12" />
                </g>

                {/* Flowing Data Particles */}
                <g fill="none" strokeWidth="3" filter="url(#glow-strong)" style={{ animation: 'data-flow 3s linear infinite' }}>
                  <path d="M 50 250 Q 150 100 250 250 T 450 250" stroke="url(#trace-grad-1)" strokeDasharray="20 400" />
                  <path d="M 100 400 Q 250 300 250 250 T 400 100" stroke="url(#trace-grad-2)" strokeDasharray="30 350" style={{ animationDelay: '-1s' }} />
                  <path d="M 150 150 Q 250 250 350 150" stroke="#f43f5e" strokeDasharray="15 200" style={{ animationDelay: '-0.5s' }} />
                  <path d="M 150 350 Q 250 250 350 350" stroke="#4f46e5" strokeDasharray="25 250" style={{ animationDelay: '-2s' }} />
                  <circle cx="250" cy="250" r="100" stroke="#10b981" strokeDasharray="10 300" style={{ animationDelay: '-1.5s' }} />
                </g>

                {/* Nodes & Labels */}
                <g filter="url(#glow-light)">
                  {/* Core Node */}
                  <circle cx="250" cy="250" r="12" fill="#0d0f12" stroke="#f43f5e" strokeWidth="3" />
                  <circle cx="250" cy="250" r="6" fill="#f43f5e" style={{ animation: 'node-burst 3s infinite 1.5s' }} />
                  <text x="250" y="230" fill="#a1a1aa" fontSize="12" fontWeight="bold" textAnchor="middle" letterSpacing="1">MODEL CORE</text>

                  {/* Peripheral Nodes */}
                  <circle cx="50" cy="250" r="5" fill="#6366f1" style={{ animation: 'node-burst 3s infinite 0s' }} />
                  <text x="50" y="235" fill="#71717a" fontSize="10" textAnchor="middle">Ingestion</text>

                  <circle cx="150" cy="150" r="6" fill="#f43f5e" style={{ animation: 'node-burst 3s infinite 0.5s' }} />
                  <text x="150" y="135" fill="#71717a" fontSize="10" textAnchor="middle">VAE Encoder</text>

                  <circle cx="150" cy="350" r="5" fill="#4f46e5" style={{ animation: 'node-burst 3s infinite 1s' }} />
                  <text x="150" y="370" fill="#71717a" fontSize="10" textAnchor="middle">MC Dropout</text>

                  <circle cx="350" cy="150" r="6" fill="#10b981" style={{ animation: 'node-burst 3s infinite 2s' }} />
                  <text x="350" y="135" fill="#71717a" fontSize="10" textAnchor="middle">Drift Score</text>

                  <circle cx="350" cy="350" r="5" fill="#f59e0b" style={{ animation: 'node-burst 3s infinite 2.5s' }} />
                  <text x="350" y="370" fill="#71717a" fontSize="10" textAnchor="middle">SHAP Explainer</text>

                  <circle cx="450" cy="250" r="6" fill="#6366f1" style={{ animation: 'node-burst 3s infinite 3s' }} />
                  <text x="450" y="235" fill="#71717a" fontSize="10" textAnchor="middle">Retrain</text>

                  <circle cx="100" cy="400" r="5" fill="#10b981" style={{ animation: 'node-burst 3s infinite 0.5s' }} />
                  <circle cx="400" cy="100" r="6" fill="#3b82f6" style={{ animation: 'node-burst 3s infinite 2.5s' }} />

                  {/* Orbital Nodes */}
                  <circle cx="150" cy="250" r="4" fill="#a855f7" />
                  <circle cx="350" cy="250" r="4" fill="#a855f7" />
                  <circle cx="250" cy="150" r="4" fill="#14b8a6" />
                  <circle cx="250" cy="350" r="4" fill="#14b8a6" />
                </g>
              </svg>
            </div>
          </div>
        </div>
      </main>

      <div className="border-t border-white/5 bg-black/20" id="how-it-works">
        <div className="max-w-7xl mx-auto px-6 py-16">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-white mb-4">How the pipeline works</h2>
            <p className="text-zinc-400 max-w-2xl mx-auto">Model Decay Radar runs completely invisibly in the background. Your application only talks to the `/predict` endpoint, while the orchestrator handles the heavy lifting.</p>
          </div>

          <div className="grid md:grid-cols-3 gap-6">
            {/* Step 1 */}
            <div className="glass-card p-8 rounded-2xl border border-zinc-800/50 hover:border-zinc-700 transition-colors relative overflow-hidden group">
              <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
              <div className="font-mono text-xs text-indigo-400 font-bold tracking-widest mb-6">STEP 01</div>
              <Activity className="w-10 h-10 text-indigo-400 mb-6" />
              <h3 className="text-xl font-bold text-white mb-3">Buffer & Monitor</h3>
              <p className="text-sm text-zinc-400 leading-relaxed">Incoming predictions are buffered in memory. Every 500 samples, the data is pushed through a 7-layer validation gauntlet without blocking the main prediction thread.</p>
            </div>

            {/* Step 2 */}
            <div className="glass-card p-8 rounded-2xl border border-zinc-800/50 hover:border-zinc-700 transition-colors relative overflow-hidden group">
              <div className="absolute inset-0 bg-gradient-to-br from-amber-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
              <div className="font-mono text-xs text-amber-400 font-bold tracking-widest mb-6">STEP 02</div>
              <ShieldAlert className="w-10 h-10 text-amber-400 mb-6" />
              <h3 className="text-xl font-bold text-white mb-3">Detect & Explain</h3>
              <p className="text-sm text-zinc-400 leading-relaxed">A Variational Autoencoder detects structural data drift, while MC Dropout quantifies uncertainty. SHAP explicitly attributes the drift to specific features.</p>
            </div>

            {/* Step 3 */}
            <div className="glass-card p-8 rounded-2xl border border-zinc-800/50 hover:border-zinc-700 transition-colors relative overflow-hidden group">
              <div className="absolute inset-0 bg-gradient-to-br from-rose-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
              <div className="font-mono text-xs text-rose-400 font-bold tracking-widest mb-6">STEP 03</div>
              <BrainCircuit className="w-10 h-10 text-rose-400 mb-6" />
              <h3 className="text-xl font-bold text-white mb-3">Score & Retrain</h3>
              <p className="text-sm text-zinc-400 leading-relaxed">A Model Health Score is calculated. If the score drops below 85%, a candidate model is trained automatically on the new distribution and evaluated against the registry.</p>
            </div>
          </div>

          {/* Terminal Typing Snippet */}
          <div className="max-w-7xl mx-auto mt-16 pt-16 border-t border-zinc-800/50">
            <div className="grid lg:grid-cols-3 gap-6">
              
              {/* Terminal 1: Orchestrator */}
              <div className="glass-card rounded-xl border border-zinc-700 overflow-hidden shadow-2xl flex flex-col">
                <div className="bg-zinc-900 border-b border-zinc-800 px-4 py-3 flex items-center gap-2 shrink-0">
                  <div className="w-3 h-3 rounded-full bg-rose-500/20 border border-rose-500/50"></div>
                  <div className="w-3 h-3 rounded-full bg-amber-500/20 border border-amber-500/50"></div>
                  <div className="w-3 h-3 rounded-full bg-emerald-500/20 border border-emerald-500/50"></div>
                  <div className="ml-4 font-mono text-[10px] text-zinc-500 tracking-widest">orchestrator.log</div>
                </div>
                <div className="p-6 font-mono text-xs md:text-sm leading-relaxed bg-[#0a0d10] text-zinc-300 flex-1">
                  {terminalStep >= 1 && (
                    <div className="mb-2">
                      <span className="text-emerald-400">[INFO]</span> <span className="text-zinc-500">21:40:02</span> Window #42 filled. Triggering Layer 1 (Ingestion)...
                    </div>
                  )}
                  {terminalStep >= 2 && (
                    <div className="mb-2">
                      <span className="text-emerald-400">[INFO]</span> <span className="text-zinc-500">21:40:03</span> Layer 3: VAE Recon Error: <span className="text-rose-400">0.824</span> (Threshold: 0.150)
                    </div>
                  )}
                  {terminalStep >= 3 && (
                    <div className="mb-2">
                      <span className="text-amber-400">[WARN]</span> <span className="text-zinc-500">21:40:04</span> Drift confirmed (p &lt; 0.05). Running SHAP attribution...<br />
                      <span className="text-zinc-500">... Top drifted sensors: LIT101, DPIT301, P402</span>
                    </div>
                  )}
                  {terminalStep >= 4 && (
                    <div className="mb-2 animate-type">
                      <span className="text-rose-400">[ALERT]</span> Model Health Score dropped to 72%. Auto-Retrain...<span className="border-r-2 border-white ml-1 animate-pulse"></span>
                    </div>
                  )}
                  {terminalStep < 4 && (
                    <div className="flex items-center">
                      <span className="w-2 h-4 bg-zinc-400 animate-pulse ml-1"></span>
                    </div>
                  )}
                </div>
              </div>

              {/* Terminal 2: VAE Detector */}
              <div className="glass-card rounded-xl border border-zinc-700 overflow-hidden shadow-2xl flex flex-col">
                <div className="bg-zinc-900 border-b border-zinc-800 px-4 py-3 flex items-center gap-2 shrink-0">
                  <div className="w-3 h-3 rounded-full bg-rose-500/20 border border-rose-500/50"></div>
                  <div className="w-3 h-3 rounded-full bg-amber-500/20 border border-amber-500/50"></div>
                  <div className="w-3 h-3 rounded-full bg-emerald-500/20 border border-emerald-500/50"></div>
                  <div className="ml-4 font-mono text-[10px] text-zinc-500 tracking-widest">detector.py</div>
                </div>
                <div className="p-6 font-mono text-xs md:text-sm leading-relaxed bg-[#0a0d10] text-zinc-300 flex-1">
                  <div className="mb-2 text-indigo-400">import torch<br/>from models import VAE</div>
                  {terminalStep >= 2 && (
                    <>
                      <div className="mb-2">
                        <span className="text-zinc-500"># Evaluating batched tensor stream...</span><br/>
                        &gt;&gt;&gt; z_mean, z_log_var = encoder(x_batch)<br/>
                        &gt;&gt;&gt; recon_loss = F.mse_loss(recon, x_batch)<br/>
                        &gt;&gt;&gt; kl_loss = -0.5 * sum(1 + z_log_var)
                      </div>
                      <div className="text-rose-400 animate-type">RuntimeWarning: kl spike detected!</div>
                    </>
                  )}
                  {terminalStep < 2 && (
                    <div className="flex items-center">
                      <span className="w-2 h-4 bg-zinc-400 animate-pulse ml-1"></span>
                    </div>
                  )}
                </div>
              </div>

              {/* Terminal 3: Explainer */}
              <div className="glass-card rounded-xl border border-zinc-700 overflow-hidden shadow-2xl flex flex-col">
                <div className="bg-zinc-900 border-b border-zinc-800 px-4 py-3 flex items-center gap-2 shrink-0">
                  <div className="w-3 h-3 rounded-full bg-rose-500/20 border border-rose-500/50"></div>
                  <div className="w-3 h-3 rounded-full bg-amber-500/20 border border-amber-500/50"></div>
                  <div className="w-3 h-3 rounded-full bg-emerald-500/20 border border-emerald-500/50"></div>
                  <div className="ml-4 font-mono text-[10px] text-zinc-500 tracking-widest">shap_explainer.py</div>
                </div>
                <div className="p-6 font-mono text-xs md:text-sm leading-relaxed bg-[#0a0d10] text-zinc-300 flex-1">
                  <div className="mb-2 text-indigo-400">import shap<br/>explainer = shap.Explainer(model)</div>
                  {terminalStep >= 3 && (
                    <>
                      <div className="mb-2">
                        <span className="text-zinc-500"># Generating local explanations...</span><br/>
                        &gt;&gt;&gt; shap_values = explainer(X_drift)
                      </div>
                      <div className="mb-2 animate-type">
                        [ <span className="text-amber-400">Feature Importance</span> ]<br/>
                        LIT101:  <span className="text-rose-400">████████</span> 0.42<br/>
                        P402:    <span className="text-rose-400">████</span> 0.21<br/>
                        DPIT301: <span className="text-rose-400">██</span> 0.15<br/>
                      </div>
                    </>
                  )}
                  {terminalStep < 3 && (
                    <div className="flex items-center">
                      <span className="w-2 h-4 bg-zinc-400 animate-pulse ml-1"></span>
                    </div>
                  )}
                </div>
              </div>

            </div>
          </div>

        </div>
      </div>

      <div className="border-t border-white/5 py-16 px-6" id="features">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center gap-4 mb-8">
            <div className="w-12 h-12 rounded-xl bg-rose-500/10 border border-rose-500/20 flex items-center justify-center">
              <Layers className="w-6 h-6 text-rose-400" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white">Platform Capabilities</h2>
              <p className="text-sm text-zinc-400">Everything needed to monitor models in production.</p>
            </div>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">

            <div className="glass-card p-6 rounded-xl border border-zinc-800 hover:border-rose-500/30 hover:bg-zinc-900/80 transition-all cursor-default">
              <Terminal className="w-6 h-6 text-emerald-400 mb-4" />
              <div className="font-mono text-[10px] text-emerald-400 uppercase tracking-widest mb-2">Layer 1-3</div>
              <h4 className="text-base font-bold text-white mb-2">VAE Drift Detection</h4>
              <p className="text-xs text-zinc-400 leading-relaxed">Identifies structural shifts in input data using a deep autoencoder and permutation tests to reject false positives.</p>
            </div>

            <div className="glass-card p-6 rounded-xl border border-zinc-800 hover:border-indigo-500/30 hover:bg-zinc-900/80 transition-all cursor-default">
              <Settings2 className="w-6 h-6 text-indigo-400 mb-4" />
              <div className="font-mono text-[10px] text-indigo-400 uppercase tracking-widest mb-2">Layer 4</div>
              <h4 className="text-base font-bold text-white mb-2">MC Dropout</h4>
              <p className="text-xs text-zinc-400 leading-relaxed">Runs 50 stochastic forward passes per sample to precisely calculate the model's epistemic uncertainty during inference.</p>
            </div>

            <div className="glass-card p-6 rounded-xl border border-zinc-800 hover:border-amber-500/30 hover:bg-zinc-900/80 transition-all cursor-default">
              <Activity className="w-6 h-6 text-amber-400 mb-4" />
              <div className="font-mono text-[10px] text-amber-400 uppercase tracking-widest mb-2">Layer 5</div>
              <h4 className="text-base font-bold text-white mb-2">SHAP Explanations</h4>
              <p className="text-xs text-zinc-400 leading-relaxed">Passes VAE gradients to a KernelExplainer to pinpoint exactly which sensors or features caused the drift event.</p>
            </div>

            <div className="glass-card p-6 rounded-xl border border-zinc-800 hover:border-rose-500/30 hover:bg-zinc-900/80 transition-all cursor-default">
              <Workflow className="w-6 h-6 text-rose-400 mb-4" />
              <div className="font-mono text-[10px] text-rose-400 uppercase tracking-widest mb-2">Layer 7</div>
              <h4 className="text-base font-bold text-white mb-2">Auto-Retraining</h4>
              <p className="text-xs text-zinc-400 leading-relaxed">Triggers a background retrain of a candidate model, utilizing SMOTE for class imbalance and comparing against the registry.</p>
            </div>

          </div>
        </div>
      </div>

      {/* Footer / Final CTA */}
      <footer className="border-t border-white/5 py-16 text-center relative overflow-hidden">
        <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-full h-[300px] bg-rose-500/5 blur-[100px] pointer-events-none" />
        <div className="max-w-2xl mx-auto px-6 relative z-10">
          <h2 className="text-3xl font-bold text-white mb-6">Ready to see it in action?</h2>
          <p className="text-zinc-400 mb-8">
            Deploy the orchestrator and watch as it autonomously detects injected dataset drift in real-time.
          </p>
          <Link href="/dashboard" className="inline-flex items-center justify-center gap-2 text-sm font-semibold text-white px-8 py-4 rounded-xl bg-gradient-to-r from-rose-600 to-rose-500 hover:from-rose-500 hover:to-rose-400 transition-all shadow-[0_0_20px_rgba(244,63,94,0.3)] hover:shadow-[0_0_25px_rgba(244,63,94,0.5)]">
            Open the Dashboard <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
      </footer>

    </div>
  );
}
