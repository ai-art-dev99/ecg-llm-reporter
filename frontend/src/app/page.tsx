'use client'
import { useState, useCallback, useRef } from 'react'
import { Upload, Activity, Cpu, ChevronDown, Loader2, AlertCircle } from 'lucide-react'
import clsx from 'clsx'
import ECGChart from '../components/charts/ECGChart'
import ReportDisplay from '../components/ReportDisplay'
import MetricsDashboard from '../components/MetricsDashboard'
import {
    analyzeUpload, analyzeSynthetic, generateUploadReport, generateSyntheticReport,
    AnalysisResult, ReportResult, SyntheticRequest,
} from '../lib/api'

// ── Types ─────────────────────────────────────────────────────────────────────

type Mode = 'upload' | 'synthetic'
type Step = 'idle' | 'analysing' | 'generating' | 'done' | 'error'

const RHYTHMS = ['normal', 'tachycardia', 'bradycardia', 'afib']

// ── Main Page ─────────────────────────────────────────────────────────────────

export default function HomePage() {
    const [mode, setMode] = useState<Mode>('synthetic')
    const [step, setStep] = useState<Step>('idle')
    const [error, setError] = useState('')
    const [file, setFile] = useState<File | null>(null)
    const [dragOver, setDragOver] = useState(false)

    // Synthetic params
    const [synth, setSynth] = useState<SyntheticRequest>({
        heart_rate: 72, duration: 10, rhythm: 'normal', noise: 0.01,
    })

    // Results
    const [ecgSignal, setEcgSignal] = useState<number[]>([])
    const [analysis, setAnalysis] = useState<AnalysisResult | null>(null)
    const [report, setReport] = useState<ReportResult | null>(null)

    const fileRef = useRef<HTMLInputElement>(null)

    // ── Handlers ─────────────────────────────────────────────────────────────

    const handleRun = useCallback(async () => {
        setError('')
        setReport(null)
        setAnalysis(null)
        setEcgSignal([])

        try {
            // Step 1: Analyse
            setStep('analysing')
            let result: AnalysisResult

            if (mode === 'upload') {
                if (!file) { setError('Please select a CSV file.'); setStep('error'); return }
                result = await analyzeUpload(file)
            } else {
                result = await analyzeSynthetic(synth)
            }
            setAnalysis(result)

            // Simulate ECG signal from prompt context for display
            // (real signal isn't returned by API — we generate a matching synthetic one)
            const { default: nk } = await import('../lib/syntheticECG')
            const signal = nk.simulate({ hr: result.heart_rate_mean, n: result.sampling_rate * result.duration_seconds })
            setEcgSignal(signal)

            // Step 2: Generate LLM Report
            setStep('generating')
            let rep: ReportResult

            if (mode === 'upload') {
                rep = await generateUploadReport(file!)
            } else {
                rep = await generateSyntheticReport(synth)
            }
            setReport(rep)
            setStep('done')

        } catch (e: any) {
            setError(e.message || 'Something went wrong')
            setStep('error')
        }
    }, [mode, file, synth])

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault()
        setDragOver(false)
        const f = e.dataTransfer.files[0]
        if (f?.name.endsWith('.csv')) setFile(f)
        else setError('Please upload a .csv file')
    }, [])

    const busy = step === 'analysing' || step === 'generating'

    // ── Render ────────────────────────────────────────────────────────────────

    return (
        <div className="min-h-screen bg-slate-950">

            {/* Navbar */}
            <header className="border-b border-slate-800 px-6 py-4 flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <div className="p-2 bg-emerald-500/10 rounded-lg">
                        <Activity className="w-5 h-5 text-emerald-400" />
                    </div>
                    <div>
                        <h1 className="text-base font-semibold text-white">ECG LLM Reporter</h1>
                        <p className="text-xs text-slate-500">AI-powered clinical ECG analysis</p>
                    </div>
                </div>
                <div className="flex items-center gap-2">
                    <span className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse-slow" />
                    <span className="text-xs text-slate-400">API connected</span>
                </div>
            </header>

            <main className="max-w-6xl mx-auto px-6 py-8 space-y-8">

                {/* Mode Toggle */}
                <div className="flex gap-2 p-1 bg-slate-900 rounded-xl w-fit border border-slate-800">
                    {(['synthetic', 'upload'] as Mode[]).map(m => (
                        <button
                            key={m}
                            onClick={() => { setMode(m); setStep('idle'); setError('') }}
                            className={clsx(
                                'px-4 py-2 rounded-lg text-sm font-medium transition-all capitalize',
                                mode === m
                                    ? 'bg-brand-600 text-white shadow-lg shadow-brand-600/20'
                                    : 'text-slate-400 hover:text-white',
                            )}
                        >
                            {m === 'synthetic' ? '⚡ Synthetic ECG' : '📂 Upload CSV'}
                        </button>
                    ))}
                </div>

                {/* Input Panel */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

                    {/* Controls */}
                    <div className="card space-y-5">
                        <h2 className="text-sm font-semibold text-slate-300">
                            {mode === 'synthetic' ? 'Synthetic ECG Parameters' : 'Upload ECG File'}
                        </h2>

                        {mode === 'synthetic' ? (
                            <>
                                {/* Heart Rate */}
                                <div>
                                    <div className="flex justify-between mb-1">
                                        <label className="text-xs text-slate-400">Heart Rate</label>
                                        <span className="text-xs font-mono text-emerald-400">{synth.heart_rate} bpm</span>
                                    </div>
                                    <input type="range" min={30} max={200} value={synth.heart_rate}
                                        onChange={e => setSynth(s => ({ ...s, heart_rate: +e.target.value }))}
                                        className="w-full accent-emerald-400"
                                    />
                                    <div className="flex justify-between text-xs text-slate-600 mt-1">
                                        <span>30</span><span>200 bpm</span>
                                    </div>
                                </div>

                                {/* Duration */}
                                <div>
                                    <div className="flex justify-between mb-1">
                                        <label className="text-xs text-slate-400">Duration</label>
                                        <span className="text-xs font-mono text-emerald-400">{synth.duration}s</span>
                                    </div>
                                    <input type="range" min={5} max={60} value={synth.duration}
                                        onChange={e => setSynth(s => ({ ...s, duration: +e.target.value }))}
                                        className="w-full accent-emerald-400"
                                    />
                                </div>

                                {/* Rhythm */}
                                <div>
                                    <label className="text-xs text-slate-400 block mb-1.5">Rhythm Pattern</label>
                                    <div className="relative">
                                        <select
                                            value={synth.rhythm}
                                            onChange={e => setSynth(s => ({ ...s, rhythm: e.target.value }))}
                                            className="w-full bg-slate-800 border border-slate-700 text-slate-200 text-sm rounded-lg px-3 py-2 appearance-none focus:outline-none focus:border-brand-500"
                                        >
                                            {RHYTHMS.map(r => (
                                                <option key={r} value={r}>{r.charAt(0).toUpperCase() + r.slice(1)}</option>
                                            ))}
                                        </select>
                                        <ChevronDown className="absolute right-3 top-2.5 w-4 h-4 text-slate-400 pointer-events-none" />
                                    </div>
                                </div>

                                {/* Noise */}
                                <div>
                                    <div className="flex justify-between mb-1">
                                        <label className="text-xs text-slate-400">Signal Noise</label>
                                        <span className="text-xs font-mono text-slate-400">{synth.noise.toFixed(3)}</span>
                                    </div>
                                    <input type="range" min={0.001} max={0.1} step={0.001} value={synth.noise}
                                        onChange={e => setSynth(s => ({ ...s, noise: +e.target.value }))}
                                        className="w-full accent-slate-400"
                                    />
                                    <div className="flex justify-between text-xs text-slate-600 mt-1">
                                        <span>Clean</span><span>Noisy</span>
                                    </div>
                                </div>
                            </>
                        ) : (
                            /* Upload Zone */
                            <div
                                onDrop={handleDrop}
                                onDragOver={e => { e.preventDefault(); setDragOver(true) }}
                                onDragLeave={() => setDragOver(false)}
                                onClick={() => fileRef.current?.click()}
                                className={clsx(
                                    'border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all',
                                    dragOver ? 'border-brand-500 bg-brand-500/5' : 'border-slate-700 hover:border-slate-500',
                                )}
                            >
                                <Upload className="w-8 h-8 text-slate-500 mx-auto mb-3" />
                                {file ? (
                                    <div>
                                        <p className="text-sm font-medium text-emerald-400">{file.name}</p>
                                        <p className="text-xs text-slate-500 mt-1">{(file.size / 1024).toFixed(1)} KB</p>
                                    </div>
                                ) : (
                                    <div>
                                        <p className="text-sm text-slate-400">Drop CSV here or click to browse</p>
                                        <p className="text-xs text-slate-600 mt-1">One numeric column = Lead II signal</p>
                                    </div>
                                )}
                                <input ref={fileRef} type="file" accept=".csv" className="hidden"
                                    onChange={e => { const f = e.target.files?.[0]; if (f) setFile(f) }}
                                />
                            </div>
                        )}

                        {/* Run Button */}
                        <button
                            onClick={handleRun}
                            disabled={busy}
                            className={clsx(
                                'w-full py-3 rounded-xl text-sm font-semibold flex items-center justify-center gap-2 transition-all',
                                busy
                                    ? 'bg-slate-700 text-slate-400 cursor-not-allowed'
                                    : 'bg-gradient-to-r from-brand-600 to-emerald-600 hover:from-brand-500 hover:to-emerald-500 text-white shadow-lg shadow-emerald-900/30',
                            )}
                        >
                            {busy ? (
                                <>
                                    <Loader2 className="w-4 h-4 animate-spin" />
                                    {step === 'analysing' ? 'Analysing ECG…' : 'Generating Report…'}
                                </>
                            ) : (
                                <>
                                    <Cpu className="w-4 h-4" />
                                    Analyse + Generate Report
                                </>
                            )}
                        </button>

                        {error && (
                            <div className="flex items-start gap-2 text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg p-3">
                                <AlertCircle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
                                {error}
                            </div>
                        )}
                    </div>

                    {/* ECG Chart */}
                    <div className="lg:col-span-2 card">
                        <h2 className="text-sm font-semibold text-slate-300 mb-4">ECG Signal</h2>
                        {ecgSignal.length > 0 ? (
                            <ECGChart data={ecgSignal} samplingRate={analysis?.sampling_rate ?? 360} />
                        ) : (
                            <div className="h-48 flex items-center justify-center text-slate-600 text-sm border border-slate-800 rounded-lg border-dashed">
                                {busy ? (
                                    <div className="flex items-center gap-2">
                                        <Loader2 className="w-4 h-4 animate-spin text-emerald-500" />
                                        <span className="text-slate-500">Processing signal…</span>
                                    </div>
                                ) : (
                                    'Signal will appear here after analysis'
                                )}
                            </div>
                        )}

                        {/* Quick Stats Bar */}
                        {analysis && (
                            <div className="grid grid-cols-4 gap-3 mt-4">
                                {[
                                    { label: 'Heart Rate', value: `${analysis.heart_rate_mean.toFixed(0)} bpm` },
                                    { label: 'Duration', value: `${analysis.duration_seconds.toFixed(1)}s` },
                                    { label: 'R-peaks', value: analysis.r_peak_count.toString() },
                                    { label: 'Quality', value: `${(analysis.signal_quality * 100).toFixed(0)}%` },
                                ].map(({ label, value }) => (
                                    <div key={label} className="bg-slate-800/60 rounded-lg p-3 text-center">
                                        <p className="text-lg font-bold font-mono text-white">{value}</p>
                                        <p className="text-xs text-slate-500 mt-0.5">{label}</p>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </div>

                {/* Results */}
                {report && (
                    <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">

                        {/* Report */}
                        <div className="lg:col-span-3">
                            <h2 className="text-sm font-semibold text-slate-300 mb-4 flex items-center gap-2">
                                <span className="w-1.5 h-1.5 bg-emerald-400 rounded-full" />
                                Clinical Report
                            </h2>
                            <ReportDisplay result={report} />
                        </div>

                        {/* Metrics */}
                        <div className="lg:col-span-2">
                            <h2 className="text-sm font-semibold text-slate-300 mb-4 flex items-center gap-2">
                                <span className="w-1.5 h-1.5 bg-brand-500 rounded-full" />
                                Analysis Metrics
                            </h2>
                            <MetricsDashboard
                                features={report.features}
                                timing={report.timing_ms}
                                rag={report.rag}
                            />
                        </div>
                    </div>
                )}

                {/* Rhythm Annotation */}
                {analysis && (
                    <div className="card border-slate-700/30">
                        <p className="text-xs text-slate-500 mb-1">Rhythm Classification</p>
                        <p className="text-base text-slate-200 font-medium">{analysis.rhythm}</p>
                        {analysis.summary && (
                            <p className="text-sm text-slate-400 mt-1">{analysis.summary}</p>
                        )}
                    </div>
                )}

            </main>
        </div>
    )
}