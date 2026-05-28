'use client'
import { Activity, Brain, Database, Zap } from 'lucide-react'

interface Props {
    features: {
        heart_rate: number
        rhythm: string
        anomalies: string[]
        quality: number
    }
    timing: { ecg: number; rag: number; llm: number; total: number }
    rag: { chunks_retrieved: number; sources: string[] }
}

export default function MetricsDashboard({ features, timing, rag }: Props) {
    const qualityPct = Math.round(features.quality * 100)
    const qualityColor = qualityPct >= 70 ? 'text-emerald-400' : qualityPct >= 40 ? 'text-amber-400' : 'text-red-400'

    return (
        <div className="grid grid-cols-2 gap-3">

            {/* Heart Rate */}
            <div className="card flex items-start gap-3">
                <div className="p-2 bg-red-500/10 rounded-lg mt-0.5">
                    <Activity className="w-4 h-4 text-red-400" />
                </div>
                <div>
                    <p className="text-xs text-slate-400 mb-0.5">Heart Rate</p>
                    <p className="text-2xl font-bold text-white font-mono">{features.heart_rate.toFixed(0)}</p>
                    <p className="text-xs text-slate-500">bpm</p>
                </div>
            </div>

            {/* Signal Quality */}
            <div className="card flex items-start gap-3">
                <div className="p-2 bg-brand-500/10 rounded-lg mt-0.5">
                    <Zap className="w-4 h-4 text-brand-500" />
                </div>
                <div className="flex-1">
                    <p className="text-xs text-slate-400 mb-1">Signal Quality</p>
                    <div className="flex items-center gap-2">
                        <div className="flex-1 bg-slate-800 rounded-full h-2">
                            <div
                                className="h-2 rounded-full bg-gradient-to-r from-emerald-500 to-brand-500 transition-all"
                                style={{ width: `${qualityPct}%` }}
                            />
                        </div>
                        <span className={`text-sm font-bold font-mono ${qualityColor}`}>{qualityPct}%</span>
                    </div>
                </div>
            </div>

            {/* RAG */}
            <div className="card flex items-start gap-3">
                <div className="p-2 bg-purple-500/10 rounded-lg mt-0.5">
                    <Database className="w-4 h-4 text-purple-400" />
                </div>
                <div>
                    <p className="text-xs text-slate-400 mb-0.5">Guidelines Used</p>
                    <p className="text-2xl font-bold text-white font-mono">{rag.chunks_retrieved}</p>
                    <p className="text-xs text-slate-500">chunks retrieved</p>
                </div>
            </div>

            {/* Pipeline Timing */}
            <div className="card flex items-start gap-3">
                <div className="p-2 bg-amber-500/10 rounded-lg mt-0.5">
                    <Brain className="w-4 h-4 text-amber-400" />
                </div>
                <div className="w-full">
                    <p className="text-xs text-slate-400 mb-1.5">Pipeline Timing</p>
                    <div className="space-y-1">
                        {[
                            { label: 'ECG', ms: timing.ecg, color: 'bg-emerald-500' },
                            { label: 'RAG', ms: timing.rag, color: 'bg-purple-500' },
                            { label: 'LLM', ms: timing.llm, color: 'bg-amber-500' },
                        ].map(({ label, ms, color }) => (
                            <div key={label} className="flex items-center gap-2">
                                <span className="text-xs text-slate-500 w-7">{label}</span>
                                <div className="flex-1 bg-slate-800 rounded-full h-1.5">
                                    <div
                                        className={`h-1.5 rounded-full ${color}`}
                                        style={{ width: `${Math.min(100, (ms / timing.total) * 100)}%` }}
                                    />
                                </div>
                                <span className="text-xs text-slate-400 font-mono w-14 text-right">{ms.toFixed(0)}ms</span>
                            </div>
                        ))}
                    </div>
                    <p className="text-xs text-slate-500 mt-1.5 text-right font-mono">total {timing.total.toFixed(0)}ms</p>
                </div>
            </div>

            {/* Anomalies */}
            {features.anomalies.length > 0 && (
                <div className="card col-span-2">
                    <p className="text-xs text-slate-400 mb-2">Anomaly Flags</p>
                    <div className="flex flex-wrap gap-2">
                        {features.anomalies.map(a => (
                            <span key={a} className="text-xs px-2 py-1 rounded-full bg-red-500/15 text-red-400 border border-red-500/25 font-mono">
                                {a.replace(/_/g, ' ')}
                            </span>
                        ))}
                    </div>
                </div>
            )}

            {/* RAG Sources */}
            {rag.sources.length > 0 && (
                <div className="card col-span-2">
                    <p className="text-xs text-slate-400 mb-2">Guidelines Referenced</p>
                    <div className="flex flex-wrap gap-2">
                        {rag.sources.map(s => (
                            <span key={s} className="text-xs px-2 py-1 rounded-full bg-purple-500/15 text-purple-300 border border-purple-500/25">
                                {s}
                            </span>
                        ))}
                    </div>
                </div>
            )}

        </div>
    )
}