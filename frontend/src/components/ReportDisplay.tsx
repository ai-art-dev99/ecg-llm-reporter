'use client'
import { ReportResult } from '../lib/api'
import { AlertTriangle, CheckCircle, AlertCircle, FileText, Download } from 'lucide-react'
import clsx from 'clsx'

interface Props { result: ReportResult }

const URGENCY_CONFIG = {
    routine: { icon: CheckCircle, label: 'Routine', cls: 'badge-routine' },
    urgent: { icon: AlertTriangle, label: 'Urgent', cls: 'badge-urgent' },
    emergency: { icon: AlertCircle, label: 'Emergency', cls: 'badge-emergency' },
}

export default function ReportDisplay({ result }: Props) {
    const { report } = result
    const urg = URGENCY_CONFIG[report.urgency] ?? URGENCY_CONFIG.routine
    const Icon = urg.icon

    const download = () => {
        const blob = new Blob([result.text], { type: 'text/plain' })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `ecg-report-${report.record_name}-${Date.now()}.txt`
        a.click()
        URL.revokeObjectURL(url)
    }

    return (
        <div className="space-y-4">

            {/* Header */}
            <div className="card flex items-start justify-between gap-4">
                <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2 flex-wrap">
                        <span className={clsx('inline-flex items-center gap-1.5 text-xs font-semibold px-2.5 py-1 rounded-full', urg.cls)}>
                            <Icon className="w-3.5 h-3.5" />
                            {urg.label}
                        </span>
                        <span className="text-xs text-slate-500 font-mono">confidence: {report.confidence}</span>
                        <span className="text-xs text-slate-500 font-mono">{report.model_used}</span>
                    </div>
                    <h3 className="text-lg font-semibold text-white leading-snug">{report.primary_diagnosis}</h3>
                </div>
                <button
                    onClick={download}
                    className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-white border border-slate-700 hover:border-slate-500 px-3 py-1.5 rounded-lg transition-colors"
                >
                    <Download className="w-3.5 h-3.5" />
                    Export
                </button>
            </div>

            {/* Key Abnormalities */}
            {report.key_abnormalities.length > 0 && (
                <div className="card border-amber-500/30 bg-amber-500/5">
                    <p className="text-xs font-semibold text-amber-400 uppercase tracking-wide mb-2">Key Findings</p>
                    <ul className="space-y-1">
                        {report.key_abnormalities.map((a, i) => (
                            <li key={i} className="flex items-start gap-2 text-sm text-slate-300">
                                <span className="text-amber-400 mt-0.5">•</span>{a}
                            </li>
                        ))}
                    </ul>
                </div>
            )}

            {/* Report Sections */}
            {[
                { label: 'Findings', text: report.report.findings },
                { label: 'Interpretation', text: report.report.interpretation },
                { label: 'Differentials', text: report.report.differentials },
                { label: 'Recommendations', text: report.report.recommendations },
            ].map(({ label, text }) => (
                <div key={label} className="card">
                    <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">{label}</p>
                    <p className="text-sm text-slate-300 leading-relaxed">{text}</p>
                </div>
            ))}

            {/* Limitations */}
            <div className="card border-slate-700/30 bg-slate-800/30">
                <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1.5">⚠ Limitations</p>
                <p className="text-xs text-slate-500 leading-relaxed">{report.report.limitations}</p>
            </div>

        </div>
    )
}