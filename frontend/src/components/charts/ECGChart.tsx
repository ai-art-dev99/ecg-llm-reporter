'use client'
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, ReferenceLine, Tooltip } from 'recharts'

interface Props {
    data: number[]
    samplingRate: number
    height?: number
}

export default function ECGChart({ data, samplingRate, height = 180 }: Props) {
    // Downsample for performance — show max 1000 points
    const step = Math.max(1, Math.floor(data.length / 1000))
    const sampled = data.filter((_, i) => i % step === 0)
    const chartData = sampled.map((v, i) => ({
        t: +((i * step / samplingRate).toFixed(3)),
        v: +v.toFixed(4),
    }))

    const min = Math.min(...sampled)
    const max = Math.max(...sampled)
    const pad = (max - min) * 0.15

    return (
        <div className="w-full bg-slate-950 rounded-lg p-3 border border-slate-700/40">
            <div className="flex items-center gap-2 mb-2">
                <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse-slow" />
                <span className="text-xs text-slate-400 font-mono">Lead II — {data.length} samples @ {samplingRate}Hz</span>
            </div>
            <ResponsiveContainer width="100%" height={height}>
                <LineChart data={chartData} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
                    <XAxis
                        dataKey="t"
                        tick={{ fontSize: 10, fill: '#64748b' }}
                        tickFormatter={v => `${v}s`}
                        interval="preserveStartEnd"
                    />
                    <YAxis
                        domain={[min - pad, max + pad]}
                        tick={{ fontSize: 10, fill: '#64748b' }}
                        tickFormatter={v => v.toFixed(2)}
                        width={42}
                    />
                    <ReferenceLine y={0} stroke="#334155" strokeDasharray="3 3" />
                    <Tooltip
                        contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, fontSize: 11 }}
                        formatter={(v: number) => [v.toFixed(4) + ' mV', 'Amplitude']}
                        labelFormatter={l => `t = ${l}s`}
                    />
                    <Line
                        type="monotone"
                        dataKey="v"
                        stroke="#34d399"
                        strokeWidth={1.5}
                        dot={false}
                        isAnimationActive={false}
                    />
                </LineChart>
            </ResponsiveContainer>
        </div>
    )
}