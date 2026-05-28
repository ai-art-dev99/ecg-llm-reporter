const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export interface AnalysisResult {
    record_name: string
    sampling_rate: number
    duration_seconds: number
    signal_quality: number
    heart_rate_mean: number
    heart_rate_classification: string
    rhythm: string
    r_peak_count: number
    anomalies: string[]
    summary: string
    prompt_context: string
    processing_time_ms: number
}

export interface ReportResult {
    report: {
        record_name: string
        generated_at: string
        model_used: string
        primary_diagnosis: string
        urgency: 'routine' | 'urgent' | 'emergency'
        confidence: string
        key_abnormalities: string[]
        rag_sources_used: string[]
        processing_time_ms: number
        report: {
            findings: string
            interpretation: string
            differentials: string
            recommendations: string
            limitations: string
        }
    }
    text: string
    features: {
        heart_rate: number
        rhythm: string
        anomalies: string[]
        quality: number
    }
    rag: { chunks_retrieved: number; sources: string[] }
    timing_ms: { ecg: number; rag: number; llm: number; total: number }
}

export interface StatusResult {
    rag: { ready: boolean; total_chunks: number }
    llm: { ready: boolean; model: string }
    tracing: { langsmith: boolean }
    data: { records: string[] }
}

export interface SyntheticRequest {
    heart_rate: number
    duration: number
    rhythm: string
    noise: number
}

// ── API calls ────────────────────────────────────────────────────────────────

export async function getStatus(): Promise<StatusResult> {
    const r = await fetch(`${API}/status`, { cache: 'no-store' })
    if (!r.ok) throw new Error('API unreachable')
    return r.json()
}

export async function analyzeRecord(name: string): Promise<AnalysisResult> {
    const r = await fetch(`${API}/records/${name}/analyze`, { cache: 'no-store' })
    if (!r.ok) throw new Error(await r.text())
    return r.json()
}

export async function analyzeUpload(file: File, samplingRate = 360): Promise<AnalysisResult> {
    const fd = new FormData()
    fd.append('file', file)
    const r = await fetch(`${API}/analyze/upload?sampling_rate=${samplingRate}`, {
        method: 'POST', body: fd,
    })
    if (!r.ok) throw new Error(await r.text())
    return r.json()
}

export async function analyzeSynthetic(req: SyntheticRequest): Promise<AnalysisResult> {
    const r = await fetch(`${API}/analyze/synthetic`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req),
    })
    if (!r.ok) throw new Error(await r.text())
    return r.json()
}

export async function generateReport(name: string): Promise<ReportResult> {
    const r = await fetch(`${API}/records/${name}/report`, { cache: 'no-store' })
    if (!r.ok) throw new Error(await r.text())
    return r.json()
}

export async function generateSyntheticReport(req: SyntheticRequest): Promise<ReportResult> {
    const r = await fetch(`${API}/analyze/synthetic/report`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...req, include_rag: true }),
    })
    if (!r.ok) throw new Error(await r.text())
    return r.json()
}

export async function generateUploadReport(file: File, samplingRate = 360): Promise<ReportResult> {
    const fd = new FormData()
    fd.append('file', file)
    const r = await fetch(`${API}/analyze/upload/report?sampling_rate=${samplingRate}`, {
        method: 'POST', body: fd,
    })
    if (!r.ok) throw new Error(await r.text())
    return r.json()
}