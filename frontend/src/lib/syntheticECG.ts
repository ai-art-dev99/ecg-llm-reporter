/**
 * Lightweight ECG signal simulator for the browser.
 * Used to display a signal waveform matching the analysed ECG.
 * No dependencies — pure math.
 */

interface SimOptions {
    hr: number   // heart rate bpm
    n: number   // total samples
    fs?: number   // sampling rate (default 360)
}

function simulate({ hr, n, fs = 360 }: SimOptions): number[] {
    const signal: number[] = new Array(Math.round(n)).fill(0)
    const rr = (60 / hr) * fs   // RR interval in samples

    for (let i = 0; i < signal.length; i++) {
        const t = (i % rr) / rr   // 0..1 within each beat

        // P wave (0.08–0.18)
        signal[i] += 0.15 * gauss(t, 0.13, 0.02)

        // Q (0.27–0.30) — small negative
        signal[i] -= 0.08 * gauss(t, 0.285, 0.008)

        // R peak (0.30–0.33)
        signal[i] += 1.20 * gauss(t, 0.315, 0.012)

        // S (0.33–0.36) — negative
        signal[i] -= 0.25 * gauss(t, 0.345, 0.010)

        // T wave (0.45–0.65)
        signal[i] += 0.30 * gauss(t, 0.55, 0.045)

        // Baseline wander
        signal[i] += 0.01 * Math.sin(2 * Math.PI * t * 0.3)

        // Noise
        signal[i] += (Math.random() - 0.5) * 0.015
    }

    return signal
}

function gauss(x: number, mu: number, sigma: number): number {
    return Math.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
}

export default { simulate }