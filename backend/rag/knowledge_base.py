"""
ECG Medical Knowledge Base
---------------------------
Built-in clinical guidelines extracted from:
  - ESC (European Society of Cardiology)
  - AHA (American Heart Association)
  - ACC (American College of Cardiology)

This is always available — no internet or PDF required.
PDFs and web content ADD to this base, not replace it.
"""

KNOWLEDGE_BASE: list[dict] = [

    # ── Normal ECG ─────────────────────────────────────────────────────────
    {
        "id": "ecg_normal_001",
        "source": "AHA Guidelines",
        "category": "normal",
        "title": "Normal ECG Parameters",
        "content": """
Normal ECG values in adults:
Heart Rate: 60–100 beats per minute.
PR interval: 120–200 ms (3–5 small squares).
QRS duration: 80–120 ms (2–3 small squares).
QT interval: 350–440 ms; varies with heart rate.
QTc (Bazett corrected): <440 ms in males, <460 ms in females.
P wave: Upright in leads I, II, aVF. Duration <120 ms. Amplitude <2.5 mm.
T wave: Upright in leads I, II, V3–V6. Inverted in aVR.
ST segment: Isoelectric (flat). Elevation or depression >1 mm is abnormal.
Axis: Normal frontal QRS axis is -30° to +90°.
"""
    },

    # ── Heart Rate Disorders ────────────────────────────────────────────────
    {
        "id": "ecg_tachy_001",
        "source": "ESC Guidelines 2019",
        "category": "arrhythmia",
        "title": "Sinus Tachycardia",
        "content": """
Sinus Tachycardia definition: Heart rate >100 bpm originating from the sinus node.
ECG features: Normal P waves preceding each QRS; PR interval normal; regular rhythm.
Common causes: fever, pain, anxiety, dehydration, anemia, hyperthyroidism, heart failure,
pulmonary embolism, medications (beta-agonists, caffeine, cocaine).
Management: Treat underlying cause. Beta-blockers if symptomatic and no contraindication.
Inappropriate sinus tachycardia: HR >100 bpm at rest without physiological cause; 
treated with ivabradine or beta-blockers.
"""
    },
    {
        "id": "ecg_brady_001",
        "source": "ESC Guidelines 2021",
        "category": "arrhythmia",
        "title": "Sinus Bradycardia",
        "content": """
Sinus Bradycardia definition: Heart rate <60 bpm from the sinus node.
ECG features: Normal P wave before each QRS; regular rhythm; all intervals normal.
Common causes: Athletes (physiological), vagal tone, sleep, hypothyroidism, 
hypothermia, inferior MI, sick sinus syndrome, beta-blockers, calcium channel blockers,
digoxin toxicity.
Symptoms: Usually asymptomatic. May cause dizziness, syncope, fatigue if severe (<40 bpm).
Management: No treatment if asymptomatic. Atropine 0.5–1 mg IV for symptomatic cases.
Permanent pacemaker if symptomatic and no reversible cause.
"""
    },

    # ── Atrial Fibrillation ─────────────────────────────────────────────────
    {
        "id": "ecg_afib_001",
        "source": "ESC AFib Guidelines 2020",
        "category": "arrhythmia",
        "title": "Atrial Fibrillation",
        "content": """
Atrial Fibrillation (AF): Most common sustained cardiac arrhythmia.
ECG features: Absent P waves replaced by irregular fibrillatory baseline (f-waves, 350–600/min).
Irregularly irregular R-R intervals. QRS typically narrow unless aberrant conduction.
Ventricular rate usually 100–160 bpm if untreated.
Classification: Paroxysmal (<7 days), Persistent (>7 days), Long-standing persistent (>1 year), Permanent.
Stroke risk: CHA2DS2-VASc score guides anticoagulation.
Score ≥2 (men) or ≥3 (women): oral anticoagulation recommended.
Rate control: Target HR <110 bpm at rest. Use beta-blockers or non-dihydropyridine CCBs.
Rhythm control: Cardioversion, antiarrhythmics (flecainide, amiodarone), ablation.
"""
    },
    {
        "id": "ecg_afib_002",
        "source": "ESC AFib Guidelines 2020",
        "category": "arrhythmia",
        "title": "Atrial Fibrillation — Differential Diagnosis",
        "content": """
Differentiating AF from other irregular rhythms:
AF vs Atrial Flutter: Flutter has regular sawtooth waves at 300/min; 
ventricular rate typically 150/min (2:1 block). AF has completely irregular baseline.
AF vs Multifocal Atrial Tachycardia (MAT): MAT has ≥3 distinct P wave morphologies, 
irregular rate but P waves visible. Common in COPD patients.
AF vs Frequent PACs: PACs show early P waves with different morphology; can look like AF 
but has identifiable P waves.
Ashman phenomenon: Wide QRS beats in AF due to aberrant conduction after long-short RR sequence.
"""
    },

    # ── QT Interval ────────────────────────────────────────────────────────
    {
        "id": "ecg_qt_001",
        "source": "AHA/ACC Guidelines",
        "category": "intervals",
        "title": "QT Interval and QTc",
        "content": """
QT Interval measurement: Start of Q wave to end of T wave. Measured in V5 or II.
QTc calculation (Bazett formula): QTc = QT / √(RR interval in seconds).
Normal QTc: <440 ms (males), <460 ms (females).
Borderline: 440–460 ms (males), 460–480 ms (females).
Prolonged QTc: >460 ms (males), >480 ms (females). Risk of Torsades de Pointes.
Short QTc: <340 ms. Associated with increased arrhythmia risk.
Causes of prolonged QT:
  Congenital: Long QT syndrome (LQTS1, LQTS2, LQTS3).
  Acquired: Hypokalemia, hypomagnesemia, hypocalcemia.
  Drugs: Amiodarone, sotalol, haloperidol, erythromycin, chloroquine, methadone.
  Cardiac: Myocardial ischemia, myocarditis, complete heart block.
Torsades de Pointes (TdP): Polymorphic VT associated with long QT; 
ECG shows twisting QRS complexes around isoelectric line.
Management: Correct electrolytes, stop causative drugs, IV magnesium 2g for TdP.
"""
    },

    # ── Bundle Branch Blocks ────────────────────────────────────────────────
    {
        "id": "ecg_lbbb_001",
        "source": "ESC Guidelines",
        "category": "conduction",
        "title": "Left Bundle Branch Block (LBBB)",
        "content": """
LBBB ECG criteria (all must be present):
1. QRS duration ≥120 ms.
2. Broad notched or slurred R wave in lateral leads (I, aVL, V5, V6).
3. Absent septal Q waves in I, V5, V6.
4. RS or rS pattern in V1.
5. ST-T waves discordant (opposite direction to QRS).
Clinical significance: LBBB is almost always pathological.
Causes: Coronary artery disease (most common), hypertension, cardiomyopathy, 
aortic valve disease, cardiac surgery.
New LBBB: Treat as STEMI equivalent if chest pain present (Sgarbossa criteria).
LBBB with chest pain: Emergent cardiology evaluation; consider primary PCI.
Sgarbossa criteria: ST elevation ≥1mm concordant with QRS scores 5 points (specific for MI).
"""
    },
    {
        "id": "ecg_rbbb_001",
        "source": "ESC Guidelines",
        "category": "conduction",
        "title": "Right Bundle Branch Block (RBBB)",
        "content": """
RBBB ECG criteria:
1. QRS duration ≥120 ms (complete RBBB); 100–119 ms (incomplete RBBB).
2. RSR' pattern (M-shaped) in V1-V2 (rabbit ears).
3. Wide slurred S wave in lateral leads (I, V5, V6).
4. T wave inversion in V1-V3 (secondary repolarisation change).
Clinical significance: Can be normal variant, especially incomplete RBBB.
Complete RBBB causes: Right heart strain (PE, cor pulmonale), right heart catheterisation,
congenital heart disease, myocardial ischemia, Brugada syndrome.
New RBBB: Consider pulmonary embolism in appropriate clinical context.
Brugada pattern: RBBB-like pattern with ST elevation in V1–V2 (coved or saddle-back type);
associated with risk of sudden cardiac death; genetic sodium channelopathy.
"""
    },

    # ── ST Changes ─────────────────────────────────────────────────────────
    {
        "id": "ecg_stemi_001",
        "source": "ESC STEMI Guidelines 2017",
        "category": "ischemia",
        "title": "ST Elevation Myocardial Infarction (STEMI)",
        "content": """
STEMI ECG criteria: New ST elevation at J-point in ≥2 contiguous leads.
Threshold: ≥1 mm in all leads except V2–V3.
V2–V3 threshold: ≥2.5 mm in men <40 years, ≥2 mm in men ≥40 years, ≥1.5 mm in women.
STEMI localisation:
  Anterior: V1–V4 (LAD occlusion).
  Inferior: II, III, aVF (RCA or LCx occlusion).
  Lateral: I, aVL, V5–V6 (LCx occlusion).
  Posterior: ST depression V1–V3 + upright T; dominant R in V1–V2.
  Right ventricular: ST elevation V4R.
Reciprocal changes: ST depression in leads opposite to elevation (supports STEMI diagnosis).
Evolution: Hyperacute T waves → ST elevation → Q waves → T wave inversion → Q waves persist.
Management: Primary PCI within 90 minutes (door-to-balloon). 
Fibrinolysis if PCI unavailable within 120 minutes.
Dual antiplatelet therapy: Aspirin + P2Y12 inhibitor (ticagrelor or prasugrel preferred).
"""
    },
    {
        "id": "ecg_nstemi_001",
        "source": "ESC NSTE-ACS Guidelines 2020",
        "category": "ischemia",
        "title": "NSTEMI and Unstable Angina",
        "content": """
NSTEMI/UA ECG features:
ST depression: Horizontal or downsloping ≥0.5 mm in ≥2 contiguous leads.
T wave inversion: ≥1 mm in leads with dominant R wave.
No ST elevation (distinguishes from STEMI).
Wellens syndrome: Biphasic or deeply inverted T waves in V2–V3; 
indicates critical proximal LAD stenosis; high risk of anterior MI.
De Winter T waves: Upsloping ST depression in V1–V6 with peaked T waves; 
anterior STEMI equivalent requiring urgent PCI.
Management: Risk stratification with GRACE or TIMI score.
High risk: Invasive strategy within 24 hours.
Intermediate risk: Invasive within 72 hours.
Anticoagulation: Fondaparinux preferred (unless PCI planned, then UFH).
"""
    },

    # ── Hypertrophy ─────────────────────────────────────────────────────────
    {
        "id": "ecg_lvh_001",
        "source": "AHA Guidelines",
        "category": "hypertrophy",
        "title": "Left Ventricular Hypertrophy (LVH)",
        "content": """
LVH ECG criteria:
Sokolow-Lyon: S in V1 + R in V5 or V6 ≥35 mm (sensitivity 22%, specificity 100%).
Cornell voltage: R in aVL + S in V3 >28 mm (men) or >20 mm (women).
Cornell product: Cornell voltage × QRS duration >2436 mm·ms.
LVH pattern: Tall R in lateral leads, deep S in right precordial leads.
Strain pattern: ST depression + T inversion in I, aVL, V5–V6 (indicates LVH with repolarisation abnormality).
Causes: Hypertension (most common), aortic stenosis, hypertrophic cardiomyopathy, 
obesity, athletic heart.
Significance: LVH is independent predictor of cardiovascular events.
Echocardiography is gold standard for LVH diagnosis (ECG has low sensitivity).
"""
    },

    # ── AV Blocks ──────────────────────────────────────────────────────────
    {
        "id": "ecg_avblock_001",
        "source": "ESC Pacing Guidelines 2021",
        "category": "conduction",
        "title": "Atrioventricular (AV) Blocks",
        "content": """
First-degree AV block: PR interval >200 ms; every P wave conducts; usually benign.
Causes: Enhanced vagal tone, inferior MI, myocarditis, digoxin, beta-blockers.

Second-degree AV block — Mobitz Type I (Wenckebach):
Progressive PR prolongation until a P wave is not conducted (dropped QRS).
Grouped beating pattern. Usually benign; often in inferior MI or athletes.

Second-degree AV block — Mobitz Type II:
Constant PR interval with sudden non-conducted P waves.
QRS often wide (bundle branch block pattern).
More serious than Mobitz I; risk of progression to complete heart block.
Requires pacemaker evaluation.

Third-degree (Complete) AV block:
P waves and QRS complexes are completely dissociated.
Atrial rate > ventricular rate.
Escape rhythm: Junctional (narrow QRS, 40–60 bpm) or ventricular (wide QRS, 20–40 bpm).
Causes: Inferior MI, anterior MI, Lyme disease, digoxin toxicity, post-cardiac surgery.
Emergency: Transcutaneous pacing; temporary pacing; permanent pacemaker.
"""
    },

    # ── HRV Clinical Significance ──────────────────────────────────────────
    {
        "id": "ecg_hrv_001",
        "source": "ESC/NASPE HRV Guidelines",
        "category": "hrv",
        "title": "Heart Rate Variability (HRV) Clinical Interpretation",
        "content": """
HRV is a measure of variation in time between heartbeats (RR intervals).
Reflects autonomic nervous system balance (sympathetic/parasympathetic).

Time-domain measures:
SDNN: Standard deviation of NN intervals. Normal >100 ms. 
  <50 ms: high risk; 50–100 ms: intermediate risk.
RMSSD: Root mean square of successive differences. Reflects parasympathetic activity.
  Normal: 20–50 ms. Reduced in autonomic neuropathy, heart failure, post-MI.
pNN50: % NN intervals differing >50 ms. Normal >3%. Reflects vagal tone.

Clinical significance of reduced HRV:
Post-MI: Reduced HRV predicts mortality (SDNN <70 ms associated with 5x mortality increase).
Heart failure: SDNN <50 ms indicates poor prognosis.
Diabetic autonomic neuropathy: Reduced RMSSD and pNN50.
Depression and PTSD: Associated with reduced HRV.

Increased HRV: Athletes, young age, good cardiovascular fitness.
"""
    },

    # ── Channelopathies ────────────────────────────────────────────────────
    {
        "id": "ecg_brugada_001",
        "source": "ESC Guidelines on Channelopathies 2022",
        "category": "channelopathy",
        "title": "Brugada Syndrome",
        "content": """
Brugada Syndrome: Autosomal dominant sodium channelopathy (SCN5A mutation, 20–30%).
ECG pattern (Type 1 — diagnostic): Coved ST elevation ≥2 mm in ≥1 of V1–V3, 
followed by negative T wave. Spontaneous or drug-induced.
Type 2 (saddle-back): ST elevation ≥2 mm, positive or biphasic T wave (not diagnostic alone).
Prevalence: 1–5 per 10,000. More common in Asian populations.
Clinical features: Ventricular fibrillation, sudden cardiac death, often at rest or sleep.
Age of presentation: Typically 3rd–4th decade. Male predominance (8:1).
Triggers: Fever, sodium channel blockers (flecainide, ajmaline provocation test), 
large meals, vagotonia.
Management: ICD for symptomatic patients. Quinidine for recurrent VF.
Asymptomatic: ICD controversial; close follow-up recommended.
"""
    },
    {
        "id": "ecg_wpw_001",
        "source": "ESC Guidelines",
        "category": "channelopathy",
        "title": "Wolff-Parkinson-White (WPW) Syndrome",
        "content": """
WPW: Pre-excitation syndrome with accessory pathway (Bundle of Kent) bypassing the AV node.
ECG features:
1. Short PR interval (<120 ms).
2. Delta wave: Slurred upstroke at onset of QRS.
3. Wide QRS (>120 ms total with delta wave).
4. Secondary ST-T changes (discordant).
Risk: AF with rapid conduction down accessory pathway → VF → sudden death.
Risk factors for SCD: Shortest pre-excited RR interval <250 ms during AF, 
symptomatic tachycardia, multiple accessory pathways.
Management: Catheter ablation of accessory pathway (curative, >95% success).
Avoid AV nodal blocking drugs (digoxin, verapamil, adenosine) in WPW with AF —
risk of accelerated conduction and VF.
"""
    },

    # ── Electrolyte Abnormalities ──────────────────────────────────────────
    {
        "id": "ecg_electrolyte_001",
        "source": "Clinical ECG Interpretation",
        "category": "metabolic",
        "title": "Electrolyte Abnormalities on ECG",
        "content": """
Hyperkalaemia ECG changes (progressive with rising K+):
K+ 5.5–6.5: Tall peaked T waves (tented), shortened QT.
K+ 6.5–7.5: Widened QRS, prolonged PR, flattened P waves.
K+ >7.5: Sine wave pattern, VF, asystole.
Treatment: Calcium gluconate (membrane stabilisation), insulin+glucose, 
sodium bicarbonate, salbutamol, dialysis.

Hypokalaemia ECG changes:
Flattened/inverted T waves, prominent U waves (positive deflection after T wave in V2–V3),
ST depression, prolonged QU interval, increased risk of TdP.
Treatment: IV or oral potassium replacement.

Hypercalcaemia: Shortened QT interval, shortened ST segment.
Hypocalcaemia: Prolonged QT interval, prolonged ST segment.

Hypomagnesaemia: Prolonged PR, QRS, QT; predisposes to TdP and AF.
Treatment: IV magnesium sulfate 2g over 10 minutes.
"""
    },
]


def get_all_documents() -> list[dict]:
    """Return all built-in knowledge base documents."""
    return KNOWLEDGE_BASE


def get_by_category(category: str) -> list[dict]:
    """Filter documents by clinical category."""
    return [d for d in KNOWLEDGE_BASE if d["category"] == category]


def get_categories() -> list[str]:
    """Return all unique categories."""
    return list({d["category"] for d in KNOWLEDGE_BASE})