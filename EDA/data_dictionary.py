"""Plain-language data dictionary + feature interdependencies for the dashboard.

This is **reference content**, not computed statistics — it answers "what is this
column and how does it relate to the others?" so a reader can understand the 49
raw fields without combing the CSV. The live numbers (distributions, missingness,
correlations) come from ``profile.py`` / ``aspects.py``; this module supplies the
human meaning, how to read the values, and the curated interdependency graph that
the **Data Dictionary** page renders.

Provenance: the per-column ``meaning`` / ``reads_as`` / ``links`` were drafted per
column-family and then adversarially fact-checked against the live correlation and
target-association numbers (empirical links must match a real number; mechanical /
clinical / governance links must be definitionally or domain sound). The
interdependency graph collapses the 23 sparse drug columns into one ``medications``
node while keeping ``insulin`` (the only well-populated drug) separate.

Link ``kind`` legend:
  - empirical  : backed by a measured correlation / target-association number
  - mechanical : definitional or computed-from (e.g. number_diagnoses tallies diag_*)
  - clinical   : a real-world medical relationship
  - governance : matters for the graded pipeline (grouped split, row filter, target)
"""

from __future__ import annotations

# Visual metadata for the four link kinds and the four edge strengths.
KIND_META: dict[str, dict[str, str]] = {
    "empirical":  {"icon": "📊", "color": "#2563eb", "label": "empirical (measured)"},
    "mechanical": {"icon": "⚙️", "color": "#0891b2", "label": "mechanical (computed-from)"},
    "clinical":   {"icon": "🩺", "color": "#7c3aed", "label": "clinical (medical)"},
    "governance": {"icon": "⚖️", "color": "#d97706", "label": "governance (pipeline rule)"},
}

STRENGTH_META: dict[str, dict] = {
    "strong":     {"width": 4.0, "dash": "solid", "color": "#1d4ed8", "label": "strong |r|≥0.35"},
    "moderate":   {"width": 2.6, "dash": "solid", "color": "#3b82f6", "label": "moderate 0.2–0.35"},
    "weak":       {"width": 1.6, "dash": "solid", "color": "#93c5fd", "label": "weak |r|<0.2"},
    "conceptual": {"width": 1.6, "dash": "dot",   "color": "#a78bfa", "label": "conceptual link"},
}

COLUMNS = {   'encounter_id': {   'family': 'Identifier',
                        'meaning': 'Administrative primary key for one hospital encounter '
                                   '(admission to discharge). No clinical content; labels a '
                                   'single row.',
                        'reads_as': 'int64, unique per row (101,766 distinct, 0 missing). '
                                    'Audit-log / join key only, not a feature.',
                        'links': [   {   'to': 'patient_nbr',
                                         'why': 'Many encounter_ids nest under one patient_nbr; '
                                                'this hierarchy defines the grouping unit for '
                                                'the split.',
                                         'kind': 'governance'},
                                     {   'to': 'readmitted',
                                         'why': 'Never a feature: unique per row lets a tree '
                                                'memorise rows and overfit the target.',
                                         'kind': 'governance'}]},
    'patient_nbr': {   'family': 'Identifier',
                       'meaning': 'Stable de-identified person ID recurring across visits; links '
                                  'an encounter to a patient, not a single admission.',
                       'reads_as': 'int64, 71,518 distinct (~1.42 visits/patient); 46.21% of '
                                   'rows are repeat patients, max 40.',
                       'links': [   {   'to': 'encounter_id',
                                        'why': 'One patient_nbr groups its multiple encounter '
                                               'rows, defining the visit-within-patient '
                                               'hierarchy.',
                                        'kind': 'governance'},
                                    {   'to': 'readmitted',
                                        'why': 'Group split on patient_nbr stops the same '
                                               'patient leaking across train/test and biasing '
                                               'the target.',
                                        'kind': 'governance'},
                                    {   'to': 'number_inpatient',
                                        'why': 'Keys leakage-safe prior-visit history built only '
                                               "from a patient's earlier encounters.",
                                        'kind': 'governance'}]},
    'race': {   'family': 'Demographic',
                'meaning': 'Self-reported/administratively recorded patient race. A weak '
                           'clinical covariate, used mainly as the protected attribute for '
                           'slicing the fairness audit.',
                'reads_as': '5 levels; ~94% Caucasian/AfricanAmerican; Hispanic/Other/Asian '
                            "tail; '?' missing (2.2%) loaded as NaN.",
                'links': [   {   'to': 'gender',
                                 'why': 'Co-protected attribute; both feed Fairlearn MetricFrame '
                                        'slicing, neither is a real predictor',
                                 'kind': 'governance'},
                             {   'to': 'age',
                                 'why': 'Co-protected attribute; the three demographics together '
                                        'define fairness subgroups',
                                 'kind': 'governance'},
                             {   'to': 'payer_code',
                                 'why': 'Payer/insurance is a socioeconomic proxy that '
                                        'correlates with race for the audit',
                                 'kind': 'clinical'}]},
    'gender': {   'family': 'Demographic',
                  'meaning': 'Patient sex recorded at registration. A pre-discharge protected '
                             'attribute for the fairness audit; not a useful standalone '
                             'predictor.',
                  'reads_as': "Female ~53.8%, Male ~46.2%, plus 3 degenerate 'Unknown/Invalid' "
                              'rows; 0% missing.',
                  'links': [   {   'to': 'race',
                                   'why': 'Co-protected attribute; both slice the Fairlearn '
                                          'fairness audit, not the model',
                                   'kind': 'governance'},
                               {   'to': 'age',
                                   'why': 'Co-protected attribute; jointly define demographic '
                                          'fairness subgroups',
                                   'kind': 'governance'}]},
    'age': {   'family': 'Demographic',
               'meaning': 'Patient age as a 10-year decade bucket. Clinically meaningful '
                          '(comorbidity load rises with age) and a protected attribute for '
                          'fairness slicing.',
               'reads_as': '10 ordered decade buckets [0-10)..[90-100); skews older ([70-80) '
                           'largest); 0% missing; stored as string interval.',
               'links': [   {   'to': 'number_diagnoses',
                                'why': 'r~0.243, older patients carry more coded comorbidities',
                                'kind': 'empirical'},
                            {   'to': 'time_in_hospital',
                                'why': 'r~0.108, older patients stay longer',
                                'kind': 'empirical'},
                            {   'to': 'race',
                                'why': 'Co-protected attribute; jointly define fairness '
                                       'subgroups for the audit',
                                'kind': 'governance'},
                            {   'to': 'gender',
                                'why': 'Co-protected attribute; jointly define fairness '
                                       'subgroups for the audit',
                                'kind': 'governance'},
                            {   'to': 'payer_code',
                                'why': 'Medicare payer skews elderly, so age and payer overlap '
                                       'as cohort proxies',
                                'kind': 'clinical'}]},
    'weight': {   'family': 'Demographic',
                  'meaning': 'Patient weight in 25-lb buckets, a potential obesity/BMI proxy, '
                             'but ~97% unrecorded so it carries almost no usable information.',
                  'reads_as': "96.9% '?' (missing); only ~3,197 rows valued, mostly [50-100); "
                              'drop-recommended.',
                  'links': [   {   'to': 'age',
                                   'why': "Both demographic covariates; weight's near-total "
                                          'missingness makes age the usable body/risk proxy',
                                   'kind': 'clinical'},
                               {   'to': 'race',
                                   'why': 'Recorded-weight rows are a non-random slice, biasing '
                                          'any demographic comparison',
                                   'kind': 'governance'}]},
    'admission_type_id': {   'family': 'Administrative ID (coded)',
                             'meaning': 'Integer code for admission acuity set at intake '
                                        '(Emergency, Urgent, Elective, plus rare '
                                        'Trauma/Newborn). Pre-discharge, no leakage. Near-zero '
                                        'target signal.',
                             'reads_as': '8 codes; 1=Emergency(53%), 3=Elective(19%), '
                                         '2=Urgent(18%); 5/6/8=NULL/NotAvailable/NotMapped '
                                         'pseudo-missing (~10%).',
                             'links': [   {   'to': 'admission_source_id',
                                              'why': 'intake pair; Emergency type co-occurs with '
                                                     'ER source, both fixed at admission',
                                              'kind': 'clinical'},
                                          {   'to': 'discharge_disposition_id',
                                              'why': 'intake acuity vs discharge destination '
                                                     "together sketch the encounter's severity "
                                                     'arc',
                                              'kind': 'clinical'},
                                          {   'to': 'readmitted',
                                              'why': "essentially no signal: Cramer's V 0.014, "
                                                     'per-level rate flat (lift ~1.03)',
                                              'kind': 'empirical'}]},
    'discharge_disposition_id': {   'family': 'Administrative ID (coded)',
                                    'meaning': 'Integer code for where/how the patient left '
                                               '(home, SNF, rehab, hospice, expired, transfer). '
                                               'Encodes frailty and drives the required '
                                               'expired/hospice row filter.',
                                    'reads_as': '26 codes; 1=home(59%), 3=SNF(14%), '
                                                '6=home-health(13%); 11/13/14/19-21=expired or '
                                                'hospice (filter out).',
                                    'links': [   {   'to': 'readmitted',
                                                     'why': "strongest admin correlate: Cramer's "
                                                            'V 0.124; sicker destinations '
                                                            'readmit far more than home',
                                                     'kind': 'empirical'},
                                                 {   'to': 'patient_nbr',
                                                     'why': 'both govern data prep: dispo '
                                                            'filters expired/hospice rows, '
                                                            'patient_nbr governs the grouped '
                                                            'split',
                                                     'kind': 'governance'},
                                                 {   'to': 'number_inpatient',
                                                     'why': 'discharge to SNF/rehab flags '
                                                            'frailty also reflected in prior '
                                                            'inpatient stays, the top predictor',
                                                     'kind': 'clinical'},
                                                 {   'to': 'medical_specialty',
                                                     'why': 'rehab/psychiatric destinations map '
                                                            'to corresponding high-risk care '
                                                            'specialties',
                                                     'kind': 'clinical'}]},
    'admission_source_id': {   'family': 'Administrative ID (coded)',
                               'meaning': 'Integer code for where the admission came from (ER, '
                                          'physician/clinic referral, transfer from '
                                          'hospital/SNF). Recorded at intake, available by '
                                          'discharge. Weak signal.',
                               'reads_as': '17 codes; 7=ER(57%), 1=PhysicianReferral(29%); '
                                           '4/5/6=transfers; 9/15/17=NULL/NotAvailable/NotMapped '
                                           '(~7%).',
                               'links': [   {   'to': 'admission_type_id',
                                                'why': 'intake pair; ER source aligns with '
                                                       'Emergency type, both fixed at admission',
                                                'kind': 'clinical'},
                                            {   'to': 'discharge_disposition_id',
                                                'why': 'transfer-in source vs transfer-out '
                                                       'destination together trace the care '
                                                       'pathway',
                                                'kind': 'clinical'},
                                            {   'to': 'readmitted',
                                                'why': "weak signal: Cramer's V 0.019; large "
                                                       'levels barely separate from the base '
                                                       'rate',
                                                'kind': 'empirical'}]},
    'payer_code': {   'family': 'Administrative text',
                      'meaning': "Billing code for the encounter's primary insurer (Medicare, "
                                 'Medicaid, BlueCross, HMO, Self-Pay, etc.). A finance field, '
                                 'not clinical; rough proxy for insurance type.',
                      'reads_as': '17 codes; ~60% populated, ~40% missing. MC=Medicare dominates '
                                  '(~53% of populated rows).',
                      'links': [   {   'to': 'medical_specialty',
                                       'why': 'Sibling registration-captured text field; both '
                                              '~40-50% missing in the same encounters '
                                              '(structural co-missingness)',
                                       'kind': 'mechanical'},
                                   {   'to': 'readmitted',
                                       'why': 'Barely separates target: assoc 0.028, lift 1.18; '
                                              'all payer levels hug the 11% base rate',
                                       'kind': 'empirical'},
                                   {   'to': 'age',
                                       'why': 'Medicare (MC) payer skews elderly, so payer '
                                              'partly re-encodes the age-cohort signal',
                                       'kind': 'clinical'},
                                   {   'to': 'race',
                                       'why': 'Both proxy socioeconomic access-to-care, so both '
                                              'are protected-attribute slices in the fairness '
                                              'audit',
                                       'kind': 'governance'}]},
    'medical_specialty': {   'family': 'Administrative text',
                             'meaning': 'Department/specialty of the admitting physician '
                                        '(InternalMedicine, Cardiology, Nephrology, '
                                        'Oncology...). Proxies the clinical reason for admission '
                                        'and care setting.',
                             'reads_as': '72 levels, ~51% populated. InternalMedicine top; '
                                         'Oncology/Nephrology ~15-19% readmit vs 11% base.',
                             'links': [   {   'to': 'readmitted',
                                              'why': 'Strongest in this family: assoc 0.048, '
                                                     'lift 1.732; Oncology/Nephrology fan well '
                                                     'above base',
                                              'kind': 'empirical'},
                                          {   'to': 'diag_1',
                                              'why': 'Specialty proxies the primary diagnosis '
                                                     'context (Nephrology->renal, '
                                                     'Oncology->cancer), overlapping diag_1',
                                              'kind': 'clinical'},
                                          {   'to': 'discharge_disposition_id',
                                              'why': 'Both encode encounter '
                                                     'care-intensity/severity; rehab/oncology '
                                                     'specialties align with sicker dispositions',
                                              'kind': 'clinical'},
                                          {   'to': 'payer_code',
                                              'why': 'Sibling registration-captured text field; '
                                                     'both ~40-50% missing in the same '
                                                     'encounters (structural co-missingness)',
                                              'kind': 'mechanical'}]},
    'time_in_hospital': {   'family': 'Utilization (numeric)',
                            'meaning': 'Length of stay in days for this encounter, admission to '
                                       'discharge. A severity/operational proxy: sicker, more '
                                       'complex patients tend to stay longer.',
                            'reads_as': 'Integer days 1-14 (14 is a censoring cap); median 4, '
                                        'mean ~4.4, right-skewed.',
                            'links': [   {   'to': 'num_medications',
                                             'why': 'r=0.47, strongest numeric pair: longer '
                                                    'stays accumulate more distinct meds',
                                             'kind': 'empirical'},
                                         {   'to': 'num_lab_procedures',
                                             'why': 'r=0.32: longer stays allow more lab workup',
                                             'kind': 'empirical'},
                                         {   'to': 'number_diagnoses',
                                             'why': 'r=0.22: more documented comorbidities '
                                                    'co-occur with longer stays',
                                             'kind': 'empirical'},
                                         {   'to': 'num_procedures',
                                             'why': 'r=0.19: more inpatient procedures co-occur '
                                                    'with longer stays',
                                             'kind': 'empirical'}]},
    'num_lab_procedures': {   'family': 'Utilization (numeric)',
                              'meaning': 'Count of lab tests run during this encounter. A '
                                         'workup-intensity/diagnostic-effort proxy; diabetic '
                                         'inpatients almost always get metabolic panels.',
                              'reads_as': 'Integer 1-132, near-symmetric (slight left skew); '
                                          'median 44, mean ~43; long upper tail.',
                              'links': [   {   'to': 'time_in_hospital',
                                               'why': 'r=0.32: longer stays accrue more labs',
                                               'kind': 'empirical'},
                                           {   'to': 'num_medications',
                                               'why': 'r=0.27: more workup co-occurs with more '
                                                      'meds',
                                               'kind': 'empirical'},
                                           {   'to': 'number_diagnoses',
                                               'why': 'r=0.15: more coded diagnoses co-occur '
                                                      'with more tests',
                                               'kind': 'empirical'}]},
    'num_procedures': {   'family': 'Utilization (numeric)',
                          'meaning': 'Count of non-lab (surgical/interventional) procedures this '
                                     'encounter. An intensity/severity proxy; nearly half of '
                                     'stays have none.',
                          'reads_as': 'Integer 0-6 small ordinal; ~46% are 0, median 1, mean '
                                      "~1.3. 0 is a real 'none', not missing.",
                          'links': [   {   'to': 'num_medications',
                                           'why': 'r=0.39: more procedures co-occur with more '
                                                  'meds',
                                           'kind': 'empirical'},
                                       {   'to': 'time_in_hospital',
                                           'why': 'r=0.19: more procedures co-occur with longer '
                                                  'stays',
                                           'kind': 'empirical'}]},
    'num_medications': {   'family': 'Utilization (numeric)',
                           'meaning': 'Count of distinct generic drugs given this encounter. A '
                                      'polypharmacy/treatment-complexity proxy tracking '
                                      'comorbidity burden and care intensity.',
                           'reads_as': 'Integer 1-81; median 15, mean ~16, right-skewed with '
                                       'extreme-polypharmacy outliers.',
                           'links': [   {   'to': 'time_in_hospital',
                                            'why': 'r=0.47, strongest numeric pair: longer stays '
                                                   'carry more meds',
                                            'kind': 'empirical'},
                                        {   'to': 'num_procedures',
                                            'why': 'r=0.39: more procedures co-occur with more '
                                                   'meds',
                                            'kind': 'empirical'},
                                        {   'to': 'number_diagnoses',
                                            'why': 'r=0.26: more diagnoses co-occur with more '
                                                   'drugs',
                                            'kind': 'empirical'}]},
    'number_outpatient': {   'family': 'Utilization (numeric)',
                             'meaning': "Patient's outpatient visits in the prior year, before "
                                        'this admission. A pre-admission utilization-history '
                                        'signal, not a current-stay measure.',
                             'reads_as': 'Integer 0-42; ~84% are 0, mean ~0.37, extremely '
                                         'right-skewed. 0 means none, not missing.',
                             'links': [   {   'to': 'number_inpatient',
                                              'why': 'r=0.11: prior-utilization channels co-move '
                                                     'per patient',
                                              'kind': 'empirical'},
                                          {   'to': 'number_emergency',
                                              'why': 'r=0.09: same pre-admission prior-year '
                                                     'utilization block',
                                              'kind': 'empirical'},
                                          {   'to': 'patient_nbr',
                                              'why': 'prior-visit history is per-person; grouped '
                                                     'split prevents history leakage',
                                              'kind': 'governance'}]},
    'number_emergency': {   'family': 'Utilization (numeric)',
                            'meaning': "Patient's emergency-department visits in the prior year. "
                                       'A pre-admission acute-instability signal: frequent prior '
                                       'ER use flags uncontrolled disease.',
                            'reads_as': 'Integer 0-76; ~89% are 0, mean ~0.2, extremely '
                                        'right-skewed. >=2 prior ER visits roughly doubles '
                                        'readmit rate.',
                            'links': [   {   'to': 'number_inpatient',
                                             'why': 'r=0.27, strongest prior-utilization pair: '
                                                    'ER and admission histories track together',
                                             'kind': 'empirical'},
                                         {   'to': 'readmitted',
                                             'why': 'target-assoc 0.061: prior ER instability '
                                                    'weakly raises 30-day readmit',
                                             'kind': 'empirical'},
                                         {   'to': 'number_outpatient',
                                             'why': 'r=0.09: shared prior-year utilization block',
                                             'kind': 'empirical'},
                                         {   'to': 'patient_nbr',
                                             'why': 'prior-year counts are per-person; grouped '
                                                    'split avoids leakage',
                                             'kind': 'governance'}]},
    'number_inpatient': {   'family': 'Utilization (numeric)',
                            'meaning': "Patient's hospital admissions in the prior year. The "
                                       'strongest single readmission signal here: prior '
                                       'admissions predict future ones.',
                            'reads_as': 'Integer 0-21; ~66% are 0, mean ~0.6, right-skewed. Tail '
                                        'flags chronic high-utilizers; 0 means none.',
                            'links': [   {   'to': 'readmitted',
                                             'why': 'target-assoc 0.165, highest of all columns '
                                                    'in the dataset',
                                             'kind': 'empirical'},
                                         {   'to': 'number_emergency',
                                             'why': 'r=0.27: prior ER and admission histories '
                                                    'co-move',
                                             'kind': 'empirical'},
                                         {   'to': 'number_outpatient',
                                             'why': 'r=0.11: shared prior-year utilization '
                                                    'signal',
                                             'kind': 'empirical'},
                                         {   'to': 'patient_nbr',
                                             'why': 'prior-admission count is per-person; '
                                                    'grouped split prevents history leakage',
                                             'kind': 'governance'}]},
    'number_diagnoses': {   'family': 'Utilization (numeric)',
                            'meaning': 'Number of coded diagnoses on this encounter. A '
                                       'comorbidity-burden/case-complexity proxy; the coding '
                                       'system caps the recorded count at 9.',
                            'reads_as': 'Integer 1-16 but hard-capped at 9 (~49% pile up at 9 = '
                                        "'9 or more'); median 8, mean ~7.4.",
                            'links': [   {   'to': 'diag_1',
                                             'why': "this count tallies the encounter's coded "
                                                    'diagnoses, the diag_1/2/3 slots',
                                             'kind': 'mechanical'},
                                         {   'to': 'num_medications',
                                             'why': 'r=0.26: more diagnoses co-occur with more '
                                                    'drugs',
                                             'kind': 'empirical'},
                                         {   'to': 'age',
                                             'why': 'r=0.24 (vs age_ordinal): comorbidity load '
                                                    'rises with age',
                                             'kind': 'empirical'}]},
    'diag_1': {   'family': 'Diagnosis (ICD-9)',
                  'meaning': 'Primary/principal ICD-9-CM diagnosis: the main clinical reason for '
                             'the hospitalization. The most informative of the three diagnosis '
                             'slots.',
                  'reads_as': 'ICD-9 code string, 716 levels; top: 428=CHF, 414=ischemic heart, '
                              '410=acute MI, 250.x=diabetes.',
                  'links': [   {   'to': 'readmitted',
                                   'why': "Cramer's V 0.066, lift 1.698; strongest diag "
                                          'separator of the target',
                                   'kind': 'empirical'},
                               {   'to': 'number_diagnoses',
                                   'why': 'the principal diagnosis is one of the codes tallied '
                                          'into number_diagnoses',
                                   'kind': 'mechanical'},
                               {   'to': 'medical_specialty',
                                   'why': 'principal diagnosis drives the attending department '
                                          '(Nephrology->renal, Oncology->cancer)',
                                   'kind': 'clinical'},
                               {   'to': 'diag_2',
                                   'why': 'same encounter; principal vs comorbid coding, partly '
                                          'overlapping condition profile',
                                   'kind': 'clinical'},
                               {   'to': 'diag_3',
                                   'why': "together the three slots describe the encounter's "
                                          'full diagnosis profile',
                                   'kind': 'clinical'}]},
    'diag_2': {   'family': 'Diagnosis (ICD-9)',
                  'meaning': 'Secondary ICD-9-CM diagnosis: a comorbid condition beyond the '
                             'principal one, adding disease-burden signal.',
                  'reads_as': 'ICD-9 code string, 748 levels; chronic comorbidities dominate: '
                              '276=electrolyte, 428=CHF, 250=diabetes, 427=dysrhythmia.',
                  'links': [   {   'to': 'readmitted',
                                   'why': "Cramer's V 0.058, lift 1.452; weakest diag separator "
                                          'but still meaningful',
                                   'kind': 'empirical'},
                               {   'to': 'number_diagnoses',
                                   'why': 'counted into number_diagnoses, the comorbidity-burden '
                                          'tally',
                                   'kind': 'mechanical'},
                               {   'to': 'diag_1',
                                   'why': 'comorbidity coded alongside the principal diagnosis; '
                                          'correlated condition profile',
                                   'kind': 'clinical'},
                               {   'to': 'diag_3',
                                   'why': 'both are comorbidity slots with overlapping '
                                          'renal/diabetic codes',
                                   'kind': 'clinical'}]},
    'diag_3': {   'family': 'Diagnosis (ICD-9)',
                  'meaning': 'Third ICD-9-CM diagnosis slot: further comorbidity load. The most '
                             'often empty of the three.',
                  'reads_as': 'ICD-9 code string, 789 levels; most missing (1.398%); '
                              '250=diabetes is the largest level.',
                  'links': [   {   'to': 'readmitted',
                                   'why': "Cramer's V 0.070, highest of the three; lift 1.593, "
                                          'renal/diabetic codes drive risk',
                                   'kind': 'empirical'},
                               {   'to': 'number_diagnoses',
                                   'why': 'contributes to the number_diagnoses count of coded '
                                          'diagnoses',
                                   'kind': 'mechanical'},
                               {   'to': 'diag_2',
                                   'why': 'both comorbidity slots; chronic renal/diabetic codes '
                                          'dominate both',
                                   'kind': 'clinical'},
                               {   'to': 'diag_1',
                                   'why': 'completes the encounter diagnosis profile alongside '
                                          'the principal diagnosis',
                                   'kind': 'clinical'}]},
    'max_glu_serum': {   'family': 'Lab result',
                         'meaning': 'In-stay blood glucose (serum) lab result, banded into '
                                    'clinical ranges. Recorded to gauge acute glycemic state '
                                    'during the encounter.',
                         'reads_as': 'Norm / >200 / >300 mg/dL; NaN (94.7%) = test not ordered, '
                                     'NOT normal glucose.',
                         'links': [   {   'to': 'A1Cresult',
                                          'why': 'paired glycemic labs sharing the '
                                                 "'not-ordered=NaN' mechanism; point opposite "
                                                 'ways on risk',
                                          'kind': 'mechanical'},
                                      {   'to': 'insulin',
                                          'why': 'an elevated glucose reading prompts an in-stay '
                                                 'insulin dose adjustment',
                                          'kind': 'clinical'},
                                      {   'to': 'readmitted',
                                          'why': "weak (assoc 0.011): 'test ordered' = +1.3pp "
                                                 'readmit, with a monotonic dose-response',
                                          'kind': 'empirical'},
                                      {   'to': 'diag_1',
                                          'why': 'a diabetes primary diagnosis is what prompts '
                                                 'ordering this glucose test',
                                          'kind': 'clinical'}]},
    'A1Cresult': {   'family': 'Lab result',
                     'meaning': 'In-stay HbA1c lab result (~3-month average glycemic control), '
                                'banded. Recorded to judge longer-term diabetes control near '
                                'discharge.',
                     'reads_as': 'Norm / >7 / >8 (% control); NaN (83.3%) = A1c not ordered, NOT '
                                 'normal control.',
                     'links': [   {   'to': 'max_glu_serum',
                                      'why': 'both glycemic labs; A1c is long-term control vs '
                                             "glucose's point-in-time, same NaN mechanism",
                                      'kind': 'mechanical'},
                                  {   'to': 'insulin',
                                      'why': 'a poorly-controlled A1c (>8) drives starting or '
                                             'up-titrating insulin',
                                      'kind': 'clinical'},
                                  {   'to': 'readmitted',
                                      'why': "weak (assoc 0.018): 'A1c checked' = -1.6pp "
                                             'readmit, opposite sign to glucose',
                                      'kind': 'empirical'},
                                  {   'to': 'diabetesMed',
                                      'why': 'ordering A1c co-occurs with active pharmacologic '
                                             'diabetes management',
                                      'kind': 'clinical'}]},
    'metformin': {   'family': 'Medication (dose change)',
                     'meaning': 'First-line oral biguanide: whether given this stay and whether '
                                'its dose changed. The highest-coverage drug in the family.',
                     'reads_as': "No 80% / Steady 18% / Up 1% / Down 0.6%. 'No' = not on the "
                                 'drug.',
                     'links': [   {   'to': 'diabetesMed',
                                      'why': "any non-No here helps set diabetesMed='Yes', the "
                                             'any-drug roll-up',
                                      'kind': 'mechanical'},
                                  {   'to': 'change',
                                      'why': 'an Up/Down here is one of the events that sets '
                                             "change='Ch'",
                                      'kind': 'mechanical'},
                                  {   'to': 'insulin',
                                      'why': 'oral agent often co-prescribed with an insulin '
                                             'regimen',
                                      'kind': 'clinical'},
                                  {   'to': 'readmitted',
                                      'why': "weak assoc Cramer's V=0.023, strongest drug after "
                                             'insulin',
                                      'kind': 'empirical'}]},
    'repaglinide': {   'family': 'Medication (dose change)',
                       'meaning': 'Meglitinide (short-acting insulin secretagogue): given this '
                                  'stay and dose-change direction. A rare second-line oral '
                                  'agent.',
                       'reads_as': 'No 98.5% / Steady 1.4% / tiny Up,Down. Non-No is only ~1.5% '
                                   'of rows.',
                       'links': [   {   'to': 'diabetesMed',
                                        'why': 'non-No state contributes to the any-diabetic-med '
                                               'roll-up flag',
                                        'kind': 'mechanical'},
                                    {   'to': 'change',
                                        'why': 'rare Up/Down contributes to the '
                                               'medication-change indicator',
                                        'kind': 'mechanical'},
                                    {   'to': 'nateglinide',
                                        'why': 'same meglitinide class; both rare secretagogues '
                                               'here',
                                        'kind': 'clinical'}]},
    'nateglinide': {   'family': 'Medication (dose change)',
                       'meaning': 'Meglitinide (rapid insulin secretagogue): given this stay and '
                                  'dose-change direction. Very rarely prescribed in this cohort.',
                       'reads_as': 'No 99.3% / Steady 0.7% / negligible Up,Down. A thin '
                                   'Steady-vs-No flag.',
                       'links': [   {   'to': 'diabetesMed',
                                        'why': 'non-No state feeds the any-diabetic-med roll-up',
                                        'kind': 'mechanical'},
                                    {   'to': 'change',
                                        'why': 'dose moves feed the change indicator; here '
                                               'near-zero',
                                        'kind': 'mechanical'},
                                    {   'to': 'repaglinide',
                                        'why': 'same meglitinide secretagogue class, both rare '
                                               'here',
                                        'kind': 'clinical'}]},
    'chlorpropamide': {   'family': 'Medication (dose change)',
                          'meaning': 'First-generation sulfonylurea (older, rarely used): given '
                                     'this stay and dose direction. Near-constant here.',
                          'reads_as': 'No 99.9% / 86 non-No rows total. Effectively constant; '
                                      'Up,Down are noise.',
                          'links': [   {   'to': 'diabetesMed',
                                           'why': 'any non-No feeds the diabetic-med roll-up, '
                                                  'but only ~86 rows',
                                           'kind': 'mechanical'},
                                       {   'to': 'glimepiride',
                                           'why': 'same sulfonylurea class; this obsolete agent '
                                                  'rarely used vs newer ones',
                                           'kind': 'clinical'}]},
    'glimepiride': {   'family': 'Medication (dose change)',
                       'meaning': 'Third-generation sulfonylurea: given this stay and '
                                  'dose-change direction. A moderately common oral agent with '
                                  'usable coverage.',
                       'reads_as': 'No 95% / Steady 4.6% / Up 0.3% / Down 0.2%. Steady is '
                                   'modelable.',
                       'links': [   {   'to': 'diabetesMed',
                                        'why': 'non-No state feeds the any-diabetic-med roll-up',
                                        'kind': 'mechanical'},
                                    {   'to': 'change',
                                        'why': "Up/Down here set the change='Ch' indicator",
                                        'kind': 'mechanical'},
                                    {   'to': 'glipizide',
                                        'why': 'same sulfonylurea class; usually one '
                                               'sulfonylurea per patient',
                                        'kind': 'clinical'}]},
    'acetohexamide': {   'family': 'Medication (dose change)',
                         'meaning': 'Obsolete first-generation sulfonylurea. Only No/Steady '
                                    'appear; the most degenerate column in the family. Drop '
                                    'before modelling.',
                         'reads_as': 'No 99.999% / Steady = exactly 1 row. Constant for '
                                     'practical purposes.',
                         'links': [   {   'to': 'diabetesMed',
                                          'why': 'the single Steady row would feed the '
                                                 'diabetic-med roll-up',
                                          'kind': 'mechanical'},
                                      {   'to': 'glimepiride',
                                          'why': 'same sulfonylurea class, superseded by newer '
                                                 'agents',
                                          'kind': 'clinical'}]},
    'glipizide': {   'family': 'Medication (dose change)',
                     'meaning': 'Second-generation sulfonylurea: given this stay and dose '
                                'direction. Second-best coverage after metformin.',
                     'reads_as': 'No 87.5% / Steady 11.2% / Up 0.8% / Down 0.6%. Down (n=560) is '
                                 'a credible cell.',
                     'links': [   {   'to': 'diabetesMed',
                                      'why': 'non-No state feeds the any-diabetic-med roll-up',
                                      'kind': 'mechanical'},
                                  {   'to': 'change',
                                      'why': 'Up/Down here set the change indicator',
                                      'kind': 'mechanical'},
                                  {   'to': 'glyburide',
                                      'why': 'same sulfonylurea class; opposite dose-Down '
                                             'readmit gradient',
                                      'kind': 'clinical'},
                                  {   'to': 'readmitted',
                                      'why': 'weak assoc V=0.009; dose-Down cell CI excludes '
                                             'base, flags higher risk',
                                      'kind': 'empirical'}]},
    'glyburide': {   'family': 'Medication (dose change)',
                     'meaning': 'Second-generation sulfonylurea: given this stay and dose '
                                'direction. Coverage comparable to glipizide; signal nearly '
                                'flat.',
                     'reads_as': 'No 89.5% / Steady 9.1% / Up 0.8% / Down 0.6%.',
                     'links': [   {   'to': 'diabetesMed',
                                      'why': 'non-No state feeds the any-diabetic-med roll-up',
                                      'kind': 'mechanical'},
                                  {   'to': 'change',
                                      'why': 'Up/Down here set the change indicator',
                                      'kind': 'mechanical'},
                                  {   'to': 'glipizide',
                                      'why': 'same sulfonylurea class; one usually substitutes '
                                             'for the other',
                                      'kind': 'clinical'},
                                  {   'to': 'glyburide-metformin',
                                      'why': 'glyburide is the sulfonylurea component of that '
                                             'fixed-dose combo',
                                      'kind': 'clinical'}]},
    'tolbutamide': {   'family': 'Medication (dose change)',
                       'meaning': 'Obsolete first-generation sulfonylurea. Only No/Steady '
                                  'appear, so a binary flag. Near-constant; drop.',
                       'reads_as': 'No 99.98% / Steady = 23 rows. De-facto constant for '
                                   'modelling.',
                       'links': [   {   'to': 'diabetesMed',
                                        'why': 'any Steady row feeds the diabetic-med roll-up; '
                                               'only 23 rows',
                                        'kind': 'mechanical'},
                                    {   'to': 'chlorpropamide',
                                        'why': 'both obsolete first-gen sulfonylureas, '
                                               'near-constant here',
                                        'kind': 'clinical'}]},
    'pioglitazone': {   'family': 'Medication (dose change)',
                        'meaning': 'Thiazolidinedione (insulin sensitizer): given this stay and '
                                   'dose direction. Moderately common; Steady dominates the '
                                   'non-No mass.',
                        'reads_as': 'No 93% / Steady 6.9% / Up 0.2% / Down 0.1%. Direction '
                                    'signal not reliable.',
                        'links': [   {   'to': 'diabetesMed',
                                         'why': 'non-No state feeds the any-diabetic-med roll-up',
                                         'kind': 'mechanical'},
                                     {   'to': 'change',
                                         'why': 'Up/Down here set the change indicator',
                                         'kind': 'mechanical'},
                                     {   'to': 'rosiglitazone',
                                         'why': 'same thiazolidinedione class; one TZD usually '
                                                'substitutes for the other',
                                         'kind': 'clinical'}]},
    'rosiglitazone': {   'family': 'Medication (dose change)',
                         'meaning': 'Thiazolidinedione (insulin sensitizer): given this stay and '
                                    'dose direction. Usable coverage but flat signal.',
                         'reads_as': 'No 93.7% / Steady 6% / tiny Up,Down. Steady dominates '
                                     'non-No mass.',
                         'links': [   {   'to': 'diabetesMed',
                                          'why': 'non-No state feeds the any-diabetic-med '
                                                 'roll-up',
                                          'kind': 'mechanical'},
                                      {   'to': 'change',
                                          'why': 'Up/Down here set the change indicator',
                                          'kind': 'mechanical'},
                                      {   'to': 'pioglitazone',
                                          'why': 'same thiazolidinedione class; typically not '
                                                 'co-prescribed',
                                          'kind': 'clinical'},
                                      {   'to': 'metformin-rosiglitazone',
                                          'why': 'rosiglitazone is the TZD component of that '
                                                 'fixed-dose combo',
                                          'kind': 'clinical'}]},
    'acarbose': {   'family': 'Medication (dose change)',
                    'meaning': 'Alpha-glucosidase inhibitor: given this stay and dose direction. '
                               'Uncommon agent, near-constant; drop.',
                    'reads_as': 'No 99.7% / Steady 0.3% / negligible Up,Down. Direction is noise '
                                'only.',
                    'links': [   {   'to': 'diabetesMed',
                                     'why': 'any non-No feeds the diabetic-med roll-up; ~308 '
                                            'rows',
                                     'kind': 'mechanical'},
                                 {   'to': 'miglitol',
                                     'why': 'same alpha-glucosidase-inhibitor class, both rare '
                                            'here',
                                     'kind': 'clinical'}]},
    'miglitol': {   'family': 'Medication (dose change)',
                    'meaning': 'Alpha-glucosidase inhibitor: given this stay and dose direction. '
                               'Near-constant in this cohort; drop.',
                    'reads_as': 'No 99.96% / 38 non-No rows. Non-No per-level rates are pure '
                                'noise.',
                    'links': [   {   'to': 'diabetesMed',
                                     'why': 'any non-No feeds the diabetic-med roll-up; only ~38 '
                                            'rows',
                                     'kind': 'mechanical'},
                                 {   'to': 'acarbose',
                                     'why': 'same alpha-glucosidase-inhibitor class, both '
                                            'near-constant',
                                     'kind': 'clinical'}]},
    'troglitazone': {   'family': 'Medication (dose change)',
                        'meaning': 'Thiazolidinedione withdrawn from the US market in 2000 for '
                                   'liver toxicity; fits the 1999-2008 window. Essentially '
                                   'constant; drop.',
                        'reads_as': 'No 99.997% / Steady = 3 rows. No Up/Down levels exist.',
                        'links': [   {   'to': 'diabetesMed',
                                         'why': 'the 3 Steady rows would feed the diabetic-med '
                                                'roll-up',
                                         'kind': 'mechanical'},
                                     {   'to': 'pioglitazone',
                                         'why': 'same thiazolidinedione class; troglitazone its '
                                                'withdrawn predecessor',
                                         'kind': 'clinical'}]},
    'tolazamide': {   'family': 'Medication (dose change)',
                      'meaning': 'First-generation sulfonylurea: given this stay and dose '
                                 'direction (No/Steady/Up only). Near-constant; drop.',
                      'reads_as': "No 99.96% / 39 non-No rows. No 'Down' observed; rates are "
                                  'noise.',
                      'links': [   {   'to': 'diabetesMed',
                                       'why': 'any non-No feeds the diabetic-med roll-up; only '
                                              '~39 rows',
                                       'kind': 'mechanical'},
                                   {   'to': 'glimepiride',
                                       'why': 'same sulfonylurea class, superseded by newer '
                                              'agents',
                                       'kind': 'clinical'}]},
    'examide': {   'family': 'Medication (dose change)',
                   'meaning': 'Antidiabetic drug in the schema but every encounter is coded '
                              "'No'; carries no information. True zero variance; drop.",
                   'reads_as': 'No 100%. Single level, n_unique=1, is_constant=true.',
                   'links': [   {   'to': 'citoglipton',
                                    'why': 'both true zero-variance columns, identical '
                                           'situation, drop both',
                                    'kind': 'mechanical'},
                                {   'to': 'diabetesMed',
                                    'why': 'always-No means it never affects the diabetic-med '
                                           'roll-up',
                                    'kind': 'mechanical'}]},
    'citoglipton': {   'family': 'Medication (dose change)',
                       'meaning': 'Antidiabetic drug in the schema but every encounter is coded '
                                  "'No'; conveys no information. True zero variance; drop.",
                       'reads_as': 'No 100%. Single level, n_unique=1, is_constant=true.',
                       'links': [   {   'to': 'examide',
                                        'why': 'both true zero-variance columns, identical '
                                               'situation, drop both',
                                        'kind': 'mechanical'},
                                    {   'to': 'diabetesMed',
                                        'why': 'always-No means it never affects the '
                                               'diabetic-med roll-up',
                                        'kind': 'mechanical'}]},
    'insulin': {   'family': 'Medication (dose change)',
                   'meaning': 'Insulin therapy status this stay; the most clinically central '
                              'diabetes drug. Up/Down reflect active glycemic adjustments by the '
                              'team.',
                   'reads_as': 'No 47% / Steady 30% / Down 12% / Up 11%. The only well-populated '
                               'drug here.',
                   'links': [   {   'to': 'diabetesMed',
                                    'why': 'non-No state strongly drives the any-diabetic-med '
                                           'roll-up',
                                    'kind': 'mechanical'},
                                {   'to': 'change',
                                    'why': 'insulin Up/Down is a major contributor to '
                                           "change='Ch'",
                                    'kind': 'mechanical'},
                                {   'to': 'A1Cresult',
                                    'why': 'poor A1c clinically prompts an insulin start or dose '
                                           'increase',
                                    'kind': 'clinical'},
                                {   'to': 'max_glu_serum',
                                    'why': 'elevated inpatient glucose prompts insulin dose '
                                           'adjustment',
                                    'kind': 'clinical'},
                                {   'to': 'readmitted',
                                    'why': 'assoc V=0.043, strongest drug; rate rises '
                                           'No<Steady<Up<Down',
                                    'kind': 'empirical'}]},
    'glyburide-metformin': {   'family': 'Medication (dose change)',
                               'meaning': 'Fixed-dose combination of glyburide plus metformin: '
                                          'given this stay and dose direction. Near-constant; '
                                          'drop.',
                               'reads_as': 'No 99.3% / Steady 0.7% / tiny Up,Down. '
                                           'Best-populated combo, still <0.7%.',
                               'links': [   {   'to': 'diabetesMed',
                                                'why': 'non-No state feeds the any-diabetic-med '
                                                       'roll-up',
                                                'kind': 'mechanical'},
                                            {   'to': 'glyburide',
                                                'why': "this combo's sulfonylurea component is "
                                                       'the glyburide drug',
                                                'kind': 'clinical'},
                                            {   'to': 'metformin',
                                                'why': "this combo's biguanide component is the "
                                                       'metformin drug',
                                                'kind': 'clinical'}]},
    'glipizide-metformin': {   'family': 'Medication (dose change)',
                               'meaning': 'Fixed-dose combination of glipizide plus metformin: '
                                          'given this stay (No/Steady only). Effectively '
                                          'constant; drop.',
                               'reads_as': 'No 99.99% / Steady = 13 rows. No Up/Down; near-zero '
                                           'variance.',
                               'links': [   {   'to': 'diabetesMed',
                                                'why': 'the 13 Steady rows would feed the '
                                                       'diabetic-med roll-up',
                                                'kind': 'mechanical'},
                                            {   'to': 'glipizide',
                                                'why': "this combo's sulfonylurea component is "
                                                       'the glipizide drug',
                                                'kind': 'clinical'},
                                            {   'to': 'metformin',
                                                'why': "this combo's biguanide component is the "
                                                       'metformin drug',
                                                'kind': 'clinical'}]},
    'glimepiride-pioglitazone': {   'family': 'Medication (dose change)',
                                    'meaning': 'Fixed-dose combination of glimepiride plus '
                                               'pioglitazone. Virtually constant (one row); '
                                               'drop.',
                                    'reads_as': 'No 99.999% / Steady = 1 row. A single encounter '
                                                'records it.',
                                    'links': [   {   'to': 'glimepiride',
                                                     'why': "this combo's sulfonylurea component "
                                                            'is the glimepiride drug',
                                                     'kind': 'clinical'},
                                                 {   'to': 'pioglitazone',
                                                     'why': "this combo's TZD component is the "
                                                            'pioglitazone drug',
                                                     'kind': 'clinical'},
                                                 {   'to': 'metformin-pioglitazone',
                                                     'why': 'both single-row TZD combos, '
                                                            'near-constant, drop both',
                                                     'kind': 'mechanical'}]},
    'metformin-rosiglitazone': {   'family': 'Medication (dose change)',
                                   'meaning': 'Fixed-dose combination of metformin plus '
                                              'rosiglitazone. Virtually constant (two rows); '
                                              'drop.',
                                   'reads_as': 'No 99.998% / Steady = 2 rows. Effectively '
                                               'constant.',
                                   'links': [   {   'to': 'metformin',
                                                    'why': "this combo's biguanide component is "
                                                           'the metformin drug',
                                                    'kind': 'clinical'},
                                                {   'to': 'rosiglitazone',
                                                    'why': "this combo's TZD component is the "
                                                           'rosiglitazone drug',
                                                    'kind': 'clinical'},
                                                {   'to': 'metformin-pioglitazone',
                                                    'why': 'both near-constant metformin+TZD '
                                                           'combos, drop both',
                                                    'kind': 'mechanical'}]},
    'metformin-pioglitazone': {   'family': 'Medication (dose change)',
                                  'meaning': 'Fixed-dose combination of metformin plus '
                                             'pioglitazone. Virtually constant (one row); drop.',
                                  'reads_as': 'No 99.999% / Steady = 1 row. A single encounter '
                                              'records it.',
                                  'links': [   {   'to': 'metformin',
                                                   'why': "this combo's biguanide component is "
                                                          'the metformin drug',
                                                   'kind': 'clinical'},
                                               {   'to': 'pioglitazone',
                                                   'why': "this combo's TZD component is the "
                                                          'pioglitazone drug',
                                                   'kind': 'clinical'},
                                               {   'to': 'glimepiride-pioglitazone',
                                                   'why': 'both single-row TZD combos, '
                                                          'mirror-image, drop both',
                                                   'kind': 'mechanical'}]},
    'change': {   'family': 'Indicator',
                  'meaning': "Binary flag: was the patient's antidiabetic regimen changed during "
                             'this stay? An encounter-level roll-up over the 23 per-drug columns '
                             '(any add/stop/dose-change).',
                  'reads_as': "Two levels: 'Ch' = at least one drug changed "
                              "(Up/Down/start/stop), 'No' = unchanged. ~46% Ch.",
                  'links': [   {   'to': 'diabetesMed',
                                   'why': 'Same 23-drug roll-up; a change needs being on a drug, '
                                          "so Ch implies diabetesMed='Yes'",
                                   'kind': 'mechanical'},
                               {   'to': 'insulin',
                                   'why': 'Only densely-populated dose-change drug; its Up/Down '
                                          "adjustments are the main driver of change='Ch'",
                                   'kind': 'mechanical'},
                               {   'to': 'metformin',
                                   'why': 'Highest-coverage oral drug; its Up/Down dose '
                                          'adjustments feed the change roll-up',
                                   'kind': 'mechanical'},
                               {   'to': 'A1Cresult',
                                   'why': 'An elevated measured A1c (>7/>8) can prompt the team '
                                          'to adjust the regimen this stay',
                                   'kind': 'clinical'},
                               {   'to': 'readmitted',
                                   'why': 'Weak: readmit 0.118 (Ch) vs 0.106 (No), assoc 0.019; '
                                          'tweaked regimen flags marginally higher risk',
                                   'kind': 'empirical'}]},
    'diabetesMed': {   'family': 'Indicator',
                       'meaning': 'Binary flag: was the patient on any antidiabetic medication '
                                  "this encounter? An encounter-level roll-up: 'Yes' iff any of "
                                  "the 23 drug columns != 'No'.",
                       'reads_as': "Two levels: 'Yes' = on >=1 antidiabetic drug, 'No' = none. "
                                   '~77% Yes.',
                       'links': [   {   'to': 'change',
                                        'why': "Same 23-drug roll-up; change='Ch' can only occur "
                                               "when diabetesMed='Yes'",
                                        'kind': 'mechanical'},
                                    {   'to': 'insulin',
                                        'why': "insulin != 'No' forces diabetesMed='Yes'; it is "
                                               'the densest contributor to the roll-up',
                                        'kind': 'mechanical'},
                                    {   'to': 'metformin',
                                        'why': "metformin != 'No' (~20% of rows) is a major "
                                               "driver of diabetesMed='Yes'",
                                        'kind': 'mechanical'},
                                    {   'to': 'readmitted',
                                        'why': 'Real but weak: readmit 0.116 (Yes) vs 0.096 '
                                               '(No), assoc 0.027; treated patients readmit '
                                               'slightly more',
                                        'kind': 'empirical'}]}}

CLUSTERS = [   {   'id': 'patient-identity',
        'label': 'Patient Identity & Keys',
        'members': ['encounter_id', 'patient_nbr'],
        'summary': 'Administrative keys: encounter_id is the per-row primary key; patient_nbr '
                   'groups encounters per person and governs the leakage-safe grouped split.'},
    {   'id': 'demographics',
        'label': 'Demographics',
        'members': ['race', 'gender', 'age', 'weight'],
        'summary': 'Registration-time patient attributes; race/gender/age are protected fairness '
                   'slices, age is also clinical, weight is ~97% missing and droppable.'},
    {   'id': 'encounter-context',
        'label': 'Encounter Context',
        'members': [   'admission_type_id',
                       'discharge_disposition_id',
                       'admission_source_id',
                       'payer_code',
                       'medical_specialty'],
        'summary': 'Pre-discharge administrative context: intake acuity/source, discharge '
                   'destination (drives expired/hospice filter), insurer, and admitting '
                   'specialty proxying clinical reason.'},
    {   'id': 'utilization-intensity',
        'label': 'Utilization Intensity',
        'members': [   'time_in_hospital',
                       'num_lab_procedures',
                       'num_procedures',
                       'num_medications',
                       'number_outpatient',
                       'number_emergency',
                       'number_inpatient',
                       'number_diagnoses'],
        'summary': 'Numeric counts of current-stay workup and prior-year visits; intercorrelated '
                   'severity proxies, with prior inpatient/ER visits the strongest readmission '
                   'signals.'},
    {   'id': 'diagnoses',
        'label': 'Diagnoses (ICD-9)',
        'members': ['diag_1', 'diag_2', 'diag_3'],
        'summary': 'Three ICD-9 diagnosis slots (principal plus two comorbidities) describing '
                   "the encounter's clinical profile; tallied into number_diagnoses, modest "
                   'target signal.'},
    {   'id': 'glucose-management',
        'label': 'Glucose Management',
        'members': [   'max_glu_serum',
                       'A1Cresult',
                       'metformin',
                       'repaglinide',
                       'nateglinide',
                       'chlorpropamide',
                       'glimepiride',
                       'acetohexamide',
                       'glipizide',
                       'glyburide',
                       'tolbutamide',
                       'pioglitazone',
                       'rosiglitazone',
                       'acarbose',
                       'miglitol',
                       'troglitazone',
                       'tolazamide',
                       'examide',
                       'citoglipton',
                       'insulin',
                       'glyburide-metformin',
                       'glipizide-metformin',
                       'glimepiride-pioglitazone',
                       'metformin-rosiglitazone',
                       'metformin-pioglitazone',
                       'change',
                       'diabetesMed'],
        'summary': 'Glycemic labs, the 23 antidiabetic drug dose-change columns, and the '
                   'change/diabetesMed roll-ups; the clinical treatment block, with insulin '
                   'carrying most signal.'},
    {   'id': 'target',
        'label': 'Target',
        'members': ['readmitted'],
        'summary': 'The 30-day readmission outcome (collapsed to binary); the label every other '
                   'column is evaluated against, never a feature itself.'}]

GRAPH_NODES = [   {'id': 'patient_nbr', 'label': 'Patient ID', 'family': 'Identifier', 'kind': 'column'},
    {'id': 'encounter_id', 'label': 'Encounter ID', 'family': 'Identifier', 'kind': 'column'},
    {'id': 'race', 'label': 'Race', 'family': 'Demographic', 'kind': 'column'},
    {'id': 'gender', 'label': 'Gender', 'family': 'Demographic', 'kind': 'column'},
    {'id': 'age', 'label': 'Age', 'family': 'Demographic', 'kind': 'column'},
    {'id': 'weight', 'label': 'Weight', 'family': 'Demographic', 'kind': 'column'},
    {   'id': 'admission_type_id',
        'label': 'Admission Type',
        'family': 'Administrative ID (coded)',
        'kind': 'column'},
    {   'id': 'discharge_disposition_id',
        'label': 'Discharge Disposition',
        'family': 'Administrative ID (coded)',
        'kind': 'column'},
    {   'id': 'admission_source_id',
        'label': 'Admission Source',
        'family': 'Administrative ID (coded)',
        'kind': 'column'},
    {   'id': 'payer_code',
        'label': 'Payer Code',
        'family': 'Administrative text',
        'kind': 'column'},
    {   'id': 'medical_specialty',
        'label': 'Medical Specialty',
        'family': 'Administrative text',
        'kind': 'column'},
    {   'id': 'time_in_hospital',
        'label': 'Length of Stay',
        'family': 'Utilization (numeric)',
        'kind': 'column'},
    {   'id': 'num_lab_procedures',
        'label': 'Lab Procedures',
        'family': 'Utilization (numeric)',
        'kind': 'column'},
    {   'id': 'num_procedures',
        'label': 'Procedures',
        'family': 'Utilization (numeric)',
        'kind': 'column'},
    {   'id': 'num_medications',
        'label': 'Medication Count',
        'family': 'Utilization (numeric)',
        'kind': 'column'},
    {   'id': 'number_outpatient',
        'label': 'Prior Outpatient',
        'family': 'Utilization (numeric)',
        'kind': 'column'},
    {   'id': 'number_emergency',
        'label': 'Prior Emergency',
        'family': 'Utilization (numeric)',
        'kind': 'column'},
    {   'id': 'number_inpatient',
        'label': 'Prior Inpatient',
        'family': 'Utilization (numeric)',
        'kind': 'column'},
    {   'id': 'number_diagnoses',
        'label': 'Diagnosis Count',
        'family': 'Utilization (numeric)',
        'kind': 'column'},
    {   'id': 'diag_1',
        'label': 'Diagnosis 1 (primary)',
        'family': 'Diagnosis (ICD-9)',
        'kind': 'column'},
    {'id': 'diag_2', 'label': 'Diagnosis 2', 'family': 'Diagnosis (ICD-9)', 'kind': 'column'},
    {'id': 'diag_3', 'label': 'Diagnosis 3', 'family': 'Diagnosis (ICD-9)', 'kind': 'column'},
    {'id': 'max_glu_serum', 'label': 'Glucose Serum', 'family': 'Lab result', 'kind': 'column'},
    {'id': 'A1Cresult', 'label': 'A1C Result', 'family': 'Lab result', 'kind': 'column'},
    {'id': 'insulin', 'label': 'Insulin', 'family': 'Medication (dose change)', 'kind': 'column'},
    {   'id': 'medications',
        'label': 'Medications (23 drugs)',
        'family': 'Medication (dose change)',
        'kind': 'group'},
    {'id': 'change', 'label': 'Med Change Flag', 'family': 'Indicator', 'kind': 'column'},
    {'id': 'diabetesMed', 'label': 'Diabetes Med Flag', 'family': 'Indicator', 'kind': 'column'},
    {'id': 'readmitted', 'label': 'Readmitted <30d', 'family': 'Target', 'kind': 'column'}]

GRAPH_EDGES = [   {   'source': 'patient_nbr',
        'target': 'encounter_id',
        'label': 'groups encounters per patient',
        'strength': 'conceptual'},
    {   'source': 'patient_nbr',
        'target': 'readmitted',
        'label': 'grouped split prevents leakage',
        'strength': 'conceptual'},
    {   'source': 'patient_nbr',
        'target': 'number_inpatient',
        'label': 'keys prior-visit history',
        'strength': 'conceptual'},
    {   'source': 'encounter_id',
        'target': 'readmitted',
        'label': 'never a feature; memorises rows',
        'strength': 'conceptual'},
    {   'source': 'time_in_hospital',
        'target': 'num_medications',
        'label': 'longer stay, more meds',
        'strength': 'strong'},
    {   'source': 'num_procedures',
        'target': 'num_medications',
        'label': 'more procedures, more meds',
        'strength': 'moderate'},
    {   'source': 'time_in_hospital',
        'target': 'num_lab_procedures',
        'label': 'longer stay, more labs',
        'strength': 'moderate'},
    {   'source': 'num_lab_procedures',
        'target': 'num_medications',
        'label': 'more workup, more meds',
        'strength': 'moderate'},
    {   'source': 'number_emergency',
        'target': 'number_inpatient',
        'label': 'prior-utilization histories co-move',
        'strength': 'moderate'},
    {   'source': 'num_medications',
        'target': 'number_diagnoses',
        'label': 'more diagnoses, more drugs',
        'strength': 'moderate'},
    {   'source': 'number_diagnoses',
        'target': 'age',
        'label': 'comorbidity load rises with age',
        'strength': 'moderate'},
    {   'source': 'time_in_hospital',
        'target': 'number_diagnoses',
        'label': 'longer stay, more diagnoses',
        'strength': 'moderate'},
    {   'source': 'time_in_hospital',
        'target': 'num_procedures',
        'label': 'longer stay, more procedures',
        'strength': 'weak'},
    {   'source': 'number_outpatient',
        'target': 'number_inpatient',
        'label': 'shared prior-year utilization',
        'strength': 'weak'},
    {   'source': 'number_inpatient',
        'target': 'readmitted',
        'label': 'strongest single readmit signal',
        'strength': 'weak'},
    {   'source': 'number_emergency',
        'target': 'readmitted',
        'label': 'prior ER instability raises risk',
        'strength': 'weak'},
    {   'source': 'number_diagnoses',
        'target': 'diag_1',
        'label': 'tallies the diagnosis slots',
        'strength': 'conceptual'},
    {   'source': 'number_diagnoses',
        'target': 'diag_2',
        'label': 'tallies the diagnosis slots',
        'strength': 'conceptual'},
    {   'source': 'number_diagnoses',
        'target': 'diag_3',
        'label': 'tallies the diagnosis slots',
        'strength': 'conceptual'},
    {   'source': 'diag_1',
        'target': 'medical_specialty',
        'label': 'diagnosis drives attending department',
        'strength': 'conceptual'},
    {   'source': 'diag_1',
        'target': 'readmitted',
        'label': 'strongest diagnosis separator',
        'strength': 'weak'},
    {   'source': 'discharge_disposition_id',
        'target': 'readmitted',
        'label': 'strongest admin correlate; expired/hospice filter',
        'strength': 'weak'},
    {   'source': 'discharge_disposition_id',
        'target': 'number_inpatient',
        'label': 'SNF/rehab dispo flags frailty',
        'strength': 'conceptual'},
    {   'source': 'A1Cresult',
        'target': 'insulin',
        'label': 'poor A1c up-titrates insulin',
        'strength': 'conceptual'},
    {   'source': 'max_glu_serum',
        'target': 'insulin',
        'label': 'high glucose prompts insulin dose',
        'strength': 'conceptual'},
    {   'source': 'A1Cresult',
        'target': 'max_glu_serum',
        'label': 'paired glycemic labs, same NaN',
        'strength': 'conceptual'},
    {   'source': 'insulin',
        'target': 'change',
        'label': 'insulin Up/Down drives change',
        'strength': 'conceptual'},
    {   'source': 'insulin',
        'target': 'diabetesMed',
        'label': 'insulin forces diabetesMed=Yes',
        'strength': 'conceptual'},
    {   'source': 'change',
        'target': 'diabetesMed',
        'label': 'same 23-drug roll-up',
        'strength': 'conceptual'},
    {   'source': 'medications',
        'target': 'diabetesMed',
        'label': 'any drug feeds roll-up',
        'strength': 'conceptual'}]


def cluster_of(column: str) -> dict | None:
    """Return the cluster dict a column belongs to (or None)."""
    for c in CLUSTERS:
        if column in c["members"]:
            return c
    return None


def links_out(column: str) -> list[dict]:
    """Interdependencies this column declares toward others."""
    return COLUMNS.get(column, {}).get("links", [])


def links_in(column: str) -> list[dict]:
    """Interdependencies OTHER columns declare toward this one (reverse edges)."""
    out = []
    for src, entry in COLUMNS.items():
        for lk in entry.get("links", []):
            if lk["to"] == column:
                out.append({"from": src, "why": lk["why"], "kind": lk["kind"]})
    return out
