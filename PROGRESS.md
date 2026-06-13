# Project Progress — How I Tackled This

A plain-language log of the steps I took and the decisions I made, in order — written
so I (and anyone watching the demo) can follow the *why*, not just the *what*.
*(Updated only when I explicitly ask for it.)*

---

## A note on where I'm putting my effort

Before any code, one thing shaped every choice below: **most of the grade is for the
production system around the model — the pipeline, the API, monitoring, governance —
not for the model's raw accuracy.** On top of that, this dataset has a well-known
accuracy ceiling that no amount of clever modeling really breaks through. So I made a
deliberate call: keep the modeling lean and trustworthy, and spend my real energy on
the parts that count. A lot of the decisions below are me *choosing the simple,
defensible option on purpose* rather than experimenting my way to a tiny gain.

---

## Step 1 — Put the raw data under version control with DVC

**The goal:** treat the ~19 MB dataset like code, but without bloating the git repo.

I ran `dvc init` and then `dvc add` on the raw CSV. After that, git only stores a tiny
**pointer file** — basically a fingerprint of the data — while the real file lives in a
local DVC cache. Anyone who clones the repo runs `dvc pull` to fetch the exact same
bytes.

**Why it matters:** the data is now reproducible and tracked. If the file ever changes,
its fingerprint changes and git shows me immediately. No more "which version of the
CSV was this trained on?"

---

## Step 2 — Clean the data into one trustworthy base table

**The goal:** one solid, honest table that everything later is built on. Each decision
here quietly guards against a way the project could go wrong.

- **Missing values were hidden as `?`.** In this file, a blank isn't blank — it's the
  character `?`. I convert those to real "missing" markers on load, so the model never
  mistakes `?` for a genuine category.

- **A skipped lab test is *information*, not a blank.** The `A1Cresult` and
  `max_glu_serum` columns are mostly empty — but that emptiness means "the doctor
  didn't order this test," which is itself a clue about the patient's care. So instead
  of treating those as missing, I keep them as an explicit `NotMeasured` value. (This
  is also why I was careful on load: the word "None" in these columns means "not
  ordered," a real fact, and I made sure it didn't get silently wiped.)

- **Dropped dead weight.** Three columns carry no usable signal: `weight` (empty for
  97% of patients) and `examide` + `citoglipton` (the exact same value for every
  single row). Gone.

- **Removed patients who couldn't possibly be readmitted.** Some discharge codes mean
  the patient died or went to hospice. Those people can't come back within 30 days, so
  leaving them in would teach the model a fake "answer." I drop discharge codes
  {11, 13, 14, 19, 20, 21}. (A handful of rows with an invalid gender value went too —
  negligible, but tidy.)

- **Treated "missing" as its own category for `payer_code`, `medical_specialty`, and
  `race`.** These are missing up to ~half the time. Rather than throw the whole column
  away or delete those patients, I label the gaps as `"Unknown"` and keep them. The
  reasoning: *the fact that something is missing can be a clue.* A blank specialty, for
  example, often means the patient arrived through a different care path — that's worth
  keeping, not erasing.

- **Kept only each patient's first visit (and proved it worked).** The same person can
  appear many times across years of hospital stays. If the same patient lands in both
  my training data and my test data, the model can "cheat" by recognizing them — and
  I'd get a score that looks great but lies. To stop that, I keep only the **earliest
  visit per patient** and drop the repeats. Then I added a one-line check that asserts
  every patient now appears exactly once. If that check ever fails, I find out
  instantly.

- **One nice consequence: I could drop a whole layer of complexity.** Because every
  patient now shows up only once, there's no risk of one person spanning both data
  piles anymore. That means I *don't* need the special "grouped" splitting machinery
  people usually add for this dataset — a normal balanced split is already safe. Fewer
  moving parts, same protection.

---

## Decision — The thing I'm predicting: a simple yes/no

The raw `readmitted` column has three values: `<30` (came back within 30 days),
`>30` (came back later), and `NO` (didn't come back). I collapsed this to a plain
**yes/no**: `<30` = 1, everything else = 0.

**Why:**
- The care team's actual decision at discharge is binary — does this patient get extra
  30-day follow-up, or not? The difference between "came back later" and "never came
  back" doesn't change that decision.
- The whole brief is about *30-day* readmission, so `<30` is the event I care about.
  Lumping `<30` and `>30` together would quietly answer a different question.
- Keeping three classes also makes the rare-event imbalance worse and the metrics
  harder to read.

---

## Decision — Inventing fake patients (SMOTE): considered, rejected

Only about 9% of patients are readmitted, so the classes are very lopsided. One common
trick (SMOTE) balances them by *generating synthetic extra examples* of the rare group.
I looked at it and decided against it.

**Why:** to balance this data I'd have to fabricate roughly **54,000 patients who never
existed** — hard to stand behind for a tool that informs real hospital care. And the
published work on this exact dataset shows SMOTE actually makes the results *worse*
here. I'll handle the imbalance with **class weighting** instead, which tells the model
to take the rare cases more seriously *without* inventing anyone. I'm recording the
rejection on purpose — showing I weighed it matters more than silently skipping it.

---

## Decision — How I'm grouping the diagnosis codes: Strack-9, chosen not tested

Each visit has up to three diagnosis codes drawn from hundreds of possibilities — too
many to use directly, so they need to be grouped into a handful of clinical categories.
There are fancier options (validated "comorbidity index" scores, even using a medical
AI model to classify the codes), and I considered running a bake-off between them.

**I chose not to.** I'm going with the well-established **Strack-9** grouping — the
standard for this exact dataset — decided by *research, not experimentation*. The
fancier alternatives promised only tiny gains that tend to vanish into normal
run-to-run noise, and the assignment rewards the system around the model far more than
squeezing out a fraction of a percent. So one proven grouping, picked deliberately, and
I move on. (I left the door open: if everything else finishes early, adding a fancier
option later is a small change — I just chose not to spend the time now.)

---

## Where things stand (the numbers)

| Stage | Rows |
|---|---|
| Raw CSV | 101,766 |
| After removing expired/hospice (and bad-gender) rows | 99,340 |
| After keeping only first visit per patient | **69,987** |

- Columns: 47 (dropped `weight`, `examide`, `citoglipton`, and the original
  `readmitted`; added the yes/no `target`).
- Readmission rate: **~9%** — slightly below the often-quoted ~11% because first visits
  skew toward earlier, lower-risk stays. Expected, not a bug.
- Patient-uniqueness check: **passed.**
- The whole cleaning step runs as one reproducible command.

---

## Up next

- **Featurization:** turn the cleaned columns into model-ready signals — Strack-9
  diagnosis buckets; how many medications were changed; whether key lab tests were even
  ordered (a surprisingly strong clue); the prior-visit / "service utilization" counts;
  and tidied-up age and admission/discharge categories.
- **Modeling:** start with a simple logistic-regression baseline (the bar to beat),
  then a stronger CatBoost model tuned with Optuna — with every experiment logged to
  MLflow.

---

## Step — Building the model, honestly
Before training, I set aside a fifth of the patients as a final test set the model
never sees until the very end — touched exactly once — so the score I report isn't
one I quietly tuned toward.

I built two models, in order. First a plain **logistic regression** as the bar to
beat: it's simple and fully explainable, and it scored an honest ~0.65 (ROC-AUC).
Then a stronger **CatBoost** model, which reached ~0.67. Both weight the rare
"readmitted" cases up so they aren't ignored, and every run is logged to MLflow so
the comparison is fair and reproducible.

Two things worth saying plainly:
- These scores sound modest, but they're *right*. This dataset has a well-known
  ceiling around 0.66–0.70 — the signal genuinely is weak. I set a tripwire at 0.75:
  anything higher would mean a leak, not a triumph. Both models passed.
- CatBoost earned its place. At the same precision it catches ~40% more of the
  patients who actually return — a real operational gain — so I carried it forward
  and kept logistic regression as an explainable fallback.

**Up next:** calibrate the model's probabilities, choose the risk threshold that
fits the cost trade-off, then register it for serving.

---

## Step — Tuning both models, and choosing which to trust
I gave both models a proper tune-up (an automated search over their settings,
optimizing for AUPRC — the honest metric when only ~9% of patients are readmitted),
and tuned them side by side on the exact same train/test split so the comparison
was fair.

The result was clarifying, even though it went against my initial preference. I'd
hoped the simple logistic regression could be my headline model, because it explains
itself. But tuning showed it had already hit its ceiling — it couldn't get better,
because a straight-line model has squeezed out all the signal it can from this data.
CatBoost pulled further ahead: at the same precision it catches about 2.6× as many
of the patients who actually return.

So I'm leading with CatBoost — but I'm not giving up explainability to do it. CatBoost
supports SHAP, which explains every individual prediction ("this patient scored high
because of prior inpatient visits, not being discharged home..."). And I'm keeping the
logistic regression alongside it as a transparent cross-check: when the simple model's
reasoning agrees with CatBoost's explanations, that's strong evidence the model is
picking up real clinical signal, not noise.

**Up next:** make the model's probabilities trustworthy (calibration), choose the
risk cutoff that fits the cost of a missed readmission, and register the model.

---

## Step — Making the risk scores honest, and setting the cutoff
A model can rank patients well but still output numbers that don't mean what they say —
a "0.30" that doesn't actually correspond to a 30% chance. So I added a calibration
step that corrects the scores so the number means what it claims. I tried two methods
and picked the simpler one (they tied), and it made the probabilities ~21% more honest
without changing how the model ranks patients at all.

Then I set the decision cutoff — the line above which a patient is flagged for extra
follow-up. Because this is a safety-net screening tool, I leaned toward catching more
true cases: at the chosen cutoff the model catches about half of the patients who will
actually be readmitted. The trade-off is that it flags roughly a third of all patients
to do that, most of whom won't return — an unavoidable reality when only ~9% are
readmitted. I documented a lower-sensitivity setting too, in case the hospital's
follow-up capacity is tighter.

Finally I registered the finished model so the live service can load it, and so I can
roll back to a previous version by flipping its status if needed.

**Up next:** wrap the model in an API that returns a risk score plus the reasons behind
it, then package it to run anywhere.

---

## Step — Turning the model into a service, then a container
I wrapped the model in a small web service. You send it one patient's discharge
details and it returns three things: the calibrated risk, a yes/no follow-up flag (is
the risk above our chosen cutoff), and the top reasons behind that score in plain
feature terms — what pushed this particular patient's risk up or down.

The thing I was most careful about is the trap that quietly ruins ML systems:
"train/serve skew" — where the live service prepares the data even slightly
differently from how the model was trained, so it scores the same patient differently.
I avoided it by making the service reuse the *exact same* preparation code as training,
not a re-write. Then I proved it: the same patient scores identically — to the last
digit — in training and through the live API. That exact-match check is now an
automated test that must never break.

Finally I packaged the whole thing into a container so it runs the same anywhere. The
one wrinkle: the experiment-tracking store keeps file paths that only make sense on my
machine, so instead of pointing the container at it, I bake the finished model directly
into the image. The "which model is live" label still travels with it, so rolling back
to a previous model stays simple. The container scores the same patient identically to
the local version — the move into a container introduced no drift.

**Up next:** push the image to the cloud and set up automated build-and-deploy, then
add monitoring and the fairness/governance checks.

---

## Step — Adding a data quality gate (Great Expectations)
Before trusting any data through the pipeline, I added an automatic quality checkpoint.
It's a set of rules describing what valid data looks like — the right columns, sensible
value ranges, the categories each field is allowed to take, how missing values are
coded — and the pipeline refuses to proceed if the data breaks them.

I check at two points, because they catch different problems. One set of rules guards
the *raw* file as it comes in (did a column vanish, did an unexpected code appear). A
second set guards the *cleaned* output (did my own pipeline produce something wrong —
duplicates that shouldn't exist, a readmission rate that drifted out of range). A
failing check stops the pipeline loudly rather than letting bad data slip through — and
I proved that by feeding it a deliberately broken row (a patient with gender "Martian")
and watching it halt.

The same rulebook does double duty: it's exported as a shared "contract" that the live
service's input checks are built from, so the pipeline and the API agree on what valid
input means instead of drifting apart. Later, the drift monitoring will lean on it too.

## Step — Slimming the container so it's deployable
The first container I built was ~7.4 GB — it had accidentally packed in the entire
training-and-experimentation toolkit, none of which the live service needs. That size
would make cloud uploads slow and start-up sluggish. So I split out a serving-only set
of dependencies (just what's needed to answer a prediction) and rebuilt.

The result: **943 MB — about 87% smaller** — and, the part that mattered most, it
still scores the test patient to the exact same digit (0.074595) as before. Cutting the
fat changed nothing about the answers; it just made the thing light enough to ship
sensibly. I verified the slim version against that "golden number" before keeping it,
so I never traded correctness for size.

**Up next:** push the image to the cloud's container registry with an automated
build-and-test pipeline (every change runs the test suite before it can ship), deploy
it so it's reachable at a URL, then add live monitoring and the fairness/governance
checks.

---

## Step — Automating the build and shipping it to the cloud
I set up a pipeline that runs automatically every time I push a change. It does the
work in a strict order and refuses to cut corners: first it runs the fast input-checks,
then it builds the container, then it starts that container and tests the *real running
service* — including the all-important check that the test patient still scores exactly
0.074595 — and only if every one of those passes does it upload the finished image to
the cloud's container registry. Tests are the gate: if anything fails, nothing ships.
This is what "every change is automatically tested before it can go live" actually
looks like in practice.

The reason this works cleanly is the choice I made earlier to bake the model into the
image. The automation runs on a fresh, empty machine each time — it has none of my
local setup — so anything that depended on my laptop would break. Because the model
travels inside the image, the pipeline can rebuild and test the exact same service from
nothing but the code.

Getting it green was an honest debugging journey, and each failure was a real-world
lesson: first a leftover setup step clashed with the build environment; then it
couldn't talk to the cloud until I wired in credentials the safe way (stored as secrets,
never written into the code); then the cloud refused the upload until I granted the
right permissions. Each is exactly the kind of unglamorous plumbing that separates "runs
on my machine" from "ships automatically" — and it now ships automatically.

**Up next:** write the reflection, then the governance work — a fairness audit across
age, gender and race, clear explanations for each prediction, an audit trail, and the
model card — followed by live monitoring for drift.

---

## Step — Making it trustworthy, watchable, and easy to show
With the service shipping automatically, I turned to the work that makes a model
*responsible* rather than just functional. I audited fairness across age, gender and
race and reported it honestly: the model treats the sexes evenhandedly, the racial
differences are small and mostly down to tiny sample sizes, but a single cutoff is much
better at catching readmissions in the elderly than in middle-aged patients — a real gap
I wrote down rather than hid. I added clear per-patient explanations (which factors
pushed a given prediction up or down), an audit log that records every scored request so
any decision can be traced later, and a model card that states plainly what the model is
for, how well it does, where it fails, and that it must never be the sole decider — a
clinician reviews every case.

Then I made the running service *watchable*. The API now publishes live operational
metrics, and I stood up the standard monitoring pair — Prometheus to collect them and
Grafana to chart them — with a dashboard showing request rate, response times, error
rate and how many predictions have been served, plus alerts that would warn a human if
the service started erroring or slowing down. Separately, I built a drift detector: it
compares fresh data against the patient mix the model learned from. To prove it actually
works I fed it a deliberately shifted batch (an older, sicker population) and confirmed
it lit up, while an unchanged batch stayed quiet — a detector that only cries wolf is
worthless. That feeds a concrete, written retrain rule: specific numbers that say "the
world has moved enough — it's time to retrain," not a vague intention.

Finally, I built a simple demonstration interface so the whole thing can be *seen*. It's
a thin front-end that just asks the API for a prediction — it does no thinking of its own,
which keeps it honest. You can fill in a patient, or load a real past patient the model
never trained on and watch the model's guess line up against what actually happened to
them, marked right or wrong — including the wrong calls, shown openly. It carries a clear
"demonstration, not a medical device" banner throughout.

**Up next:** a final clean-checkout rehearsal — make sure someone starting from nothing
can rebuild and run everything from the README — then push it all up and submit.