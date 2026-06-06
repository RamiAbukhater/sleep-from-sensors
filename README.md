# Predicting Sleep from Your Pocket: Can Daytime Sensor Data Forecast Tonight's Rest?

**Authors:** Rami Abukhater, Minh Thai  
**Course:** DSC 80, Spring 2026, UC San Diego

---

## Introduction

How much you sleep tonight may be written in the sensors you carry all day. This project investigates whether **smartphone and smartwatch data collected during waking hours** can predict how many hours a person will sleep that night — all without dedicated sleep-tracking hardware.

The dataset is the **UCSD ExtraSensory Dataset**, collected from 60 university participants over several weeks. Each row represents a **1-minute sensing window** for one user and contains pre-computed features from accelerometers, audio (MFCC coefficients), GPS, and phone-state sensors, plus self-reported activity labels submitted via a mobile app.

**Research Question:** Can daytime behavioral sensor signals — movement intensity, phone usage, audio environment, and spatial mobility — predict how many hours a person will sleep on a given night?

**Dataset size:** 144,232 rows × 143 columns across **25 participants** (a subset of the full 60-participant dataset).

**Relevant columns:**

| Column | Description |
|---|---|
| `uuid` | Anonymized participant identifier |
| `timestamp` | Unix timestamp (seconds) of the 1-minute window |
| `label:SLEEPING` | 1 = participant reported sleeping, 0 = not sleeping, NaN = no report |
| `raw_acc:magnitude_stats:mean` | Mean accelerometer magnitude (movement intensity) |
| `audio_naive:mfcc0:mean` | Mean of first MFCC coefficient (overall audio energy) |
| `location:log_diameter` | Log of the spatial diameter traversed (mobility proxy) |
| `discrete:app_state:is_active` | 1 if phone screen was active during the window |

---

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning

The raw data required several cleaning steps:

1. **Column filtering**: Only label columns, accelerometer statistics, the first 10 MFCC audio features, location features, and discrete phone-state indicators were retained — 143 columns total.
2. **Type conversion**: All sensor and label columns were cast to `float` using `pd.to_numeric(..., errors='coerce')`, converting any stray string values to NaN.
3. **Datetime parsing**: The Unix `timestamp` column was converted to a UTC-aware `datetime` column. `date` (calendar day) and `hour` were extracted for aggregation and filtering.
4. **Daily aggregation for modeling**: 1-minute windows were aggregated to user-day records. Only **daytime windows (7 AM – 9 PM)** were used to compute features, preventing any leakage from overnight sleep windows. Days where `label:SLEEPING` was entirely unlabeled (all NaN) were excluded, since a sum of NaN would spuriously yield 0 sleep hours.

**Cleaned DataFrame (first 5 rows, selected columns):**

| uuid | timestamp | label:SLEEPING | label:SITTING | label:FIX_walking | accel_mean | mfcc0_mean | loc_log_diam |
|:-----|----------:|---------------:|--------------:|------------------:|-----------:|-----------:|-------------:|
| 00EABED2... | 1444079161 | 0 | 1 | 0 | 0.9968 | -4.220 | 2.313 |
| 00EABED2... | 1444079221 | 0 | 1 | 0 | 0.9969 | -7.506 | 2.263 |
| 00EABED2... | 1444079281 | 0 | 1 | 0 | 0.9968 | -7.967 | -0.565 |
| 00EABED2... | 1444079341 | 0 | 1 | 0 | 0.9969 | -5.368 | 0.741 |
| 00EABED2... | 1444079431 | 0 | 1 | 0 | 0.9974 | -13.416 | 1.613 |

### Univariate Analysis

The plot below shows how often each activity label was explicitly reported across all 1-minute windows. Sitting and lying down dominate, reflecting the amount of time participants spent labeled in sedentary states. Sleeping is surprisingly under-represented — many sleep windows go unlabeled because participants are asleep and cannot respond to prompts.

<iframe
  src="assets/activity_label_distribution.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

### Bivariate Analysis

Distinct activities cluster clearly in (movement, audio energy) space. Walking and running occupy the high-movement region while sleeping occupies the low-movement, quiet region. This confirms that sensor features contain discriminative signal for activity recognition — and by extension, for predicting sleep.

<iframe
  src="assets/accel_vs_audio.html"
  width="800"
  height="520"
  frameborder="0"
></iframe>

### Interesting Aggregates

Sleep behavior varies widely across participants. Some users report close to 0 hours of sleep on labeled days (they may label only during waking hours), while others consistently report 5–7 hours. The `mobile_rate` column shows what fraction of their windows were labeled as walking or running.

| User (truncated) | mean_sleep | std_sleep | days_observed | mobile_rate |
|:---|---:|---:|---:|---:|
| 00EABED2... | 0.88 | 2.50 | 9 | 0.071 |
| 098A72A5... | 1.94 | 4.08 | 15 | 0.032 |
| 0A986513... | 4.77 | 3.56 | 4 | 0.050 |
| 11B5EC4D... | 5.26 | 3.50 | 8 | 0.019 |
| 59EEFAE0... | 4.32 | 2.99 | 8 | 0.081 |

The high standard deviation within each user reflects the bimodal nature of the target: most days have either ~0 or ~7–8 hours of reported sleep, depending on how consistently the user labeled their overnight windows.

---

## Assessment of Missingness

### NMAR Analysis

We believe the column `label:SLEEPING` is likely **NMAR** (Not Missing At Random). In the ExtraSensory study, participants labeled their activity by responding to phone prompts. A user who is sleeping at the time of the prompt physically cannot respond — meaning the probability of the label being missing is directly related to the unobserved value (i.e., whether they were sleeping). This is the defining characteristic of NMAR: the missing data mechanism depends on the missing data itself.

To transform this into MAR, we would need additional observed data such as: whether the phone screen was turned on at the time of the prompt, whether a phone call or alarm woke the participant, or a continuous physiological signal (e.g., wrist heart rate) indicating wakefulness. With such data, missingness could be explained by observed variables rather than the unobserved sleep state.

### Missingness Dependency

We analyzed the missingness in **`location:log_diameter`** (16% missing). This GPS feature is absent when the phone cannot obtain a location fix — typically when the user is indoors or has GPS disabled.

**Test 1 — DEPENDS ON `label:SLEEPING` (p ≈ 0.000):**  
When a window is labeled as sleeping (stationary, likely indoors), the GPS log-diameter is far more likely to be missing than when the user is labeled as awake. The observed difference in mean sleep-label between GPS-missing and GPS-present windows is 0.107, far outside the null distribution. This is consistent with MAR: GPS missingness is explained by the sleep label.

<iframe
  src="assets/missingness_gps_sleep.html"
  width="800"
  height="450"
  frameborder="0"
></iframe>

**Test 2 — DOES NOT DEPEND ON `raw_acc:magnitude_stats:value_entropy` (p = 0.774):**  
The value entropy of the accelerometer signal measures how uniformly distributed the raw readings are within a window. Whether you are indoors or outdoors (which determines GPS availability) has no systematic relationship to the shape of your movement distribution. The observed difference in mean entropy between GPS-missing and GPS-present windows (0.0012) is negligible and falls well within the null distribution — we fail to reject MCAR with respect to accel value entropy.

<iframe
  src="assets/missingness_gps_entropy.html"
  width="800"
  height="450"
  frameborder="0"
></iframe>

**Conclusion:** `location:log_diameter` missingness is **MAR** — it depends on the sleep label (sleeping → stationary/indoors → no GPS) but not on accelerometer value entropy (GPS availability is unrelated to the statistical shape of movement readings).

---

## Hypothesis Testing

**Question:** Do windows labeled as **mobile** (walking or running) have higher audio energy (MFCC0) than windows labeled as **stationary** (sitting or standing)?

Intuitively, mobile activity generates more ambient noise (footsteps, crowds, traffic), while sedentary environments (offices, homes) are quieter. Testing this lets us validate that audio features carry meaningful behavioral signal.

- **Null Hypothesis (H₀):** Mobile and stationary windows have the same mean MFCC0; any observed difference is due to random chance.
- **Alternative Hypothesis (H₁):** Mobile windows have higher mean MFCC0 than stationary windows (one-sided).
- **Test Statistic:** Difference in group means: mean(mobile MFCC0) − mean(stationary MFCC0)
- **Significance Level:** α = 0.05
- **Method:** Permutation test with 5,000 shuffles

**Result:** Observed difference = **+1.83** (mobile windows are louder); p-value ≈ **0.000**.

We reject H₀. The data are consistent with the hypothesis that mobile activity environments are systematically louder than sedentary ones. This result — statistically significant and physically interpretable — confirms that audio features encode real behavioral signal for activity recognition.

*Note: This is a permutation test on observational data, not a randomized experiment. We cannot conclude that mobility *causes* louder audio; we can only say the association is unlikely to be due to chance.*

<iframe
  src="assets/hypothesis_permutation.html"
  width="800"
  height="450"
  frameborder="0"
></iframe>

---

## Framing a Prediction Problem

**Prediction Problem:** Regression — predict the continuous variable `sleep_hours` (total hours of reported sleep for a user on a given calendar day).

**Response variable:** `sleep_hours` = count of 1-minute windows where `label:SLEEPING == 1` for a user-day, divided by 60. Only days where the sleep label was explicitly observed (at least one non-NaN window) are included, to avoid confounding "no labels submitted" with "zero sleep."

**Why this variable?** Sleep duration is the most direct, interpretable measure of nightly rest available in the dataset. Predicting binary "slept / didn't sleep" would ignore the important variation in duration.

**Features (all derived from daytime windows, 7 AM – 9 PM):**

| Feature | Derivation | Justification |
|---|---|---|
| `daytime_accel_mean` | Mean accelerometer magnitude | Activity level during the day |
| `prior_day_sleep` | Previous calendar day's sleep_hours | Sleep is autocorrelated across days |
| `location_variance_daily` | Mean `location:log_diameter` during the day | Spatial mobility (indoor vs outdoor patterns) |
| `phone_active_fraction` | Fraction of windows with screen active | Phone usage intensity |
| `audio_variance_daytime` | Variance of `mfcc0` across daytime windows | Audio environment variability (social richness) |

Using only daytime features prevents data leakage: we predict tonight's sleep from today's daytime behavior, mimicking real-time deployment.

**Evaluation Metric:** Root Mean Squared Error (RMSE) in hours. RMSE penalizes large errors more heavily than MAE, which is appropriate here — being off by 3 hours is substantially worse than being off by 30 minutes.

**Train/Test Split:** 80% train / 20% test, stratified by random seed 42 at the user-day level. The same split is reused for both baseline and final model to enable fair comparison.

---

## Baseline Model

**Model:** Linear Regression with **2 features:**
- `daytime_accel_mean` (quantitative)
- `prior_day_sleep` (quantitative)

Both features are numeric and used as-is (no encoding required). `prior_day_sleep` captures momentum in sleep duration — people who slept well yesterday tend to sleep well today. `daytime_accel_mean` captures physical activity level, which correlates with fatigue and sleep drive.

**Performance:** RMSE = **3.36 hours** on the held-out test set.

This baseline is intentionally weak — with only two features and a linear model, the regressor predicts values clustered near the mean (~4.5–6 hours) and misses both short and long sleep nights. The scatter plot shows a horizontal band of predictions rather than a diagonal — that is exactly what a limited baseline looks like, and it gives a clear floor to improve upon.

---

## Final Model

**New Features Added (on top of baseline):**

| Feature | Derivation | Why it should help |
|---|---|---|
| `evening_accel_mean` | Mean accel 8 PM–1 AM | Low movement in the evening signals physical wind-down → more sleep |
| `evening_phone_fraction` | Fraction of 8 PM–1 AM windows with screen active | Heavy late-night phone use delays sleep onset → shorter/later sleep |
| `location_variance_daily` | Mean GPS log-diameter (daytime) | More mobility → outdoor activity → physical fatigue → longer sleep |
| `phone_active_fraction` | Screen-active fraction (daytime) | Overall phone usage intensity as a proxy for sedentary/active lifestyle |
| `audio_variance_daytime` | Variance of MFCC0 (daytime) | High audio variance → socially rich environments → different sleep patterns |
| `n_labeled_windows` | Count of recorded sensor windows | Data coverage per day; days with more sensor coverage have richer features |

These features are motivated by the **data generating process**: evening behavior directly precedes sleep onset; daytime features capture the fatigue and lifestyle factors that build up across the day.

**Feature Transforms (implemented in the sklearn Pipeline):**
- `StandardScaler` on direct sensor and ratio features (daytime_accel_mean, prior_day_sleep, phone_active_fraction, n_labeled_windows, evening_accel_mean, evening_phone_fraction)
- `QuantileTransformer(output_distribution='normal')` on right-skewed distribution features (location_variance_daily, audio_variance_daytime) — redistributing these to a normal distribution allows the forest to split the feature space more evenly

**Modeling Algorithm:** Random Forest Regressor inside a `Pipeline([ColumnTransformer, RandomForestRegressor])`. Unlike linear regression, the forest can capture nonlinear interactions (e.g., high phone use only disrupts sleep past a threshold) and benefits from the QuantileTransformer's redistribution of skewed features.

**Hyperparameter Tuning:** 5-fold GridSearchCV on the training set over:
- `model__n_estimators` ∈ {100, 200}
- `model__max_depth` ∈ {3, 5, None}

**Best hyperparameters:** `n_estimators=100`, `max_depth=5`

**Final Model RMSE: 2.94 hours** — an improvement of **0.42 hours (12.4%)** over the baseline (3.36 hours).

Crucially, the final model's predictions now span the full range 1.1–8.9 hours, closely matching the actual range. The baseline predicted only 4.5–6.0 hours regardless of actual sleep.

<iframe
  src="assets/feature_importances.html"
  width="800"
  height="450"
  frameborder="0"
></iframe>

---

## Fairness Analysis

**Question:** Does the final model perform equally well on **high-activity** vs. **low-activity** users?

Users were split at the median of their `daytime_accel_mean` across the dataset:
- **Group X (High-activity):** users with mean daytime accel ≥ median (11 test records)
- **Group Y (Low-activity):** users with mean daytime accel < median (16 test records)

Low-activity users might have less variation in their sensor signals day-to-day, making prediction harder or easier depending on whether the model has learned their patterns.

**Evaluation Metric:** RMSE (same metric as the overall model)

**Hypotheses:**
- **H₀:** The model is fair — RMSE is the same for high and low activity groups; any difference is due to chance.
- **H₁:** The model is unfair — RMSE differs between the two groups.

**Test Statistic:** |RMSE(high-activity) − RMSE(low-activity)|  
**Significance Level:** α = 0.05  
**Method:** Permutation test (1,000 shuffles of group labels)

**Results:**
- RMSE (high-activity): 3.67 hours
- RMSE (low-activity): 1.63 hours
- Observed |difference|: 2.03 hours
- **p-value: 0.065**

**Conclusion:** We **fail to reject H₀** at α = 0.05 (p=0.065). The observed RMSE gap could plausibly arise by chance with this sample size. There is no statistically significant evidence of unfairness, though the trend (model is somewhat more accurate for low-activity users) may warrant investigation with a larger dataset.

*Note: With only 27 test records across two groups, the permutation test has modest power. These findings should be validated on a larger dataset before drawing strong policy conclusions.*

---

*This project was completed as part of DSC 80: Practice and Application of Data Science at UC San Diego, Spring 2026.*
