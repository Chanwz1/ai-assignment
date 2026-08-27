# Report source notes

Audience: technical. Delivery mode: portable HTML generated from the canonical report artifact.

Required-structure mapping:

- Title: `title`
- Technical summary: `technical_summary`
- Key findings with visual evidence: performance, confusion-matrix, and BMI sections
- Scope, data, and metric definitions: `scope_definitions`
- Model specification and validation: `methodology` plus the tuning table
- Limitations and robustness: BMI comparison and deterministic synthetic-rule section
- Recommended next steps: `next_steps`
- Further questions: `further_questions`

Chart map:

| Section | Question | Family/type | Fields | Claim | Palette | Delivery |
|---|---|---|---|---|---|---|
| Performance | Does SVM beat baseline and 80%? | Comparison / bar | metric, accuracy | 96.15% test accuracy materially clears both | Default accessible single series | `svm_analysis_report.html` |
| BMI | Which BMI treatment is defensible? | Comparison / bar | variant, mean_cv_accuracy | Removing BMI is slightly best and avoids a corrupted field | Default accessible single series | `svm_analysis_report.html` |

Both visuals use reviewed rows derived from `static/metrics.json`, include richer tooltip context, and have adjacent narrative interpretation. No trend chart is used because the evidence is a discrete model comparison rather than a time series.
