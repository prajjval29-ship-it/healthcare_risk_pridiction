-- Healthcare Risk Analyzer — example analytics queries
-- Run these in DB Browser for SQLite, or use as reference for the Python API.
-- Database file: healthcare_risk.db (created by scripts/setup_project.py)

-- ---------------------------------------------------------------------------
-- 1) High-risk patients (model says "high" OR clinical positive label)
-- ---------------------------------------------------------------------------
SELECT id, age, glucose, bmi, ml_probability, risk_level, outcome
FROM diabetes_patients
WHERE risk_level = 'high' OR outcome = 1
ORDER BY ml_probability DESC;

SELECT id, age, sex, blood_pressure, cholesterol, ml_probability, risk_level, heart_disease
FROM heart_patients
WHERE risk_level = 'high' OR heart_disease = 1
ORDER BY ml_probability DESC;

-- ---------------------------------------------------------------------------
-- 2) Common risk factors — compare averages when label is positive vs negative
-- ---------------------------------------------------------------------------
SELECT
    AVG(CASE WHEN outcome = 1 THEN glucose END) AS avg_glucose_if_diabetes,
    AVG(CASE WHEN outcome = 0 THEN glucose END) AS avg_glucose_if_not,
    AVG(CASE WHEN outcome = 1 THEN bmi END) AS avg_bmi_if_diabetes,
    AVG(CASE WHEN outcome = 0 THEN bmi END) AS avg_bmi_if_not
FROM diabetes_patients;

SELECT
    AVG(CASE WHEN heart_disease = 1 THEN blood_pressure END) AS avg_bp_if_disease,
    AVG(CASE WHEN heart_disease = 0 THEN blood_pressure END) AS avg_bp_if_not,
    AVG(CASE WHEN heart_disease = 1 THEN cholesterol END) AS avg_chol_if_disease,
    AVG(CASE WHEN heart_disease = 0 THEN cholesterol END) AS avg_chol_if_not
FROM heart_patients;

-- ---------------------------------------------------------------------------
-- 3) Age-wise analysis (diabetes cohort has no gender column in source CSV)
-- ---------------------------------------------------------------------------
SELECT
    CASE
        WHEN age < 30 THEN 'under_30'
        WHEN age < 45 THEN '30_44'
        WHEN age < 60 THEN '45_59'
        ELSE '60_plus'
    END AS age_bucket,
    COUNT(*) AS patients,
    SUM(outcome) AS diabetes_positive_count
FROM diabetes_patients
GROUP BY age_bucket
ORDER BY age_bucket;

-- Gender-wise + age-wise (heart dataset)
SELECT
    CASE WHEN sex = 1 THEN 'male' ELSE 'female' END AS gender,
    COUNT(*) AS patients,
    SUM(heart_disease) AS heart_disease_count
FROM heart_patients
GROUP BY sex;

SELECT
    CASE
        WHEN age < 40 THEN 'under_40'
        WHEN age < 55 THEN '40_54'
        ELSE '55_plus'
    END AS age_bucket,
    COUNT(*) AS patients,
    SUM(heart_disease) AS heart_disease_count
FROM heart_patients
GROUP BY age_bucket
ORDER BY age_bucket;

-- ---------------------------------------------------------------------------
-- Risk level distribution (for dashboards)
-- ---------------------------------------------------------------------------
SELECT risk_level, COUNT(*) FROM diabetes_patients GROUP BY risk_level;
SELECT risk_level, COUNT(*) FROM heart_patients GROUP BY risk_level;
