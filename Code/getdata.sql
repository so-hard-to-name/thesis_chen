WITH demographic AS (
    SELECT
    ad.subject_id
    , ad.hadm_id
    , ad.admittime
    -- , pa.anchor_age
    -- , pa.anchor_year
    -- calculate the age as anchor_age (60) plus difference between
    -- admit year and the anchor year.
    -- the noqa retains the extra long line so the 
    -- convert to postgres bash script works
    , pa.anchor_age + DATETIME_DIFF(ad.admittime, DATETIME(pa.anchor_year, 1, 1, 0, 0, 0), YEAR) AS age -- noqa: L016
    , pa.gender AS gender
    FROM `physionet-data.mimiciv_hosp.admissions` ad
    INNER JOIN `physionet-data.mimiciv_hosp.patients` pa
        ON ad.subject_id = pa.subject_id
    )
, vitalsign AS (
    SELECT
    ie.subject_id
    , ie.stay_id
    , MIN(heart_rate) AS heart_rate_min
    , MAX(heart_rate) AS heart_rate_max
    , AVG(heart_rate) AS heart_rate_mean
    , MIN(sbp) AS sbp_min
    , MAX(sbp) AS sbp_max
    , AVG(sbp) AS sbp_mean
    , MIN(dbp) AS dbp_min
    , MAX(dbp) AS dbp_max
    , AVG(dbp) AS dbp_mean
    , MIN(mbp) AS mbp_min
    , MAX(mbp) AS mbp_max
    , AVG(mbp) AS mbp_mean
    , MIN(resp_rate) AS resp_rate_min
    , MAX(resp_rate) AS resp_rate_max
    , AVG(resp_rate) AS resp_rate_mean
    , MIN(temperature) AS temperature_min
    , MAX(temperature) AS temperature_max
    , AVG(temperature) AS temperature_mean
    , MIN(spo2) AS spo2_min
    , MAX(spo2) AS spo2_max
    , AVG(spo2) AS spo2_mean
    , MIN(glucose) AS glucose_min
    , MAX(glucose) AS glucose_max
    , AVG(glucose) AS glucose_mean
    FROM `physionet-data.mimiciv_icu.icustays` ie
    LEFT JOIN `physionet-data.mimiciv_derived.vitalsign` ce
        ON ie.stay_id = ce.stay_id
            AND ce.charttime >= DATETIME_SUB(ie.intime, INTERVAL '6' HOUR)
            AND ce.charttime <= DATETIME_ADD(ie.intime, INTERVAL '1' DAY)
    GROUP BY ie.subject_id, ie.stay_id
)

SELECT 
    demographic.subject_id
    , demographic.hadm_id
    , demographic.admittime
    , demographic.age
    , demographic.gender
    ,
    FROM demographic
