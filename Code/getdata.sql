WITH demographic AS (
    SELECT
    ie.subject_id
    , ie.stay_id
    , ie.intime
    -- , pa.anchor_age
    -- , pa.anchor_year
    -- calculate the age as anchor_age (60) plus difference between
    -- admit year and the anchor year.
    -- the noqa retains the extra long line so the 
    -- convert to postgres bash script works
    , pa.anchor_age + DATETIME_DIFF(ie.intime, DATETIME(pa.anchor_year, 1, 1, 0, 0, 0), YEAR) AS age -- noqa: L016
    , pa.gender AS gender
    FROM `physionet-data.mimiciv_icu.icustays` ie
    LEFT JOIN `physionet-data.mimiciv_hosp.patients` pa
        ON ie.subject_id = pa.subject_id
    WHERE pa.anchor_age + DATETIME_DIFF(ie.intime, DATETIME(pa.anchor_year, 1, 1, 0, 0, 0), YEAR) > 18
        AND ie.outtime > DATETIME_ADD(ie.intime, INTERVAL '1' DAY)
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
    HAVING COUNT(heart_rate) > 3
        AND COUNT(sbp) > 3
        AND COUNT(dbp) > 3
        AND COUNT(mbp) > 3
        AND COUNT(resp_rate) > 3
        AND COUNT(temperature) > 3
        AND COUNT(spo2) > 3
        AND COUNT(glucose) > 3
    )
  , gcs AS (
    SELECT
        ie.subject_id, ie.stay_id
        , g.gcs AS gcs_value
        , g.gcs_motor
        , g.gcs_verbal
        , g.gcs_eyes
        , g.gcs_unable
        -- This sorts the data by GCS
        -- rn = 1 is the the lowest total GCS value
        , ROW_NUMBER() OVER
        (
            PARTITION BY g.stay_id
            ORDER BY g.gcs
        ) AS gcs_seq
    FROM `physionet-data.mimiciv_icu.icustays` ie
    -- Only get data for the first 24 hours
    LEFT JOIN `physionet-data.mimiciv_derived.gcs` g
        ON ie.stay_id = g.stay_id
            AND g.charttime >= DATETIME_SUB(ie.intime, INTERVAL '6' HOUR)
            AND g.charttime <= DATETIME_ADD(ie.intime, INTERVAL '1' DAY)
)

, urinefinal AS (
    SELECT
        -- patient identifiers
        ie.subject_id
        , ie.stay_id
        , SUM(urineoutput) AS urineoutput
    FROM `physionet-data.mimiciv_icu.icustays` ie
    -- Join to the outputevents table to get urine output
    LEFT JOIN `physionet-data.mimiciv_derived.urine_output` uo
        ON ie.stay_id = uo.stay_id
            -- ensure the data occurs during the first day
            AND uo.charttime >= ie.intime
            AND uo.charttime <= DATETIME_ADD(ie.intime, INTERVAL '1' DAY)
    GROUP BY ie.subject_id, ie.stay_id
)

, tm AS (
    SELECT stay_id, charttime
    FROM `physionet-data.mimiciv_derived.ventilator_setting`
    UNION DISTINCT
    SELECT stay_id, charttime
    FROM `physionet-data.mimiciv_derived.oxygen_delivery`
)

, vs AS (
    SELECT tm.stay_id, tm.charttime
        -- source data columns, here for debug
        , o2_delivery_device_1
        , COALESCE(ventilator_mode, ventilator_mode_hamilton) AS vent_mode
        -- case statement determining the type of intervention
        -- done in order of priority: trach > mech vent > NIV > high flow > o2
        , CASE
            -- tracheostomy
            WHEN o2_delivery_device_1 IN
                (
                    'Tracheostomy tube'
                    -- 1135 observations for T-Piece
                    -- could be either InvasiveVent or Tracheostomy, so omit
                    -- 'T-piece',
                    , 'Trach mask ' -- 16435 observations
                )
                THEN 'Tracheostomy'
            -- mechanical / invasive ventilation
            WHEN o2_delivery_device_1 IN
                (
                    'Endotracheal tube'
                )
                OR ventilator_mode IN
                (
                    '(S) CMV'
                    , 'APRV'
                    , 'APRV/Biphasic+ApnPress'
                    , 'APRV/Biphasic+ApnVol'
                    , 'APV (cmv)'
                    , 'Ambient'
                    , 'Apnea Ventilation'
                    , 'CMV'
                    , 'CMV/ASSIST'
                    , 'CMV/ASSIST/AutoFlow'
                    , 'CMV/AutoFlow'
                    , 'CPAP/PPS'
                    , 'CPAP/PSV'
                    , 'CPAP/PSV+Apn TCPL'
                    , 'CPAP/PSV+ApnPres'
                    , 'CPAP/PSV+ApnVol'
                    , 'MMV'
                    , 'MMV/AutoFlow'
                    , 'MMV/PSV'
                    , 'MMV/PSV/AutoFlow'
                    , 'P-CMV'
                    , 'PCV+'
                    , 'PCV+/PSV'
                    , 'PCV+Assist'
                    , 'PRES/AC'
                    , 'PRVC/AC'
                    , 'PRVC/SIMV'
                    , 'PSV/SBT'
                    , 'SIMV'
                    , 'SIMV/AutoFlow'
                    , 'SIMV/PRES'
                    , 'SIMV/PSV'
                    , 'SIMV/PSV/AutoFlow'
                    , 'SIMV/VOL'
                    , 'SYNCHRON MASTER'
                    , 'SYNCHRON SLAVE'
                    , 'VOL/AC'
                )
                OR ventilator_mode_hamilton IN
                (
                    'APRV'
                    , 'APV (cmv)'
                    , 'Ambient'
                    , '(S) CMV'
                    , 'P-CMV'
                    , 'SIMV'
                    , 'APV (simv)'
                    , 'P-SIMV'
                    , 'VS'
                    , 'ASV'
                )
                THEN 'InvasiveVent'
            -- NIV
            WHEN o2_delivery_device_1 IN
                (
                    'Bipap mask ' -- 8997 observations
                    , 'CPAP mask ' -- 5568 observations
                )
                OR ventilator_mode_hamilton IN
                (
                    'DuoPaP'
                    , 'NIV'
                    , 'NIV-ST'
                )
                THEN 'NonInvasiveVent'
            -- high flow nasal cannula
            WHEN o2_delivery_device_1 IN
                (
                    'High flow nasal cannula' -- 925 observations
                )
                THEN 'HFNC'
            -- non rebreather
            WHEN o2_delivery_device_1 IN
                (
                    'Non-rebreather' -- 5182 observations
                    , 'Face tent' -- 24601 observations
                    , 'Aerosol-cool' -- 24560 observations
                    , 'Venti mask ' -- 1947 observations
                    , 'Medium conc mask ' -- 1888 observations
                    , 'Ultrasonic neb' -- 9 observations
                    , 'Vapomist' -- 3 observations
                    , 'Oxymizer' -- 1301 observations
                    , 'High flow neb' -- 10785 observations
                    , 'Nasal cannula'
                )
                THEN 'SupplementalOxygen'
            WHEN o2_delivery_device_1 IN
                (
                    'None'
                )
                THEN 'None'
            -- not categorized: other
            ELSE NULL END AS ventilation_status
    FROM tm
    LEFT JOIN `physionet-data.mimiciv_derived.ventilator_setting` vs
        ON tm.stay_id = vs.stay_id
            AND tm.charttime = vs.charttime
    LEFT JOIN `physionet-data.mimiciv_derived.oxygen_delivery` od
        ON tm.stay_id = od.stay_id
            AND tm.charttime = od.charttime
)

, vd0 AS (
    SELECT
        stay_id, charttime
        -- source data columns, here for debug
        -- , o2_delivery_device_1
        -- , vent_mode
        -- carry over the previous charttime which had the same state
        , LAG(
            charttime, 1
        ) OVER (
            PARTITION BY stay_id, ventilation_status ORDER BY charttime
        ) AS charttime_lag
        -- bring back the next charttime, regardless of the state
        -- this will be used as the end time for state transitions
        , LEAD(charttime, 1) OVER w AS charttime_lead
        , ventilation_status
        , LAG(ventilation_status, 1) OVER w AS ventilation_status_lag
    FROM vs
    WHERE ventilation_status IS NOT NULL
    WINDOW w AS (PARTITION BY stay_id ORDER BY charttime)
)

, vd1 AS (
    SELECT
        stay_id
        , charttime
        , charttime_lag
        , charttime_lead
        , ventilation_status

        -- source data columns, here for debug
        -- , o2_delivery_device_1
        -- , vent_mode

        -- calculate the time since the last event
        , DATETIME_DIFF(charttime, charttime_lag, MINUTE) / 60 AS ventduration

        -- now we determine if the current ventilation status is "new",
        -- or continuing the previous event
        , CASE
            -- if lag is null, this is the first event for the patient
            WHEN ventilation_status_lag IS NULL THEN 1
            -- a 14 hour gap always initiates a new event
            WHEN DATETIME_DIFF(charttime, charttime_lag, HOUR) >= 14 THEN 1
            -- not a new event if identical to the last row
            WHEN ventilation_status_lag != ventilation_status THEN 1
            ELSE 0
        END AS new_ventilation_event
    FROM vd0
)

, vd2 AS (
    SELECT vd1.stay_id, vd1.charttime
        , vd1.charttime_lead, vd1.ventilation_status
        , ventduration, new_ventilation_event
        -- create a cumulative sum of the instances of new ventilation
        -- this results in a monotonically increasing integer assigned 
        -- to each instance of ventilation
        , SUM(new_ventilation_event) OVER
        (
            PARTITION BY stay_id
            ORDER BY charttime
        ) AS vent_seq
    FROM vd1
)

-- create the durations for each ventilation instance
, vs_final AS (
    SELECT
    stay_id
    , MIN(charttime) AS starttime
    -- for the end time of the ventilation event, the time of the *next* setting
    -- i.e. if we go NIV -> O2, the end time of NIV is the first row
    -- with a documented O2 device
    -- ... unless it's been over 14 hours,
    -- in which case it's the last row with a documented NIV.
    , MAX(
        CASE
            WHEN charttime_lead IS NULL
                OR DATETIME_DIFF(charttime_lead, charttime, HOUR) >= 14
                THEN charttime
            ELSE charttime_lead
        END
    ) AS endtime
    -- all rows with the same vent_num will have the same ventilation_status
    -- for efficiency, we use an aggregate here,
    -- but we could equally well group by this column
    , MAX(ventilation_status) AS ventilation_status
    , ROW_NUMBER() OVER
        (
            PARTITION BY stay_id
            ORDER BY CASE WHEN MAX(ventilation_status) = 'Tracheostomy' THEN 1
                          WHEN MAX(ventilation_status) = 'InvasiveVent' THEN 2
                          WHEN MAX(ventilation_status) = 'NonInvasiveVent' THEN 3
                          WHEN MAX(ventilation_status) = 'HFNC' THEN 4
                          WHEN MAX(ventilation_status) = 'SupplementalOxygen' THEN 5
                          ELSE 6 END
        ) AS ventilation_seq
    FROM vd2
    GROUP BY stay_id, vent_seq
    HAVING MIN(charttime) != MAX(charttime)
)

, cpap AS (
    SELECT ie.stay_id
        , MIN(DATETIME_SUB(charttime, INTERVAL '1' HOUR)) AS starttime
        , MAX(DATETIME_ADD(charttime, INTERVAL '4' HOUR)) AS endtime
        , MAX(CASE
                WHEN LOWER(ce.value) LIKE '%cpap%' THEN 1
                WHEN LOWER(ce.value) LIKE '%bipap mask%' THEN 1
                ELSE 0 END) AS cpap
    FROM `physionet-data.mimiciv_icu.icustays` ie
    INNER JOIN `physionet-data.mimiciv_icu.chartevents` ce
        ON ie.stay_id = ce.stay_id
            AND ce.charttime >= ie.intime
            AND ce.charttime <= DATETIME_ADD(ie.intime, INTERVAL '1' DAY)
    WHERE itemid = 226732
        AND (
            LOWER(ce.value) LIKE '%cpap%' OR LOWER(ce.value) LIKE '%bipap mask%'
        )
    GROUP BY ie.stay_id
)

, pafi1 AS (
    -- join blood gas to ventilation durations to determine if patient was vent
    -- also join to cpap table for the same purpose
    SELECT ie.stay_id, bg.charttime
        , pao2fio2ratio
        , CASE WHEN vd.stay_id IS NOT NULL THEN 1 ELSE 0 END AS vent
        , CASE WHEN cp.stay_id IS NOT NULL THEN 1 ELSE 0 END AS cpap
    FROM `physionet-data.mimiciv_derived.bg` bg
    INNER JOIN `physionet-data.mimiciv_icu.icustays` ie
        ON bg.hadm_id = ie.hadm_id
            AND bg.charttime >= ie.intime AND bg.charttime < ie.outtime
    LEFT JOIN `physionet-data.mimiciv_derived.ventilation` vd
        ON ie.stay_id = vd.stay_id
            AND bg.charttime >= vd.starttime
            AND bg.charttime <= vd.endtime
            AND vd.ventilation_status = 'InvasiveVent'
    LEFT JOIN cpap cp
        ON ie.stay_id = cp.stay_id
            AND bg.charttime >= cp.starttime
            AND bg.charttime <= cp.endtime
)

, pafi2 AS (
    -- get the minimum PaO2/FiO2 ratio *only for ventilated/cpap patients*
    SELECT stay_id
        , MIN(pao2fio2ratio) AS pao2fio2_vent_min
    FROM pafi1
    WHERE vent = 1 OR cpap = 1
    GROUP BY stay_id
)

, cohort AS (
    SELECT ie.subject_id
        , ie.hadm_id
        , ie.stay_id
        , ie.intime
        , ie.outtime

        , gcs.gcs_min
        , vital.heart_rate_max
        , vital.heart_rate_min
        , vital.sbp_max
        , vital.sbp_min

        -- this value is non-null iff the patient is on vent/cpap
        , pf.pao2fio2_vent_min

        , labs.bun_max
        , labs.bun_min
        , labs.wbc_max
        , labs.wbc_min
        , labs.bilirubin_total_max AS bilirubin_max
        , labs.creatinine_max
        , labs.pt_min
        , labs.pt_max
        , labs.platelets_min AS platelet_min

        , uo.urineoutput

    FROM `physionet-data.mimiciv_icu.icustays` ie
    INNER JOIN `physionet-data.mimiciv_hosp.admissions` adm
        ON ie.hadm_id = adm.hadm_id
    INNER JOIN `physionet-data.mimiciv_hosp.patients` pat
        ON ie.subject_id = pat.subject_id

    -- join to above view to get pao2/fio2 ratio
    LEFT JOIN pafi2 pf
        ON ie.stay_id = pf.stay_id

    -- join to custom tables to get more data....
    LEFT JOIN `physionet-data.mimiciv_derived.first_day_gcs` gcs
        ON ie.stay_id = gcs.stay_id
    LEFT JOIN `physionet-data.mimiciv_derived.first_day_vitalsign` vital
        ON ie.stay_id = vital.stay_id
    LEFT JOIN `physionet-data.mimiciv_derived.first_day_urine_output` uo
        ON ie.stay_id = uo.stay_id
    LEFT JOIN `physionet-data.mimiciv_derived.first_day_lab` labs
        ON ie.stay_id = labs.stay_id
)

, scorecomp AS (
    SELECT
        cohort.*
  -- Below code calculates the component scores needed for SAPS

        -- neurologic
        , CASE
            WHEN gcs_min IS NULL THEN null
            WHEN gcs_min < 3 THEN null -- erroneous value/on trach
            WHEN gcs_min <= 5 THEN 5
            WHEN gcs_min <= 8 THEN 3
            WHEN gcs_min <= 13 THEN 1
            ELSE 0
        END AS neurologic

        -- cardiovascular
        , CASE
            WHEN heart_rate_max IS NULL
                AND sbp_min IS NULL THEN null
            WHEN heart_rate_min < 30 THEN 5
            WHEN sbp_min < 40 THEN 5
            WHEN sbp_min < 70 THEN 3
            WHEN sbp_max >= 270 THEN 3
            WHEN heart_rate_max >= 140 THEN 1
            WHEN sbp_max >= 240 THEN 1
            WHEN sbp_min < 90 THEN 1
            ELSE 0
        END AS cardiovascular

        -- renal
        , CASE
            WHEN bun_max IS NULL
                OR urineoutput IS NULL
                OR creatinine_max IS NULL
                THEN null
            WHEN urineoutput < 500.0 THEN 5
            WHEN bun_max >= 56.0 THEN 5
            WHEN creatinine_max >= 1.60 THEN 3
            WHEN urineoutput < 750.0 THEN 3
            WHEN bun_max >= 28.0 THEN 3
            WHEN urineoutput >= 10000.0 THEN 3
            WHEN creatinine_max >= 1.20 THEN 1
            WHEN bun_max >= 17.0 THEN 1
            WHEN bun_max >= 7.50 THEN 1
            ELSE 0
        END AS renal

        -- pulmonary
        , CASE
            WHEN pao2fio2_vent_min IS NULL THEN 0
            WHEN pao2fio2_vent_min >= 150 THEN 1
            WHEN pao2fio2_vent_min < 150 THEN 3
            ELSE null
        END AS pulmonary

        -- hematologic
        , CASE
            WHEN wbc_max IS NULL
                AND platelet_min IS NULL
                THEN null
            WHEN wbc_min < 1.0 THEN 3
            WHEN wbc_min < 2.5 THEN 1
            WHEN platelet_min < 50.0 THEN 1
            WHEN wbc_max >= 50.0 THEN 1
            ELSE 0
        END AS hematologic

        -- hepatic
        -- We have defined the "standard" PT as 12 seconds.
        -- This is an assumption and subsequent analyses may be
        -- affected by this assumption.
        , CASE
            WHEN pt_max IS NULL
                AND bilirubin_max IS NULL
                THEN null
            WHEN bilirubin_max >= 2.0 THEN 1
            WHEN pt_max > (12 + 3) THEN 1
            WHEN pt_min < (12 * 0.25) THEN 1
            ELSE 0
        END AS hepatic

    FROM cohort
)

SELECT 
    demographic.subject_id
    , demographic.stay_id
    , demographic.intime
    , demographic.age
    , demographic.gender
    , vitalsign.heart_rate_min
    , vitalsign.heart_rate_max
    , vitalsign.heart_rate_mean
    , vitalsign.sbp_min
    , vitalsign.sbp_max
    , vitalsign.sbp_mean
    , vitalsign.dbp_min
    , vitalsign.dbp_max
    , vitalsign.dbp_mean
    , vitalsign.mbp_min
    , vitalsign.mbp_max
    , vitalsign.mbp_mean
    , vitalsign.resp_rate_min
    , vitalsign.resp_rate_max
    , vitalsign.resp_rate_mean
    , vitalsign.temperature_min
    , vitalsign.temperature_max
    , vitalsign.temperature_mean
    , vitalsign.spo2_min
    , vitalsign.spo2_max
    , vitalsign.spo2_mean
    , vitalsign.glucose_min
    , vitalsign.glucose_max
    , vitalsign.glucose_mean
    , urinefinal.urineoutput
    , gcs.gcs_value
    , gcs.gcs_motor
    , gcs.gcs_verbal
    , gcs.gcs_eyes
    , gcs.gcs_unable
    , vs_final.ventilation_status
    , COALESCE(neurologic, 0)
    + COALESCE(cardiovascular, 0)
    + COALESCE(renal, 0)
    + COALESCE(pulmonary, 0)
    + COALESCE(hematologic, 0)
    + COALESCE(hepatic, 0)
    AS lods
    , neurologic
    , cardiovascular
    , renal
    , pulmonary
    , hematologic
    , hepatic
    FROM demographic
    JOIN vitalsign 
        ON demographic.stay_id = vitalsign.stay_id
    JOIN urinefinal
        ON demographic.stay_id = urinefinal.stay_id
    JOIN gcs
        ON demographic.stay_id = gcs.stay_id
            AND gcs.gcs_seq = 1
    JOIN vs_final
        ON demographic.stay_id = vs_final.stay_id
            AND vs_final.starttime >= DATETIME_SUB(demographic.intime, INTERVAL '6' HOUR)
            AND vs_final.starttime <= DATETIME_ADD(demographic.intime, INTERVAL '1' DAY)
            AND vs_final.ventilation_seq = 1
    LEFT JOIN scorecomp s
          ON demographic.stay_id = s.stay_id
    WHERE vitalsign.heart_rate_min IS NOT NULL
      AND vitalsign.sbp_min IS NOT NULL
      AND vitalsign.dbp_min IS NOT NULL
      AND vitalsign.mbp_min IS NOT NULL
      AND vitalsign.resp_rate_min IS NOT NULL
      AND vitalsign.temperature_min IS NOT NULL
      AND vitalsign.spo2_min IS NOT NULL
      AND urinefinal.urineoutput IS NOT NULL
      AND gcs.gcs_value IS NOT NULL
      AND gcs.gcs_motor IS NOT NULL
      AND gcs.gcs_eyes IS NOT NULL