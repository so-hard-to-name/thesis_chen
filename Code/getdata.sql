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
, gcsscore AS (
    WITH gcs_final AS (
    SELECT
        ie.subject_id, ie.stay_id
        , g.gcs
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

    SELECT
        ie.subject_id
        , ie.stay_id
        -- The minimum GCS is determined by the above row partition
        -- we only join if gcs_seq = 1
        , gcs AS gcs_min
        , gcs_motor
        , gcs_verbal
        , gcs_eyes
        , gcs_unable
    FROM `physionet-data.mimiciv_icu.icustays` ie
    LEFT JOIN gcs_final gs
        ON ie.stay_id = gs.stay_id
            AND gs.gcs_seq = 1
    )
, urine AS (
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
, ventilation AS (      -- Not first day, add time settings
    WITH tm AS (
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
        FROM vd2
        GROUP BY stay_id, vent_seq
        HAVING MIN(charttime) != MAX(charttime)
        )

SELECT 
    demographic.subject_id
    , demographic.hadm_id
    , demographic.admittime
    , demographic.age
    , demographic.gender
    ,
    FROM demographic
