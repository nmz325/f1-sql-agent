with results as (

    select * from {{ ref('stg_f1__results') }}

),

status as (

    select * from {{ ref('stg_f1__status') }}

),

result_codes as (

    select * from {{ ref('result_codes') }}

),

time_corrections as (

    select * from {{ ref('race_time_corrections') }}

),

enriched as (

    select
        results.result_id,
        results.race_id,
        results.driver_id,
        results.constructor_id,
        results.car_number,
        results.grid_position,
        results.finish_position,
        results.position_order,
        results.position_text,
        -- Numeric codes are finishing positions; letter codes come from the seed.
        case
            when regexp_full_match(results.position_text, '\d+') then 'classified'
            else result_codes.result_type
        end as result_type,
        coalesce(result_codes.is_race_start, true) as is_race_start,
        results.finish_position is not null as is_classified,
        coalesce(results.finish_position = 1, false) as is_win,
        coalesce(results.finish_position <= 3, false) as is_podium,
        results.points,
        results.laps_completed,
        -- Official total time: classified drivers only, known source errors fixed.
        case
            when results.finish_position is not null
                then results.race_time_ms_raw + coalesce(time_corrections.time_offset_ms, 0)
        end as race_time_ms,
        results.fastest_lap_number,
        results.fastest_lap_rank,
        results.fastest_lap_time_ms,
        results.fastest_lap_speed_kph,
        status.status
    from results
    left join status on status.status_id = results.status_id
    left join result_codes on result_codes.position_text = results.position_text
    left join time_corrections on time_corrections.race_id = results.race_id

)

select
    *,
    -- Gap to the official winner (finish_position = 1), not to the lowest time.
    race_time_ms
        - max(case when finish_position = 1 then race_time_ms end)
            over (partition by race_id) as gap_to_winner_ms
from enriched
