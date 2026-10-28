with sprint_results as (

    select * from {{ ref('stg_f1__sprint_results') }}

),

status as (

    select * from {{ ref('stg_f1__status') }}

),

result_codes as (

    select * from {{ ref('result_codes') }}

),

enriched as (

    select
        sprint_results.sprint_result_id,
        sprint_results.race_id,
        sprint_results.driver_id,
        sprint_results.constructor_id,
        sprint_results.car_number,
        sprint_results.grid_position,
        sprint_results.finish_position,
        sprint_results.position_order,
        sprint_results.position_text,
        case
            when regexp_full_match(sprint_results.position_text, '\d+') then 'classified'
            else result_codes.result_type
        end as result_type,
        coalesce(result_codes.is_race_start, true) as is_sprint_start,
        sprint_results.finish_position is not null as is_classified,
        coalesce(sprint_results.finish_position = 1, false) as is_sprint_win,
        sprint_results.points,
        sprint_results.laps_completed,
        case
            when sprint_results.finish_position is not null
                then sprint_results.sprint_time_ms_raw
        end as sprint_time_ms,
        sprint_results.fastest_lap_number,
        sprint_results.fastest_lap_time_ms,
        status.status
    from sprint_results
    left join status on status.status_id = sprint_results.status_id
    left join result_codes on result_codes.position_text = sprint_results.position_text

)

select
    *,
    sprint_time_ms
        - max(case when finish_position = 1 then sprint_time_ms end)
            over (partition by race_id) as gap_to_winner_ms
from enriched
