-- The official winner must have the lowest race time among classified finishers.
with race_times as (

    select
        race_id,
        min(race_time_ms) as lowest_time_ms,
        min(case when finish_position = 1 then race_time_ms end) as winner_time_ms
    from {{ ref('fct_race_results') }}
    group by race_id

)

select *
from race_times
where winner_time_ms > lowest_time_ms
