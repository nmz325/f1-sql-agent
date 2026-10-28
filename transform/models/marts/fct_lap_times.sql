with lap_times as (

    select * from {{ ref('stg_f1__lap_times') }}

),

race_entries as (

    select race_id, driver_id, constructor_id
    from {{ ref('stg_f1__results') }}

)

select
    lap_times.race_id,
    lap_times.driver_id,
    race_entries.constructor_id,
    lap_times.lap_number,
    lap_times.position_on_lap,
    lap_times.lap_time_ms
from lap_times
left join race_entries
    on race_entries.race_id = lap_times.race_id
    and race_entries.driver_id = lap_times.driver_id
