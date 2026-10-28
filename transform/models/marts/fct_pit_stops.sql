with pit_stops as (

    select * from {{ ref('stg_f1__pit_stops') }}

),

race_entries as (

    select race_id, driver_id, constructor_id
    from {{ ref('stg_f1__results') }}

)

select
    pit_stops.race_id,
    pit_stops.driver_id,
    race_entries.constructor_id,
    pit_stops.stop_number,
    pit_stops.lap_number,
    pit_stops.pit_stop_duration_ms
from pit_stops
left join race_entries
    on race_entries.race_id = pit_stops.race_id
    and race_entries.driver_id = pit_stops.driver_id
