with source as (

    select * from {{ source('f1', 'pit_stops') }}

),

renamed as (

    select
        cast(raceId as integer) as race_id,
        cast(driverId as integer) as driver_id,
        cast(stop as integer) as stop_number,
        cast(lap as integer) as lap_number,
        -- Time spent in the pit lane. Red-flag stoppages can produce very long values.
        cast(milliseconds as integer) as pit_stop_duration_ms
    from source

)

select * from renamed
