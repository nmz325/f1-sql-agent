with source as (

    select * from {{ source('f1', 'sprint_results') }}

),

renamed as (

    select
        cast(resultId as integer) as sprint_result_id,
        cast(raceId as integer) as race_id,
        cast(driverId as integer) as driver_id,
        cast(constructorId as integer) as constructor_id,
        cast(number as integer) as car_number,
        cast(grid as integer) as grid_position,
        cast("position" as integer) as finish_position,
        positionText as position_text,
        cast(positionOrder as integer) as position_order,
        cast(points as double) as points,
        cast(laps as integer) as laps_completed,
        cast(milliseconds as bigint) as sprint_time_ms_raw,
        cast(fastestLap as integer) as fastest_lap_number,
        {{ lap_time_to_ms('fastestLapTime') }} as fastest_lap_time_ms,
        cast(statusId as integer) as status_id
    from source

)

select * from renamed
