with source as (

    select * from {{ source('f1', 'qualifying') }}

),

renamed as (

    select
        cast(qualifyId as integer) as qualifying_id,
        cast(raceId as integer) as race_id,
        cast(driverId as integer) as driver_id,
        cast(constructorId as integer) as constructor_id,
        cast(number as integer) as car_number,
        cast("position" as integer) as qualifying_position,
        {{ lap_time_to_ms('q1') }} as q1_time_ms,
        {{ lap_time_to_ms('q2') }} as q2_time_ms,
        {{ lap_time_to_ms('q3') }} as q3_time_ms
    from source

)

select * from renamed
