with source as (

    select * from {{ source('f1', 'circuits') }}

),

renamed as (

    select
        cast(circuitId as integer) as circuit_id,
        circuitRef as circuit_ref,
        name as circuit_name,
        location as locality,
        country,
        cast(lat as double) as latitude,
        cast(lng as double) as longitude,
        cast(alt as integer) as altitude_m
    from source

)

select * from renamed
