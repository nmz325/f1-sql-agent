with source as (

    select * from {{ source('f1', 'races') }}

),

renamed as (

    select
        cast(raceId as integer) as race_id,
        cast("year" as integer) as season,
        cast("round" as integer) as round,
        cast(circuitId as integer) as circuit_id,
        name as race_name,
        cast("date" as date) as race_date
    from source

)

select * from renamed
