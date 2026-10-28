with source as (

    select * from {{ source('f1', 'constructor_standings') }}

),

renamed as (

    select
        cast(constructorStandingsId as integer) as constructor_standing_id,
        cast(raceId as integer) as race_id,
        cast(constructorId as integer) as constructor_id,
        cast(points as double) as championship_points,
        cast("position" as integer) as championship_position,
        positionText as championship_position_text,
        cast(wins as integer) as wins_to_date
    from source

)

select * from renamed
