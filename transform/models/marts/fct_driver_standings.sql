with standings as (

    select * from {{ ref('stg_f1__driver_standings') }}

)

select
    driver_standing_id,
    race_id,
    driver_id,
    championship_points,
    championship_position,
    championship_position_text,
    wins_to_date
from standings
