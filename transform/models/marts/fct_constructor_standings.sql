with standings as (

    select * from {{ ref('stg_f1__constructor_standings') }}

)

select
    constructor_standing_id,
    race_id,
    constructor_id,
    championship_points,
    championship_position,
    championship_position_text,
    wins_to_date
from standings
