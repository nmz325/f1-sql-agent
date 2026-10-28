with races as (

    select * from {{ ref('stg_f1__races') }}

),

circuits as (

    select * from {{ ref('stg_f1__circuits') }}

),

sprint_races as (

    select distinct race_id from {{ ref('stg_f1__sprint_results') }}

)

select
    races.race_id,
    races.season,
    races.round,
    races.race_name,
    races.race_date,
    races.round = max(races.round) over (partition by races.season) as is_season_finale,
    sprint_races.race_id is not null as has_sprint,
    circuits.circuit_id,
    circuits.circuit_ref,
    circuits.circuit_name,
    circuits.locality as circuit_locality,
    circuits.country as circuit_country,
    circuits.latitude as circuit_latitude,
    circuits.longitude as circuit_longitude,
    circuits.altitude_m as circuit_altitude_m
from races
left join circuits on circuits.circuit_id = races.circuit_id
left join sprint_races on sprint_races.race_id = races.race_id
