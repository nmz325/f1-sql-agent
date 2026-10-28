-- The official winner must have the lowest time among classified finishers,
-- in Grands Prix and in sprints.
with session_times as (

    select 'race' as session, race_id, finish_position, race_time_ms as time_ms
    from {{ ref('fct_race_results') }}

    union all

    select 'sprint', race_id, finish_position, sprint_time_ms
    from {{ ref('fct_sprint_results') }}

),

summary as (

    select
        session,
        race_id,
        min(time_ms) as lowest_time_ms,
        min(case when finish_position = 1 then time_ms end) as winner_time_ms
    from session_times
    group by session, race_id

)

select *
from summary
where winner_time_ms > lowest_time_ms
