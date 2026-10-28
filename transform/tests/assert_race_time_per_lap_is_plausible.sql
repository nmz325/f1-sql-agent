-- A classified time below 50 s per lap is impossible (the fastest lap in the
-- dataset is about 55 s) and points to a source error such as Belgium 2024.
select 'race' as session, result_id as entry_id, race_id, laps_completed, race_time_ms as time_ms
from {{ ref('fct_race_results') }}
where race_time_ms is not null
  and laps_completed > 0
  and race_time_ms / laps_completed < 50000

union all

select 'sprint', sprint_result_id, race_id, laps_completed, sprint_time_ms
from {{ ref('fct_sprint_results') }}
where sprint_time_ms is not null
  and laps_completed > 0
  and sprint_time_ms / laps_completed < 50000
