-- A classified race time below 50 s per lap is impossible (the fastest lap in the
-- dataset is about 55 s) and points to a source error such as Belgium 2024.
select result_id, race_id, laps_completed, race_time_ms
from {{ ref('fct_race_results') }}
where race_time_ms is not null
  and laps_completed > 0
  and race_time_ms / laps_completed < 50000
