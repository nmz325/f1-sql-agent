with qualifying as (

    select * from {{ ref('stg_f1__qualifying') }}

)

select
    qualifying_id,
    race_id,
    driver_id,
    constructor_id,
    car_number,
    qualifying_position,
    coalesce(qualifying_position = 1, false) as is_pole,
    q1_time_ms,
    q2_time_ms,
    q3_time_ms,
    -- least() ignores NULLs: best time among the sessions the driver took part in.
    least(q1_time_ms, q2_time_ms, q3_time_ms) as best_time_ms
from qualifying
