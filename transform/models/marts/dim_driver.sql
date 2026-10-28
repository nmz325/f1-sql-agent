with drivers as (

    select * from {{ ref('stg_f1__drivers') }}

)

select
    driver_id,
    driver_ref,
    driver_code,
    permanent_number,
    first_name,
    last_name,
    full_name,
    date_of_birth,
    driver_nationality
from drivers
