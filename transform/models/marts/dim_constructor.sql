with constructors as (

    select * from {{ ref('stg_f1__constructors') }}

)

select
    constructor_id,
    constructor_ref,
    constructor_name,
    constructor_nationality
from constructors
