with source as (

    select * from {{ source('f1', 'drivers') }}

),

renamed as (

    select
        cast(driverId as integer) as driver_id,
        driverRef as driver_ref,
        code as driver_code,
        cast(number as integer) as permanent_number,
        forename as first_name,
        surname as last_name,
        concat_ws(' ', forename, surname) as full_name,
        cast(dob as date) as date_of_birth,
        nationality as driver_nationality
    from source

)

select * from renamed
