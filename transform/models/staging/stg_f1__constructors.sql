with source as (

    select * from {{ source('f1', 'constructors') }}

),

renamed as (

    select
        cast(constructorId as integer) as constructor_id,
        constructorRef as constructor_ref,
        name as constructor_name,
        nationality as constructor_nationality
    from source

)

select * from renamed
