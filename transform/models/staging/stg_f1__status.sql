with source as (

    select * from {{ source('f1', 'status') }}

),

renamed as (

    select
        cast(statusId as integer) as status_id,
        status
    from source

)

select * from renamed
