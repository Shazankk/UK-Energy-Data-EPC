{{ config(materialized='table') }}

with source as (
    select * from {{ ref('stg_epc__domestic') }}
),

unique_locations as (
    select distinct
        postcode,
        post_town,
        local_authority,
        constituency,
        county
    from source
    where postcode is not null
),

final as (
    select
        {{ dbt_utils.generate_surrogate_key(['postcode']) }} as location_id,
        postcode,
        post_town,
        local_authority,
        constituency,
        county
    from unique_locations
)

select * from final
