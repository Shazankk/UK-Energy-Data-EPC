{{ config(materialized='table') }}

with source as (
    select * from {{ ref('stg_epc__domestic') }}
),

ranked_properties as (
    select
        uprn,
        property_type,
        built_form,
        construction_age_band,
        tenure,
        total_floor_area_sqm,
        count_habitable_rooms,
        count_heated_rooms,
        count_extensions,
        -- Take the latest assessment date for each property to get the most recent state
        row_number() over (
            partition by uprn 
            order by inspection_at desc, lodgement_at desc
        ) as property_rank
    from source
    where uprn is not null
),

latest_characteristics as (
    select
        {{ dbt_utils.generate_surrogate_key(['uprn']) }} as property_id,
        uprn,
        property_type,
        built_form,
        construction_age_band,
        tenure,
        total_floor_area_sqm,
        count_habitable_rooms,
        count_heated_rooms,
        count_extensions
    from ranked_properties
    where property_rank = 1
)

select * from latest_characteristics
