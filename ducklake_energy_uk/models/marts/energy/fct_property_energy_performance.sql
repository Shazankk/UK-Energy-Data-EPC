{{ config(materialized='table') }}

with staging as (
    select * from {{ ref('stg_epc__domestic') }}
),

calculated_gains as (
    select
        certificate_id,
        uprn,
        address1,
        postcode,
        energy_rating_current,
        energy_rating_potential,
        energy_efficiency_current,
        energy_efficiency_potential,
        (energy_efficiency_potential - energy_efficiency_current) as efficiency_gain_potential,
        total_floor_area_sqm,
        inspection_at,
        lodgement_at,
        property_type,
        built_form,
        heating_cost_current,
        hot_water_cost_current,
        lighting_cost_current,
        (heating_cost_current + hot_water_cost_current + lighting_cost_current) as total_estimated_annual_cost_current
    from staging
)

select * from calculated_gains
