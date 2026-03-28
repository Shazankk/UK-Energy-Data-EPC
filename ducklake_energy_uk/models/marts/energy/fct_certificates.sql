{{ config(materialized='table') }}

with source as (
    select * from {{ ref('stg_epc__domestic') }}
),

final as (
    select
        {{ dbt_utils.generate_surrogate_key(['certificate_id']) }} as certificate_key,
        certificate_id,
        {{ dbt_utils.generate_surrogate_key(['uprn']) }} as property_id,
        {{ dbt_utils.generate_surrogate_key(['postcode']) }} as location_id,
        
        -- Ratings
        energy_rating_current,
        energy_rating_potential,
        energy_efficiency_current,
        energy_efficiency_potential,
        
        -- Environmental
        env_impact_current,
        env_impact_potential,
        energy_consumption_current,
        energy_consumption_potential,
        co2_emissions_current_tonnes_per_year,
        co2_emissions_potential_tonnes_per_year,
        
        -- Costs
        lighting_cost_current,
        heating_cost_current,
        hot_water_cost_current,
        
        -- Dates
        inspection_at,
        lodgement_at,
        
        -- Metadata
        source_file
        
    from source
    where certificate_id is not null
)

select * from final
