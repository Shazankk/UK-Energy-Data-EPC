with source as (
    select * from {{ source('epc_raw', 'epc_domestic') }}
),

renamed as (
    select
        -- Identifiers
        lmk_key as certificate_id,
        uprn as uprn,
        building_reference_number as building_ref,
        
        -- Address Information
        address1,
        address2,
        address3,
        postcode,
        posttown as post_town,
        local_authority_label as local_authority,
        constituency_label as constituency,
        county,
        
        -- Energy Performance (Ratings)
        current_energy_rating as energy_rating_current,
        potential_energy_rating as energy_rating_potential,
        cast(current_energy_efficiency as integer) as energy_efficiency_current,
        cast(potential_energy_efficiency as integer) as energy_efficiency_potential,
        
        -- Property Characteristics
        property_type,
        built_form,
        cast(total_floor_area as decimal(10,2)) as total_floor_area_sqm,
        construction_age_band,
        tenure,
        transaction_type,
        
        -- Dates
        cast(inspection_date as date) as inspection_at,
        cast(lodgement_date as date) as lodgement_at,
        
        -- Environment & Energy
        cast(environment_impact_current as integer) as env_impact_current,
        cast(environment_impact_potential as integer) as env_impact_potential,
        cast(energy_consumption_current as integer) as energy_consumption_current,
        cast(energy_consumption_potential as integer) as energy_consumption_potential,
        cast(co2_emissions_current as decimal(10,2)) as co2_emissions_current_tonnes_per_year,
        cast(co2_emissions_potential as decimal(10,2)) as co2_emissions_potential_tonnes_per_year,
        
        -- Costs
        cast(lighting_cost_current as decimal(10,2)) as lighting_cost_current,
        cast(heating_cost_current as decimal(10,2)) as heating_cost_current,
        cast(hot_water_cost_current as decimal(10,2)) as hot_water_cost_current,
        
        -- Technical details
        main_fuel,
        floor_level,
        cast(number_habitable_rooms as integer) as count_habitable_rooms,
        cast(number_heated_rooms as integer) as count_heated_rooms,
        cast(extension_count as integer) as count_extensions,
        
        -- Metadata
        filename as source_file
        
    from source
)

select * from renamed
