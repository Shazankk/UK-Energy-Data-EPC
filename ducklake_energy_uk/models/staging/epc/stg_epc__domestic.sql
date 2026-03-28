with source as (
    select * from {{ source('epc_raw', 'epc_domestic') }}
),

renamed as (
    select
        -- Identifiers
        {{ dbt_utils.generate_surrogate_key(['lmk_key']) }} as certificate_key,
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
        
        -- Ratings (Cleaned A-G)
        case 
            when current_energy_rating in ('A', 'B', 'C', 'D', 'E', 'F', 'G') then current_energy_rating 
            else null 
        end as energy_rating_current,
        case 
            when potential_energy_rating in ('A', 'B', 'C', 'D', 'E', 'F', 'G') then potential_energy_rating 
            else null 
        end as energy_rating_potential,
        cast(current_energy_efficiency as integer) as energy_efficiency_current,
        cast(potential_energy_efficiency as integer) as energy_efficiency_potential,
        
        -- Property Characteristics
        property_type,
        built_form,
        cast(total_floor_area as decimal(10,2)) as total_floor_area_sqm,
        
        -- Standardized Construction Age Bands
        case
            when construction_age_band like '%before 1900%' then 'Pre-1900'
            when construction_age_band like '%1900-1929%' then '1900-1929'
            when construction_age_band like '%1930-1949%' then '1930-1949'
            when construction_age_band like '%1950-1966%' or construction_age_band like '%1967-1975%' then '1950-1975'
            when construction_age_band like '%1976-1982%' or construction_age_band like '%1983-1990%' then '1976-1990'
            when construction_age_band like '%1991-1995%' or construction_age_band like '%1996-2002%' then '1991-2002'
            when construction_age_band like '%2003-2006%' or construction_age_band like '%2007%' then '2003-2011'
            when construction_age_band like '%2012%' or construction_age_band like '%onwards%' then '2012-Present'
            when construction_age_band ~ '^[0-9]{4}$' then 
                case 
                    when cast(construction_age_band as integer) < 1900 then 'Pre-1900'
                    when cast(construction_age_band as integer) between 1900 and 1929 then '1900-1929'
                    when cast(construction_age_band as integer) between 1930 and 1949 then '1930-1949'
                    when cast(construction_age_band as integer) between 1950 and 1975 then '1950-1975'
                    when cast(construction_age_band as integer) between 1976 and 1990 then '1976-1990'
                    when cast(construction_age_band as integer) between 1991 and 2002 then '1991-2002'
                    when cast(construction_age_band as integer) between 2003 and 2011 then '2003-2011'
                    else '2012-Present'
                end
            else 'Unknown'
        end as construction_age_band,
        
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
        
        -- Standardized Fuel Types
        case
            when lower(main_fuel) like '%gas%' then 'Mains Gas'
            when lower(main_fuel) like '%electricity%' then 'Electricity'
            when lower(main_fuel) like '%oil%' then 'Heating Oil'
            when lower(main_fuel) like '%lpg%' then 'LPG'
            when lower(main_fuel) like '%coal%' or lower(main_fuel) like '%anthracite%' then 'Solid Fuel'
            when lower(main_fuel) like '%wood%' or lower(main_fuel) like '%biomass%' then 'Biomass'
            else 'Other/Unknown'
        end as main_fuel,
        
        floor_level,
        cast(number_habitable_rooms as integer) as count_habitable_rooms,
        cast(number_heated_rooms as integer) as count_heated_rooms,
        cast(extension_count as integer) as count_extensions,
        
        -- Metadata
        filename as source_file
        
    from source
    where current_energy_rating is not null
      and current_energy_rating != 'INVALID!'
      and uprn is not null
      and inspection_date is not null
)

select * from renamed
