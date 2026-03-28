{{ config(materialized='table') }}

with certificates as (
    select * from {{ ref('fct_certificates') }}
),

properties as (
    select * from {{ ref('dim_properties') }}
),

locations as (
    select * from {{ ref('dim_locations') }}
),

joined as (
    select
        f.energy_rating_current,
        f.energy_efficiency_current,
        f.co2_emissions_current_tonnes_per_year,
        p.property_type,
        p.tenure,
        p.construction_age_band,
        l.county,
        date_trunc('year', f.inspection_at) as inspection_year
    from certificates f
    left join properties p on f.property_id = p.property_id
    left join locations l on f.location_id = l.location_id
),

final as (
    select
        county,
        property_type,
        tenure,
        construction_age_band,
        inspection_year,
        count(*) as certificate_count,
        round(avg(energy_efficiency_current), 2) as avg_energy_efficiency,
        round(sum(co2_emissions_current_tonnes_per_year), 2) as total_co2_emissions
    from joined
    group by 1, 2, 3, 4, 5
)

select * from final
