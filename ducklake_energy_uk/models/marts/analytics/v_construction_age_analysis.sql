{{ config(materialized='view') }}

with analytics as (
    select
        p.construction_age_band,
        p.property_type,
        count(*) as total_certificates,
        avg(f.energy_efficiency_current) as avg_efficiency,
        avg(f.energy_consumption_current) as avg_consumption,
        avg(f.co2_emissions_current_tonnes_per_year) as avg_co2
    from {{ ref('fct_certificates') }} f
    join {{ ref('dim_properties') }} p on f.property_id = p.property_id
    where p.construction_age_band is not null
    group by 1, 2
)

select
    *
from analytics
order by construction_age_band desc
