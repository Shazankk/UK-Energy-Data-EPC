{{ config(materialized='view') }}

with combined as (
    select 
        f.*,
        l.county,
        l.post_town,
        l.local_authority
    from {{ ref('fct_certificates') }} f
    join {{ ref('dim_locations') }} l on f.location_id = l.location_id
)

select
    county,
    local_authority,
    post_town,
    count(*) as total_certificates,
    avg(energy_efficiency_current) as avg_current_efficiency,
    avg(energy_efficiency_potential) as avg_potential_efficiency,
    avg(co2_emissions_current_tonnes_per_year) as avg_co2_current,
    avg(co2_emissions_potential_tonnes_per_year) as avg_co2_potential,
    -- Calculate the average gap
    avg(energy_efficiency_potential - energy_efficiency_current) as avg_efficiency_gap
from combined
group by 1, 2, 3
order by avg_current_efficiency desc
