{{ config(materialized='view') }}

with potential_gaps as (
    select
        f.certificate_id,
        p.uprn,
        p.property_type,
        p.built_form,
        f.energy_efficiency_current,
        f.energy_efficiency_potential,
        (f.energy_efficiency_potential - f.energy_efficiency_current) as efficiency_gap,
        (f.co2_emissions_current_tonnes_per_year - f.co2_emissions_potential_tonnes_per_year) as co2_reduction_potential_tonnes
    from {{ ref('fct_certificates') }} f
    join {{ ref('dim_properties') }} p on f.property_id = p.property_id
)

select
    *
from potential_gaps
where efficiency_gap > 30  -- Focus on high-impact properties
order by efficiency_gap desc
