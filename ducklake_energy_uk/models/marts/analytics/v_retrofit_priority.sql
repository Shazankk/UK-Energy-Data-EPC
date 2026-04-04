{{ config(materialized='table') }}

/*
Retrofit Priority Score — per property type × construction age band × local authority
======================================================================================

Measures how urgent and impactful it would be to retrofit a given group of properties.

Score components (0-100 each, globally normalised):
  - current_inefficiency (35%): How far below perfect is this property today?
        = (100 - energy_efficiency_current)
        A G-rated property (~20 SAP) scores ~80. An A-rated (~95 SAP) scores ~5.

  - efficiency_gap (40%): How much SAP improvement is achievable with upgrades?
        = (energy_efficiency_potential - energy_efficiency_current)
        Highest weight: a property with a large achievable gap is more actionable
        than one already close to its potential.

  - co2_saving (25%): How many tonnes of CO₂ per year would be saved post-retrofit?
        = (co2_current - co2_potential)
        Captures the climate impact of the retrofit.

Composite score = 0.35 * current_inefficiency_norm
               + 0.40 * efficiency_gap_norm
               + 0.25 * co2_saving_norm

Score is then multiplied by 100, resulting in a 0-100 scale.
A score of 100 = theoretically the most impactful retrofit candidate.
A score of 0   = property already at its maximum efficiency potential.

Grain: one row per (property_type × construction_age_band × county × local_authority)
*/

with base as (
    select
        p.property_type,
        p.construction_age_band,
        p.tenure,
        l.county,
        l.local_authority,
        f.energy_efficiency_current,
        f.energy_efficiency_potential,
        f.co2_emissions_current_tonnes_per_year      as co2_current,
        f.co2_emissions_potential_tonnes_per_year    as co2_potential,

        -- Raw score components per certificate
        (100.0 - f.energy_efficiency_current)                                   as current_inefficiency,
        (f.energy_efficiency_potential - f.energy_efficiency_current)           as efficiency_gap,
        (f.co2_emissions_current_tonnes_per_year
            - f.co2_emissions_potential_tonnes_per_year)                        as co2_saving

    from {{ ref('fct_certificates') }} f
    join {{ ref('dim_properties') }} p on f.property_id = p.property_id
    join {{ ref('dim_locations') }}   l on f.location_id = l.location_id

    where f.energy_efficiency_current  > 0
      and f.energy_efficiency_potential > f.energy_efficiency_current
      and f.co2_emissions_current_tonnes_per_year is not null
      and p.property_type              is not null
      and p.construction_age_band      is not null
      and l.local_authority            is not null
),

-- Compute global maxima for normalisation — ensures all scores are on the same 0-1 scale
global_stats as (
    select
        max(current_inefficiency) as max_inefficiency,
        max(efficiency_gap)       as max_efficiency_gap,
        max(co2_saving)           as max_co2_saving
    from base
),

scored as (
    select
        b.*,
        -- Normalised components (0-1)
        b.current_inefficiency / nullif(s.max_inefficiency,    0) as inefficiency_norm,
        b.efficiency_gap       / nullif(s.max_efficiency_gap,  0) as gap_norm,
        b.co2_saving           / nullif(s.max_co2_saving,      0) as co2_norm,

        -- Composite retrofit priority score (0-100)
        round(
            (0.35 * (b.current_inefficiency / nullif(s.max_inefficiency,   0))
           + 0.40 * (b.efficiency_gap       / nullif(s.max_efficiency_gap, 0))
           + 0.25 * (b.co2_saving           / nullif(s.max_co2_saving,     0))
            ) * 100,
        1) as retrofit_priority_score

    from base b
    cross join global_stats s
)

select
    property_type,
    construction_age_band,
    tenure,
    county,
    local_authority,

    count(*)                                              as property_count,

    -- Efficiency metrics
    round(avg(energy_efficiency_current),  1)            as avg_efficiency_current,
    round(avg(energy_efficiency_potential),1)            as avg_efficiency_potential,
    round(avg(efficiency_gap),             1)            as avg_efficiency_gap,

    -- CO₂ metrics
    round(avg(co2_current),                3)            as avg_co2_current_tonnes,
    round(avg(co2_saving),                 3)            as avg_co2_saving_per_property_tonnes,
    round(sum(co2_saving),                 1)            as total_co2_saving_potential_tonnes,

    -- The composite score
    round(avg(retrofit_priority_score),    1)            as avg_retrofit_priority_score,
    round(max(retrofit_priority_score),    1)            as max_retrofit_priority_score

from scored
group by 1, 2, 3, 4, 5
