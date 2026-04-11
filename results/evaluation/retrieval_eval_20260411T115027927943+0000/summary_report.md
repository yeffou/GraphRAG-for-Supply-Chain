# Retrieval Evaluation Summary

- Evaluation ID: `retrieval_eval_20260411T115027927943+0000`
- Questions: `24`
- Top-k compared: `3`
- Scoring version: `retrieval_eval_v1`

## Aggregate Scores

| Method | Overall | Relevance | Directness | Groundedness | Correctness | Wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tfidf | 0.564 | 0.615 | 0.513 | 0.565 | 0.562 | 15 |
| dense | 0.460 | 0.513 | 0.408 | 0.465 | 0.456 | 10 |
| graph | 0.246 | 0.280 | 0.216 | 0.242 | 0.246 | 3 |

## Per-Question Winners

| Question ID | Category | Winners |
| --- | --- | --- |
| q01_strategic_manufacturing_foreign_inputs | direct_factual | tfidf |
| q02_three_building_blocks | direct_factual | tfidf |
| q03_triple_a_qualities | direct_factual | dense, tfidf |
| q04_services_for_adaptable_supply_chains | direct_factual | dense |
| q05_four_external_transport_disruptions | direct_factual | tfidf |
| q06_early_warning_examples | direct_factual | tfidf |
| q07_worldbank_spread_or_transfer | direct_factual | dense |
| q08_diversification_improves_resilience | causal | dense |
| q09_trade_cooperation_improves_resilience | causal | tfidf |
| q10_relocalisation_vs_resilience | causal | tfidf |
| q11_digitalisation_improves_resilience | causal | dense |
| q12_collaboration_strengthens_resilience | causal | graph |
| q13_wto_mitigation_strategies | mitigation | tfidf |
| q14_strengthen_service_sectors | mitigation | tfidf |
| q15_trade_facilitation_agility | mitigation | graph |
| q16_early_warning_and_intervention | mitigation | tfidf |
| q17_efficiency_resilience_tradeoff | trade_off | dense, graph, tfidf |
| q18_resilience_efficiency_sustainability_tradeoffs | trade_off | tfidf |
| q19_aligning_sustainability_efficiency_resilience | trade_off | dense |
| q20_crm_concentration_export_restrictions | multi_hop | tfidf |
| q21_trade_facilitation_new_suppliers | multi_hop | tfidf |
| q22_information_sharing_visibility_disruptions | multi_hop | dense |
| q23_sustainability_rules_and_foreign_suppliers | multi_hop | dense |
| q24_trade_in_services_and_resilience | multi_hop | dense, tfidf |

## Method Notes

### tfidf
- Strongest categories: `direct_factual` (0.686), `trade_off` (0.623)
- Weakest categories: `causal` (0.407), `multi_hop` (0.520)
- Failure modes: `competitive`=15, `indirect_context_only`=7, `broad_context_not_direct`=1, `weak_evidence_support`=1
- Questions won: q01_strategic_manufacturing_foreign_inputs, q02_three_building_blocks, q03_triple_a_qualities, q05_four_external_transport_disruptions, q06_early_warning_examples, q09_trade_cooperation_improves_resilience, q10_relocalisation_vs_resilience, q13_wto_mitigation_strategies, q14_strengthen_service_sectors, q16_early_warning_and_intervention, q17_efficiency_resilience_tradeoff, q18_resilience_efficiency_sustainability_tradeoffs, q20_crm_concentration_export_restrictions, q21_trade_facilitation_new_suppliers, q24_trade_in_services_and_resilience

### dense
- Strongest categories: `trade_off` (0.628), `multi_hop` (0.555)
- Weakest categories: `mitigation` (0.274), `direct_factual` (0.439)
- Failure modes: `competitive`=10, `indirect_context_only`=8, `broad_context_not_direct`=2, `weak_evidence_support`=2, `wrong_document_cluster`=2
- Questions won: q03_triple_a_qualities, q04_services_for_adaptable_supply_chains, q07_worldbank_spread_or_transfer, q08_diversification_improves_resilience, q11_digitalisation_improves_resilience, q17_efficiency_resilience_tradeoff, q19_aligning_sustainability_efficiency_resilience, q22_information_sharing_visibility_disruptions, q23_sustainability_rules_and_foreign_suppliers, q24_trade_in_services_and_resilience

### graph
- Strongest categories: `causal` (0.364), `mitigation` (0.239)
- Weakest categories: `direct_factual` (0.182), `multi_hop` (0.230)
- Failure modes: `indirect_context_only`=8, `wrong_document_cluster`=8, `competitive`=4, `weak_evidence_support`=4
- Questions won: q12_collaboration_strengthens_resilience, q15_trade_facilitation_agility, q17_efficiency_resilience_tradeoff
