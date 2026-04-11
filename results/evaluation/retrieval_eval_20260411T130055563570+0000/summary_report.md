# Retrieval Evaluation Summary

- Evaluation ID: `retrieval_eval_20260411T130055563570+0000`
- Questions: `24`
- Top-k compared: `3`
- Scoring version: `retrieval_eval_v2_dual_track`

## Question Set Analysis

- Category distribution: `causal`=5, `direct_factual`=7, `mitigation`=4, `multi_hop`=5, `trade_off`=3
- Track distribution: `exact_retrieval`=8, `graph_stressing`=16
- Explanation-oriented questions: `16`

## Aggregate Scores

| Method | Overall | Relevance | Directness | Groundedness | Correctness | Top-1 Doc | Top-k Gold Chunk | Wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tfidf | 0.616 | 0.673 | 0.614 | 0.575 | 0.601 | 0.958 | 0.625 | 13 |
| hybrid_graph | 0.554 | 0.610 | 0.567 | 0.503 | 0.536 | 0.792 | 0.542 | 7 |
| dense | 0.516 | 0.579 | 0.522 | 0.465 | 0.497 | 0.792 | 0.458 | 5 |
| graph | 0.466 | 0.506 | 0.498 | 0.408 | 0.452 | 0.625 | 0.375 | 7 |

## Track Results

| Method | Exact Retrieval | Graph-Stressing | Exact Wins | Graph Wins |
| --- | ---: | ---: | ---: | ---: |
| tfidf | 0.621 | 0.613 | 6 | 7 |
| hybrid_graph | 0.436 | 0.613 | 1 | 6 |
| dense | 0.384 | 0.581 | 3 | 2 |
| graph | 0.220 | 0.589 | 0 | 7 |

## Per-Question Winners

| Question ID | Category | Track | Winners |
| --- | --- | --- | --- |
| q01_strategic_manufacturing_foreign_inputs | direct_factual | exact_retrieval | tfidf |
| q02_three_building_blocks | direct_factual | exact_retrieval | tfidf |
| q03_triple_a_qualities | direct_factual | exact_retrieval | dense, hybrid_graph, tfidf |
| q04_services_for_adaptable_supply_chains | direct_factual | exact_retrieval | dense |
| q05_four_external_transport_disruptions | direct_factual | exact_retrieval | tfidf |
| q06_early_warning_examples | direct_factual | exact_retrieval | tfidf |
| q07_worldbank_spread_or_transfer | direct_factual | exact_retrieval | dense |
| q08_diversification_improves_resilience | causal | graph_stressing | graph, hybrid_graph |
| q09_trade_cooperation_improves_resilience | causal | graph_stressing | graph, tfidf |
| q10_relocalisation_vs_resilience | causal | graph_stressing | tfidf |
| q11_digitalisation_improves_resilience | causal | graph_stressing | graph |
| q12_collaboration_strengthens_resilience | causal | graph_stressing | graph |
| q13_wto_mitigation_strategies | mitigation | exact_retrieval | tfidf |
| q14_strengthen_service_sectors | mitigation | graph_stressing | tfidf |
| q15_trade_facilitation_agility | mitigation | graph_stressing | dense, hybrid_graph |
| q16_early_warning_and_intervention | mitigation | graph_stressing | tfidf |
| q17_efficiency_resilience_tradeoff | trade_off | graph_stressing | tfidf |
| q18_resilience_efficiency_sustainability_tradeoffs | trade_off | graph_stressing | hybrid_graph, tfidf |
| q19_aligning_sustainability_efficiency_resilience | trade_off | graph_stressing | dense, graph, hybrid_graph |
| q20_crm_concentration_export_restrictions | multi_hop | graph_stressing | tfidf |
| q21_trade_facilitation_new_suppliers | multi_hop | graph_stressing | graph |
| q22_information_sharing_visibility_disruptions | multi_hop | graph_stressing | hybrid_graph |
| q23_sustainability_rules_and_foreign_suppliers | multi_hop | graph_stressing | hybrid_graph |
| q24_trade_in_services_and_resilience | multi_hop | graph_stressing | graph |

## Method Notes

### tfidf
- Strongest categories: `trade_off` (0.730), `direct_factual` (0.686)
- Weakest categories: `mitigation` (0.515), `causal` (0.540)
- Track performance: `exact_retrieval` (0.621), `graph_stressing` (0.613)
- Collective support: concepts=0.636, relations=0.749, reasoning=0.538
- Failure modes: `competitive`=12, `concept_coverage_gap`=5, `indirect_context_only`=3, `weak_evidence_support`=3, `broad_context_not_direct`=1
- Questions won: q01_strategic_manufacturing_foreign_inputs, q02_three_building_blocks, q03_triple_a_qualities, q05_four_external_transport_disruptions, q06_early_warning_examples, q09_trade_cooperation_improves_resilience, q10_relocalisation_vs_resilience, q13_wto_mitigation_strategies, q14_strengthen_service_sectors, q16_early_warning_and_intervention, q17_efficiency_resilience_tradeoff, q18_resilience_efficiency_sustainability_tradeoffs, q20_crm_concentration_export_restrictions

### hybrid_graph
- Strongest categories: `trade_off` (0.733), `multi_hop` (0.701)
- Weakest categories: `mitigation` (0.374), `direct_factual` (0.499)
- Track performance: `exact_retrieval` (0.436), `graph_stressing` (0.613)
- Collective support: concepts=0.547, relations=0.830, reasoning=0.525
- Failure modes: `competitive`=13, `concept_coverage_gap`=5, `broad_context_not_direct`=2, `wrong_document_cluster`=2, `indirect_context_only`=1, `weak_evidence_support`=1
- Questions won: q03_triple_a_qualities, q08_diversification_improves_resilience, q15_trade_facilitation_agility, q18_resilience_efficiency_sustainability_tradeoffs, q19_aligning_sustainability_efficiency_resilience, q22_information_sharing_visibility_disruptions, q23_sustainability_rules_and_foreign_suppliers

### dense
- Strongest categories: `trade_off` (0.703), `multi_hop` (0.623)
- Weakest categories: `mitigation` (0.357), `direct_factual` (0.439)
- Track performance: `exact_retrieval` (0.384), `graph_stressing` (0.581)
- Collective support: concepts=0.499, relations=0.783, reasoning=0.474
- Failure modes: `competitive`=11, `concept_coverage_gap`=6, `broad_context_not_direct`=2, `indirect_context_only`=2, `wrong_document_cluster`=2, `weak_evidence_support`=1
- Questions won: q03_triple_a_qualities, q04_services_for_adaptable_supply_chains, q07_worldbank_spread_or_transfer, q15_trade_facilitation_agility, q19_aligning_sustainability_efficiency_resilience

### graph
- Strongest categories: `trade_off` (0.660), `causal` (0.640)
- Weakest categories: `direct_factual` (0.251), `mitigation` (0.303)
- Track performance: `exact_retrieval` (0.220), `graph_stressing` (0.589)
- Collective support: concepts=0.347, relations=0.878, reasoning=0.460
- Failure modes: `competitive`=9, `wrong_document_cluster`=8, `indirect_context_only`=4, `concept_coverage_gap`=3
- Questions won: q08_diversification_improves_resilience, q09_trade_cooperation_improves_resilience, q11_digitalisation_improves_resilience, q12_collaboration_strengthens_resilience, q19_aligning_sustainability_efficiency_resilience, q21_trade_facilitation_new_suppliers, q24_trade_in_services_and_resilience
