# Evaluation results

## Cell prediction (held-out test games)

|                  |   roc_auc |   log_loss |   brier |   next_shot_hit_rate |
|:-----------------|----------:|-----------:|--------:|---------------------:|
| prior            |    0.5562 |     0.5522 |  0.1826 |               0.2687 |
| density (search) |    0.7473 |     0.8991 |  0.1864 |               0.6123 |
| neural           |    0.7894 |     0.4245 |  0.1467 |               0.6707 |
| neural + search  |    0.755  |     0.9585 |  0.1896 |               0.6396 |

## Shots to sink the fleet (300 identical boards)

| agent         |   mean |   std |   min |   median |   max |
|:--------------|-------:|------:|------:|---------:|------:|
| neural_search |  47.78 |  3.45 |    38 |     48   |    58 |
| density       |  47.82 |  3.61 |    37 |     48   |    59 |
| neural        |  50.2  |  3.79 |    40 |     50   |    61 |
| hunt_target   |  51.57 |  4.33 |    42 |     51   |    65 |
| random        |  97.68 |  2.63 |    86 |     98.5 |   100 |

## Head to head: neural_search vs. density

- neural_search better: 45.7%
- draw: 6.7%
- density better: 47.7%
