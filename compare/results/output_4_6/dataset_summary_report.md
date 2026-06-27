# 4.6 Structural Analysis Summary

Note: this analysis intentionally focuses on hands and pose to match the structural-analysis goal.
Face keypoints are not used even though the newer extraction has much lower face missing rates.

## handshape_top5
- ASL: top components cover 75.47%
  - 2 / neutral_proxy: 19.65% (3297)
  - 3 / neutral_proxy: 18.38% (3084)
  - 5 / closed_proxy: 15.56% (2611)
  - 0 / closed_proxy: 11.90% (1997)
  - 6 / closed_proxy: 9.98% (1675)
- VSL: top components cover 86.01%
  - 2 / neutral_proxy: 27.51% (1926)
  - 3 / neutral_proxy: 23.44% (1641)
  - 6 / closed_proxy: 15.61% (1093)
  - 5 / closed_proxy: 10.96% (767)
  - 4 / curved_proxy: 8.49% (594)

## location
- ASL: top components cover 100.00%
  - shoulder: 99.95% (16772)
  - below_waist: 0.04% (7)
  - head: 0.01% (1)
- VSL: top components cover 100.00%
  - shoulder: 99.71% (6980)
  - below_waist: 0.13% (9)
  - head: 0.13% (9)
  - chest: 0.03% (2)

## one_two_hand
- ASL: top components cover 100.00%
  - two_hand: 64.73% (7755)
  - low_hand_activity: 32.22% (3860)
  - right_one_hand: 2.98% (357)
  - left_one_hand: 0.07% (8)
- VSL: top components cover 100.00%
  - two_hand: 72.83% (3177)
  - low_hand_activity: 19.92% (869)
  - right_one_hand: 7.24% (316)

## orientation
- ASL: top components cover 62.91%
  - left / backward: 18.67% (3132)
  - up / right: 13.62% (2286)
  - up / backward: 10.67% (1790)
  - left / right: 10.02% (1682)
  - up / forward: 9.93% (1667)
- VSL: top components cover 59.63%
  - down / backward: 21.59% (1511)
  - left / backward: 11.20% (784)
  - up / right: 9.56% (669)
  - down / up: 9.34% (654)
  - left / up: 7.94% (556)
