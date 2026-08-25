# High-Resolution Multi-View Tiling Ablation

Created: 2026-08-25

## Problem

Ultra-high-resolution remote-sensing images are heavily downscaled when they are forced into a fixed visual-token budget. Small objects, counts, colors, and local spatial details can be lost in the global view.

## Method

The tested mechanism is fixed `global + 2x2 local tiles`: one complete global overview plus top-left, top-right, bottom-left, and bottom-right crops of the same source image. The model is unchanged: Qwen3-VL-2B-Instruct + LoRA v1 + INT4 NF4.

The prompt explicitly states that the five images are multiple views of the same remote-sensing scene and that the local crops are only for recovering details. Ground truth and evaluators are unchanged.

## XLRS A/B Result

| Mode | Samples | Accuracy | Parse failures | Mean latency s | P95 latency s | Peak VRAM MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Single-view | 1848 | 0.389069 | 0 | 0.555370 | 0.681128 | 1901.08 |
| Global+2x2 | 1848 | 0.405303 | 1 | 2.709781 | 2.868999 | 2920.86 |

- Accuracy gain: +0.016234
- Latency multiplier: 4.88x
- Peak VRAM increase: +1019.78 MB
- Single wrong -> tiling correct: 285
- Single correct -> tiling wrong: 255

XLRS by category:

| Group | Samples | Single acc | Tiling acc | Delta |
| --- | ---: | ---: | ---: | ---: |
| Complex reasoning/Anomaly Detection and Interpretation | 100 | 0.680000 | 0.650000 | -0.030000 |
| Complex reasoning/Environmental condition reasoning | 100 | 0.770000 | 0.720000 | -0.050000 |
| Counting/Counting with complex reasoning | 100 | 0.410000 | 0.390000 | -0.020000 |
| Counting/Overall counting | 60 | 0.216667 | 0.233333 | +0.016667 |
| Land use classification/Regional Land use classification | 200 | 0.640000 | 0.705000 | +0.065000 |
| Object properties/Object color | 728 | 0.274725 | 0.291209 | +0.016484 |
| Object properties/Object motion state | 60 | 0.516667 | 0.583333 | +0.066667 |
| Object spatial relationship/Object spatial relationship | 500 | 0.322000 | 0.342000 | +0.020000 |

XLRS by resolution bucket:

| Group | Samples | Single acc | Tiling acc | Delta |
| --- | ---: | ---: | ---: | ---: |
| 4-16MP | 8 | 0.250000 | 0.250000 | +0.000000 |
| <4MP | 3 | 0.333333 | 0.333333 | +0.000000 |
| >16MP | 1837 | 0.389766 | 0.406097 | +0.016331 |

## MME-RS A/B Result

| Mode | Samples | Accuracy | Parse failures | Mean latency s | P95 latency s | Peak VRAM MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Single-view | 150 | 0.466667 | 0 | 0.569578 | 0.693621 | 1850.04 |
| Global+2x2 | 150 | 0.473333 | 0 | 2.587324 | 2.777234 | 2869.18 |

- Accuracy gain: +0.006666
- Latency multiplier: 4.54x
- Peak VRAM increase: +1019.14 MB
- Single wrong -> tiling correct: 24
- Single correct -> tiling wrong: 23

MME-RS by category:

| Group | Samples | Single acc | Tiling acc | Delta |
| --- | ---: | ---: | ---: | ---: |
| color | 50 | 0.580000 | 0.620000 | +0.040000 |
| count | 50 | 0.200000 | 0.220000 | +0.020000 |
| position | 50 | 0.620000 | 0.580000 | -0.040000 |

MME-RS by resolution bucket:

| Group | Samples | Single acc | Tiling acc | Delta |
| --- | ---: | ---: | ---: | ---: |
| 4-16MP | 7 | 0.285714 | 0.428571 | +0.142857 |
| <4MP | 8 | 0.500000 | 0.625000 | +0.125000 |
| >16MP | 135 | 0.474074 | 0.466667 | -0.007407 |

## Positive And Negative Samples

XLRS single wrong -> tiling correct:

| ID | Category | Size | GT | Single | Tiling | Question |
| --- | --- | ---: | --- | --- | --- | --- |
| 11 | Complex reasoning/Anomaly Detection and Interpretation | 4741x5976 | A | B | A | What can be inferred from the presence of the breakwater in the image? |
| 35 | Complex reasoning/Anomaly Detection and Interpretation | 10000x10000 | C | D | C | Why are there so many vacant lots in this area of the image? |
| 49 | Complex reasoning/Anomaly Detection and Interpretation | 4096x4096 | C | D | C | A detailed analysis of the image can reveal? |
| 57 | Complex reasoning/Anomaly Detection and Interpretation | 10000x10000 | B | D | B | Why are the sports fields located to the bottom right of the largest residential area on the left side of the image so c |
| 83 | Complex reasoning/Anomaly Detection and Interpretation | 7360x4912 | D | A | D | What in the picture reflects the level of urban activity and traffic flow? |
| 84 | Complex reasoning/Anomaly Detection and Interpretation | 10000x10000 | D | C | D | Why does the forest area in the lower left corner of the picture have circular routes and straight, scattered paths? |
| 109 | Complex reasoning/Environmental condition reasoning | 10000x10000 | A | B | A | Impact of Large Areas of Trees on Residential Mobility? |
| 111 | Complex reasoning/Environmental condition reasoning | 10000x10000 | B | D | B | What is the impact of the road in the lower right corner of the image on the surrounding residential buildings? |
| 115 | Complex reasoning/Environmental condition reasoning | 10000x10000 | B | D | B | Can the farmland located beneath the runway in the large housing cluster at the top right corner of the map be transform |
| 144 | Complex reasoning/Environmental condition reasoning | 10000x10000 | C | A | C | What is the impact of the trees and river shown in the picture on the environment? |
| 201 | Object spatial relationship/Object spatial relationship | 10000x10000 | A | C | A | What is the position of the triangular parking lot within the town, relative to the white rectangular building in the ce |
| 212 | Object spatial relationship/Object spatial relationship | 7360x4912 | C | D | C | In the image, at the bottom left on the left side of the highway, what is the position of the gray cylindrical roofed bu |

XLRS single correct -> tiling wrong:

| ID | Category | Size | GT | Single | Tiling | Question |
| --- | --- | ---: | --- | --- | --- | --- |
| 10 | Complex reasoning/Anomaly Detection and Interpretation | 10000x10000 | B | B | D | Why is there a river in the image? |
| 14 | Complex reasoning/Anomaly Detection and Interpretation | 10000x10000 | D | D | C | The overall color of the picture is dark, yet the area slightly to the right above the river appears brighter. Why does  |
| 17 | Complex reasoning/Anomaly Detection and Interpretation | 10000x10000 | C | C | A | Why is there so much empty space in the middle of the image? |
| 55 | Complex reasoning/Anomaly Detection and Interpretation | 10000x10000 | D | D | A | Why are the farmlands and cultivated fields at the top of the picture neatly divided by rows of trees? |
| 66 | Complex reasoning/Anomaly Detection and Interpretation | 7360x4912 | D | D | B | What in the picture reflects the consistency and modernization level of urban planning? |
| 67 | Complex reasoning/Anomaly Detection and Interpretation | 10000x10000 | A | A | D | The presence of a lake and extensive green areas in the picture suggests? |
| 68 | Complex reasoning/Anomaly Detection and Interpretation | 10000x10000 | A | A | D | Why are there several lakes next to the forest on the right side of the picture? |
| 86 | Complex reasoning/Anomaly Detection and Interpretation | 10000x10000 | A | A | C | In the image, the residential area at the top has three ponds clustered together in the upper left corner. Why is this t |
| 88 | Complex reasoning/Anomaly Detection and Interpretation | 4096x4096 | B | B | D | Can it be inferred from the image that buildings are sparse and primarily located in areas with less vegetation? |
| 117 | Complex reasoning/Environmental condition reasoning | 5616x3744 | A | A | B | Based on the satellite imagery and considering the existing infrastructure and layout, is the area at the center of the  |
| 133 | Complex reasoning/Environmental condition reasoning | 4096x4096 | A | A | D | Based on satellite imagery, considering the presence of industrial facilities and a port in the central area, what poten |
| 141 | Complex reasoning/Environmental condition reasoning | 10000x10000 | A | A | C | What could be the reason for the stark difference in color between the teal lake located towards the upper right-center  |

MME-RS single wrong -> tiling correct:

| ID | Category | Size | GT | Single | Tiling | Question |
| --- | --- | ---: | --- | --- | --- | --- |
| perception/remote_sensing/position/1567 | position | 2432x3244 | B | A | B | Where is the tennis court located in this picture? |
| perception/remote_sensing/position/3118 | position | 7360x4912 | C | D | C | Where is the small green boat in the picture? |
| perception/remote_sensing/position/3151 | position | 7360x4912 | A | B | A | Where is the square with colored circular patterns and decorations in the picture? |
| perception/remote_sensing/position/2294 | position | 7360x4912 | B | C | B | Where is the green football field located in this picture? |
| perception/remote_sensing/position/2414 | position | 7360x4912 | A | E | A | Where is the yellow excavator in the picture? |
| perception/remote_sensing/position/0922 | position | 4097x4096 | C | A | C | Where is the cargo ship with a white container in the picture? |
| perception/remote_sensing/position/1829 | position | 7360x4912 | C | B | C | Where is the house with a yellow roof in the picture? |
| perception/remote_sensing/color/1577 | color | 4096x4097 | C | D | C | What color is the big ship in the middle of the top of this picture? |
| perception/remote_sensing/color/1824 | color | 7360x4912 | C | B | C | What color is the small triangle on the triangle roof in the bottom right corner of the picture? |
| perception/remote_sensing/color/3565 | color | 4096x4096 | D | C | D | What color is the building at the end of the path on the grass in the upper right corner of the picture? |
| perception/remote_sensing/color/1922 | color | 7360x4912 | C | A | C | What color car is next to the red car at the corner of the road in the upper right corner of the picture? |
| perception/remote_sensing/color/2076 | color | 7360x4912 | D | B | D | What color is the shed on the right window of the house with solar panels on the roof in the left area of the picture? |

MME-RS single correct -> tiling wrong:

| ID | Category | Size | GT | Single | Tiling | Question |
| --- | --- | ---: | --- | --- | --- | --- |
| perception/remote_sensing/position/2512 | position | 4096x4096 | C | C | B | Where are the red and white cell towers in the picture? |
| perception/remote_sensing/position/0794 | position | 4097x4096 | B | B | C | Where is the biggest football field located in the picture? |
| perception/remote_sensing/position/1037 | position | 4097x4096 | D | D | B | Where is the building with black-and-white in the picture? |
| perception/remote_sensing/position/1915 | position | 921x1062 | B | B | A | Where are the red cars in the picture? |
| perception/remote_sensing/position/3376 | position | 7360x4912 | B | B | C | Where are the trees with pink flowers in the picture? |
| perception/remote_sensing/position/3705 | position | 6000x6000 | D | D | C | Where is the white-roofed building with the letter H in the middle in the picture? |
| perception/remote_sensing/position/1604 | position | 7360x4912 | B | B | C | Where are the white tables and yellow and black chairs located in this picture? |
| perception/remote_sensing/position/3471 | position | 7360x4912 | D | D | C | Where is the basketball court located in this picture? |
| perception/remote_sensing/position/0948 | position | 4096x4096 | A | A | D | Where is the sky blue ship in the picture? |
| perception/remote_sensing/color/1044 | color | 7360x4912 | A | A | D | What color is the fourth car in the train on the far right of the picture? |
| perception/remote_sensing/color/0949 | color | 4096x4096 | D | D | C | What color is the house on the far left of the picture along the edge? |
| perception/remote_sensing/color/3185 | color | 7360x4912 | B | B | D | What color is the top car on the road along the river in the picture? |

## Deployment Decision

- The mechanism is technically feasible on the RTX 5060 Laptop GPU 8GB with INT4 NF4. There were no inference failures in the full XLRS partial run or MME-RS full run.
- The accuracy gains are small: XLRS +0.016234 and MME-RS +0.006666.
- The resource cost is large: XLRS mean latency is 4.88x single-view and MME-RS mean latency is 4.54x single-view. Peak VRAM rises by about 1.0 GB.
- The benefit is not cleanly concentrated in all high-resolution images because almost all XLRS local samples are already >16MP. Category-level gains are task-dependent.
- A simple adaptive route based only on megapixels is not justified by this evidence. A more selective route would need a task-aware trigger, but that would be a new design beyond this fixed ablation.

## Recommendation

Do not make global+2x2 tiling the default deployment mode for the competition submission. Keep it as a reproducible innovation ablation and optional high-resolution fallback for selected tasks where latency is acceptable.

It can be described as a tested resource-aware multi-scale inference mechanism, but not as the final default model path because the small accuracy gain does not justify the latency multiplier.

## Limitations

- XLRS is still a local 44/74-shard partial result, not the full official Lite benchmark.
- The first version uses fixed 2x2 tiles with 0 overlap and no learned or detector-based routing.
- The prompt is an added multi-view wrapper around official MCQ instructions; no evaluator or ground truth was changed.
- No LoRA retraining or prompt search was performed.
