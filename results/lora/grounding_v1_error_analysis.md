# LoRA v1 Grounding Error Analysis

Generated: 2026-08-24T07:28:19

## Summary

| Metric | Count |
|---|---:|
| Grounding samples | 30 |
| Improved, delta IoU > 0.02 | 9 |
| Declined, delta IoU < -0.02 | 14 |
| Basically unchanged, abs delta <= 0.02 | 7 |
| LoRA parse failures | 1 |

Category counts use geometry-only heuristics from prediction bbox vs GT bbox; multi-target and target-understanding labels are inferred, not manually image-inspected.

| Category | Count |
|---|---:|
| bbox整体偏移 | 1 |
| bbox边界不准 | 3 |
| bbox过大 | 4 |
| bbox过小 | 4 |
| parse失败 | 1 |
| 基本正确 | 13 |
| 目标理解错误/多目标选错 | 4 |

## All Grounding Samples

| sample_id | base prediction | LoRA prediction | GT | base IoU | LoRA IoU | delta IoU | LoRA parse | category |
|---|---|---|---:|---:|---:|---:|---|---|
| grounding_ext_001 | <code>```json {   "box": [464, 677, 627, 821] } ```</code> | <code>{<45><66><63><82>}</code> | [46,68,63,82] | 0.933257 | 0.826389 | -0.106868 | yes | 基本正确 |
| grounding_ext_002 | <code>```json {   "box": "585,473,741,633" } ```</code> | <code>{<57><47><74><68>}</code> | [56,48,73,66] | 0.664719 | 0.768000 | +0.103281 | yes | 基本正确 |
| grounding_ext_003 | <code>```json {   "box": [0, 0, 669, 619] } ```</code> | <code>{<0><0><67><62>}</code> | [0,0,80,67] | 0.772595 | 0.775000 | +0.002405 | yes | 基本正确 |
| grounding_ext_004 | <code>```json {   "box": [325, 529, 685, 771],   "label": "The small bridge spanning the river is located towards the middle-left of the image." } ```</code> | <code>{<14><47><18><50>}</code> | [14,46,20,47] | 0.000000 | 0.000000 | +0.000000 | yes | bbox边界不准 |
| grounding_ext_005 | <code>```json [ 	{"bbox_2d": [108, 158, 148, 206], "label": "The swimming pool located at the top-middle of the image."} ] ```</code> | <code>{<2><15><13><21>}</code> | [41,5,44,9] | 0.000000 | 0.000000 | +0.000000 | yes | 目标理解错误/多目标选错 |
| grounding_ext_006 | <code>```json {   "box": [697, 456, 830, 997] } ```</code> | <code>{<71><45><79><77>}</code> | [71,44,89,100] | 0.602042 | 0.253968 | -0.348074 | yes | bbox过小 |
| grounding_ext_007 | <code>```json {   "box": [307, 820, 445, 908] } ```</code> | <code>{<39><83><45><91>}</code> | [35,84,57,91] | 0.306393 | 0.262500 | -0.043893 | yes | bbox过小 |
| grounding_ext_008 | <code>```json {   "box": [0, 0, 998, 1000] } ```</code> | <code>{<10><29><48><66>}</code> | [10,28,47,65] | 0.137174 | 0.923077 | +0.785903 | yes | 基本正确 |
| grounding_ext_009 | <code>```json {   "box": [647, 574, 999, 997] } ```</code> | <code>{<64><57><100><99>}</code> | [66,58,100,100] | 0.940332 | 0.901682 | -0.038650 | yes | 基本正确 |
| grounding_ext_010 | <code>```json {   "box": [384, 511, 548, 639] } ```</code> | <code>{<44><81><54><93>}</code> | [48,83,83,100] | 0.000000 | 0.091603 | +0.091603 | yes | bbox过小 |
| grounding_ext_011 | <code>```json {   "box": [67, 789, 408, 998] } ```</code> | <code>{<6><80><38><100>}</code> | [7,80,37,100] | 0.826504 | 0.937500 | +0.110996 | yes | 基本正确 |
| grounding_ext_012 | <code>```json {   "box": [538, 79, 601, 154] } ```</code> | <code>{52><9><60><16><10><13></code> | [52,7,60,16] | 0.639175 | 0.000000 | -0.639175 | no | parse失败 |
| grounding_ext_013 | <code>```json {   "box": [157, 669, 280, 789],   "label": "The small bridge in the bottom-left of the image spans diagonally across the water." } ```</code> | <code>{<37><85><43><93>}</code> | [0,96,14,100] | 0.000000 | 0.000000 | +0.000000 | yes | 目标理解错误/多目标选错 |
| grounding_ext_014 | <code>```json {   "box": [439, 410, 998, 998] } ```</code> | <code>{<50><41><100><99>}</code> | [44,41,100,100] | 0.991287 | 0.877724 | -0.113563 | yes | 基本正确 |
| grounding_ext_015 | <code>```json {   "box": [569, 0, 760, 704] } ```</code> | <code>{56><0><76><70>}</code> | [62,0,75,68] | 0.657425 | 0.631429 | -0.025996 | yes | 基本正确 |
| grounding_ext_016 | <code>```json {   "box": [111, 115, 737, 494] } ```</code> | <code>{<11><12><74><50>}</code> | [17,18,66,42] | 0.495671 | 0.491228 | -0.004443 | yes | bbox过大 |
| grounding_ext_017 | <code>```json {   "box": [621, 348, 838, 765] } ```</code> | <code>{<68><72><88><93>}</code> | [70,68,89,88] | 0.100463 | 0.562500 | +0.462037 | yes | 基本正确 |
| grounding_ext_018 | <code>```json {   "box": [288, 180, 488, 579] } ```</code> | <code>{<29><17><48><57>}</code> | [29,18,47,57] | 0.879699 | 0.923684 | +0.043985 | yes | 基本正确 |
| grounding_ext_019 | <code>```json {   "box_2d": [685, 798, 774, 958] } ```</code> | <code>{<85><87><96><100>}</code> | [89,87,95,100] | 0.000000 | 0.545455 | +0.545455 | yes | 基本正确 |
| grounding_ext_020 | <code>```json [ 	{"bbox_2d": [801, 625, 847, 756], "label": "The small ship is docked near the top edge of the harbor."} ] ```</code> | <code>{<79><61><85><76>}</code> | [80,62,84,75] | 0.767596 | 0.577778 | -0.189818 | yes | 基本正确 |
| grounding_ext_021 | <code>```json {   "box": [174, 550, 275, 645] } ```</code> | <code>{<19><56><23><62>}</code> | [20,59,27,63] | 0.291819 | 0.209302 | -0.082517 | yes | bbox边界不准 |
| grounding_ext_022 | <code>```json {   "box": [777, 0, 1000, 640] } ```</code> | <code>{<80><37><100><66>}</code> | [75,23,100,71] | 0.533773 | 0.483333 | -0.050440 | yes | bbox过小 |
| grounding_ext_023 | <code>```json {   "box": [516, 111, 664, 197] } ```</code> | <code>{<50><11><65><20>}</code> | [55,13,65,18] | 0.392835 | 0.370370 | -0.022465 | yes | bbox过大 |
| grounding_ext_024 | <code>```json {   "box": [100, 485, 198, 590],   "label": "The bridge located towards the middle-left side of the image." } ```</code> | <code>{<8><45><11><51>}</code> | [7,48,10,50] | 0.000000 | 0.200000 | +0.200000 | yes | bbox过大 |
| grounding_ext_025 | <code>```json {   "box": [531, 838, 554, 875] } ```</code> | <code>{<53><0><65><5>}</code> | [35,0,39,4] | 0.000000 | 0.000000 | +0.000000 | yes | 目标理解错误/多目标选错 |
| grounding_ext_026 | <code>```json {   "box": [616, 0, 724, 45] } ```</code> | <code>{<61><0><72><6>}</code> | [2,7,5,19] | 0.000000 | 0.000000 | +0.000000 | yes | 目标理解错误/多目标选错 |
| grounding_ext_027 | <code>```json {   "box_1": {     "x1": 124,     "y1": 556,     "x2": 188,     "y2": 606,     "label": "The vehicle positioned towards the middle-left of the image."   } } ```</code> | <code>{<6><71><11><76>}</code> | [13,55,15,60] | 0.265060 | 0.000000 | -0.265060 | yes | bbox过大 |
| grounding_ext_028 | <code>```json {   "box": [98, 202, 418, 538],   "label": "The airplane situated at the top-left of the image." } ```</code> | <code>{<10><1><44><55>}</code> | [6,20,46,57] | 0.726486 | 0.559737 | -0.166749 | yes | bbox整体偏移 |
| grounding_ext_029 | <code>```json { "bbox_3d": [872, 532, 999, 999] } ```</code> | <code>{<88><52><95><99>}</code> | [89,54,100,96] | 0.766462 | 0.467532 | -0.298930 | yes | bbox边界不准 |
| grounding_ext_030 | <code>```json {   "box": [565, 133, 770, 385] } ```</code> | <code>{<57><20><75><37>}</code> | [58,21,74,34] | 0.402633 | 0.679739 | +0.277106 | yes | 基本正确 |

## Top 10 IoU Improvements

- grounding_ext_008: delta=+0.785903, base=0.137174, lora=0.923077, category=基本正确, pred=<code>{<10><29><48><66>}</code>
- grounding_ext_019: delta=+0.545455, base=0.000000, lora=0.545455, category=基本正确, pred=<code>{<85><87><96><100>}</code>
- grounding_ext_017: delta=+0.462037, base=0.100463, lora=0.562500, category=基本正确, pred=<code>{<68><72><88><93>}</code>
- grounding_ext_030: delta=+0.277106, base=0.402633, lora=0.679739, category=基本正确, pred=<code>{<57><20><75><37>}</code>
- grounding_ext_024: delta=+0.200000, base=0.000000, lora=0.200000, category=bbox过大, pred=<code>{<8><45><11><51>}</code>
- grounding_ext_011: delta=+0.110996, base=0.826504, lora=0.937500, category=基本正确, pred=<code>{<6><80><38><100>}</code>
- grounding_ext_002: delta=+0.103281, base=0.664719, lora=0.768000, category=基本正确, pred=<code>{<57><47><74><68>}</code>
- grounding_ext_010: delta=+0.091603, base=0.000000, lora=0.091603, category=bbox过小, pred=<code>{<44><81><54><93>}</code>
- grounding_ext_018: delta=+0.043985, base=0.879699, lora=0.923684, category=基本正确, pred=<code>{<29><17><48><57>}</code>
- grounding_ext_003: delta=+0.002405, base=0.772595, lora=0.775000, category=基本正确, pred=<code>{<0><0><67><62>}</code>

## Top 10 IoU Declines

- grounding_ext_012: delta=-0.639175, base=0.639175, lora=0.000000, category=parse失败, pred=<code>{52><9><60><16><10><13></code>
- grounding_ext_006: delta=-0.348074, base=0.602042, lora=0.253968, category=bbox过小, pred=<code>{<71><45><79><77>}</code>
- grounding_ext_029: delta=-0.298930, base=0.766462, lora=0.467532, category=bbox边界不准, pred=<code>{<88><52><95><99>}</code>
- grounding_ext_027: delta=-0.265060, base=0.265060, lora=0.000000, category=bbox过大, pred=<code>{<6><71><11><76>}</code>
- grounding_ext_020: delta=-0.189818, base=0.767596, lora=0.577778, category=基本正确, pred=<code>{<79><61><85><76>}</code>
- grounding_ext_028: delta=-0.166749, base=0.726486, lora=0.559737, category=bbox整体偏移, pred=<code>{<10><1><44><55>}</code>
- grounding_ext_014: delta=-0.113563, base=0.991287, lora=0.877724, category=基本正确, pred=<code>{<50><41><100><99>}</code>
- grounding_ext_001: delta=-0.106868, base=0.933257, lora=0.826389, category=基本正确, pred=<code>{<45><66><63><82>}</code>
- grounding_ext_021: delta=-0.082517, base=0.291819, lora=0.209302, category=bbox边界不准, pred=<code>{<19><56><23><62>}</code>
- grounding_ext_022: delta=-0.050440, base=0.533773, lora=0.483333, category=bbox过小, pred=<code>{<80><37><100><66>}</code>

## Parse Failure

- grounding_ext_012: prediction=<code>{52><9><60><16><10><13></code>, reason=no supported bbox found, question=请找出图像中的目标：The vehicle located towards the top-middle area of the image, is the right-most vehicle present.，并输出其边界框坐标。

## Protocol Notes

- LoRA v1 training assistant target for grounding is <code>{&lt;x1&gt;&lt;y1&gt;&lt;x2&gt;&lt;y2&gt;}</code> in normalized 0-100 xyxy space.
- Prompt v1 asks for one bounding box in normalized 0-100 xyxy format: <code>{&lt;x1&gt;&lt;y1&gt;&lt;x2&gt;&lt;y2&gt;}</code>.
- Evaluator parser accepts JSON arrays/objects and free-form strings containing four numbers, with 0-1, 0-100, or 0-1000 coordinate spaces normalized to 0-100.
- Training, inference prompt, and evaluator are broadly compatible, but evaluator is more permissive than the training/prompt protocol.
