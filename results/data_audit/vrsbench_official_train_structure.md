# VRSBench Official Train Structure

Generated: 2026-08-23T22:08:46

## Official Data Paths

| Item | Path | Count/Size |
|---|---|---:|
| Images_train | `G:\Images_train\Images_train` | 20264 images |
| VRSBench_train.json | `\\wsl.localhost\Ubuntu\home\lhs\spacers_m3\data\raw\vrsbench_official\VRSBench_train.json` | 142390 rows |
| Annotations_train | `\\wsl.localhost\Ubuntu\home\lhs\spacers_m3\data\raw\vrsbench_official\Annotations_train.zip` | 20264 json files |

## VRSBench_train.json

| Metric | Count |
|---|---:|
| Unique images | 20262 |
| Caption samples | 20264 |
| VQA samples | 85813 |
| Grounding samples | 25564 |
| Invalid grounding rows skipped | 10749 |

Top-level keys: `{'id': 142390, 'image': 142390, 'conversations': 142390}`

Conversation keys: `{'from': 284780, 'value': 284780}`

Grounding bbox is parsed from the assistant value in `[refer]` rows. Parsed range: `0` to `100`.

## Annotations_train.zip

| Metric | Value |
|---|---:|
| Present | True |
| JSON files | 20264 |
| obj_coord range | 0.0 to 1.97 |
| obj_corner range | -0.734375 to 1.96875 |

Annotation top-level keys: `{'caption': 20264, 'objects': 20264, 'qa_pairs': 20264, 'image': 20264}`

Object keys: `{'obj_id': 36313, 'referring_sentence': 36313, 'obj_cls': 36313, 'obj_corner': 36313, 'obj_coord': 36313, 'is_unique': 36313, 'obj_position': 36313, 'obj_rel_position': 36313, 'obj_size': 36313, 'obj_rel_size': 36313, 'flag': 36311}`

QA pair keys: `{'ques_id': 85813, 'question': 85813, 'answer': 85813, 'type': 85813}`

Notes:

- `VRSBench_train.json` is the final source for LoRA manifest construction.
- `[caption]`, `[vqa]`, and `[refer]` task tags are read from the human conversation text.
- `[refer]` assistant values are normalized 0-100 xyxy boxes such as `{<45><45><59><59>}`.
