# High-Resolution Input Audit

Created: 2026-08-25

Scope: XLRS-Bench-lite local 44-shard partial run and MME-RealWorld-Lite Remote Sensing subset.

Processor `max_pixels` used by current benchmark scripts: `1048576`.
The existing XLRS single-view predictions record `image_grid_thw`; MME-RS dimensions are audited from image files.

## XLRS-Bench-lite Local 44-Shard Partial

| Metric | Value |
| --- | ---: |
| Samples | 1848 |
| Width min / median / mean / max | 1361 / 10000 / 8291.8 / 11500 |
| Height min / median / mean / max | 1196 / 10000 / 7903.0 / 10000 |
| Megapixels min / median / mean / max | 1.916 / 100.000 / 71.775 / 100.000 |
| Compression to 1MP median / mean / max | 95.37x / 68.45x / 95.37x |

Resolution buckets:

| Bucket | Samples |
| --- | ---: |
| 4-16MP | 8 |
| <4MP | 3 |
| >16MP | 1837 |

Most common XLRS processor `image_grid_thw` values:

| image_grid_thw | Samples | Approx visual tokens |
| --- | ---: | ---: |
| `[[1, 64, 64]]` | 1532 | 4096 |
| `[[1, 50, 76]]` | 298 | 3800 |
| `[[1, 72, 56]]` | 3 | 4032 |
| `[[1, 64, 62]]` | 3 | 3968 |
| `[[1, 50, 78]]` | 3 | 3900 |
| `[[1, 52, 76]]` | 3 | 3952 |
| `[[1, 48, 82]]` | 2 | 3936 |
| `[[1, 54, 74]]` | 1 | 3996 |

## MME-RealWorld-Lite Remote Sensing

| Metric | Value |
| --- | ---: |
| Samples | 150 |
| Width min / median / mean / max | 921 / 7360 / 5707.6 / 11500 |
| Height min / median / mean / max | 1062 / 4912 / 4397.1 / 7500 |
| Megapixels min / median / mean / max | 0.978 / 36.152 / 26.753 / 86.250 |
| Compression to 1MP median / mean / max | 34.48x / 25.51x / 82.25x |

Resolution buckets:

| Bucket | Samples |
| --- | ---: |
| 4-16MP | 7 |
| <4MP | 8 |
| >16MP | 135 |

## Samples Most Exposed To Downscaling Detail Loss

XLRS largest images:

| Row | Category | Size | MP | Grid | Single correct | Question |
| --- | --- | ---: | ---: | --- | --- | --- |
| 1 | Complex reasoning/Anomaly Detection and Interpretation | 10000x10000 | 100.00 | `[[1, 64, 64]]` | False | What caused the formation of the blue-green lake in the upper left corner of the picture? |
| 2 | Complex reasoning/Anomaly Detection and Interpretation | 10000x10000 | 100.00 | `[[1, 64, 64]]` | False | Why does the image show more farmland and fewer residential areas? |
| 3 | Complex reasoning/Anomaly Detection and Interpretation | 10000x10000 | 100.00 | `[[1, 64, 64]]` | True | Why are the buildings in the two large housing clusters separated by a road in the top left corner of the picture so dif |
| 6 | Complex reasoning/Anomaly Detection and Interpretation | 10000x10000 | 100.00 | `[[1, 64, 64]]` | True | The presence of multifunctional areas in the image suggests? |
| 7 | Complex reasoning/Anomaly Detection and Interpretation | 10000x10000 | 100.00 | `[[1, 64, 64]]` | True | Why are the houses in the multi-unit residential image built in a spiral shape? |
| 8 | Complex reasoning/Anomaly Detection and Interpretation | 10000x10000 | 100.00 | `[[1, 64, 64]]` | True | In the middle on the right side, the lower left corner, and on both sides of the highway in the farmland areas shown in  |
| 9 | Complex reasoning/Anomaly Detection and Interpretation | 10000x10000 | 100.00 | `[[1, 64, 64]]` | True | Why is there a river in the image? |
| 11 | Complex reasoning/Anomaly Detection and Interpretation | 10000x10000 | 100.00 | `[[1, 64, 64]]` | True | The residential areas near the sea are densely populated. Why is this the case? |
| 13 | Complex reasoning/Anomaly Detection and Interpretation | 10000x10000 | 100.00 | `[[1, 64, 64]]` | True | The overall color of the picture is dark, yet the area slightly to the right above the river appears brighter. Why does  |
| 15 | Complex reasoning/Anomaly Detection and Interpretation | 10000x10000 | 100.00 | `[[1, 64, 64]]` | True | Why are there many small islands on both sides of the river in the picture? |
| 16 | Complex reasoning/Anomaly Detection and Interpretation | 10000x10000 | 100.00 | `[[1, 64, 64]]` | True | Why is there so much empty space in the middle of the image? |
| 17 | Complex reasoning/Anomaly Detection and Interpretation | 10000x10000 | 100.00 | `[[1, 64, 64]]` | True | Is the rectangular area near the water in the picture farmland or something else, and why does this occur? |

MME-RS largest images:

| Question ID | Category | Size | MP | Question |
| --- | --- | ---: | ---: | --- |
| perception/remote_sensing/position/0003 | position | 11500x7500 | 86.25 | Where is the Y-fork cross in the picture? |
| perception/remote_sensing/position/2527 | position | 7360x4912 | 36.15 | Where is the yellow crane in the picture? |
| perception/remote_sensing/position/1379 | position | 7360x4912 | 36.15 | Where is the white tower in the picture? |
| perception/remote_sensing/position/1433 | position | 7360x4912 | 36.15 | Where is the gray billboard with gorillas in the picture? |
| perception/remote_sensing/position/1199 | position | 7360x4912 | 36.15 | Where is the largest vacant lot in the picture? |
| perception/remote_sensing/position/2983 | position | 7360x4912 | 36.15 | Where is the circular mini amusement park with amusement facilities located in the picture? |
| perception/remote_sensing/position/1205 | position | 7360x4912 | 36.15 | Where the river is in the picture? |
| perception/remote_sensing/position/3118 | position | 7360x4912 | 36.15 | Where is the small green boat in the picture? |
| perception/remote_sensing/position/1826 | position | 7360x4912 | 36.15 | Where is the house with a triangular roof in the picture located? |
| perception/remote_sensing/position/3151 | position | 7360x4912 | 36.15 | Where is the square with colored circular patterns and decorations in the picture? |
| perception/remote_sensing/position/2294 | position | 7360x4912 | 36.15 | Where is the green football field located in this picture? |
| perception/remote_sensing/position/1022 | position | 7360x4912 | 36.15 | Where is the cross swimming pool in the picture? |

## Audit Conclusion

- XLRS contains many true ultra-high-resolution samples. In the existing single-view run, 100MP images are commonly represented as `image_grid_thw=[[1, 64, 64]]`, which is clear evidence of strong downscaling into the model input budget.
- MME-RS also contains large remote-sensing images up to 11500x7500 pixels, so it is a useful cross-benchmark check for the same tiling idea.
- The first tiling ablation should focus on >16MP images and categories involving counting, object properties, and spatial relationships, because these are most likely to lose fine details after global resizing.
