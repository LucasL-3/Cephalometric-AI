# Keypoint order and “points look wrong”

If **some photos don’t get analyzed** or **the points on the image look in the wrong places**, it’s usually one of these:

---

## 1. No analysis for some photos

- The model may not detect anything in those images (low confidence).
- **What we did:** Detection uses a lower confidence threshold (`CONF_THRESHOLD = 0.15` in `image_processor.py`) so more X-rays get at least one detection. You can change that value (e.g. `0.1` for more detections, `0.25` for stricter).
- If it still skips images: check image size, contrast, and that the cephalometric region is visible and similar to your training data.

---

## 2. Point identification is “super off”

The **order of keypoints** from your YOLO model must match the order used for Steiner analysis. If they don’t, the same indices are used for different landmarks and measurements look wrong.

- **Where the order is defined:**  
  `yolov8/ceph_pose_keypoints.yaml` (and the `LANDMARKS` mapping in `image_processor.py`).
- **What you need:**  
  The **exact order** in which keypoints were labeled when you created the dataset (e.g. in your annotation tool or script). That order is:  
  **index 0 = first landmark, index 1 = second, …, index 18 = 19th landmark.**

### How to fix wrong points

1. **Find your dataset’s keypoint order**  
   From your labeling protocol, annotation tool, or the script that generated YOLO labels: which landmark is point 0, which is point 1, …, which is point 18?

2. **Align with the YAML**  
   Edit `yolov8/ceph_pose_keypoints.yaml` so that the **names** next to each index match **your** order, for example:

   ```yaml
   keypoints:
     0: S      # ← must be whatever YOUR dataset uses as first point
     1: N      # ← second point in your dataset
     2: A      # etc.
     ...
   ```

   The names in that file are the ones used for Steiner (S, N, A, B, Go, Me, U1_tip, …). So you’re not changing the **analysis** names; you’re saying “in my dataset, index 0 is S, index 1 is N, …”. If your dataset has “index 0 = N” and “index 1 = S”, then you must swap them in the YAML (e.g. `0: N`, `1: S`) or the points and measurements will be wrong.

3. **Optional: list your order in a file**  
   If you have a list like “0=S, 1=N, 2=A, 3=B, …” from your training data, you can keep it in the repo and make sure `ceph_pose_keypoints.yaml` matches it. Then re-run the app and check the overlay again.

---

## 3. Improving point accuracy (inference and data)

- **Inference resolution:** In `image_processor.py`, `IMGSZ = 640` is used for prediction. If your model was trained at a different size (e.g. 1280), set `IMGSZ` to match; matching training size usually gives the best keypoint accuracy.
- **Keypoint offset (cutter.py):** If your labels were built with `cutter.py`, the first two keypoints in the label file are dummies `(0.5,0.5)` and `(1.0,1.0)`. In that case the **anatomical** landmarks start at index 2. Set `keypoint_offset: 2` in `ceph_pose_keypoints.yaml` only if your model actually outputs **21** keypoints (2 dummies + 19 anatomical). With the default 19-keypoint model, keep `keypoint_offset: 0`.

---

## Summary

- **Some photos not analyzed:** Lower `CONF_THRESHOLD` in `image_processor.py` (e.g. 0.1); already 0.15.
- **Points in wrong places:** Align **training keypoint order** with `ceph_pose_keypoints.yaml`; use `keypoint_offset` if you have leading dummy points.
- **Better accuracy:** Set `IMGSZ` in `image_processor.py` to match your training resolution.
