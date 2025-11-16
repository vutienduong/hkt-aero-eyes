def postprocess_track(raw_dets, confidence_threshold=0.4, max_gap=3, min_len=5):
    """
    raw_dets: list of dicts: {frame, x1, y1, x2, y2, score}
    Returns: dict in challenge JSON format:
    {
      "detections": [
        {"bboxes": [ {frame, x1, y1, x2, y2}, ... ]},
        ...
      ]
    }
    """
    # 1) Filter by confidence
    dets = [d for d in raw_dets if d["score"] >= confidence_threshold]
    dets.sort(key=lambda d: d["frame"])

    # 2) Group into intervals by frame gaps
    intervals = []
    current = []

    for d in dets:
        if not current:
            current.append(d)
            continue
        if d["frame"] - current[-1]["frame"] <= max_gap:
            current.append(d)
        else:
            if len(current) >= min_len:
                intervals.append(current)
            current = [d]
    if current and len(current) >= min_len:
        intervals.append(current)

    # 3) Build JSON-ish structure
    result = {"detections": []}
    for interval in intervals:
        bboxes = [
            {
                "frame": d["frame"],
                "x1": d["x1"],
                "y1": d["y1"],
                "x2": d["x2"],
                "y2": d["y2"],
            }
            for d in interval
        ]
        result["detections"].append({"bboxes": bboxes})

    return result
