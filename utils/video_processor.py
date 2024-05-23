import cv2

def extract_frame_at_time(video_path: str, time_ms: int):
    vidcap = cv2.VideoCapture(video_path)
    vidcap.set(cv2.CAP_PROP_POS_MSEC, time_ms)
    ret, frame = vidcap.read()
    assert ret, f"Can't read {video_path} at time {time_ms} ms"
    return frame