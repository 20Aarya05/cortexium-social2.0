try:
    import mediapipe as mp
    print("mediapipe OK")
except Exception as e:
    print(f"mediapipe FAIL: {e}")

try:
    import deepface
    print("deepface OK")
except Exception as e:
    print(f"deepface FAIL: {e}")
