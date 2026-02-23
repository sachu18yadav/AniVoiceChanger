import sys
import os
import traceback
sys.path.append(r"c:\Users\comei\DOTA 2 Voice over tool\AniVoiceChanger")
import rvc_infer

pth = r"c:\Users\comei\DOTA 2 Voice over tool\virtual_mic\models\pudge\G_5000.pth"
try:
    print(f"Loading {pth}")
    m, i = rvc_infer.load_rvc_model(pth)
    print("Success:", m is not None)
except Exception as e:
    print("FAILED")
    traceback.print_exc()
