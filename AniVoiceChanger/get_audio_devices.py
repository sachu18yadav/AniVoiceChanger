"""
Run this once to find your audio device indices.
Copy the numbers you need into your .env file.
"""
import sounddevice as sd

print("=" * 60)
print("  AUDIO DEVICES")
print("=" * 60)
print()

devices = sd.query_devices()
print("--- INPUT DEVICES (microphones) ---")
for i, d in enumerate(devices):
    if d['max_input_channels'] > 0:
        marker = ""
        name = d['name']
        if 'cable input' in name.lower():
            marker = "  ← VB-Cable Input"
        elif 'realtek' in name.lower() and 'mic' in name.lower():
            marker = "  ← Likely your microphone"
        print(f"  {i}: {name}{marker}")

print()
print("--- OUTPUT DEVICES (speakers / cables) ---")
for i, d in enumerate(devices):
    if d['max_output_channels'] > 0:
        marker = ""
        name = d['name']
        if 'cable input' in name.lower():
            marker = "  ← Set as OUTPUT_DEVICE_INDEX (Discord hears this)"
        elif 'realtek' in name.lower() or 'speaker' in name.lower():
            marker = "  ← Set as SPEAKER_DEVICE_INDEX (you hear this)"
        print(f"  {i}: {name}{marker}")

print()
print("Copy the indices into your .env file.")
print("=" * 60)