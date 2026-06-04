################################################################################


import asyncio
import numpy as np
import sys
from RealtimeSTT import AudioToTextRecorder

def test_realtimestt_feed():
    print("Initializing RealtimeSTT...")
    # Initialize without opening system mic (if possible)
    try:
        recorder = AudioToTextRecorder(
            model="tiny.en",
            language="en",
            spinner=False,
            # input_device_index=None # Might not work depending on version
        )
        print("Success initializing.")
        
        # Check for feed_audio or equivalent
        if hasattr(recorder, "feed_audio") or hasattr(recorder, "process_audio"):
            print("Detected feeding capability.")
        else:
            print("No direct feed method found. Checking dir...")
            # print(dir(recorder))
            
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as e:
        print(f"Error: {e}")
        return False
    return True

if __name__ == "__main__":
    sys.exit(0 if test_realtimestt_feed() else 1)


##
