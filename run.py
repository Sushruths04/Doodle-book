"""Launcher for DoodleBook — captures all output for debugging."""
import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(__file__))

# Redirect stderr to file
log = open("_crash.log", "w", encoding="utf-8")
sys.stderr = log
sys.stdout = log

try:
    from app import create_layout, load_sample_book, create_book
    print("Imports OK", flush=True)
    
    demo = create_layout(
        load_sample_fn=load_sample_book,
        create_book_fn=create_book,
    )
    print("Layout OK", flush=True)
    
    demo.launch(server_port=7870, prevent_thread_lock=True)
    print("Launch OK — server running on port 7870", flush=True)
    
    import time
    while True:
        time.sleep(60)
    
except Exception as e:
    print(f"ERROR: {e}", flush=True)
    traceback.print_exc(file=log)
finally:
    log.flush()
