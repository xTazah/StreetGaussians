
import mediapy as media
import numpy as np

print("Testing mediapy VideoWriter...")
try:
    height, width = 100, 100
    with media.VideoWriter("test_output.mp4", shape=(height, width), fps=10) as writer:
        for _ in range(10):
            image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            writer.add_image(image)
    print("Success! Video created.")
except Exception as e:
    print(f"Failed with error: {e}")
    import traceback
    traceback.print_exc()
