try:
    from mlx_vlm.utils import load_image

    print("mlx_vlm successfully imported.")
    print(dir(load_image))
except (ImportError, OSError, RuntimeError) as e:
    print(f"Error: {e}")
