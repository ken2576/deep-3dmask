### Evaluation Steps

1. Run ```collect_data.py``` to convert to LLFF format

2. Run ```imgs2renderpath.py``` for the first frame to generate render path

    Example:
    ```python .\imgs2renderpath_new.py [scene] [scene]\frames_to_render\[start]\outputs\test_path.txt```

3. Run LLFF or ```render_llff_video.py```