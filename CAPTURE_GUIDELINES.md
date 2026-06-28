# Capture guidelines for new classes

A quick checklist for recording/photographing items before adding them as a new class. Based on what this project's own benchmark caught: classes with shared, dominant backgrounds got confused with each other, while classes with the object filling the frame did not.

## Before recording

- **Plain or varied background.** Avoid the same wall/desk/room used for other classes. If the background can't be changed, get close enough that the object fills most of the frame.
- **Good, even lighting.** Avoid strong shadows or backlighting that hides the object's shape or color.
- **Decide: photo or video?** Video is the better default — one short video gives more angle variety than several static photos shot from the same spot.

## If recording video

- **Length:** 5-8 seconds is enough.
- **Movement:** move the camera around the object — different angles, slight distance changes. Don't hold the camera still.
- **Speed:** move slowly enough to avoid heavy motion blur.
- Extract frames with:
  ```
  python extract_frames.py <video_file> <class_name>
  ```
  Default settings (one frame every 0.5s, capped at 20 frames) work well for an 5-8s video.

## If taking photos

- Take 5-8 photos minimum.
- Vary the angle and distance between shots — don't take near-identical repeats.
- Keep the object as the dominant element in the frame.

## After capturing, before embedding

- Open the class folder and look through the images.
- Delete any that are blurry, poorly lit, or where the object isn't clearly visible.
- It's fine to have an uneven number of images per class.

## Always run after adding a class

```
python create_embeddings.py
python benchmark.py
```

Check the new class's Top-1/Top-3 accuracy and the confusion list. If the new class is confused with an existing one, the likely cause (based on this project's own history) is a shared background or too few/too-similar images — revisit the capture rather than assuming the model is at fault.
