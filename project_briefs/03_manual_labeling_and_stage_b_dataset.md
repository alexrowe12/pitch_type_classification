# Brief 3: Manual Labeling and Stage B Dataset

## Whole Project Summary

The goal of this project was to classify baseball pitches from broadcast video clips as either **fastball** or **off-speed**. The original clips came from the MLB-YouTube dataset, which contains short broadcast segments from the 2017 MLB postseason.

The project became more than just training a model. A major part of the work was building a computer vision pipeline that could turn messy broadcast footage into usable training data. The final pipeline had four main stages:

1. Download short MLB video clips.
2. Detect which frames actually show the pitch camera.
3. Manually label the release and catch frames for each pitch.
4. Train and evaluate binary classification models.

The final result was a failed-but-rigorous experiment: the models did not reliably beat a simple baseline on the test set, but the project produced useful evidence about why this task is difficult with broadcast video.

## Your Section: Manual Labeling and Stage B Dataset

Your job is to explain the dataset we manually created after Stage A found the correct camera view.

Stage B focused on the actual pitch event. For each usable clip, the goal was to identify:

- The **release frame**, meaning the first frame where the ball is out of the pitcher's hand.
- The **catch frame**, meaning the last frame before the ball disappears into the catcher's glove.

This created a cleaner sequence showing the ball traveling from the pitcher to the catcher.

## Why Manual Labeling Was Needed

Automatic detection was not reliable enough. The ball was too small, and motion-based methods often focused on the pitcher or batter instead of the ball. Even when the pitch-camera segment was correct, the model still needed to know which frames actually contained the pitch flight.

Manual labeling made it possible to create a higher-quality dataset. Instead of asking the model to learn from a full 7-second clip, we gave it a short sequence centered on the actual pitch.

## How the Labeling Worked

A local Streamlit app was built to make labeling faster. For each clip, the app showed frames from the pitch-camera segment. The user selected the release and catch frames, then saved the event.

After labeling, the pipeline exported fixed-length sequences. Even though real pitches lasted different numbers of frames, the export step sampled each release-to-catch span into the same number of frames. The final modeling sequences used **12 frames** each.

## Final Dataset Shape

After manual labeling, the project had:

- 255 final labeled pitch events
- 140 fastballs
- 115 off-speed pitches
- Train, validation, and test splits
- Each exported sequence shaped like 12 frames of cropped video

The individual off-speed types included changeup, curveball, knucklecurve, sinker, and slider, but for modeling they were grouped together as one off-speed class.

## Important Point to Communicate

This manual dataset was the cleanest version of the data. It removed most irrelevant footage and focused the model on the actual ball flight. However, the dataset was still small for a video classification task, especially because the visual differences between pitch types were subtle.

## Suggested Closing Point

Stage B gave the project a real, carefully labeled dataset. Even though the final model did not perform well enough, this step made the experiment rigorous because the model was tested on the best data we could reasonably create from the available footage.
