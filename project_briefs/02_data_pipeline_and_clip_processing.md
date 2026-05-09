# Brief 2: Data Pipeline and Clip Processing

## Whole Project Summary

The goal of this project was to classify baseball pitches from broadcast video clips as either **fastball** or **off-speed**. The original clips came from the MLB-YouTube dataset, which contains short broadcast segments from the 2017 MLB postseason.

The project became more than just training a model. A major part of the work was building a computer vision pipeline that could turn messy broadcast footage into usable training data. The final pipeline had four main stages:

1. Download short MLB video clips.
2. Detect which frames actually show the pitch camera.
3. Manually label the release and catch frames for each pitch.
4. Train and evaluate binary classification models.

The final result was a failed-but-rigorous experiment: the models did not reliably beat a simple baseline on the test set, but the project produced useful evidence about why this task is difficult with broadcast video.

## Your Section: Data Pipeline and Clip Processing

Your job is to explain how the raw video clips were turned into something we could inspect and model.

The starting point was a metadata file from the MLB-YouTube dataset. That file gave us YouTube video IDs, timestamps, and pitch labels. The download script used that information to download short video clips around each pitch.

However, those clips were not automatically ready for modeling. A 7-second broadcast clip might include the actual pitch, but it might also include:

- A closeup of the pitcher or batter
- A replay
- A scoreboard graphic
- The wrong camera angle
- Extra time before or after the pitch

Because of that, the project needed a processing pipeline before model training could happen.

## Stage A: Finding the Pitch Camera

The first major processing step was called **Stage A**. Its purpose was to identify which parts of each clip showed the standard center-field pitch camera, where the pitcher, batter, catcher, and strike zone are visible.

At first, simple motion detection was tried. That approach often found the beginning of the pitcher's motion, but it did not reliably find the actual useful pitch frames. The project then switched to a better two-stage approach:

1. Export many frames from each clip.
2. Train a small model to classify each frame as either `pitch_camera` or `non_pitch_camera`.

This worked much better because the model learned what the correct broadcast camera looked like.

## Debug Contact Sheets

A major part of the workflow was creating contact sheets. These are image grids that show frames from a clip, along with labels or prediction confidence.

The contact sheets made it possible to quickly inspect whether the pipeline was working. They were used to check:

- Whether the pitch camera was detected correctly
- Whether the selected segment started and ended in the right place
- Whether the later manual labeling step had good input data

## Important Point to Communicate

This part of the project was necessary because raw broadcast video is messy. If we had trained directly on random frames from the 7-second clips, the model would have learned from closeups, crowd shots, and non-pitch footage. Stage A made the later dataset much cleaner by narrowing each clip down to the actual pitch-camera segment.

## Suggested Closing Point

The data pipeline was one of the most successful parts of the project. It did not solve pitch classification by itself, but it created a much cleaner dataset and made the later failure analysis more meaningful.
