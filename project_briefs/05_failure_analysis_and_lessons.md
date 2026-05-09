# Brief 5: Failure Analysis and Lessons Learned

## Whole Project Summary

The goal of this project was to classify baseball pitches from broadcast video clips as either **fastball** or **off-speed**. The original clips came from the MLB-YouTube dataset, which contains short broadcast segments from the 2017 MLB postseason.

The project became more than just training a model. A major part of the work was building a computer vision pipeline that could turn messy broadcast footage into usable training data. The final pipeline had four main stages:

1. Download short MLB video clips.
2. Detect which frames actually show the pitch camera.
3. Manually label the release and catch frames for each pitch.
4. Train and evaluate binary classification models.

The final result was a failed-but-rigorous experiment: the models did not reliably beat a simple baseline on the test set, but the project produced useful evidence about why this task is difficult with broadcast video.

## Your Section: Failure Analysis and Lessons Learned

Your job is to explain why the project did not produce a strong classifier and why that is still a valid result.

The project did not fail because no work was done. It failed after building a complete pipeline, cleaning the data, labeling examples, trying multiple preprocessing strategies, training several models, and evaluating the results against a baseline.

## Main Reasons the Model Struggled

The biggest issue was that the visual signal was extremely weak. The most useful information should be the baseball's speed and movement, but the ball is very small in broadcast footage.

The model also had to deal with several problems:

- The ball is often only a few pixels wide.
- Broadcast video is compressed and sometimes blurry.
- Pitcher and batter movement dominate the frame.
- Fastballs and off-speed pitches are intentionally made to look similar from the batter's view.
- The camera angle was not designed for measuring pitch movement.
- The labeled dataset was still small for video classification.
- Some pitch labels may be visually impossible to separate from this footage alone.

## Why Preprocessing Did Not Fully Solve It

Motion-based preprocessing helped isolate movement, but it also highlighted the pitcher, batter, and catcher. Attempts to emphasize the ball were not reliable because many other white or moving objects appeared in the frame.

This meant the model could easily learn the wrong things. Instead of learning pitch type, it might learn background patterns, pitcher motion, camera noise, scorebug artifacts, or random differences between clips.

## Why This Is Still a Good Experiment

A failed result can still be rigorous if the process is careful. This project had:

- A clear classification goal
- A real dataset
- A custom data-cleaning pipeline
- Manual event labeling
- Train, validation, and test splits
- Multiple model experiments
- Baseline comparison
- Honest failure analysis

The conclusion is not "computer vision does not work for baseball." The more precise conclusion is that this specific broadcast dataset, at this size and quality, did not provide enough reliable visual signal for this binary classification task.

## What Would Be Needed Next

A stronger version of this project would likely need one or more of the following:

- More labeled examples
- Higher-resolution video
- A fixed camera angle designed for ball tracking
- Better ball detection or tracking
- Statcast-style trajectory data
- Player and ball pose estimation
- A larger pretrained video model

## Suggested Closing Point

The most important lesson is that machine learning performance depends heavily on whether the data actually contains the signal needed for the task. In this project, the pipeline became strong enough to show that the remaining problem was probably not just model choice, but the limits of the available broadcast footage.
