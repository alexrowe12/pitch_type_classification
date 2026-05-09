# Brief 1: Problem and Dataset

## Whole Project Summary

The goal of this project was to classify baseball pitches from broadcast video clips as either **fastball** or **off-speed**. The original clips came from the MLB-YouTube dataset, which contains short broadcast segments from the 2017 MLB postseason.

The project became more than just training a model. A major part of the work was building a computer vision pipeline that could turn messy broadcast footage into usable training data. The final pipeline had four main stages:

1. Download short MLB video clips.
2. Detect which frames actually show the pitch camera.
3. Manually label the release and catch frames for each pitch.
4. Train and evaluate binary classification models.

The final result was a failed-but-rigorous experiment: the models did not reliably beat a simple baseline on the test set, but the project produced useful evidence about why this task is difficult with broadcast video.

## Your Section: Problem and Dataset

Your job is to explain what the project was trying to solve and what data we used.

The classification task was simplified from many pitch types into two categories:

- **Fastball**
- **Off-speed**, including slider, curveball, changeup, sinker, and knucklecurve

This binary setup made the problem more realistic for a course project because the original pitch types were unevenly distributed. Fastballs were the largest class, while some off-speed pitch types had far fewer examples.

## Why This Is a Hard Computer Vision Problem

The key challenge is that pitch type is not visually obvious in normal broadcast footage. The ball is very small, often only a few pixels wide, and broadcast video includes compression, camera cuts, score graphics, crowd backgrounds, pitcher motion, batter motion, and catcher movement.

In theory, fastballs and off-speed pitches differ in speed and movement. In practice, those differences are hard to detect from a low-resolution broadcast clip because:

- The ball is tiny and sometimes hard to see.
- The camera angle is not designed for machine learning.
- The pitcher tries to make every pitch look similar.
- The video includes many frames that are not useful, such as closeups or replays.
- The dataset was not created specifically for pitch trajectory analysis.

## Important Point to Communicate

The dataset was not "bad," but it was not ideal for this exact task. The project showed that having labels for pitch type is not enough. For video classification, the model also needs clean visual evidence of the thing it is supposed to learn. In this case, the useful signal was the ball trajectory, and that signal was extremely small compared with everything else in the frame.

## Suggested Closing Point

The project started as a pitch classification task, but it became an experiment in whether broadcast baseball footage contains enough visual information for a small computer vision model to distinguish fastballs from off-speed pitches. Our conclusion was that the signal was probably too weak for the amount and quality of labeled data we had.
