# Brief 4: Modeling Experiments

## Whole Project Summary

The goal of this project was to classify baseball pitches from broadcast video clips as either **fastball** or **off-speed**. The original clips came from the MLB-YouTube dataset, which contains short broadcast segments from the 2017 MLB postseason.

The project became more than just training a model. A major part of the work was building a computer vision pipeline that could turn messy broadcast footage into usable training data. The final pipeline had four main stages:

1. Download short MLB video clips.
2. Detect which frames actually show the pitch camera.
3. Manually label the release and catch frames for each pitch.
4. Train and evaluate binary classification models.

The final result was a failed-but-rigorous experiment: the models did not reliably beat a simple baseline on the test set, but the project produced useful evidence about why this task is difficult with broadcast video.

## Your Section: Modeling Experiments

Your job is to explain what models and input formats were tried, and what the results showed.

The final modeling task was binary classification:

- `fastball`
- `offspeed`

The input to the model was a short sequence of 12 frames from release to catch. The frames were cropped to focus on the pitcher-to-plate action zone.

## Input Variants Tried

Several versions of the video sequences were tested:

- **RGB**: normal cropped color frames.
- **Diff**: frame differences that highlight motion.
- **RGB + diff**: color frames combined with motion information.
- **Ball-motion variants**: experimental preprocessing that tried to emphasize the baseball.

The idea was to see whether motion-focused inputs would help the model detect pitch speed or movement better than raw color frames.

## Model Types Tried

The project tested several model approaches:

- A frame-based CNN that processes each frame and pools information across time.
- A small 3D CNN designed to learn motion across frames.
- A transfer-learning baseline using a pretrained ResNet-18 frame encoder.

The transfer-learning model was included because pretrained image models can sometimes help when the dataset is small.

## Results

The results were not strong enough to claim successful pitch classification.

One CPU run using the diff input showed some validation promise, but test performance was close to or below a simple majority-class baseline. The model did not reliably generalize to unseen test clips.

A key baseline was predicting the most common class, fastball. Since the test set had 30 fastballs out of 53 examples, a fastball-only baseline would get about **56.6% test accuracy**. A useful model should clearly beat that while also identifying off-speed pitches. The tested models did not do that reliably.

## Important Point to Communicate

The model did not simply fail because of one bad training run. Several input formats and model types were tried. The broader pattern was that the model could sometimes fit or partially learn the validation data, but it did not show reliable test-set performance.

This suggests the core issue was probably the data signal, not just the model architecture.

## Suggested Closing Point

The modeling experiments were useful because they tested whether the cleaned video sequences contained enough information for classification. The answer appeared to be no, at least with this dataset size, video quality, and camera angle.
