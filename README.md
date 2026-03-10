# Browser Eye Navigation

This project explores browser tab selection from eye tracking and click-confirmed intent. The core training signal is a stream of transactions where the user looks at a browser tab and then clicks it, with the click recorded in full-monitor coordinates alongside gaze, eye-state, and head-pose data.

# Eye tracking technology that we will train and run in production

We will train and run production on a **webcam-based desktop eye tracking stack**. The stack uses a monitor-facing RGB camera and software that estimates:

- gaze direction mapped into monitor coordinates
- eye landmarks and eye-state features
- head position and head rotation relative to the screen

This technology is a good fit for the target training data because the label is not just "what the eyes looked like," but "which browser tab was selected at this monitor position." A webcam pipeline can be calibrated against the active monitor, then used to continuously estimate where the user is looking on the screen while also capturing posture and head orientation.

The production approach is:

1. Calibrate the webcam tracker to the active monitor so gaze estimates are expressed in monitor coordinates.
2. Continuously estimate gaze position, eye features, and head pose while the user browses.
3. Build a short rolling pre-click window of these features.
4. Run a model that scores visible browser tabs by intent likelihood.
5. Use confidence thresholds so the system only acts when the prediction is strong enough.

The production model should learn from both static and dynamic signals:

- current gaze point on the monitor
- short gaze trajectory before the click
- fixation stability
- head translation and rotation
- relative position of the click that confirms the chosen tab

# Eye tracking technology that we will use to collect training data

We will also use a **webcam-based desktop eye tracking stack** to collect training data over the coming days. Using the same sensing modality for collection and production keeps the data distribution aligned and avoids training on hardware that is more precise than the hardware available at runtime.

The data collection approach is:

1. Run the calibrated webcam tracker during normal browser use.
2. Detect browser tab clicks and record them as labeled intent transactions.
3. Store the click in full-monitor coordinates together with the corresponding tab identity.
4. Attach gaze, eye-state, and head-pose samples from a short time window leading up to the click.
5. Repeat this across multiple sessions and days so the dataset captures natural variation in posture, seating distance, lighting, and fatigue.

Each collected transaction should include at least:

- `timestamp`
- `monitor_width`, `monitor_height`
- `click_x`, `click_y`
- `clicked_tab_id` or `clicked_tab_index`
- `gaze_x`, `gaze_y`
- eye feature bundle for both eyes
- `head_position_xyz`
- `head_rotation_pitch_yaw_roll`
- a short pre-click sample window, such as 300-800 ms

This collection setup gives us a supervised dataset where the clicked tab is the label, and the gaze plus head/eye context is the predictive signal.

# Process

## Data collection

Start by collecting real browser usage sessions with the webcam tracker enabled. The system should log only transactions where a tab click occurs, because those clicks provide the explicit label for user intent. Each transaction should include the selected tab, the click position on the full monitor, and the short sequence of gaze and head-pose samples that led to that selection.

The collection phase should run for several days so the data includes:

- different seating positions
- different head angles
- lighting changes
- variations in fatigue and focus
- different browser layouts and tab counts

## Training

Use the collected transactions to train a supervised tab-intent model. The model input is the pre-click gaze, eye, and head-pose context, optionally combined with monitor geometry and click context. The target is the clicked tab.

The preferred training framing is tab ranking or tab classification rather than raw coordinate regression, because the product goal is to choose among visible browser tabs.

## Testing

Evaluate the model on held-out sessions collected on different days from the training data. Testing should measure:

- top-1 tab prediction accuracy
- top-k accuracy
- robustness to posture changes
- robustness to lighting changes
- robustness to different tab counts
- calibration drift sensitivity

Testing should also check failure behavior, especially whether the system correctly falls back to "no action" when confidence is low.

## Production run

Once the model is validated, run the same webcam-based tracking stack in production. Keep the calibration step, continuous gaze and head-pose estimation, and tab scoring pipeline aligned with the collection setup. This reduces train-production mismatch and makes the live predictions reflect the same feature space used during training.

In production, the system should:

- estimate the current intended tab from recent gaze and head dynamics
- avoid switching when confidence is weak
- remain stable under normal posture changes
- be easy to recalibrate when monitor position or user setup changes
