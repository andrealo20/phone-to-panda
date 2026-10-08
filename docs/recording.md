# Recording protocol

All data in this repo was recorded with one phone, by hand, following the steps
below. The target task is the LIBERO-Goal task *put the bowl on the plate*, so
the props are a bowl and a plate from my kitchen.

## Setup

- A tablet lying flat on the table shows `docs/board/board.html` full screen,
  brightness at maximum, auto-lock off. The board defines the table frame. It
  sits on the far side of the workspace, so that the hand reaching for the
  objects rarely covers it; the bowl and the plate sit on the table between
  me and the tablet, never on top of it.
- Measurements, taken once with a ruler and written to
  `data/raw/<session>/session.json` (template in `configs/session.example.json`):
  the side of one black-and-white checker square on the screen (`square_mm`,
  measured across the 7 squares of the long side and divided by 7), the screen's
  height above the table, i.e. the tablet thickness (`board_height_mm`), the
  bowl's height and the plate's height.
- Phone camera: landscape, main lens at 1x, 1080p, 30 fps, **video stabilisation off**
  (electronic stabilisation crops and warps every frame differently, which
  breaks the fixed-intrinsics camera model), HDR off, focus locked if the
  camera app allows it. The same settings are used for calibration and demos.
- The phone is held in the left hand near eye height, looking down at the
  table at roughly 45 degrees. It moves during the clip; that is expected,
  since the board gives the camera pose at every frame.

## Calibration clip (once per session)

One 30 second clip, no hands in view: slowly move the phone around the board,
at different distances (20 to 60 cm) and tilts (straight down to about 60
degrees), so that the board covers different parts of the image, corners
included.

## Demonstrations

One clip per demonstration, right hand only.

1. Hand in view from the start of the recording.
2. Reach the bowl and pinch its rim between thumb and index finger, the
   closest human grasp to a parallel jaw gripper. Stay still on the rim for
   about half a second before lifting.
3. Lift, carry, and set the bowl down on the plate. Stay still for about half
   a second before letting go.
4. Open the fingers and move the hand back up and away.

The two pauses matter: grasp and release are found from the motion (the bowl
only moves while it is held), as the last two places where the hand stops.
The first three clips were recorded before this step was written down; they
have pauses too, just shorter ones.

Between clips, move the bowl and the plate to new positions within a roughly
40 x 30 cm area beside the board, 15 to 35 cm apart, and vary which side of
the bowl gets pinched. The hand should not pass over the board, and the
bowl never goes on it: the board gives the camera pose, and losing it at the
release loses the release.

File naming: `data/raw/<session>/calib.mp4` and
`data/raw/<session>/demo_000.mp4`, `demo_001.mp4`, ...
