# taichi-2d-vof
A single file VOF fluid solver implementation in Taichi.

![VOF blog](https://user-images.githubusercontent.com/2747993/225257322-8b55cf4e-16fa-4801-912d-8f1eb89a93c5.gif)

## Usage
You can execute the script with the following arguments to control its behavior:
```bash
$ python3 2dvof.py -ic 1  # ic stands for Initial Condition; Default value is 1
$ python3 2dvof.py -s  # Save png files to the ./output/ folder
```
Currently, there are three types of initial condition settings:
1. Dam break
2. Rising bubble
3. Droping liquid

You can also tweak the `set_init_F()` kernel to implement your own. Be aware
that the solver might become unstable under various settings. The material
properties are carefully chosed to produce best results.

During the operation, press SPACE to switch visualization method. Currently,
the script can display
1. VOF field
2. U velocity
3. V velocity
4. Velocity norm

You can follow the steps below if you wish to output a video file:
```bash
$ python3 2dvof.py -ic 1 -s  # Add -s argument to enable saving
$ cd output
$ ti video  # Use Taichi's video utility to generate a mp4 file
$ ti gif -i video.mp4  # Use Taichi's gif utility to generate a gif file
```

## Implementation
WIP.

## References
WIP.
