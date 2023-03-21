# taichi-2d-vof
A single file VOF fluid solver implementation in Taichi.

![VOF blog](https://user-images.githubusercontent.com/2747993/225257322-8b55cf4e-16fa-4801-912d-8f1eb89a93c5.gif)

Simply install Taichi and run!
```bash
$ python3 -m pip install taichi
$ python3 2dvof.py
```

In the GUI window, you can press **SPACE** to switch between visualization methods, or press **q** on your keyboard to terminate the simulation.

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

During the operation, press **SPACE** to switch visualization method. Currently,
the script can display
1. VOF field
2. U velocity
3. V velocity
4. Velocity norm
5. Velocity vectors (Implemented in `flow_visualization.py` module)

![display](https://user-images.githubusercontent.com/2747993/226554195-cd767de2-f386-46aa-8be7-00f1ed0c7f7a.png)

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
1. Volume of Fluid (VOF) Method for the Dynamics of Free Boundaries, C. W. Hirt and B. D. Nichols
2. Direct Numerical Simulations of Gas–Liquid Multiphase Flows, Grétar Tryggvason, Ruben Scardovelli,  Stéphane Zaleski
3. Fully Multidimensional Flux-Corrected Transport Algorithms for Fluids, Steven T. Zalesak
4. Volume-Tracking Methods For Interfacial Flow Calculations, Murray Rudman
5. A Continuum Method for Modeling Surface Tension, J.U. Brackbill, D.B. Kothe, C. Zemach


## Acknowledgement
The code in this repository is jointly developed by @houkensjtu and @zju-zhoucl. The original early version can be found
in @zju-zhoucl's repo [here](https://github.com/zju-zhoucl/taichi_VOF).
