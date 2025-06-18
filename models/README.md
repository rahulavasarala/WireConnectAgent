### Object Asset Generation

To generate new objects for Mujoco:
- Get .obj file of the mesh 
- Run obj2mjcf (https://github.com/kevinzakka/obj2mjcf), for example:

    obj2mjcf --obj-dir . --coacd-args.threshold 0.01 --decompose --save-mjcf
    - This runs a convex decomposition for the obj file for collision, where the threshold argument is set to the finest setting (can play around with this, and verify in Blender)
- Add the xml file to the scene xml