Usage:
1. Take many images of sphere pressed against the sensor at different locations.
2. detect_circles_app.py -> Tag the locations of the circles. Make sure to update R_MAX according to your sphere size.
3. calculate_normal_maps.py -> calculate the normal maps for training.
3. pixel_mlp.py and pixel_dataset.py are the helper files for training.
4. train_mlp.py will train the network. pixel_mlp_normals.py is the trained network.
5. fullframe_height.py calculates the height for only one frame.
6. video_predict.py analyzes a video file or a camera and shows the 3d map.
7. visualize3d is a helper function for video_predict.py.