# emotions-gcn

Implementing Graph Convolutional Networks (GCNs) for emotion classification. Original graphs are made using the Delaunay method to convert face-landmarks into triangular mesh. Prior study showed that only some landmarks are relevant for emotion classification so we do not select the others.
Moreover, we propose a preprocessing pipeline to make our model robust to face size variations and implemented data-augmentation (rotation, flip, face stretching, +gaussian noise).

All neural networks blocks and models can be found in the `src/models` folder whereas preprocessing and datasets are located in `src/data`.
The model reach 0.657 accuracy on the validation set. 
