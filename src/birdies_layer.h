#ifndef BIRDIES_LAYER_H
#define BIRDIES_LAYER_H

#include "darknet.h"
#include "layer.h"
#include "network.h"

layer make_birdies_layer(int batch, int w, int h);
void forward_birdies_layer(const layer l, network net);
void backward_birdies_layer(const layer l, network net);

#ifdef GPU
void forward_birdies_layer_gpu(const layer l, network net);
void backward_birdies_layer_gpu(layer l, network net);
#endif

#endif
