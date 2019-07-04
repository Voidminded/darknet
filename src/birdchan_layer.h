#ifndef BIRDCHAN_LAYER_H
#define BIRDCHAN_LAYER_H

#include "darknet.h"
#include "layer.h"
#include "network.h"

char *get_birdChan_string(int c);
layer make_birdchan_layer(int batch, int w, int h, int n, int* channels, float scale, ACTIVATION activation, COST_TYPE cost);
void forward_birdchan_layer(const layer l, network net);
void backward_birdchan_layer(const layer l, network net);

#ifdef GPU
void forward_birdchan_layer_gpu(const layer l, network net);
void backward_birdchan_layer_gpu(layer l, network net);
#endif

#endif
