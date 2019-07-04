#ifndef BLASSIFY_LAYER_H
#define BLASSIFY_LAYER_H

#include "darknet.h"
#include "layer.h"
#include "network.h"

layer make_blassify_layer(int batch, int w, int h, ACTIVATION activation);
void forward_blassify_layer(const layer l, network net);
void backward_blassify_layer(const layer l, network net);

#ifdef GPU
void forward_blassify_layer_gpu(const layer l, network net);
void backward_blassify_layer_gpu(layer l, network net);
#endif

#endif
