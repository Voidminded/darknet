#include "blassify_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

layer make_blassify_layer(int batch, int w, int h, ACTIVATION activation)
{
    layer l = {0};
    l.type = BLASSIFY;

    l.h = h;
    l.w = w;
    l.c = 3;//12;
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.batch = batch;
    l.cost = calloc(1, sizeof(float));
    l.outputs = h*w*l.c;
    l.inputs = l.outputs;
    l.truths = l.w*l.h*l.c;
    l.delta = calloc(batch*l.outputs, sizeof(float));
    l.loss = calloc(batch*l.outputs, sizeof(float));
    l.output = calloc(batch*l.outputs, sizeof(float));
    l.scales = calloc(l.h*l.w, sizeof(float));

    l.activation = activation;
    l.forward = forward_blassify_layer;
    l.backward = backward_blassify_layer;
    l.state = calloc(3, sizeof(float));
#ifdef GPU
    l.forward_gpu = forward_blassify_layer_gpu;
    l.backward_gpu = backward_blassify_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
    l.loss_gpu = cuda_make_array(l.loss, batch*l.outputs);
#endif

    fprintf(stderr, "Blassify\n");
    srand(0);

    return l;
}

void forward_blassify_layer(const layer l, network net)
{
    //double time = what_time_is_it_now();
    int i,b,k;
#ifndef GPU
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));
    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    memset(l.loss, 0, l.outputs * l.batch * sizeof(float));
    memset(l.scales, 0, l.w * l.h * sizeof(float));

    activate_array(l.output, l.outputs*l.batch, l.activation);
#endif

    if(!net.truth)
      return;
    for (b = 0; b < l.batch; ++b){
#ifndef GPU
        l2_cpu( l.w*l.h, l.output + b*l.outputs, net.truth + b*net.outputs, l.delta + b*l.outputs, l.loss + b*l.outputs);
        memcpy( l.scales, net.truth + b*net.outputs, l.w*l.h*sizeof( float));
        for(i = 1; i < l.c; ++i){
            for(k = 0; k < l.w*l.h; ++k){
                if( l.scales[k] > 0.5)//mask is bird 
                {
                    int index = b*l.outputs + i*l.w*l.h + k;
                  l2_cpu( 1, l.output + index, net.truth + b*net.outputs + i*l.w*l.h + k, l.delta + index, l.loss + index);
                }
            }
        }
#endif
        l.state[0] += sum_array(l.loss+b*l.outputs, l.w*l.h); 
        l.state[1] += sum_array(l.loss+l.w*l.h+b*l.outputs, l.w*l.h);
        l.state[2] += sum_array(l.loss+2*l.w*l.h+b*l.outputs, l.w*l.h);
    }
    if(((*net.seen)/net.batch)%net.subdivisions == 0){
        printf("Blassify\tLoss: %12.6f  Birdness:%12.6f  Jackdaw:%12.6f  Rook:%12.6f\n", sum_array(l.state,l.c)/(net.batch*net.subdivisions), l.state[0]/(net.batch*net.subdivisions), l.state[1]/(net.batch*net.subdivisions), l.state[2]/(net.batch*net.subdivisions));
        fprintf( net.log, "%f,%f,%f", l.state[0]/(net.batch*net.subdivisions), l.state[1]/(net.batch*net.subdivisions), l.state[2]/(net.batch*net.subdivisions));
        memset(l.state, 0, 3 * sizeof( float));
    }

    *(l.cost) = sum_array(l.loss, l.batch*l.inputs); //pow(mag_array(l.delta, l.outputs * l.batch), 2);

}

void backward_blassify_layer(const layer l, network net)
{
    axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
}

#ifdef GPU

void forward_blassify_layer_gpu(const layer l, network net)
{
    copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
    l2_gpu(l.batch*l.inputs, l.output_gpu, net.truth_gpu, l.delta_gpu, l.loss_gpu);
    int b;
    for (b = 0; b < l.batch; ++b)
    {
      mul_gpu( l.w*l.h, b * net.outputs + net.truth_gpu, 1, b * l.outputs + l.loss_gpu+l.w*l.h, 1);
      mul_gpu( l.w*l.h, b * net.outputs + net.truth_gpu, 1, b * l.outputs + l.loss_gpu+2*(l.w*l.h), 1);
      mul_gpu( l.w*l.h, b * net.outputs + net.truth_gpu, 1, b * l.outputs + l.delta_gpu+l.w*l.h, 1);
      mul_gpu( l.w*l.h, b * net.outputs + net.truth_gpu, 1, b * l.outputs + l.delta_gpu+2*(l.w*l.h), 1);
    }
    //cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);
    cuda_pull_array(l.loss_gpu, l.loss, l.batch*l.outputs);
   // cuda_pull_array(l.delta_gpu, l.delta, l.batch*l.outputs);
    forward_blassify_layer(l, net);
   // cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
    //cuda_push_array(l.loss_gpu, l.loss, l.batch*l.outputs);
//    l2_gpu(l.batch*l.inputs, l.output_gpu, net.truth_gpu, l.delta_gpu, l.loss_gpu);
//    cuda_pull_array(l.loss_gpu, l.loss, l.batch*l.inputs);
//    *(l.cost) = sum_array(l.loss, l.batch*l.inputs);
}

void backward_blassify_layer_gpu(const layer l, network net)
{
//    int b;
//    for (b = 0; b < l.batch; ++b){
//        //if(l.extra) gradient_array_gpu(l.output_gpu + b*l.outputs + l.classes*l.w*l.h, l.extra*l.w*l.h, LOGISTIC, l.delta_gpu + b*l.outputs + l.classes*l.w*l.h);
//    }
    
    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif

