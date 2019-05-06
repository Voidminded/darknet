#include "birdies_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

layer make_birdies_layer(int batch, int w, int h)
{
    layer l = {0};
    l.type = BIRDIES;

    l.h = h;
    l.w = w;
    l.c = 12;
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

    l.forward = forward_birdies_layer;
    l.backward = backward_birdies_layer;
#ifdef GPU
    l.forward_gpu = forward_birdies_layer_gpu;
    l.backward_gpu = backward_birdies_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
    l.loss_gpu = cuda_make_array(l.loss, batch*l.outputs);
#endif

    fprintf(stderr, "birdies\n");
    srand(0);

    return l;
}

void forward_birdies_layer(const layer l, network net)
{
    //double time = what_time_is_it_now();
    int i,b,k;
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));
    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    memset(l.loss, 0, l.outputs * l.batch * sizeof(float));
    memset(l.scales, 0, l.w * l.h * sizeof(float));

#ifndef GPU
    for (b = 0; b < l.batch; ++b){
        for(i = 0; i < l.c; ++i){
            int index = b*l.outputs + i*l.w*l.h;
            if( i<4) // for Birdness, Spicies, and Depth
                activate_array(l.output + index, l.w*l.h, LOGISTIC);
            else
                activate_array(l.output + index, l.w*l.h, TANH);
        }
    }
#endif

    float *mse = calloc( l.c, sizeof( float));
    for (b = 0; b < l.batch; ++b){
        // Birdness thingy
        l1_cpu( l.w*l.h, l.output + b*l.outputs, net.truth + b*l.outputs, l.delta + b*l.outputs, l.loss + b*l.outputs);
        float sum = 0;
        int count;
        mse[0] += sum_array(l.loss + b*l.outputs, l.w*l.h)/(l.w*l.h);
        memcpy( l.scales, net.truth + b*l.outputs, l.w*l.h*sizeof( float));
        // Other predictions
        for(i = 1; i < l.c; ++i){
            sum = 0;
            count = 0;
            for(k = 0; k < l.w*l.h; ++k){
                if( l.scales[k] > 0.5)//mask is bird
                {
                    int index = b*l.outputs + i*l.w*l.h + k;
                    if( i < 3) // For scpicies
                       l2_cpu( 1, l.output + index, net.truth + index, l.delta + index, l.loss + index);
                    else if( i < 6)
                        smooth_l1_cpu( 1, l.output + index, net.truth + index, l.delta + index, l.loss + index);
                    else
                        l2_cpu( 1, l.output + index, net.truth + index, l.delta + index, l.loss + index);
                       // l.delta[index] = net.truth[ index] - l.output[index];
                    count++;
                    sum += l.loss[ index];
                }
            }
            mse[i] += sum/count;
        }
    }
    printf("Birdness:%15.12f Jackdaw:%15.12f Rook:%15.12f Dx:%15.12f Dy:%15.12f Dz:%15.12f Qx:%15.12f Qy:%15.12f Qz:%15.12f Qw:%15.12f WB(sin):%15.12f WB(cos):%15.12f \n", mse[0], mse[1], mse[2], mse[4], mse[5], mse[3], mse[6], mse[7], mse[8], mse[9], mse[10], mse[11]);
    free(mse);

    *(l.cost) = sum_array(l.loss, l.batch*l.inputs); //pow(mag_array(l.delta, l.outputs * l.batch), 2);
    //printf("took %lf sec\n", what_time_is_it_now() - time);
}

void backward_birdies_layer(const layer l, network net)
{
    axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
}

#ifdef GPU

void forward_birdies_layer_gpu(const layer l, network net)
{
    copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
    int b, i;
    for (b = 0; b < l.batch; ++b){
        for(i = 0; i < l.c; ++i){
            int index = b*l.outputs + i*l.w*l.h;
            if( i<4) // for Birdness, Spicies, and Depth
                activate_array_gpu(l.output_gpu + index, l.w*l.h, LOGISTIC);
            else
                activate_array_gpu(l.output_gpu + index, l.w*l.h, TANH);
        }
    }

    cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);
    forward_birdies_layer(l, net);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
    cuda_push_array(l.loss_gpu, l.loss, l.batch*l.outputs);
//    l2_gpu(l.batch*l.inputs, l.output_gpu, net.truth_gpu, l.delta_gpu, l.loss_gpu);
//    cuda_pull_array(l.loss_gpu, l.loss, l.batch*l.inputs);
//    *(l.cost) = sum_array(l.loss, l.batch*l.inputs);
}

void backward_birdies_layer_gpu(const layer l, network net)
{
//    int b;
//    for (b = 0; b < l.batch; ++b){
//        //if(l.extra) gradient_array_gpu(l.output_gpu + b*l.outputs + l.classes*l.w*l.h, l.extra*l.w*l.h, LOGISTIC, l.delta_gpu + b*l.outputs + l.classes*l.w*l.h);
//    }
    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif

